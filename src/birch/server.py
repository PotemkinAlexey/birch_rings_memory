"""BirchKM MCP server — exposes memory tools to Claude agents."""
from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP

from .memory_store import MemoryStore
from .resonance.embeddings import EmbeddingError

_DB_PATH = os.environ.get("BIRCH_DB", str(Path.home() / ".birch" / "memory.db"))
Path(_DB_PATH).parent.mkdir(parents=True, exist_ok=True)

_store = MemoryStore(db_path=_DB_PATH)

mcp = FastMCP("BirchKM")


def _embedding_error_response(exc: EmbeddingError) -> dict:
    """Wrap an EmbeddingError as the structured failure shape MCP tools
    return when the embedding provider is unreachable / misconfigured.

    Symmetric with forecast_memory's DimensionMismatchError wrapper:
    agent gets ``{"ok": False, "error": ..., "detail", "hint"}`` instead
    of a raw stacktrace, and the hint points at the three knobs the
    user actually controls.
    """
    return {
        "ok": False,
        "error": "embedding_provider_unavailable",
        "detail": str(exc),
        "hint": (
            "Start Ollama, set BIRCH_EMBED_MODEL to a model the provider "
            "knows, or set BIRCH_EMBED_PROVIDER=mock for offline use."
        ),
    }


@mcp.tool()
def query_memory(
    text: str,
    top_k: int = 5,
    session_id: Optional[str] = None,
    min_similarity: float = 0.0,
    layers: Optional[list[str]] = None,
    subject_prefix: Optional[str] = None,
    min_gravity: float = 0.0,
) -> dict:
    """USE WHEN: looking up facts relevant to a user's first message or to a
    sub-question mid-session. Pass ``session_id`` so retrieved facts are
    attributed to the open session (their gravity rises if the session
    resonates, falls if it goes toxic).

    Returns ranked hits plus ``conflicts`` — any (subject, predicate) that
    has more than one live candidate among the hits, with a ``recommended_id``
    by gravity. Use that to spot "two competing HEAD values" cases.

    Filters: ``subject_prefix`` (case-insensitive prefix on subject — startswith),
    ``min_gravity`` (drop low-confidence facts), ``layers`` (any of
    ``surface``/``kinetic``/``core``), ``min_similarity`` (cosine floor).
    Deprecated / expired facts are never returned (they live in the
    singularity, not the live layers).

    source values:
      ``surface`` / ``kinetic`` / ``core`` — live FactPassport layers
      ``hawking``      — single fact recovered from the black hole
      ``hawking_meta`` — MetaFact bundle recovered from the black hole
    """
    # Bounds: reject non-positive / oversized top_k with a structured
    # response. _store.query handles top_k=0 internally now (round 9),
    # but the MCP contract should be explicit: an agent that asks for
    # zero hits gets told why, not silently empty results.
    if top_k <= 0:
        return {"results": [], "error": "invalid_top_k",
                "_hint": "top_k must be positive"}
    requested_top_k = top_k
    if top_k > 50:
        top_k = 50
    layer_map = {"surface": 0, "kinetic": 1, "core": 2}
    if layers:
        # Validate enum: a typo like "surfase" used to silently produce an
        # empty allowed set, returning zero results with no explanation.
        # Now we reject the call with a structured error the agent can
        # parse — much better MCP UX than an empty list lie.
        unknown = [name for name in layers if name not in layer_map]
        if unknown:
            return {
                "results": [],
                "error": "unknown_layer",
                "unknown_layers": unknown,
                "allowed_layers": list(layer_map),
            }
        # Build an explicit set so layers=["surface", "core"] excludes
        # kinetic — a range filter would silently include it.
        allowed = {layer_map[name] for name in layers}
    else:
        allowed = None
    try:
        results = _store.query(
            text,
            top_k=top_k,
            hawking=True,
            session_id=session_id,
            min_similarity=min_similarity,
            subject_prefix=subject_prefix,
            min_gravity=min_gravity,
            allowed_layers=allowed,
        )
    except EmbeddingError as exc:
        return _embedding_error_response(exc)
    hits = [r.to_mcp_dict() for r in results]

    # conflict_hints: same (subject, predicate) with different objects.
    conflicts: list[dict] = []
    by_slot: dict[tuple[str, str], list[dict]] = {}
    for h in hits:
        if h.get("kind") != "fact":
            continue
        s = (h.get("subject") or "").strip().lower()
        p = (h.get("predicate") or "").strip().lower()
        by_slot.setdefault((s, p), []).append(h)
    for (_, _), group in by_slot.items():
        if len(group) <= 1:
            continue
        sorted_group = sorted(
            group,
            key=lambda x: x.get("gravity_score", 0.0),
            reverse=True,
        )
        conflicts.append({
            "subject": sorted_group[0].get("subject"),
            "predicate": sorted_group[0].get("predicate"),
            "candidates": [
                {
                    "fact_id": x.get("fact_id"),
                    "object": x.get("object"),
                    "gravity_score": x.get("gravity_score"),
                    "similarity": x.get("similarity"),
                }
                for x in sorted_group
            ],
            "recommended_id": sorted_group[0].get("fact_id"),
            "_hint": "consider set_fact(...) or supersede_fact(loser_id, recommended_id)",
        })

    if not session_id:
        hint = (
            "pass session_id to attribute these reads to a session "
            "so gravity updates on session_close"
        )
    else:
        hint = "call session_close when the conversation ends to propagate resonance to gravity"
    response: dict = {"results": hits, "_hint": hint,
                      "effective_top_k": top_k}
    if requested_top_k != top_k:
        response["_warning"] = (
            f"top_k capped at {top_k} (requested {requested_top_k})"
        )
    if conflicts:
        response["conflicts"] = conflicts
    return response


@mcp.tool()
def record_fact(
    subject: str,
    predicate: str,
    object: str,
    session_id: Optional[str] = None,
) -> dict:
    """USE WHEN: storing a new atomic SPO triple where the (subject, predicate)
    can legitimately carry several objects (e.g. "api uses Postgres" AND
    "api uses Redis"). For one-canonical-value slots — HEADs, versions,
    counts — use ``set_fact`` instead so old values auto-supersede.

    Identical triples (case-insensitive, whitespace-normalised) are deduplicated:
    the existing fact is touched and returned with ``already_existed=true``.

    The response includes ``similar_existing`` — paraphrase candidates already
    in the store at cosine ≥ 0.85 (excluding this fact). Treat them as
    "consider supersede_fact or set_fact"; the dedup in this call only catches
    exact normalised SPO matches, not "X uses Postgres" vs "X is on Postgres".
    """
    # Transaction-honest: add_fact returns created from inside its own
    # write txn, so there's no race window between a fact_exists probe and
    # the insert (same pattern set_fact uses since round 5).
    try:
        fact, created = _store.add_fact(
            subject, predicate, object,
            session_id=session_id, return_status=True,
        )
    except EmbeddingError as exc:
        return _embedding_error_response(exc)
    already_existed = not created
    similar: list[dict] = []
    if created and fact.vector:
        similar = _store._find_similar_by_vector(
            fact.vector,
            top_k=3,
            min_similarity=0.85,
            exclude_ids={fact.fact_id},
        )
    if not session_id:
        hint = (
            "open a session with session_open and pass session_id here "
            "so gravity updates when the session closes"
        )
    else:
        hint = "call session_close when the conversation ends to propagate resonance to gravity"
    return {
        "fact_id": fact.fact_id,
        "already_existed": already_existed,
        "layer": fact.layer,
        "gravity_score": round(fact.gravity_score, 3),
        "similar_existing": similar,
        "_hint": hint,
    }


@mcp.tool()
def record_facts(
    facts: list[dict],
    session_id: Optional[str] = None,
) -> dict:
    """USE WHEN: storing several SPO triples at once — one Ollama round-trip
    and one SQLite transaction beats N ``record_fact`` calls. Same semantics
    per item as ``record_fact``; exact-SPO duplicates are touched and
    returned with ``already_existed=true``. For mutable-scalar slots use
    ``set_fact`` per item instead.

    Each item must have ``subject``, ``predicate``, ``object``; per-item
    ``session_id`` overrides the top-level one.
    """
    # Per-item validation: a malformed entry used to raise a raw KeyError
    # straight through MCP. Now we collect the issues and return a typed
    # response the agent can parse and fix in one round trip.
    required = ("subject", "predicate", "object")
    invalid: list[dict] = []
    for i, f in enumerate(facts):
        if not isinstance(f, dict):
            invalid.append({
                "index": i,
                "error": "item_not_an_object",
                "got_type": type(f).__name__,
            })
            continue
        missing = [k for k in required if k not in f or f[k] in (None, "")]
        if missing:
            invalid.append({"index": i, "missing": missing})
            continue
        # Type validation: round-9 caught missing/None/""; round 11
        # catches the next layer — subject=123 / predicate=[] / object={}
        # all pass the presence check but break embedding-text formatting
        # and SPO normalisation downstream. Triples must be strings, and
        # whitespace-only strings count as empty.
        bad_type = [
            k for k in required
            if not isinstance(f[k], str) or not f[k].strip()
        ]
        if bad_type:
            invalid.append({
                "index": i,
                "error": "invalid_field_type",
                "bad_fields": bad_type,
                "got_types": {
                    k: type(f[k]).__name__ for k in bad_type
                },
            })
    if invalid:
        return {
            "ok": False,
            "error": "invalid_fact_item",
            "invalid": invalid,
            "required_fields": list(required),
        }

    triples = [
        (f["subject"], f["predicate"], f["object"])
        for f in facts
    ]
    # Per-item session_id overrides the top-level one — honour the contract
    # spelled out in the docstring. Items without their own session_id fall
    # back to the top-level argument.
    per_item_sids = [f.get("session_id") for f in facts]
    try:
        statuses = _store.add_facts(
            triples,
            session_id=session_id,
            session_ids=per_item_sids,
            return_status=True,
        )
    except EmbeddingError as exc:
        return _embedding_error_response(exc)
    if not session_id:
        hint = (
            "open a session with session_open and pass session_id here "
            "so gravity updates when the session closes"
        )
    else:
        hint = "call session_close when the conversation ends to propagate resonance to gravity"
    return {
        "facts": [
            {
                "fact_id": s["fact"].fact_id,
                # True if the SPO was already in the store BEFORE this batch.
                "already_existed": s["already_existed"],
                # True if an earlier item in this same batch already created
                # the SPO — agents that send the same triple twice in one
                # call now see it instead of getting two clean inserts.
                "duplicate_in_batch": s["duplicate_in_batch"],
                # Direct "this call created the row" signal: True iff the
                # SPO was NOT in the store before this batch AND was NOT
                # introduced by an earlier item in the batch. Cheaper for
                # the agent than computing not already_existed and not
                # duplicate_in_batch on its own.
                "created": (
                    not s["already_existed"]
                    and not s["duplicate_in_batch"]
                ),
                "layer": s["fact"].layer,
                "gravity_score": round(s["fact"].gravity_score, 3),
            }
            for s in statuses
        ],
        "_hint": hint,
    }


@mcp.tool()
def set_fact(
    subject: str,
    predicate: str,
    object: str,
    session_id: Optional[str] = None,
) -> dict:
    """USE WHEN: a (subject, predicate) slot has one canonical value that
    *replaces* whatever was there before — HEADs, version strings, current
    counts, single-valued settings. Atomic upsert with auto-supersede.

    Records the new fact and supersedes every live fact that shares the same
    ``(subject, predicate)`` — old bodies land in the singularity with
    ``deprecated_by`` pointing at the new one (lineage preserved, MetaFact +
    Hawking still possible). This is the canonical write for mutable scalars;
    use ``record_fact`` instead when several ``object``s can legitimately
    coexist on the same (subject, predicate) — for example a service that
    "uses" both Postgres and Redis.

    Returns ``{"set": true, "fact_id", "already_existed", "superseded": [...]}``.
    """
    return _store.set_fact(subject, predicate, object, session_id=session_id)


@mcp.tool()
def supersede_fact(old_fact_id: str, new_fact_id: str) -> dict:
    """
    Mark a fact as superseded by a newer one — the canonical "we now know better".

    The old fact's deprecated_by pointer is set (lineage preserved), the SPO
    slot is freed for the new claim, and the body is sent to the singularity
    immediately. The row stays in storage, so the deprecated fact can still
    fuel singularity collapse (MetaFact compression) and be Hawking-emitted
    if a future query reopens the topic.

    Use this whenever you record a replacement fact for an older one — it is
    the right path for "stale / wrong / outdated" data. Do NOT use
    delete_fact for that; delete_fact is a destructive primitive for
    secrets and accidental writes.

    Returns {"superseded": true, "old_id", "new_id", "absorbed": [...]}.
    """
    return _store.supersede_fact(old_fact_id, new_fact_id)


@mcp.tool()
def retire_fact(fact_id: str) -> dict:
    """
    Send a no-longer-relevant fact to the singularity with no replacement.

    Use when the fact's topic is simply over (a project ended, a feature
    was removed) and there is no newer fact replacing it. The body is
    marked expired and absorbed into the black hole in the same call. The
    row stays in storage, so the fact can still feed singularity collapse
    and Hawking emission.

    Prefer supersede_fact when you DO have a replacement — that preserves
    the "we used to think X, now Y" lineage. Use delete_fact only for
    truly destructive removal.

    Returns {"retired": true, "fact_id", "absorbed": [...]}.
    """
    return _store.retire_fact(fact_id)


@mcp.tool()
def forecast_memory(horizon_ticks: int = 50) -> dict:
    """
    Run the galaxy forward and write a per-body stability prediction back.

    For every live body (FactPassport AND MetaFact — both carry
    ``forecast_stability``) the galaxy simulates an orbit around the
    central black hole. After ``horizon_ticks`` integrator steps each
    body has either survived (its orbital radius gives a stability
    score in [0, 1]) or crossed the event horizon (stability = 0).
    The score lands on body.forecast_stability and is consumed by the
    adaptive gravity formula via ``w_stability`` — bodies the galaxy
    predicts to fall earn a smaller gravity contribution, bodies
    predicted safe earn more.

    The feature complements the local pre-resonance signals (freshness,
    access, graph, recent_utility) with a forecast no local feature can
    produce: it sees the body's *future*. Whether the forecast helps is
    not assumed — ``w_stability`` is learned by the same SGD pass that
    learns the other adaptive weights, so a useless forecast just sits
    at its prior and contributes nothing.

    Call at session start (or once per day) on a real store. Pure numpy,
    O(n²) per step in body count.

    Returns a summary: how many bodies were forecasted and updated, with
    a per-type split (``facts_updated_count`` / ``metas_updated_count``)
    and a coarse distribution across {safe / kinetic / near_horizon /
    predicted_fall}. The ``facts_forecasted`` / ``facts_updated`` keys
    are kept as legacy aliases for wire-format stability but actually
    count BODIES (FactPassport + MetaFact). Prefer the ``bodies_*`` keys.
    On mixed embedding dimensions in the store (after a BIRCH_EMBED_MODEL
    swap without reindex) returns ``{"ok": false, "error":
    "mixed_embedding_dimensions", "hint": "...", "detail": "..."}`` instead
    of a raw exception so the agent gets actionable diagnostic.
    """
    from .vector_index import DimensionMismatchError

    try:
        return _store.run_forecast(horizon_ticks=horizon_ticks)
    except DimensionMismatchError as exc:
        return {
            "ok": False,
            "error": "mixed_embedding_dimensions",
            "hint": (
                "Store contains vectors of different sizes — likely the "
                "embedding model changed under it. Pin BIRCH_EMBED_MODEL "
                "or rebuild/reindex before running the forecast."
            ),
            "detail": str(exc),
        }


@mcp.tool()
def delete_fact(fact_id: str) -> dict:
    """
    Hard-delete a fact — destructive primitive, the data is GONE.

    Removes the fact from all live layers, the vector index, the SPO dedup
    index, AND storage. Does NOT send it to the black hole; there is no
    lineage, no MetaFact compression, no Hawking rescue. Use only for:

      - Secrets / sensitive data that must not exist (GDPR-style removal)
      - Facts you just recorded by accident in the same session

    For "stale / wrong / outdated" data, prefer supersede_fact (when there
    is a replacement) or retire_fact (when there is not) — both preserve
    the body in the singularity so the brain can still learn from it.

    Returns {"deleted": true} if found, {"deleted": false} if not found.
    """
    deleted = _store.delete_fact(fact_id)
    return {"deleted": deleted, "fact_id": fact_id}


@mcp.tool()
def list_facts(
    subject: Optional[str] = None,
    predicate: Optional[str] = None,
    limit: int = 50,
    subject_prefix: Optional[str] = None,
    min_gravity: float = 0.0,
    layer: Optional[str] = None,
    exclude_deprecated: bool = True,
) -> list[dict]:
    """USE WHEN: auditing what the store actually holds about a topic, no
    semantic query needed. Sorted by gravity descending; ``exclude_deprecated``
    defaults true so superseded / retired bodies don't pollute the list
    (those live in the singularity).

    Filters: ``subject`` / ``predicate`` (case-insensitive substring),
    ``subject_prefix`` (case-insensitive ``startswith`` on subject — use
    when you want all facts under "my-project"),
    ``min_gravity`` (drop low-confidence facts), ``layer`` (one of
    ``surface``/``kinetic``/``core``).
    """
    # Bounds: a zero/negative limit used to fall through to the
    # "len(out) >= limit" check below AFTER the first append, so the
    # caller still got one item back. Reject explicitly. Cap upper
    # bound so an agent asking for 1M facts can't pin the server.
    if limit <= 0:
        return []
    if limit > 500:
        # list_facts returns list[dict] of fact rows; injecting a warning
        # dict at index 0 would break every consumer that iterates. Log
        # the cap server-side instead — the agent can also detect the
        # cap by len(result) == 500 against its requested limit.
        import logging
        logging.getLogger(__name__).warning(
            "list_facts: limit capped at 500 (requested %d)", limit,
        )
        limit = 500
    layer_map = {"surface": 0, "kinetic": 1, "core": 2}
    # Validate enum: a typo used to silently produce target_layer=None,
    # which then returned ALL layers — worse than empty because the
    # agent thinks the filter applied. Now we return a structured
    # error sentinel so the agent sees what went wrong.
    if layer is not None and layer not in layer_map:
        return [{
            "error": "unknown_layer",
            "got": layer,
            "allowed": list(layer_map),
        }]
    target_layer = layer_map.get(layer) if layer else None
    facts = _store.list_facts(subject=subject, predicate=predicate, limit=10_000)
    prefix = subject_prefix.lower() if subject_prefix else None
    out: list[dict] = []
    for f in facts:
        if exclude_deprecated and (f.is_deprecated or f.is_expired):
            continue
        if target_layer is not None and f.layer != target_layer:
            continue
        if f.gravity_score < min_gravity:
            continue
        if prefix and not f.subject.lower().startswith(prefix):
            continue
        out.append({
            "fact_id": f.fact_id,
            "subject": f.subject,
            "predicate": f.predicate,
            "object": f.object,
            "layer": f.layer,
            "gravity_score": round(f.gravity_score, 3),
            "source": {0: "surface", 1: "kinetic", 2: "core"}.get(f.layer, "kinetic"),
        })
        if len(out) >= limit:
            break
    return out


@mcp.tool()
def find_similar(
    text: str,
    top_k: int = 5,
    min_similarity: float = 0.85,
    subject_prefix: Optional[str] = None,
) -> dict:
    """USE WHEN: hunting for paraphrase candidates before writing or to plan a
    ``set_fact`` / ``supersede_fact`` cleanup. Read-only; never mutates.

    Returns live (non-deprecated, non-expired) facts whose embedding cosine
    to ``text`` is at or above ``min_similarity``. Higher threshold = stricter.
    ``subject_prefix`` scopes by a case-insensitive ``startswith`` on subject.

    Typical flow: ``find_similar("HEAD on master", subject_prefix="my-project")``
    surfaces several stale HEAD entries; then ``set_fact(subject, "HEAD on
    master", new_value)`` collapses them in one call.
    """
    # Bounds: explicit MCP contract. Round 9 added a top_k<=0 guard
    # at VectorIndex.search; the server layer should surface that as
    # a structured warning so the agent learns instead of silently
    # getting nothing.
    if top_k <= 0:
        return {
            "query": text,
            "hits": [],
            "_warning": "top_k must be positive",
        }
    requested_top_k = top_k
    if top_k > 50:
        top_k = 50
    try:
        hits = _store.find_similar(
            text=text,
            top_k=top_k,
            min_similarity=min_similarity,
            subject_prefix=subject_prefix,
        )
    except EmbeddingError as exc:
        return _embedding_error_response(exc)
    response: dict = {
        "query": text,
        "min_similarity": min_similarity,
        "hits": hits,
        "effective_top_k": top_k,
        "_hint": (
            "use set_fact for slot-replace, supersede_fact(old, new) when the "
            "new fact is already recorded, retire_fact when no replacement"
        ),
    }
    if requested_top_k != top_k:
        response["_warning"] = (
            f"top_k capped at {top_k} (requested {requested_top_k})"
        )
    return response


@mcp.tool()
def explain_fact(fact_id: str) -> dict:
    """USE WHEN: a fact's gravity surprises you and you need to know why.

    Decomposes the gravity score into per-feature contributions
    (``freshness``, ``access``, ``graph``, ``recent_utility``,
    ``forecast_stability``, ``resonance``) using the current adaptive
    weights. Also reports ``is_deprecated`` / ``is_expired`` /
    ``deprecated_by`` so you can tell whether the fact is even live.
    Read-only debug; never mutates.
    """
    return _store.explain_fact(fact_id)


@mcp.tool()
def session_open(
    session_id: Optional[str] = None,
    agent_id: str = "default",
    first_message: Optional[str] = None,
    record_first_message: bool = True,
) -> dict:
    """USE WHEN: starting a conversation that will read or write memory and
    you want gravity feedback to land on the right facts. Always open a
    session before the first ``query_memory`` so retrieved facts get
    attributed to this session's outcome.

    Pass ``first_message`` (the user's opening text) to trigger echo
    detection on the same call: if the topic matches a past unresolved
    problem at cosine ≥ 0.68, the past session's R is pulled into toxic
    territory and the penalty propagates to the facts that misled it. The
    echo result is returned in the response under ``echo``.

    ``record_first_message`` (default True) also pushes the first message
    into the session's trajectory so the resonance engine sees it on close
    — same as if you had called ``session_push(first_message)`` immediately
    after. Set to False if you only want the echo check and plan to push
    the opening message separately.

    Returns ``session_id`` — pass it to every subsequent ``record_fact``,
    ``record_facts``, ``set_fact``, ``query_memory``, ``session_push``. Close
    with ``session_close`` when the conversation ends; that's when the
    resonance signal propagates to gravity. If ``session_id`` is omitted a
    unique one is generated.
    """
    sid = session_id or f"{agent_id}-{int(time.time())}-{uuid.uuid4().hex[:4]}"
    _store.session_start(sid)
    response: dict = {
        "session_id": sid,
        "_hint": (
            "pass this session_id to record_fact, record_facts, query_memory, "
            "and session_push — then call session_close when done"
        ),
    }
    if first_message:
        response["echo"] = _store.check_echo(first_message, session_id=sid)
        if record_first_message:
            # Push the opening message into the trajectory so the resonance
            # engine sees it on close. Without this, an agent that uses
            # session_open(first_message=...) loses the opening turn from
            # the semantic-shift and repetition signals.
            _store.session_message(first_message, session_id=sid)
    return response


@mcp.tool()
def check_echo(first_message: str, session_id: Optional[str] = None) -> dict:
    """USE WHEN: you want to know whether the user is returning to an
    unresolved problem before responding. Compares the first message against
    past K-means session bundles; on a hit at cosine ≥ 0.68 the past
    session's R is pulled into toxic territory and the penalty propagates to
    every fact the past session touched (scaled by per-fact relevance).

    Idempotent: a second echo on the same matched session returns
    ``penalty=0``. Usually called automatically via ``session_open(first_message=...)``;
    this tool is the explicit form when you want the check without opening a
    new session, or when you've already opened one and want a late check.
    """
    return _store.check_echo(first_message, session_id=session_id)


@mcp.tool()
def session_push(text: str, session_id: str) -> dict:
    """USE WHEN: a user message arrives during an open session. Push every
    user message so the resonance engine can score the conversation
    trajectory on close.

    Pass user-side text only. Do NOT push your own assistant responses —
    behavioural / semantic / repetition signals score user closure, not
    your replies.
    """
    _store.session_message(text, session_id=session_id)
    return {
        "session_id": session_id,
        "ok": True,
        "_hint": (
            "call session_close when the conversation ends to score resonance "
            "and update gravity"
        ),
    }


@mcp.tool()
def session_close(
    session_id: str,
    sentiment: Optional[str] = None,
    r_override: Optional[float] = None,
) -> dict:
    """USE WHEN: a conversation that wrote or read facts ends. Closes the
    session, scores R from the message trajectory (or from explicit
    ``sentiment`` / ``r_override``), propagates R to every touched fact's
    gravity, runs `_absorb_dead`, may trigger background singularity
    collapse. Call exactly once per opened session — repeated closes
    corrupt the resonance signal.

    Resonance scoring — three paths:
      - Default (neither arg set): heuristic over the message text
        (behavioural + semantic + repetition). The original contract.
        Works well for natural conversation; mis-classifies grumpy-
        sounding technical summaries ("stale snapshot", "failure
        mode", "no repeats") as toxic even when the session was a
        clean win — words look bad, context was good.
      - ``sentiment``: pass one of ``"resonant"`` / ``"neutral"`` /
        ``"toxic"`` (or aliases ``"positive"`` / ``"negative"``) when
        you KNOW the outcome but the message text doesn't reflect it.
        Maps to ±0.7 / 0.0 — lands inside the label band without
        saturating.
      - ``r_override``: pass an exact float in [-1, 1] when you want
        precise control. Beats ``sentiment`` when both are set.

    Returns: ``label``, ``r_score``, ``migrations``, ``absorbed``,
    ``scoring_source`` (``"heuristic"`` / ``"sentiment"`` /
    ``"r_override"``) so the caller can confirm which path resolved R,
    and a fresh ``stats`` snapshot.
    """
    summary = _store.session_close(
        session_id=session_id,
        sentiment=sentiment,
        r_override=r_override,
    )
    return {
        "session_id": session_id,
        "label": summary.get("label"),
        "r_score": round(summary.get("r", 0.0), 3),
        "migrations": len(summary.get("migrations", [])),
        "absorbed": len(summary.get("absorbed", [])),
        "scoring_source": summary.get("scoring_source"),
        "stats": _store.stats,
    }


@mcp.tool()
def record_session(messages: list[str], agent_id: str = "default") -> dict:
    """USE WHEN: scoring a finished conversation in one call (without having
    used ``session_open`` / ``session_push`` / ``session_close``). Pass every
    user message in order; do not include assistant replies.

    Prefer the open / push / close trio when you can — it lets ``query_memory``
    and ``record_fact`` attribute reads and writes incrementally to the
    session. ``record_session`` is the fallback for "I forgot to open a
    session and now I want the resonance signal anyway".
    """
    session_id = f"{agent_id}-{int(time.time())}-{uuid.uuid4().hex[:4]}"
    _store.session_start(session_id)
    # Symmetric with session_open(first_message=...) — if there's an
    # opening message, run echo detection now so the retroactive penalty
    # path can fire for this convenience entrypoint too. Without this,
    # an agent that records a whole session in one call never benefits
    # from echo, which is the main retroactive correction feature.
    echo = None
    if messages:
        echo = _store.check_echo(messages[0], session_id=session_id)
    for msg in messages:
        _store.session_message(msg, session_id=session_id)
    summary = _store.session_close(session_id=session_id)
    response = {
        "session_id": session_id,
        "label": summary.get("label"),
        "r_score": round(summary.get("r", 0.0), 3),
        "migrations": len(summary.get("migrations", [])),
        "absorbed": len(summary.get("absorbed", [])),
        "stats": _store.stats,
    }
    if echo is not None:
        response["echo"] = echo
    return response


@mcp.tool()
def memory_stats() -> dict:
    """USE WHEN: checking the memory's health — at session start if you
    suspect drift, or on a periodic audit. Cheap, read-only.

    Reports layer distribution (surface / kinetic / core), black hole mass,
    Hawking emission count, and the live ``adaptive_weights`` (so you can
    see what the formula has actually learned). Interpret:
      ``black_hole_mass`` rising — facts are failing; review what is being stored.
      ``surface`` dropping — active knowledge declining; needs fresh input.
      ``hawking_emissions > 0`` — dead facts resurface; store may have stale info.
    """
    return _store.stats


if __name__ == "__main__":
    mcp.run()
