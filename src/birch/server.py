"""BirchKM MCP server — exposes memory tools to Claude agents."""
from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP

from .memory_store import MemoryStore

_DB_PATH = os.environ.get("BIRCH_DB", str(Path.home() / ".birch" / "memory.db"))
Path(_DB_PATH).parent.mkdir(parents=True, exist_ok=True)

_store = MemoryStore(db_path=_DB_PATH)

mcp = FastMCP("BirchKM")


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
    layer_map = {"surface": 0, "kinetic": 1, "core": 2}
    if layers:
        # Build an explicit set so layers=["surface", "core"] excludes
        # kinetic — a range filter would silently include it.
        allowed = {layer_map[name] for name in layers if name in layer_map}
    else:
        allowed = None
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
    response: dict = {"results": hits, "_hint": hint}
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
    already_existed = _store.fact_exists(subject, predicate, object)
    fact = _store.add_fact(subject, predicate, object, session_id=session_id)
    similar: list[dict] = []
    if not already_existed and fact.vector:
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
    triples = [
        (f["subject"], f["predicate"], f["object"])
        for f in facts
    ]
    # Per-item session_id overrides the top-level one — honour the contract
    # spelled out in the docstring. Items without their own session_id fall
    # back to the top-level argument.
    per_item_sids = [f.get("session_id") for f in facts]
    statuses = _store.add_facts(
        triples,
        session_id=session_id,
        session_ids=per_item_sids,
        return_status=True,
    )
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
    Run the galaxy forward and write a per-fact stability prediction back.

    For every live fact the galaxy simulates a body in orbit around the
    central black hole. After ``horizon_ticks`` integrator steps each body
    has either survived (its orbital radius gives a stability score in
    [0, 1]) or crossed the event horizon (stability = 0). The score lands
    on FactPassport.forecast_stability and is consumed by the adaptive
    gravity formula via ``w_stability`` — facts the galaxy predicts to
    fall earn a smaller gravity contribution from this feature, facts
    predicted safe earn more.

    The feature complements the local pre-resonance signals (freshness,
    access, graph, recent_utility) with a forecast no local feature can
    produce: it sees the body's *future*. Whether the forecast helps is
    not assumed — ``w_stability`` is learned by the same SGD pass that
    learns the other adaptive weights, so a useless forecast just sits
    at its prior and contributes nothing.

    Call at session start (or once per day) on a real store. Pure numpy,
    O(n²) per step in fact count.

    Returns a summary: how many facts were forecasted and updated, and a
    coarse distribution across {safe / kinetic / near_horizon / predicted_fall}.
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
    facts = _store.list_facts(subject=subject, predicate=predicate, limit=10_000)
    layer_map = {"surface": 0, "kinetic": 1, "core": 2}
    target_layer = layer_map.get(layer) if layer else None
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
    hits = _store.find_similar(
        text=text,
        top_k=top_k,
        min_similarity=min_similarity,
        subject_prefix=subject_prefix,
    )
    return {
        "query": text,
        "min_similarity": min_similarity,
        "hits": hits,
        "_hint": (
            "use set_fact for slot-replace, supersede_fact(old, new) when the "
            "new fact is already recorded, retire_fact when no replacement"
        ),
    }


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
def session_close(session_id: str) -> dict:
    """USE WHEN: a conversation that wrote or read facts ends. Closes the
    session, scores R from the message trajectory, propagates R to every
    touched fact's gravity, runs `_absorb_dead`, may trigger background
    singularity collapse. Call exactly once per opened session — repeated
    closes corrupt the resonance signal.

    Returns: ``label`` (resonant / neutral / toxic), ``r_score``,
    ``migrations`` (count of facts that moved layer), ``absorbed`` (count of
    facts that fell into the singularity), and a fresh ``stats`` snapshot.
    """
    summary = _store.session_close(session_id=session_id)
    return {
        "session_id": session_id,
        "label": summary.get("label"),
        "r_score": round(summary.get("r", 0.0), 3),
        "migrations": len(summary.get("migrations", [])),
        "absorbed": len(summary.get("absorbed", [])),
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
