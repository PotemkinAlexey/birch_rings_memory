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


def _validate_text(
    value, field_name: str = "text",
) -> Optional[dict]:
    """Reject non-string or empty text inputs at the MCP boundary.

    query_memory, find_similar, session_push, check_echo, session_open
    (first_message) all accept a ``text``-like argument that goes
    straight into the embedding path. None / list / dict / whitespace-
    only would either crash deep in embed() or, worse, embed an
    opaque value as if it were content. Caller gets a structured
    response with the failed field name so the agent can fix one
    argument instead of guessing.
    """
    if not isinstance(value, str) or not value.strip():
        return {
            "ok": False,
            "error": "invalid_text",
            "field": field_name,
            "got_type": type(value).__name__,
            "hint": f"{field_name} must be a non-empty string",
        }
    return None


def _validate_optional_text(value, field_name: str) -> Optional[dict]:
    """Reject non-string values for optional text args (subject_prefix,
    subject substring, predicate substring). ``None`` passes through;
    empty strings pass through (callers treat them as "no filter").
    Same family as ``_validate_text`` but allows omission.

    Without this guard, ``subject_prefix=123`` reaches core where
    ``.lower()`` raises raw ``AttributeError`` deep in the search
    path. The string-handling tools (query_memory, list_facts,
    find_similar) all forward these args unchanged.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        return {
            "ok": False,
            "error": "invalid_text",
            "field": field_name,
            "got_type": type(value).__name__,
            "hint": f"{field_name} must be a string or omitted",
        }
    return None


def _validate_optional_id(value, field_name: str) -> Optional[dict]:
    """Same shape as _validate_id but ``None`` passes through. For
    session_id-style args where omitted means "use current" or
    "generate one"; only the explicit non-None case must be a
    non-empty string."""
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        return {
            "ok": False,
            "error": "invalid_id",
            "field": field_name,
            "got_type": type(value).__name__,
            "hint": f"{field_name} must be a non-empty string when set",
        }
    return None


def _validate_bool(value, field_name: str) -> Optional[dict]:
    """Reject non-bool values for explicit bool flags. Python's truthy
    semantics would let ``record_first_message="false"`` evaluate True
    (non-empty string) and silently push the message — exactly the
    opposite of what the caller meant. Strict isinstance check."""
    if not isinstance(value, bool):
        return {
            "ok": False,
            "error": "invalid_bool",
            "field": field_name,
            "got_type": type(value).__name__,
            "hint": f"{field_name} must be true or false (JSON bool)",
        }
    return None


def _validate_id(value, field_name: str = "fact_id") -> Optional[dict]:
    """Reject non-string or empty id arguments at the MCP boundary.

    ID-based tools (``delete_fact``, ``delete_body``, ``supersede_fact``,
    ``retire_fact``, ``explain_fact``) used to accept whatever the
    caller passed and forward it to core. None / int / dict / "" /
    whitespace silently returned "not found" (mostly benign) or, for
    ``explain_fact``, crashed deep inside a dict lookup. Catching here
    gives a single structured failure shape per field — symmetric with
    ``_validate_text`` and ``_validate_spo_strings``.
    """
    if not isinstance(value, str) or not value.strip():
        return {
            "ok": False,
            "error": "invalid_id",
            "field": field_name,
            "got_type": type(value).__name__,
            "hint": f"{field_name} must be a non-empty string",
        }
    return None


def _env_int(
    name: str, default: int, *, lo: int = 1, hi: int = 1_000_000,
) -> int:
    """Tolerant int env var. Garbage values (``BIRCH_*=abc``) MUST NOT
    crash module import — the server has to come up so it can return
    a structured error from a tool call. Bad input falls back to the
    default; out-of-range input clamps."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, value))


# Cap on the per-call ``record_facts`` batch. Each item carries a string
# embedding round-trip and a SQLite write — a 50k-item payload is one
# accidental copy-paste away from a many-minute embed call + a huge
# transaction. 500 is a comfortable upper bound for batch embed
# endpoints (well under Ollama's default) and well above any agent's
# legitimate single-call write. Override via env if you really need
# bigger batches and have audited the cost.
_RECORD_FACTS_BATCH_CAP = _env_int(
    "BIRCH_RECORD_FACTS_BATCH_CAP", 500, lo=1, hi=10_000,
)


def _validate_int(
    value, field_name: str, *, lo: int = 1, hi: int = 500,
) -> tuple[Optional[int], Optional[dict]]:
    """Type+range validator for integer MCP inputs (top_k, limit,
    horizon_ticks). Returns ``(clamped_int, None)`` on success or
    ``(None, error_dict)``. Whitespace-only / non-numeric / None all
    return structured ``invalid_int`` so tools never raise raw
    TypeError on a string-typed JSON arg from a poorly-typed client."""
    try:
        ivalue = int(value)
    except (TypeError, ValueError):
        return None, {
            "ok": False,
            "error": "invalid_int",
            "field": field_name,
            "got_type": type(value).__name__,
            "hint": f"{field_name} must be an integer",
        }
    if ivalue < lo:
        return None, {
            "ok": False,
            "error": "invalid_int",
            "field": field_name,
            "min": lo,
            "got": ivalue,
        }
    return min(ivalue, hi), None


def _validate_float(
    value, field_name: str, *, lo: float = 0.0, hi: float = 1.0,
) -> tuple[Optional[float], Optional[dict]]:
    """Type+range+finite validator for float MCP inputs (min_similarity,
    min_gravity). NaN/Infinity are rejected the same way as in
    session_close r_override — undefined cosine comparisons are worse
    than a clear error."""
    import math
    try:
        fvalue = float(value)
    except (TypeError, ValueError):
        return None, {
            "ok": False,
            "error": "invalid_float",
            "field": field_name,
            "got_type": type(value).__name__,
            "hint": f"{field_name} must be a finite float",
        }
    if not math.isfinite(fvalue):
        return None, {
            "ok": False,
            "error": "invalid_float",
            "field": field_name,
            "detail": "NaN or Infinity",
        }
    if not lo <= fvalue <= hi:
        return None, {
            "ok": False,
            "error": "invalid_float",
            "field": field_name,
            "min": lo,
            "max": hi,
            "got": fvalue,
        }
    return fvalue, None


def _validate_spo_strings(
    subject, predicate, obj,
) -> Optional[dict]:
    """Type-validate the SPO triple at the MCP boundary.

    Mirrors the same validator across ``record_facts`` (batch path)
    and ``record_fact`` / ``set_fact`` (single paths) so the failure
    mode is symmetric: subject=123 fails the same way it would in
    a batch.

    Returns ``None`` if all three fields are non-empty strings;
    otherwise a structured error dict with ``bad_fields`` and
    ``got_types``. Whitespace-only counts as empty.
    """
    required = (("subject", subject), ("predicate", predicate),
                ("object", obj))
    bad: list[str] = []
    types: dict[str, str] = {}
    for name, val in required:
        if not isinstance(val, str) or not val.strip():
            bad.append(name)
            types[name] = type(val).__name__
    if not bad:
        return None
    return {
        "ok": False,
        "error": "invalid_fact_fields",
        "bad_fields": bad,
        "got_types": types,
        "hint": "subject, predicate, object must be non-empty strings.",
    }


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
    err = _validate_text(text, "text")
    if err is not None:
        return err
    # Numeric type validation: a poorly-typed JSON client passing
    # top_k="5" used to raise raw TypeError in the `<=` comparison
    # below. _validate_int turns that into a structured response and
    # also caps at 50 (the existing upper bound), preserving the
    # "requested vs capped" disclosure path.
    requested_top_k = top_k
    top_k_validated, err = _validate_int(top_k, "top_k", lo=1, hi=50)
    if err is not None:
        return {"results": [], **err}
    top_k = top_k_validated  # type: ignore[assignment]
    min_sim_validated, err = _validate_float(
        min_similarity, "min_similarity", lo=0.0, hi=1.0,
    )
    if err is not None:
        return {"results": [], **err}
    min_similarity = min_sim_validated  # type: ignore[assignment]
    min_grav_validated, err = _validate_float(
        min_gravity, "min_gravity", lo=0.0, hi=1.0,
    )
    if err is not None:
        return {"results": [], **err}
    min_gravity = min_grav_validated  # type: ignore[assignment]
    err = _validate_optional_text(subject_prefix, "subject_prefix")
    if err is not None:
        return {"results": [], **err}
    layer_map = {"surface": 0, "kinetic": 1, "core": 2}
    if layers is not None:
        # Shape check: layers="surface" (string instead of list) used
        # to iterate by character and report unknown_layers=["s","u",
        # "r","f","a","c","e"] — technically structured but
        # misleading. Tell the caller the actual shape problem.
        if isinstance(layers, str) or not isinstance(layers, list):
            return {
                "results": [],
                "error": "invalid_layers",
                "got_type": type(layers).__name__,
                "hint": (
                    "layers must be a list of strings; valid values "
                    "are surface, kinetic, core"
                ),
            }
        if any(not isinstance(x, str) for x in layers):
            return {
                "results": [],
                "error": "invalid_layers",
                "hint": "every item in layers must be a string",
            }
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
    err = _validate_spo_strings(subject, predicate, object)
    if err is not None:
        return err
    # Transaction-honest: add_fact returns created from inside its own
    # write txn, so there's no race window between a fact_exists probe and
    # the insert (same pattern set_fact uses).
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
        similar = _store.find_similar_by_vector(
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
    # Top-level type check: if the caller passes a string (would iterate
    # by character) or a dict (would iterate by key) instead of a list,
    # the per-item loop below would generate one "item_not_an_object"
    # error per character/key, which is technically structured but
    # useless. Reject the payload up front with a single clear error.
    if not isinstance(facts, list):
        return {
            "ok": False,
            "error": "invalid_facts_payload",
            "got_type": type(facts).__name__,
            "hint": (
                "facts must be a list of objects with subject, "
                "predicate, object"
            ),
        }
    # Batch-size cap: an agent that accidentally pastes 50k items would
    # otherwise issue one huge embed batch + one giant SQLite
    # transaction with no progress signal. Return a structured error so
    # the caller can split the work and retry.
    if len(facts) > _RECORD_FACTS_BATCH_CAP:
        return {
            "ok": False,
            "error": "batch_too_large",
            "limit": _RECORD_FACTS_BATCH_CAP,
            "got": len(facts),
            "hint": (
                f"Split into batches of at most {_RECORD_FACTS_BATCH_CAP} "
                "items. Override the cap with "
                "BIRCH_RECORD_FACTS_BATCH_CAP if you really need bigger."
            ),
        }
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
        # Type validation: presence check above catches missing/None/"";
        # this catches the next layer — subject=123 / predicate=[] /
        # object={} all pass the presence check but break embedding-text
        # formatting and SPO normalisation downstream. Triples must be
        # strings, and whitespace-only strings count as empty.
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
            continue
        # Per-item session_id type check: items can override the
        # top-level session_id, and the override flows straight into
        # _resolve_sid. A non-string (int, list, dict) would silently
        # mis-attribute the fact to a session id of the wrong type
        # or skip attribution entirely. Catch here so the agent gets
        # a structured error per item instead of a quiet
        # mis-attribution.
        if "session_id" in f and f["session_id"] is not None:
            if (
                not isinstance(f["session_id"], str)
                or not f["session_id"].strip()
            ):
                invalid.append({
                    "index": i,
                    "error": "invalid_session_id",
                    "got_type": type(f["session_id"]).__name__,
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
    err = _validate_spo_strings(subject, predicate, object)
    if err is not None:
        return err
    # set_fact -> add_fact -> embed(); wrap so an unreachable embedding
    # provider produces a structured failure instead of raw stacktrace
    # at the MCP boundary (completes the wrap coverage shared with
    # record_fact / record_facts / query_memory).
    try:
        return _store.set_fact(
            subject, predicate, object, session_id=session_id,
        )
    except EmbeddingError as exc:
        return _embedding_error_response(exc)


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

    Returns ``{"superseded": true, "old_id", "new_id", "absorbed": [...]}``
    on success. On failure returns ``{"superseded": false, "old_id",
    "new_id", "error", ...}`` — both ids are always echoed so the
    agent can key on ``result["old_id"]`` in both branches without a
    KeyError. ``error`` is ``"not_a_factpassport"`` (with ``kind`` =
    ``meta``/``singularity_fact``/``singularity_meta`` and a hint
    explaining why the lifecycle op doesn't apply) or ``"not_found"``.

    FactPassport-only by design — MetaFacts have no SPO slot, so
    "supersede a cluster" has no defined semantics. For destructive
    removal of any body kind use ``delete_body``; for stale MetaFact
    data, record contradicting facts and let next-cycle collapse
    re-aggregate.
    """
    err = _validate_id(old_fact_id, "old_fact_id")
    if err is not None:
        return err
    err = _validate_id(new_fact_id, "new_fact_id")
    if err is not None:
        return err
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

    Returns ``{"retired": true, "fact_id", "absorbed": [...]}`` on
    success. On failure returns ``{"retired": false, "fact_id",
    "error", ...}`` — ``fact_id`` is always echoed. ``error`` is
    ``"not_a_factpassport"`` (with ``kind`` and a hint) when the id
    points at a MetaFact / singularity body, or ``"not_found"``.

    FactPassport-only by design. See ``supersede_fact`` for the
    same rationale.
    """
    err = _validate_id(fact_id, "fact_id")
    if err is not None:
        return err
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

    On a concurrent-mutation race (another agent wrote to memory
    while the heavy O(n²) simulation was running outside the lock)
    returns ``{"ok": false, "error": "forecast_snapshot_stale", ...}``
    instead of writing stale scores into the post-mutation bodies.
    The agent retries; the next call sees the new universe.
    """
    from .vector_index import DimensionMismatchError

    # Numeric validation symmetric with the other tools. horizon=0/neg
    # was already handled by the catch-all ValueError below, but a
    # structured invalid_int gives the agent a cleaner contract.
    horizon_validated, err = _validate_int(
        horizon_ticks, "horizon_ticks", lo=1, hi=10_000,
    )
    if err is not None:
        return err
    horizon_ticks = horizon_validated  # type: ignore[assignment]
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
    except (ValueError, TypeError) as exc:
        # Deterministic input issues — bad vector shape sneaking past
        # _safe_vector, numpy raising on malformed body, galaxy refusing
        # a zero-mass body. Structured response so the agent gets an
        # actionable diagnostic instead of a raw stacktrace. NOT
        # catching BaseException — we don't want to hide programmer
        # bugs or KeyboardInterrupt.
        return {
            "ok": False,
            "error": "forecast_failed",
            "detail": str(exc),
            "hint": (
                "Check fact / metafact vectors for shape consistency; "
                "run memory_stats to inspect body counts."
            ),
        }


@mcp.tool()
def delete_fact(fact_id: str) -> dict:
    """
    Legacy hard-delete — handles ONLY live FactPassports.

    Use ``delete_body(body_id)`` for the polymorphic version that also
    handles live MetaFacts, singularity FactPassports, and singularity
    MetaFacts — a ``query_memory`` result's ``body_id`` may point at any
    of those. ``delete_fact`` is kept for backward compatibility and
    cases where the caller knows the id is specifically a live fact.

    Destructive primitive: data is GONE. Removes the fact from live
    layers, the vector index, the SPO dedup index, AND storage. Does
    NOT send it to the black hole; no lineage, no MetaFact compression,
    no Hawking rescue. Use only for:

      - Secrets / sensitive data that must not exist (GDPR-style removal)
      - Facts you just recorded by accident in the same session

    For "stale / wrong / outdated" data, prefer supersede_fact (when there
    is a replacement) or retire_fact (when there is not) — both preserve
    the body in the singularity so the brain can still learn from it.

    Returns {"deleted": true} if found, {"deleted": false} if not found.
    """
    err = _validate_id(fact_id, "fact_id")
    if err is not None:
        return err
    deleted = _store.delete_fact(fact_id)
    return {"deleted": deleted, "fact_id": fact_id}


@mcp.tool()
def delete_body(body_id: str) -> dict:
    """
    Polymorphic hard-delete — handles ALL four body locations.

    ``query_memory`` returns polymorphic hits (live FactPassport, live
    MetaFact, singularity FactPassport, or singularity MetaFact) under
    a single ``body_id``. ``delete_body`` checks all four locations and
    deletes wherever the id lives. Use this when you got a ``body_id``
    from a query and want to remove it regardless of what kind of body
    it is — for example GDPR removal of MetaFact-aggregated content,
    or accidental absorbed-body cleanup.

    Returns ``{"deleted": True, "kind": "fact"|"meta"|"singularity_fact"|
    "singularity_meta", "body_id": ...}`` on success, or
    ``{"deleted": False, "body_id": ...}`` if not found.

    Same destructive contract as delete_fact: data GONE, no singularity
    rescue, no MetaFact lineage. Prefer supersede_fact / retire_fact
    for stale data — both preserve the body for resonance feedback.
    """
    err = _validate_id(body_id, "body_id")
    if err is not None:
        return err
    return _store.delete_body(body_id)


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
    # Numeric validation: string limit / min_gravity from a poorly
    # typed JSON client would raise raw TypeError below. _validate_*
    # turn that into a structured response. list_facts returns list[dict];
    # the convention here is to inject the error at index 0 (existing
    # unknown_layer path does the same).
    limit_validated, err = _validate_int(limit, "limit", lo=1, hi=500)
    if err is not None:
        return [err]
    requested_limit = limit
    limit = limit_validated  # type: ignore[assignment]
    if requested_limit != limit:
        # Cap disclosure (was previously a server-side log; keeping
        # behaviour for now). Agent can detect the cap by
        # len(result) == 500 against its requested limit.
        import logging
        logging.getLogger(__name__).warning(
            "list_facts: limit capped at 500 (requested %s)",
            requested_limit,
        )
    min_grav_validated, err = _validate_float(
        min_gravity, "min_gravity", lo=0.0, hi=1.0,
    )
    if err is not None:
        return [err]
    min_gravity = min_grav_validated  # type: ignore[assignment]
    for opt_name, opt_val in (
        ("subject", subject),
        ("predicate", predicate),
        ("subject_prefix", subject_prefix),
    ):
        err = _validate_optional_text(opt_val, opt_name)
        if err is not None:
            return [err]
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
    err = _validate_text(text, "text")
    if err is not None:
        return err
    # Numeric validation: structured response on bad type / out-of-range
    # before _store.find_similar receives the input.
    requested_top_k = top_k
    top_k_validated, err = _validate_int(top_k, "top_k", lo=1, hi=50)
    if err is not None:
        return {"query": text, "hits": [], **err}
    top_k = top_k_validated  # type: ignore[assignment]
    min_sim_validated, err = _validate_float(
        min_similarity, "min_similarity", lo=0.0, hi=1.0,
    )
    if err is not None:
        return {"query": text, "hits": [], **err}
    min_similarity = min_sim_validated  # type: ignore[assignment]
    err = _validate_optional_text(subject_prefix, "subject_prefix")
    if err is not None:
        return {"query": text, "hits": [], **err}
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

    Polymorphic: handles live FactPassport, live MetaFact,
    singularity FactPassport, and singularity MetaFact (same four
    locations as ``delete_body`` and ``query_memory``). For MetaFact
    bodies the response carries ``kind="meta"``, ``weight``,
    ``source_fact_ids``, and ``source_texts`` instead of SPO fields.
    Consider ``explain_body`` as the symmetric name when working
    with a ``body_id`` from ``query_memory``.
    """
    err = _validate_id(fact_id, "fact_id")
    if err is not None:
        return err
    return _store.explain_fact(fact_id)


@mcp.tool()
def explain_body(body_id: str) -> dict:
    """USE WHEN: you got a ``body_id`` from ``query_memory`` and want to
    decompose its gravity — alias for ``explain_fact`` with the
    body-named contract.

    Same response shape as ``explain_fact``: returns ``found``,
    ``kind`` (``fact``/``meta``/``singularity_fact``/``singularity_meta``),
    per-feature ``features``, current ``weights``, per-component
    ``contributions``, and shape-specific fields (SPO for facts,
    weight + lineage for metas). Read-only debug; never mutates.
    """
    err = _validate_id(body_id, "body_id")
    if err is not None:
        return err
    return _store.explain_body(body_id)


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
    # Boundary validation: session_id (if explicitly passed) must be
    # a non-empty string; record_first_message must be a real bool
    # (Python truthy semantics on "false"/[1]/dict would silently push
    # when caller meant skip); agent_id is a name used to mint the
    # default session_id, must be a non-empty string.
    err = _validate_optional_id(session_id, "session_id")
    if err is not None:
        return err
    err = _validate_text(agent_id, "agent_id")
    if err is not None:
        return err
    err = _validate_bool(record_first_message, "record_first_message")
    if err is not None:
        return err
    sid = session_id or f"{agent_id}-{int(time.time())}-{uuid.uuid4().hex[:4]}"
    # session_start now returns whether a brand-new SessionContext was
    # created (True) or an existing in-flight one was promoted (False).
    # Retry-aware agents need to distinguish "fresh open" from
    # "recovered after a flaky open" so they don't accidentally double-
    # push the same first_message into the trajectory.
    created = _store.session_start(sid)
    response: dict = {
        "session_id": sid,
        "created": created,
        "already_open": not created,
        "_hint": (
            "pass this session_id to record_fact, record_facts, query_memory, "
            "and session_push — then call session_close when done"
        ),
    }
    if first_message is not None:
        # Type/empty guard for first_message — same contract as
        # session_push text. Invalid input rejected before the
        # session does any embed work.
        err = _validate_text(first_message, "first_message")
        if err is not None:
            # session is already opened on disk (session_start above);
            # mark partial_open so the agent can choose to close or
            # retry with a valid first_message.
            response["ok"] = False
            response["partial_open"] = True
            response["first_message_error"] = err
            return response
    if first_message:
        # Both check_echo and session_message embed the first message.
        # An unreachable provider here would abort session_open after
        # the session was already started — wrap so the agent gets a
        # structured failure and can retry. The session is left open
        # so the caller can decide: retry session_push(first_message),
        # close it, or abort. ``first_message_recorded`` tells the
        # agent which state it's in so it can't assume the message
        # already landed in the trajectory.
        try:
            response["echo"] = _store.check_echo(
                first_message, session_id=sid,
            )
            if record_first_message:
                _store.session_message(first_message, session_id=sid)
                response["first_message_recorded"] = True
            else:
                response["first_message_recorded"] = False
        except EmbeddingError as exc:
            response["echo_error"] = _embedding_error_response(exc)
            response["first_message_recorded"] = False
            # Top-level failure markers so an agent that reads only the
            # response envelope (not nested keys) cannot miss the
            # partial-open state. session_id stays — the session IS
            # opened on disk; the agent can either retry session_push
            # or call session_close to drop the empty session.
            response["ok"] = False
            response["partial_open"] = True
            response["_hint"] = (
                "session was opened but first_message was NOT recorded "
                "due to embedding failure; retry session_push or call "
                "session_close to drop the empty session"
            )
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
    err = _validate_text(first_message, "first_message")
    if err is not None:
        return err
    err = _validate_optional_id(session_id, "session_id")
    if err is not None:
        return err
    try:
        return _store.check_echo(first_message, session_id=session_id)
    except EmbeddingError as exc:
        return _embedding_error_response(exc)


@mcp.tool()
def session_push(text: str, session_id: str) -> dict:
    """USE WHEN: a user message arrives during an open session. Push every
    user message so the resonance engine can score the conversation
    trajectory on close.

    Pass user-side text only. Do NOT push your own assistant responses —
    behavioural / semantic / repetition signals score user closure, not
    your replies.
    """
    err = _validate_text(text, "text")
    if err is not None:
        return err
    err = _validate_id(session_id, "session_id")
    if err is not None:
        return err
    try:
        _store.session_message(text, session_id=session_id)
    except EmbeddingError as exc:
        return _embedding_error_response(exc)
    except KeyError as exc:
        # Unknown / expired session_id used to leak raw KeyError to the
        # MCP layer. Structured response so the agent gets an actionable
        # hint instead of a stacktrace.
        return {
            "ok": False,
            "error": "unknown_session",
            "session_id": session_id,
            "detail": str(exc),
            "hint": (
                "Call session_open first and pass the returned session_id, "
                "or check the id hasn't been closed."
            ),
        }
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
    err = _validate_id(session_id, "session_id")
    if err is not None:
        return err
    # Validate r_override BEFORE calling core. Core's float(r_override)
    # would raise ValueError for non-numeric input which the catch
    # below maps to "invalid_sentiment" — wrong error class, misleading
    # message. Validate at the boundary so the response is honest.
    # NaN and Infinity also rejected explicitly: core does
    # max(-1.0, min(1.0, r)) which on NaN gives a surprising number
    # rather than failing cleanly.
    if r_override is not None:
        import math as _math
        try:
            r_check = float(r_override)
        except (TypeError, ValueError):
            return {
                "ok": False,
                "error": "invalid_r_override",
                "session_id": session_id,
                "got_type": type(r_override).__name__,
                "hint": "r_override must be a finite float in [-1, 1]",
            }
        if not _math.isfinite(r_check):
            return {
                "ok": False,
                "error": "invalid_r_override",
                "session_id": session_id,
                "detail": "NaN or Infinity",
                "hint": "r_override must be a finite float in [-1, 1]",
            }
    try:
        summary = _store.session_close(
            session_id=session_id,
            sentiment=sentiment,
            r_override=r_override,
        )
    except ValueError as exc:
        # Core raises ValueError on unknown sentiment label. Map to a
        # structured response so the agent can read the allowed set
        # instead of getting a stacktrace. r_override path is now
        # validated above, so this catch only fires for sentiment.
        return {
            "ok": False,
            "error": "invalid_sentiment",
            "session_id": session_id,
            "detail": str(exc),
            "allowed": [
                "resonant", "positive", "neutral", "toxic", "negative",
            ],
        }
    # Core returns {} for an unknown session_id or a session that was
    # never pushed to (no messages). Without this guard the response
    # looked like a successful close with null label and r_score=0.0
    # — indistinguishable from a real close that scored neutral. Be
    # explicit so the agent can tell the close didn't actually happen.
    if not summary:
        return {
            "ok": False,
            "error": "unknown_or_empty_session",
            "session_id": session_id,
            "hint": (
                "Call session_open and session_push before session_close, "
                "or check the session_id hasn't already been closed."
            ),
            "stats": _store.stats,
        }
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
    # agent_id is used to mint the synthetic session_id below — must
    # be a non-empty string or the generated id is junk.
    err = _validate_text(agent_id, "agent_id")
    if err is not None:
        return err
    # Pre-validate the messages list before opening a session. Round
    # 15: a string passed for messages would have iterated chars; a
    # list with non-string items would have reached the embed layer
    # as an opaque value. Reject up front with a structured response
    # so we never half-open a session for a malformed input.
    if not isinstance(messages, list):
        return {
            "ok": False,
            "error": "invalid_messages",
            "got_type": type(messages).__name__,
            "hint": "messages must be a list of strings",
        }
    bad_items = [
        i for i, m in enumerate(messages)
        if not isinstance(m, str) or not m.strip()
    ]
    if bad_items:
        return {
            "ok": False,
            "error": "invalid_message_item",
            "indices": bad_items,
            "hint": "each message must be a non-empty string",
        }
    # Empty-list reject: symmetric with session_close's
    # unknown_or_empty_session envelope. record_session([]) used to
    # half-open a session, never embed anything, and quietly close
    # to a neutral r_score=0.0 — indistinguishable from a real
    # heuristic-scored neutral outcome.
    if not messages:
        return {
            "ok": False,
            "error": "empty_messages",
            "hint": "messages must contain at least one entry",
        }
    session_id = f"{agent_id}-{int(time.time())}-{uuid.uuid4().hex[:4]}"
    _store.session_start(session_id)
    # Symmetric with session_open(first_message=...) — if there's an
    # opening message, run echo detection now so the retroactive penalty
    # path can fire for this convenience entrypoint too. Without this,
    # an agent that records a whole session in one call never benefits
    # from echo, which is the main retroactive correction feature.
    echo = None
    try:
        if messages:
            echo = _store.check_echo(messages[0], session_id=session_id)
        for msg in messages:
            _store.session_message(msg, session_id=session_id)
        summary = _store.session_close(session_id=session_id)
    except EmbeddingError as exc:
        # The session was started before the failing embed. Without a
        # cleanup, it would sit in _sessions until TTL eviction and
        # leak into open_sessions on disk. abort_session drops it
        # without computing resonance — best-effort, swallows any
        # secondary error so the original embed failure surfaces.
        try:
            _store.abort_session(session_id)
        except Exception:
            pass
        return _embedding_error_response(exc)
    except Exception:
        # Any OTHER failure (storage, dim mismatch, ValueError from
        # core validators, etc.) also needs the same orphan-session
        # cleanup. Previously only EmbeddingError triggered abort;
        # a sqlite error mid-session_message would leak the open
        # session into disk until TTL eviction. Best-effort cleanup
        # then re-raise so the agent sees the original error
        # (caller decides how to react).
        try:
            _store.abort_session(session_id)
        except Exception:
            pass
        raise
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
