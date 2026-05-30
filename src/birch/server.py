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
    # Length cap — defence against DoS / billing. _MAX_FIELD_LEN is
    # evaluated at module import time AFTER this function is defined
    # in source order, so the lookup happens at call time via the
    # module globals (already in scope by the time any MCP tool fires).
    if len(value) > _MAX_FIELD_LEN:
        return {
            "ok": False,
            "error": "field_too_long",
            "field": field_name,
            "got_length": len(value),
            "limit": _MAX_FIELD_LEN,
            "hint": (
                f"{field_name} is capped at {_MAX_FIELD_LEN} chars. "
                f"Split into multiple atomic facts or raise "
                f"BIRCH_MAX_FIELD_LEN if your deployment can afford "
                f"the embedding cost."
            ),
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


# Per-field character cap for text inputs at the MCP boundary
# (S/P/O strings, query_memory text, session_message text). Defence
# against DoS / billing: an agent looping with a 10 MB log dump
# would otherwise pay full embedding-provider cost on every call,
# write a multi-megabyte row to SQLite, and return a JSON response
# the consumer has to parse and feed back into LLM context.
#
# 2000 chars is comfortable for atomic SPO triples and conversational
# message turns; tunable down for stricter deployments or up when
# the deployment is offline and the operator has audited the cost
# profile. Lower bound 128 prevents accidental tiny caps that brick
# legitimate use.
_MAX_FIELD_LEN = _env_int(
    "BIRCH_MAX_FIELD_LEN", 2000, lo=128, hi=200_000,
)


# Control characters that are forbidden in any text field flowing
# through the MCP boundary. C0 control codes (0x00-0x1F except TAB
# / LF / CR) and DEL (0x7F) make zero semantic sense in user-facing
# text — their presence almost always signals binary content, a
# broken decoder, or a deliberate attempt to smuggle invisible
# bytes past a downstream LLM's tokenizer. Symmetric for zero-width
# Unicode (ZWSP / ZWNJ / ZWJ / BOM) which is the standard prompt-
# injection smuggling vector for LLM context.
_BANNED_CONTROL = {chr(c) for c in range(0x00, 0x20)} \
    - {"\t", "\n", "\r"}
_BANNED_CONTROL.add("\x7f")          # DEL
_BANNED_ZERO_WIDTH = {"​", "‌", "‍", "﻿"}
_BANNED_CHARS = _BANNED_CONTROL | _BANNED_ZERO_WIDTH


def _sanitize_for_llm(text: str) -> str:
    """Strip dangerous invisible characters from text crossing the MCP
    boundary into or out of the store.

    This is NOT a prompt-injection silver bullet. The consumer of
    ``query_memory`` results is still responsible for wrapping
    retrieved bodies in clear structural delimiters (XML tags, JSON
    fences, etc.) before feeding them into a downstream LLM context.

    What this DOES do:
      - Removes ASCII C0 control codes (except TAB / LF / CR which
        are legitimate in body text).
      - Removes DEL (0x7F).
      - Removes zero-width Unicode (ZWSP / ZWNJ / ZWJ / BOM) — the
        standard smuggling vector for invisible "ignore previous
        instructions" payloads.

    What this does NOT do:
      - Rewrite legitimate-looking model-instruction markers (e.g.
        ``<|im_start|>``, ``[INST]``, ``<<SYS>>``). Aggressively
        replacing those breaks legitimate discussion of prompts
        and is itself a security footgun (false positives become
        a content-filtering bypass surface). The honest answer is
        "consumer must wrap"; this helper buys defence-in-depth
        on the easy invisible bytes only.
    """
    if not text:
        return text
    if not any(c in _BANNED_CHARS for c in text):
        return text     # fast path: zero alloc for the common case
    return "".join(c for c in text if c not in _BANNED_CHARS)


# Known model-instruction markers. Used for DETECTION only — never
# rewritten. query_memory / find_similar attach a ``has_instruction
# _markers=True`` flag to a result when the body contains any of
# these substrings, so the consumer knows to wrap aggressively
# before passing it into LLM context. False positives are fine —
# the flag is advisory, not a block.
_LLM_INSTRUCTION_MARKERS = (
    "<|im_start|>", "<|im_end|>", "<|endoftext|>", "<|system|>",
    "<|user|>", "<|assistant|>",
    "[INST]", "[/INST]", "<<SYS>>", "<</SYS>>",
    "<|begin_of_text|>", "<|end_of_text|>",
    "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>",
)


def _has_instruction_markers(text: str) -> bool:
    """True if the text contains any known LLM control marker.
    Detection-only — used to advise the consumer to wrap."""
    if not text:
        return False
    return any(m in text for m in _LLM_INSTRUCTION_MARKERS)


def _check_non_empty_after_sanitize(
    field_values: dict[str, str],
) -> Optional[dict]:
    """Re-validate non-emptiness AFTER ``_sanitize_for_llm`` ran.

    The MCP boundary validators (``_validate_text`` /
    ``_validate_spo_strings``) check non-emptiness BEFORE sanitisation.
    A payload of pure zero-width Unicode (ZWSP/ZWNJ/ZWJ/BOM) passes
    those validators (str.strip() doesn't strip zero-width) but the
    sanitiser strips it to ``""``. Without this re-check the write
    path would persist an empty subject/predicate/object — a
    legitimate-looking row that's actually informationless.

    Returns ``None`` if every value still has content after strip(),
    otherwise a structured ``field_empty_after_sanitization`` error
    naming the offending fields. Pairs naturally with the existing
    ``invalid_text`` / ``invalid_fact_fields`` shapes.
    """
    bad = [name for name, val in field_values.items() if not val.strip()]
    if not bad:
        return None
    return {
        "ok": False,
        "error": "field_empty_after_sanitization",
        "bad_fields": bad,
        "hint": (
            "Input contained only invisible control characters or "
            "zero-width Unicode; after sanitisation the field is "
            "empty. Provide visible non-empty content."
        ),
    }


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
    if bad:
        return {
            "ok": False,
            "error": "invalid_fact_fields",
            "bad_fields": bad,
            "got_types": types,
            "hint": "subject, predicate, object must be non-empty strings.",
        }
    # Length cap — symmetric with _validate_text. SPO triples that
    # exceed the cap are usually paste accidents (a log dump landing
    # in `object`) or malicious payload-stuffing. Either way, the
    # embedding round-trip + the SQLite row would be expensive and
    # the resulting fact unparseable. Reject with structured detail
    # so the agent knows which field to split.
    too_long: list[str] = []
    lengths: dict[str, int] = {}
    for name, val in required:
        if len(val) > _MAX_FIELD_LEN:
            too_long.append(name)
            lengths[name] = len(val)
    if too_long:
        return {
            "ok": False,
            "error": "field_too_long",
            "bad_fields": too_long,
            "got_lengths": lengths,
            "limit": _MAX_FIELD_LEN,
            "hint": (
                f"S/P/O fields are each capped at {_MAX_FIELD_LEN} "
                f"chars. Split the offending field into multiple "
                f"atomic facts, or raise BIRCH_MAX_FIELD_LEN."
            ),
        }
    return None


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
    namespace_prefix: Optional[str] = None,
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
    # MemoryBricks Step 1: namespace_prefix is a hierarchical scope
    # filter (VB-style path match). Same shape as subject_prefix.
    err = _validate_optional_text(namespace_prefix, "namespace_prefix")
    if err is not None:
        return {"results": [], **err}
    # Optional-id boundary: a non-None session_id that is not a non-empty
    # string used to flow straight into core.query(), where the session
    # lookup would either KeyError or, worse, silently miss a real
    # session whose id was stringly mis-typed by a poorly-typed client
    # (e.g. an int id from a JSON encoder that auto-coerces). Reject at
    # the MCP boundary so the failure shape is structured.
    err = _validate_optional_id(session_id, "session_id")
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
            namespace_prefix=namespace_prefix,
        )
    except EmbeddingError as exc:
        return _embedding_error_response(exc)
    hits = [r.to_mcp_dict() for r in results]

    # Prompt-injection advisory: scan retrieved bodies for known LLM
    # control markers and flag the hit so the consumer wraps it
    # before feeding into downstream LLM context. Detection-only —
    # we never rewrite the body (false-positive rewrites are
    # themselves a content-filtering bypass surface). Fields scanned
    # are everything an LLM consumer is likely to render: subject,
    # predicate, object for fact hits; summary and source_texts for
    # meta hits. ``has_instruction_markers`` is a per-hit boolean;
    # ``injection_warnings`` is the top-level list of body_ids for
    # the agent's structured handling.
    injection_warnings: list[str] = []
    for h in hits:
        fields_to_scan = (
            h.get("subject"), h.get("predicate"), h.get("object"),
            h.get("summary"),
        )
        flagged = any(
            isinstance(v, str) and _has_instruction_markers(v)
            for v in fields_to_scan
        )
        if not flagged:
            source_texts = h.get("source_texts") or []
            if isinstance(source_texts, list):
                flagged = any(
                    isinstance(t, str) and _has_instruction_markers(t)
                    for t in source_texts
                )
        if flagged:
            h["has_instruction_markers"] = True
            bid = h.get("body_id")
            if bid:
                injection_warnings.append(bid)

    # conflict_hints: same (namespace, subject, predicate) with different
    # objects. Namespace is part of the key — the same SPO in WORK/A and
    # PERSONAL are independent live rows (MemoryBricks scoping), not a conflict.
    conflicts: list[dict] = []
    by_slot: dict[tuple[str, str, str], list[dict]] = {}
    for h in hits:
        if h.get("kind") != "fact":
            continue
        ns = h.get("namespace") or ""
        s = (h.get("subject") or "").strip().lower()
        p = (h.get("predicate") or "").strip().lower()
        by_slot.setdefault((ns, s, p), []).append(h)
    for _slot, group in by_slot.items():
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
            "namespace": sorted_group[0].get("namespace") or "",
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
    if injection_warnings:
        response["injection_warnings"] = injection_warnings
        response["_injection_hint"] = (
            "Some retrieved bodies contain known LLM control markers "
            "(e.g. <|im_start|>, [INST]). Wrap them in explicit "
            "structural delimiters (XML tags, JSON fences) before "
            "feeding them into a downstream LLM context — they were "
            "stored as data but may be parsed as instructions."
        )
    return response


@mcp.tool()
def record_fact(
    subject: str,
    predicate: str,
    object: str,
    session_id: Optional[str] = None,
    namespace: Optional[str] = None,
    salient: bool = False,
) -> dict:
    """USE WHEN: storing a new atomic SPO triple where the (subject, predicate)
    can legitimately carry several objects (e.g. "api uses Postgres" AND
    "api uses Redis"). For one-canonical-value slots — HEADs, versions,
    counts — use ``set_fact`` instead so old values auto-supersede.

    Set ``salient=True`` ONLY for rare-but-critical knowledge that would be
    catastrophic to forget and may not be exercised for a long time (a yearly
    runbook step, a hard safety constraint). It pins the fact against
    disuse-absorption from the moment of writing — the one signal the system
    can't infer bottom-up. It is NOT a "this is good" rating (utility is still
    inferred); over-pinning is self-defeating (a per-namespace budget evicts
    the least-at-risk pins, and a pin that keeps surfacing without resonating
    decays away). Use sparingly.

    Identical triples (case-insensitive, whitespace-normalised) are deduplicated:
    the existing fact is touched and returned with ``already_existed=true``.
    Dedup is scoped by ``namespace`` — the same SPO under different
    namespaces are independent live rows.

    ``namespace`` (MemoryBricks Step 1) is a VB-style hierarchical path
    (e.g. ``"WORK/DataArt/Databricks"``). Empty / omitted means the
    global / unscoped root and preserves the pre-Step-1 contract.

    The response includes ``similar_existing`` — paraphrase candidates already
    in the store at cosine ≥ 0.85 (excluding this fact). Treat them as
    "consider supersede_fact or set_fact"; the dedup in this call only catches
    exact normalised SPO matches, not "X uses Postgres" vs "X is on Postgres".
    """
    err = _validate_spo_strings(subject, predicate, object)
    if err is not None:
        return err
    # Optional-id boundary: a non-None session_id that is not a non-empty
    # string used to flow into add_fact → _resolve_sid → silent skip of
    # the attribution path (resonance/gravity bookkeeping never fires).
    # Reject at the boundary so the agent gets a structured error.
    err = _validate_optional_id(session_id, "session_id")
    if err is not None:
        return err
    # MemoryBricks Step 1: namespace is optional text — same shape as
    # subject_prefix. ``None`` means "global root" downstream.
    err = _validate_optional_text(namespace, "namespace")
    if err is not None:
        return err
    err = _validate_bool(salient, "salient")
    if err is not None:
        return err
    # Strip invisible control chars + zero-width Unicode at the write
    # boundary. Stored data is permanent — any smuggled NUL / ZWSP /
    # BOM would leak into every future query_memory response and into
    # downstream LLM context. The legitimate-looking instruction
    # markers (<|im_start|>, [INST], …) are NOT rewritten; the
    # consumer is responsible for wrapping retrieved bodies. This
    # closes the easy invisible-bytes smuggling vector.
    subject = _sanitize_for_llm(subject)
    predicate = _sanitize_for_llm(predicate)
    object = _sanitize_for_llm(object)
    # Re-validate after sanitisation. _validate_spo_strings ran on the
    # ORIGINAL input; if the input was pure invisible-bytes payload
    # (ZWSP-only subject etc.) the validator passed but the sanitiser
    # just stripped it to "". Without this re-check the write path
    # persists a legitimate-looking row with empty fields.
    err = _check_non_empty_after_sanitize(
        {"subject": subject, "predicate": predicate, "object": object}
    )
    if err is not None:
        return err
    # Transaction-honest: add_fact returns created from inside its own
    # write txn, so there's no race window between a fact_exists probe and
    # the insert (same pattern set_fact uses).
    try:
        # salient is threaded into add_fact so the write + pin commit in ONE
        # transaction — a rare-critical fact is never left written-but-unpinned.
        fact, created = _store.add_fact(
            subject, predicate, object,
            session_id=session_id, return_status=True,
            namespace=(namespace or ""), salient=salient,
        )
    except EmbeddingError as exc:
        return _embedding_error_response(exc)
    already_existed = not created
    pinned = bool(salient)
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
        "pinned": pinned,
        "similar_existing": similar,
        "_hint": hint,
    }


@mcp.tool()
def record_facts(
    facts: list[dict],
    session_id: Optional[str] = None,
    namespace: Optional[str] = None,
) -> dict:
    """USE WHEN: storing several SPO triples at once — one Ollama round-trip
    and one SQLite transaction beats N ``record_fact`` calls. Same semantics
    per item as ``record_fact`` — EXCEPT ``salient``, which is only supported by
    ``record_fact`` (batch items cannot be pinned; declare criticality with a
    dedicated ``record_fact(..., salient=True)`` call). Exact-SPO duplicates are
    touched and returned with ``already_existed=true``. For mutable-scalar slots
    use ``set_fact`` per item instead.

    Each item must have ``subject``, ``predicate``, ``object``; per-item
    ``session_id`` overrides the top-level one. Per-item ``namespace``
    overrides the top-level ``namespace`` the same way. Dedup is scoped
    per namespace (MemoryBricks Step 1).
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
    # Optional-id boundary: the top-level session_id is the fallback for
    # items that don't carry their own. A non-None non-string would be
    # forwarded to add_facts and silently mis-attribute every item in
    # the batch that relies on the fallback. Per-item session_id is
    # already validated below; this is the symmetric guard for the
    # top-level one.
    err = _validate_optional_id(session_id, "session_id")
    if err is not None:
        return err
    # MemoryBricks Step 1: top-level namespace, symmetric with the
    # top-level session_id. Per-item override is validated in the loop.
    err = _validate_optional_text(namespace, "namespace")
    if err is not None:
        return err
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
        # Per-item length cap symmetric with the single-fact path.
        # A batch with one giant `object` field used to embed the
        # whole batch and dwarf the rest of the SQLite txn — reject
        # the offending item with structured detail so the agent
        # can split or trim that one and resubmit the rest.
        too_long = [
            k for k in required if len(f[k]) > _MAX_FIELD_LEN
        ]
        if too_long:
            invalid.append({
                "index": i,
                "error": "field_too_long",
                "bad_fields": too_long,
                "got_lengths": {k: len(f[k]) for k in too_long},
                "limit": _MAX_FIELD_LEN,
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
        # MemoryBricks Step 1: per-item namespace type check. None or
        # missing falls back to top-level; non-string is rejected.
        if "namespace" in f and f["namespace"] is not None:
            if not isinstance(f["namespace"], str):
                invalid.append({
                    "index": i,
                    "error": "invalid_namespace",
                    "got_type": type(f["namespace"]).__name__,
                })
    if invalid:
        return {
            "ok": False,
            "error": "invalid_fact_item",
            "invalid": invalid,
            "required_fields": list(required),
        }

    # Invisible-char strip per item + post-sanitisation non-empty
    # re-check symmetric with record_fact / set_fact. A ZWSP-only
    # input would pass per-item type validation above but collapse
    # to "" after sanitisation. Collect every offending item index
    # so the caller can fix multiple bad items in one round trip
    # (same pattern as the invalid list above).
    triples: list[tuple[str, str, str]] = []
    emptied: list[dict] = []
    for i, f in enumerate(facts):
        clean_s = _sanitize_for_llm(f["subject"])
        clean_p = _sanitize_for_llm(f["predicate"])
        clean_o = _sanitize_for_llm(f["object"])
        bad = [
            name for name, val in (
                ("subject", clean_s),
                ("predicate", clean_p),
                ("object", clean_o),
            ) if not val.strip()
        ]
        if bad:
            emptied.append({
                "index": i,
                "error": "field_empty_after_sanitization",
                "bad_fields": bad,
            })
            continue
        triples.append((clean_s, clean_p, clean_o))
    if emptied:
        return {
            "ok": False,
            "error": "invalid_fact_item",
            "invalid": emptied,
            "hint": (
                "Items contained only invisible control characters "
                "or zero-width Unicode in the listed fields; after "
                "sanitisation those fields are empty. Provide "
                "visible non-empty content."
            ),
        }
    # Per-item session_id overrides the top-level one — honour the contract
    # spelled out in the docstring. Items without their own session_id fall
    # back to the top-level argument.
    per_item_sids = [f.get("session_id") for f in facts]
    per_item_ns = [f.get("namespace") for f in facts]
    try:
        statuses = _store.add_facts(
            triples,
            session_id=session_id,
            session_ids=per_item_sids,
            return_status=True,
            namespace=(namespace or ""),
            namespaces=per_item_ns,
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
    namespace: Optional[str] = None,
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
    # Optional-id boundary: symmetric with record_fact/record_facts.
    # set_fact's session_id flows into the supersede bookkeeping —
    # a non-string id would silently drop the attribution.
    err = _validate_optional_id(session_id, "session_id")
    if err is not None:
        return err
    # MemoryBricks Step 1: namespace is optional text — same shape as
    # in record_fact. ``None`` flows as "" downstream.
    err = _validate_optional_text(namespace, "namespace")
    if err is not None:
        return err
    # Invisible-char strip symmetric with record_fact / record_facts.
    subject = _sanitize_for_llm(subject)
    predicate = _sanitize_for_llm(predicate)
    object = _sanitize_for_llm(object)
    # Re-validate post-sanitisation symmetric with record_fact —
    # ZWSP-only inputs pass _validate_spo_strings but sanitiser
    # strips them to "".
    err = _check_non_empty_after_sanitize(
        {"subject": subject, "predicate": predicate, "object": object}
    )
    if err is not None:
        return err
    # set_fact -> add_fact -> embed(); wrap so an unreachable embedding
    # provider produces a structured failure instead of raw stacktrace
    # at the MCP boundary (completes the wrap coverage shared with
    # record_fact / record_facts / query_memory).
    try:
        return _store.set_fact(
            subject, predicate, object, session_id=session_id,
            namespace=(namespace or ""),
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
    namespace_prefix: Optional[str] = None,
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
        # MemoryBricks Step 1: hierarchical scope filter (VB-style).
        ("namespace_prefix", namespace_prefix),
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
    facts = _store.list_facts(
        subject=subject, predicate=predicate, limit=10_000,
        namespace_prefix=namespace_prefix,
    )
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
            # MemoryBricks Step 1: surface namespace so agents can
            # disambiguate same-SPO facts across scopes.
            "namespace": f.namespace or "",
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
    namespace_prefix: Optional[str] = None,
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
    # MemoryBricks Step 1: hierarchical scope filter.
    err = _validate_optional_text(namespace_prefix, "namespace_prefix")
    if err is not None:
        return {"query": text, "hits": [], **err}
    try:
        hits = _store.find_similar(
            text=text,
            top_k=top_k,
            min_similarity=min_similarity,
            subject_prefix=subject_prefix,
            namespace_prefix=namespace_prefix,
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

    Pass ``first_message`` (the user's opening text) to arm *deferred* echo
    detection: if the topic matches a past session at cosine ≥ 0.68, a pending
    marker is recorded — but no penalty is applied yet. ``session_close`` then
    decides: penalise the past session's facts only if this conversation also
    ends non-resonant (returned and still unresolved); cancel if it ends
    resonant (a productive revisit). The response reports ``echo.pending`` and
    ``echo.matched_session``; the outcome shows up in ``echo_outcome`` at close.

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
        # Strip invisible / control bytes — session text reaches the same
        # query_memory responses and LLM context as facts, so the same
        # write-boundary sanitisation applies (it didn't, before).
        first_message = _sanitize_for_llm(first_message)
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
            # Deferred echo: arm a pending marker; session_close decides by
            # outcome (see peek_echo). check_echo is the apply-now path.
            response["echo"] = _store.peek_echo(
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
    ``penalty=0``. This is the IMMEDIATE (apply-now) path. ``session_open(
    first_message=...)`` and ``record_session`` do NOT call it — they
    ``peek_echo`` (arm a pending marker) and let ``session_close`` decide by
    the current session's outcome. Use this tool when you deliberately want a
    detect-and-apply right now, without opening a session.
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
    text = _sanitize_for_llm(text)  # strip invisible/control bytes (write boundary)
    # Post-sanitize empty guard — symmetric with record_fact. _validate_text
    # checks non-emptiness BEFORE sanitize, so a pure zero-width payload
    # ("​") passes it, then strips to "" here. Without this re-check an
    # informationless empty message would land in the scored trajectory.
    err = _check_non_empty_after_sanitize({"text": text})
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
    except RuntimeError as exc:
        # The closing-session race gate in session_message raises
        # RuntimeError("session_closing: ...") when a push targets a
        # sid currently mid-close. Surface that as a structured
        # response with a clear retry hint instead of a raw
        # stacktrace; without this catch the new gate would still
        # do the right thing (reject the push) but the agent would
        # see an unstructured error and not know to wait or open
        # a fresh session.
        msg = str(exc)
        if "session_closing" in msg:
            return {
                "ok": False,
                "error": "session_closing",
                "session_id": session_id,
                "detail": msg,
                "hint": (
                    "session_close is in progress for this id. Wait "
                    "for it to complete (the close pops the sid; a "
                    "follow-up session_message will then get "
                    "unknown_session) or open a new session_id for "
                    "any further messages."
                ),
            }
        raise
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
        # confidence ∈ [0,1] and effective_r = r·confidence (what moved gravity).
        "confidence": summary.get("confidence"),
        "effective_r": summary.get("effective_r"),
        # none / applied / cancelled / noop
        "echo_outcome": summary.get("echo_outcome"),
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
    # Strip invisible/control bytes from every message (write boundary) —
    # session text reaches the same downstream context as facts.
    messages = [_sanitize_for_llm(m) for m in messages]
    # Post-sanitize empty guard — symmetric with the pre-sanitize
    # invalid_message_item check above and with record_fact. A pure
    # zero-width message ("​") passes the .strip() check above, then
    # strips to "" here; without this re-check it would still land in the
    # scored trajectory as a phantom neutral turn.
    emptied = [i for i, m in enumerate(messages) if not m.strip()]
    if emptied:
        return {
            "ok": False,
            "error": "field_empty_after_sanitization",
            "indices": emptied,
            "hint": (
                "one or more messages contained only invisible control "
                "characters or zero-width Unicode; after sanitisation they "
                "are empty. Provide visible non-empty content."
            ),
        }
    session_id = f"{agent_id}-{int(time.time())}-{uuid.uuid4().hex[:4]}"
    _store.session_start(session_id)
    # Peek the echo (arm a pending marker) and let session_close decide by
    # outcome — record_session has the whole conversation, so applying
    # immediately would penalise a productive revisit. Like session_open.
    echo = None
    try:
        if messages:
            echo = _store.peek_echo(messages[0], session_id=session_id)
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
        # Full close contract, symmetric with session_close (no one-shot drift):
        "scoring_source": summary.get("scoring_source"),
        "confidence": summary.get("confidence"),
        "effective_r": summary.get("effective_r"),
        # none / applied / cancelled / noop
        "echo_outcome": summary.get("echo_outcome"),
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
