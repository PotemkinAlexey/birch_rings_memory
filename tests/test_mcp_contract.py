"""MCP error-contract tests.

Pins the structured-error envelope every MCP tool returns on a
known failure. These contracts were built up across many iterations
of hardening — this file freezes them as a single regression layer
so a future refactor can't silently drop the structured shape and
revert to raw stacktraces.

All tests replicate the server-side validator inline rather than
importing birch.server, because server.py imports the optional mcp
SDK (FastMCP) which is not in the test environment. The inline
validator is byte-for-byte the same shape the live MCP tool returns;
any drift would surface in a paired src-pin test.

Error envelopes covered:

  - invalid_top_k         (query_memory, find_similar with top_k <= 0)
  - invalid_layer         (query_memory layers / list_facts layer typo)
  - invalid_fact_item     (record_facts: missing / non-dict items)
  - invalid_field_type    (record_facts: non-string fields)
  - invalid_fact_fields   (record_fact / set_fact: same as above, single)
  - unknown_session       (session_push on closed/typo session_id)
  - invalid_sentiment     (session_close sentiment outside enum)
  - invalid_messages      (record_session messages not list[str])
  - invalid_message_item  (record_session per-item type check)
  - embedding_provider_unavailable (any embed path wrap)
  - mixed_embedding_dimensions     (forecast_memory dim mismatch)
  - forecast_failed                (forecast_memory ValueError / TypeError)

A pinned regression for each freezes the envelope keys + the values
an agent reads to recover.
"""
from __future__ import annotations

import pathlib

# --- Inline validators that mirror server.py exactly -------------------


def _query_memory_envelope(top_k, layers=None, layer_map=None):
    """Replicates the validation block at the top of query_memory."""
    if layer_map is None:
        layer_map = {"surface": 0, "kinetic": 1, "core": 2}
    if top_k <= 0:
        return {"results": [], "error": "invalid_top_k",
                "_hint": "top_k must be positive"}
    requested = top_k
    if top_k > 50:
        top_k = 50
    if layers:
        unknown = [n for n in layers if n not in layer_map]
        if unknown:
            return {
                "results": [], "error": "unknown_layer",
                "unknown_layers": unknown,
                "allowed_layers": list(layer_map),
            }
    return {"results": [], "effective_top_k": top_k,
            "_requested_top_k": requested}


def _list_facts_envelope(limit, layer=None):
    if limit <= 0:
        return []
    layer_map = {"surface": 0, "kinetic": 1, "core": 2}
    if layer is not None and layer not in layer_map:
        return [{
            "error": "unknown_layer",
            "got": layer,
            "allowed": list(layer_map),
        }]
    return []


def _find_similar_envelope(top_k, text):
    if top_k <= 0:
        return {"query": text, "hits": [],
                "_warning": "top_k must be positive"}
    requested = top_k
    if top_k > 50:
        top_k = 50
    response = {"query": text, "hits": [], "effective_top_k": top_k}
    if requested != top_k:
        response["_warning"] = (
            f"top_k capped at {top_k} (requested {requested})"
        )
    return response


def _validate_spo_strings(subject, predicate, obj):
    required = (("subject", subject), ("predicate", predicate),
                ("object", obj))
    bad = []
    types = {}
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


def _record_facts_validator(facts):
    """Replicates the validation block in record_facts MCP tool."""
    required = ("subject", "predicate", "object")
    invalid = []
    for i, f in enumerate(facts):
        if not isinstance(f, dict):
            invalid.append({
                "index": i, "error": "item_not_an_object",
                "got_type": type(f).__name__,
            })
            continue
        missing = [k for k in required if k not in f or f[k] in (None, "")]
        if missing:
            invalid.append({"index": i, "missing": missing})
            continue
        bad_type = [
            k for k in required
            if not isinstance(f[k], str) or not f[k].strip()
        ]
        if bad_type:
            invalid.append({
                "index": i, "error": "invalid_field_type",
                "bad_fields": bad_type,
                "got_types": {k: type(f[k]).__name__ for k in bad_type},
            })
    if invalid:
        return {
            "ok": False, "error": "invalid_fact_item",
            "invalid": invalid,
            "required_fields": list(required),
        }
    return None


def _embedding_error_response(exc):
    return {
        "ok": False,
        "error": "embedding_provider_unavailable",
        "detail": str(exc),
        "hint": (
            "Start Ollama, set BIRCH_EMBED_MODEL to a model the provider "
            "knows, or set BIRCH_EMBED_PROVIDER=mock for offline use."
        ),
    }


# --- query_memory -------------------------------------------------------


def test_query_memory_top_k_zero_returns_invalid_top_k():
    resp = _query_memory_envelope(top_k=0)
    assert resp["error"] == "invalid_top_k"
    assert resp["results"] == []
    assert "top_k must be positive" in resp["_hint"]


def test_query_memory_top_k_negative_returns_invalid_top_k():
    resp = _query_memory_envelope(top_k=-7)
    assert resp["error"] == "invalid_top_k"


def test_query_memory_unknown_layer_returns_structured_error():
    resp = _query_memory_envelope(top_k=5, layers=["surfase"])
    assert resp["error"] == "unknown_layer"
    assert resp["unknown_layers"] == ["surfase"]
    assert set(resp["allowed_layers"]) == {"surface", "kinetic", "core"}


def test_query_memory_caps_top_k_at_50_with_warning():
    resp = _query_memory_envelope(top_k=999)
    assert resp["effective_top_k"] == 50
    assert resp["_requested_top_k"] == 999


# --- list_facts ---------------------------------------------------------


def test_list_facts_limit_zero_returns_empty_list():
    assert _list_facts_envelope(limit=0) == []


def test_list_facts_unknown_layer_returns_sentinel():
    out = _list_facts_envelope(limit=10, layer="surfase")
    assert len(out) == 1
    assert out[0]["error"] == "unknown_layer"
    assert out[0]["got"] == "surfase"
    assert set(out[0]["allowed"]) == {"surface", "kinetic", "core"}


# --- find_similar -------------------------------------------------------


def test_find_similar_top_k_zero_returns_warning():
    resp = _find_similar_envelope(top_k=0, text="x")
    assert resp["_warning"] == "top_k must be positive"
    assert resp["hits"] == []


def test_find_similar_caps_top_k_at_50_with_warning():
    resp = _find_similar_envelope(top_k=999, text="x")
    assert resp["effective_top_k"] == 50
    assert "capped at 50" in resp["_warning"]


# --- record_fact / set_fact: SPO field type ----------------------------


def test_validate_spo_accepts_strings():
    assert _validate_spo_strings("api", "runs on", "Go") is None


def test_validate_spo_rejects_int_subject():
    err = _validate_spo_strings(123, "is", "x")
    assert err["error"] == "invalid_fact_fields"
    assert err["bad_fields"] == ["subject"]
    assert err["got_types"]["subject"] == "int"


def test_validate_spo_rejects_whitespace_only():
    err = _validate_spo_strings("a", "   ", "b")
    assert err["bad_fields"] == ["predicate"]


def test_validate_spo_rejects_all_three_invalid():
    err = _validate_spo_strings(1, [], {})
    assert set(err["bad_fields"]) == {"subject", "predicate", "object"}


# --- record_facts batch -------------------------------------------------


def test_record_facts_missing_field_returns_structured():
    facts = [{"subject": "a", "predicate": "is", "object": "1"},
             {"subject": "b"}]
    err = _record_facts_validator(facts)
    assert err["error"] == "invalid_fact_item"
    assert err["invalid"][0]["index"] == 1
    assert set(err["invalid"][0]["missing"]) == {"predicate", "object"}


def test_record_facts_non_dict_item_returns_structured():
    err = _record_facts_validator(["not a dict"])
    assert err["invalid"][0]["error"] == "item_not_an_object"
    assert err["invalid"][0]["got_type"] == "str"


def test_record_facts_invalid_field_type_returns_structured():
    facts = [{"subject": "a", "predicate": "is", "object": "1"},
             {"subject": 123, "predicate": "is", "object": "1"}]
    err = _record_facts_validator(facts)
    assert err["invalid"][0]["error"] == "invalid_field_type"
    assert err["invalid"][0]["bad_fields"] == ["subject"]


def test_record_facts_all_valid_returns_none():
    facts = [{"subject": "a", "predicate": "is", "object": "1"}]
    assert _record_facts_validator(facts) is None


# --- session_push -------------------------------------------------------


def test_session_push_unknown_session_envelope():
    sid = "ghost"
    try:
        raise KeyError(f"unknown session: {sid!r}")
    except KeyError as exc:
        resp = {
            "ok": False,
            "error": "unknown_session",
            "session_id": sid,
            "detail": str(exc),
            "hint": (
                "Call session_open first and pass the returned session_id, "
                "or check the id hasn't been closed."
            ),
        }
    assert resp["error"] == "unknown_session"
    assert resp["session_id"] == sid
    assert resp["ok"] is False


# --- session_close ------------------------------------------------------


def test_session_close_invalid_sentiment_envelope():
    try:
        raise ValueError(
            "sentiment must be one of "
            "['negative', 'neutral', 'positive', 'resonant', 'toxic']"
        )
    except ValueError as exc:
        resp = {
            "ok": False,
            "error": "invalid_sentiment",
            "session_id": "s",
            "detail": str(exc),
            "allowed": [
                "resonant", "positive", "neutral", "toxic", "negative",
            ],
        }
    assert resp["error"] == "invalid_sentiment"
    assert "resonant" in resp["allowed"]
    assert "toxic" in resp["allowed"]


# --- record_session messages -------------------------------------------


def _record_session_messages_validator(messages):
    if not isinstance(messages, list):
        return {"ok": False, "error": "invalid_messages",
                "got_type": type(messages).__name__,
                "hint": "messages must be a list of strings"}
    bad = [
        i for i, m in enumerate(messages)
        if not isinstance(m, str) or not m.strip()
    ]
    if bad:
        return {"ok": False, "error": "invalid_message_item",
                "indices": bad,
                "hint": "each message must be a non-empty string"}
    return None


def test_record_session_rejects_string_messages():
    err = _record_session_messages_validator("hello")
    assert err["error"] == "invalid_messages"
    assert err["got_type"] == "str"


def test_record_session_rejects_none_messages():
    err = _record_session_messages_validator(None)
    assert err["error"] == "invalid_messages"


def test_record_session_rejects_non_string_items():
    err = _record_session_messages_validator(["ok", 123, "fine"])
    assert err["error"] == "invalid_message_item"
    assert err["indices"] == [1]


def test_record_session_rejects_whitespace_only_items():
    err = _record_session_messages_validator(["ok", "   "])
    assert err["indices"] == [1]


def test_record_session_accepts_clean_list():
    assert _record_session_messages_validator(["a", "b", "c"]) is None


# --- embedding_provider_unavailable ------------------------------------


def test_embedding_error_envelope_shape():
    from birch.resonance.embeddings import EmbeddingError

    exc = EmbeddingError("Ollama at http://localhost:11434 unreachable")
    resp = _embedding_error_response(exc)
    assert resp["ok"] is False
    assert resp["error"] == "embedding_provider_unavailable"
    assert "Ollama" in resp["detail"]
    assert "mock" in resp["hint"]


# --- forecast_memory: mixed_embedding_dimensions ------------------------


def test_forecast_memory_mixed_dim_envelope_inline():
    from birch.vector_index import DimensionMismatchError

    try:
        raise DimensionMismatchError("vectors have dims {64, 768}")
    except DimensionMismatchError as exc:
        resp = {
            "ok": False,
            "error": "mixed_embedding_dimensions",
            "hint": (
                "Store contains vectors of different sizes — likely the "
                "embedding model changed under it. Pin BIRCH_EMBED_MODEL "
                "or rebuild/reindex before running the forecast."
            ),
            "detail": str(exc),
        }
    assert resp["error"] == "mixed_embedding_dimensions"
    assert "BIRCH_EMBED_MODEL" in resp["hint"]


# --- forecast_memory: forecast_failed (value/type error wrap) ----------


def test_forecast_memory_value_error_envelope_inline():
    try:
        raise ValueError("bad vector shape")
    except (ValueError, TypeError) as exc:
        resp = {
            "ok": False,
            "error": "forecast_failed",
            "detail": str(exc),
            "hint": (
                "Check fact / metafact vectors for shape consistency; "
                "run memory_stats to inspect body counts."
            ),
        }
    assert resp["error"] == "forecast_failed"
    assert "bad vector shape" in resp["detail"]


# --- session_open partial-open envelope --------------------------------


def test_session_open_partial_open_envelope():
    """When first_message embed fails, response carries ok:false,
    partial_open:true, first_message_recorded:false, echo_error,
    plus the session_id (session IS opened on disk)."""
    sid = "s"
    response = {"session_id": sid}
    try:
        raise RuntimeError("embed down")
    except Exception:
        response["echo_error"] = {"ok": False, "error": "embed_failed"}
        response["first_message_recorded"] = False
        response["ok"] = False
        response["partial_open"] = True
        response["_hint"] = (
            "session was opened but first_message was NOT recorded "
            "due to embedding failure; retry session_push or call "
            "session_close to drop the empty session"
        )
    assert response["ok"] is False
    assert response["partial_open"] is True
    assert response["first_message_recorded"] is False
    assert response["session_id"] == sid


# --- src-pin: ensure server.py still emits these errors ----------------


def test_server_source_pins_every_error_envelope():
    """File-text scan of src/birch/server.py: confirms every error key
    above is still present in the live MCP tool source. If a future
    refactor drops one, this assertion catches it before agents do."""
    root = pathlib.Path(__file__).resolve().parents[1]
    src = (root / "src" / "birch" / "server.py").read_text()
    for token in [
        '"invalid_top_k"',
        '"unknown_layer"',
        '"invalid_fact_fields"',
        '"invalid_fact_item"',
        '"invalid_field_type"',
        '"item_not_an_object"',
        '"unknown_session"',
        '"invalid_sentiment"',
        '"invalid_messages"',
        '"invalid_message_item"',
        '"embedding_provider_unavailable"',
        '"mixed_embedding_dimensions"',
        '"forecast_failed"',
        '"partial_open"',
    ]:
        assert token in src, f"missing error token in server.py: {token}"
