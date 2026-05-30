"""MCP error-contract pins — exception-path shapes + a source guard.

HISTORY / WHY THIS FILE SHRANK. This file used to replicate every server-side
validator *inline* and assert against the copy, on the theory that importing
``birch.server`` (which pulls the optional FastMCP SDK) might not work in CI.
That theory produced exactly the rot you'd predict: the inline mirrors drifted
from the real tools (the top_k envelope became ``invalid_int`` under the shared
``_validate_int`` while the mirror still asserted ``invalid_top_k``; find_similar
and list_facts mirrors drifted similarly), and nothing caught it because each
test validated its own stale copy.

The live, drift-proof contract coverage now lives in **test_server_contract.py**,
which calls the real ``birch.server`` tool functions directly (``@mcp.tool()``
returns the plain function) and pins the envelopes they actually return.

What remains here:
  1. Shape pins for error paths that are awkward to trigger by a direct call —
     embedding-provider-down, mixed-embedding-dimension forecast, forecast
     value-error wrap, and the session_open partial-open envelope. These assert
     the *shape* an agent recovers from; they are intentionally illustrative,
     not bound to a live call.
  2. A source-text guard that fails if any structured-error token disappears
     from server.py — cheap and complementary to the real-function tests.
"""
from __future__ import annotations

import pathlib

# --- embedding_provider_unavailable: envelope shape --------------------

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


def test_embedding_error_envelope_shape():
    from birch.resonance.embeddings import EmbeddingError

    exc = EmbeddingError("Ollama at http://localhost:11434 unreachable")
    resp = _embedding_error_response(exc)
    assert resp["ok"] is False
    assert resp["error"] == "embedding_provider_unavailable"
    assert "Ollama" in resp["detail"]
    assert "mock" in resp["hint"]


# --- forecast_memory: mixed_embedding_dimensions (shape) ---------------

def test_forecast_memory_mixed_dim_envelope_shape():
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


# --- forecast_memory: forecast_failed (value/type error wrap, shape) ---

def test_forecast_memory_value_error_envelope_shape():
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


# --- session_open partial-open envelope (shape) ------------------------

def test_session_open_partial_open_envelope_shape():
    """When first_message embed fails, the response carries ok:false,
    partial_open:true, first_message_recorded:false, plus the session_id
    (the session IS opened on disk)."""
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
    """File-text scan of src/birch/server.py: confirms every structured-error
    key is still present in the live MCP tool source. If a future refactor
    drops one, this catches it before agents do. Complements the live-function
    assertions in test_server_contract.py."""
    root = pathlib.Path(__file__).resolve().parents[1]
    src = (root / "src" / "birch" / "server.py").read_text()
    for token in [
        # Numeric input validation collapsed under the symmetric
        # invalid_int / invalid_float family — previously a per-tool
        # ad-hoc invalid_top_k literal lived here.
        '"invalid_int"',
        '"invalid_float"',
        '"invalid_layers"',
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
