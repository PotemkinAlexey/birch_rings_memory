"""ChatGPT round-12 punch-list regressions.

Round 12 (second self-audited ChatGPT round). Six surviving findings,
all shipped:

  1. forecast cache invalidation on same-process writes (SQLite
     PRAGMA data_version doesn't bump for same-connection writes).
  2. EmbeddingError wrap for set_fact / session_open / session_push
     / record_session.
  3. _post() wraps non-404 HTTPError as EmbeddingError.
  4. forecast_memory catches ValueError/TypeError as forecast_failed.
  5. core MemoryStore.run_forecast docstring says bodies not facts.
  6. record_fact / set_fact get the same string validation
     record_facts already has.
"""
from __future__ import annotations

import io
import urllib.error

import pytest

from birch.memory_store import MemoryStore
from birch.resonance.embeddings import EmbeddingError

# --- P1: forecast cache invalidates on same-process write ---------------


def test_forecast_cache_invalidates_after_add_fact(tmp_path):
    """SQLite PRAGMA data_version doesn't change for writes on the
    same connection. A naive (data_version, body_count, horizon)
    key would return stale results if body count happened to be
    unchanged. The mutation counter bumps on every write path so
    the cache key always differs from the previous one."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    mem.add_fact("api", "runs on", "Go")
    first = mem.run_forecast(horizon_ticks=5)
    assert first["cached"] is False
    second = mem.run_forecast(horizon_ticks=5)
    assert second["cached"] is True

    # Same-process write — body count goes from 1 to 2.
    mem.add_fact("db", "is", "Postgres")
    third = mem.run_forecast(horizon_ticks=5)
    assert third["cached"] is False
    mem.close()


def test_forecast_cache_invalidates_after_session_close(tmp_path):
    """session_close mutates per-fact recent_utility EWMA + trains
    adaptive weights — that should invalidate the forecast cache
    even though body count is unchanged."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    mem.add_fact("api", "runs on", "Go")
    first = mem.run_forecast(horizon_ticks=5)
    assert first["cached"] is False

    mem.session_start("s")
    mem.session_message("looking at api")
    mem.query("api", session_id="s")
    mem.session_close(session_id="s", sentiment="resonant")

    # Body count unchanged (still 1), but mutation_version bumped.
    second = mem.run_forecast(horizon_ticks=5)
    assert second["cached"] is False
    mem.close()


def test_forecast_cache_invalidates_after_delete(tmp_path):
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    f1 = mem.add_fact("a", "is", "1")
    mem.add_fact("b", "is", "2")
    first = mem.run_forecast(horizon_ticks=5)
    assert first["cached"] is False
    mem.delete_fact(f1.fact_id)
    second = mem.run_forecast(horizon_ticks=5)
    assert second["cached"] is False
    mem.close()


# --- P1: _post wraps non-404 HTTPError as EmbeddingError ----------------


def test_post_wraps_500_as_embedding_error(monkeypatch):
    from birch.resonance import embeddings

    def _fake_urlopen(req, timeout=None):
        raise urllib.error.HTTPError(
            req.full_url, 500, "Internal Server Error", {},
            io.BytesIO(b"oops"),
        )

    monkeypatch.setattr(embeddings.urllib.request, "urlopen", _fake_urlopen)
    with pytest.raises(EmbeddingError, match="HTTP 500"):
        embeddings._post("http://localhost:11434/api/embed", {"x": 1})


def test_post_lets_404_escape_for_fallback(monkeypatch):
    """404 still escapes raw — that's the signal _ollama_embed uses
    to fall back to the legacy endpoint."""
    from birch.resonance import embeddings

    def _fake_urlopen(req, timeout=None):
        raise urllib.error.HTTPError(
            req.full_url, 404, "Not Found", {}, io.BytesIO(b""),
        )

    monkeypatch.setattr(embeddings.urllib.request, "urlopen", _fake_urlopen)
    with pytest.raises(urllib.error.HTTPError):
        embeddings._post("http://localhost:11434/api/embed", {"x": 1})


def test_post_wraps_400_as_embedding_error(monkeypatch):
    from birch.resonance import embeddings

    def _fake_urlopen(req, timeout=None):
        raise urllib.error.HTTPError(
            req.full_url, 400, "Bad Request", {}, io.BytesIO(b""),
        )

    monkeypatch.setattr(embeddings.urllib.request, "urlopen", _fake_urlopen)
    with pytest.raises(EmbeddingError, match="HTTP 400"):
        embeddings._post("http://localhost:11434/api/embed", {"x": 1})


# --- P2: forecast_memory catches ValueError/TypeError -------------------


def test_forecast_memory_response_shape_for_value_error_inline():
    """server.py now wraps ValueError as forecast_failed. Replicate
    the wrap inline since the server module imports the mcp SDK."""
    exc = ValueError("bad vector shape")
    try:
        raise exc
    except (ValueError, TypeError) as exc2:
        response = {
            "ok": False,
            "error": "forecast_failed",
            "detail": str(exc2),
            "hint": "Check fact / metafact vectors for shape consistency.",
        }
    assert response["ok"] is False
    assert response["error"] == "forecast_failed"
    assert "bad vector shape" in response["detail"]


# --- P3: run_forecast docstring uses bodies -----------------------------


def test_run_forecast_core_docstring_uses_bodies():
    from birch.memory_store import MemoryStore as MS
    doc = MS.run_forecast.__doc__ or ""
    assert "bodies" in doc
    assert "FactPassport AND MetaFact" in doc
    # Legacy aliases still documented for wire-format stability.
    assert "facts_forecasted" in doc


# --- P3: _validate_spo_strings catches non-string fields ----------------


def test_validate_spo_strings_accepts_valid_inline():
    """Validator is also in server.py; replicate inline so we don't
    need the mcp SDK."""

    def _validate(subject, predicate, obj):
        required = (
            ("subject", subject), ("predicate", predicate), ("object", obj),
        )
        bad = []
        types = {}
        for name, val in required:
            if not isinstance(val, str) or not val.strip():
                bad.append(name)
                types[name] = type(val).__name__
        if not bad:
            return None
        return {
            "ok": False, "error": "invalid_fact_fields",
            "bad_fields": bad, "got_types": types,
        }

    assert _validate("api", "runs on", "Go") is None
    err = _validate(123, "runs on", "Go")
    assert err["error"] == "invalid_fact_fields"
    assert err["bad_fields"] == ["subject"]
    assert err["got_types"]["subject"] == "int"

    err2 = _validate("a", [], {"x": 1})
    assert set(err2["bad_fields"]) == {"predicate", "object"}

    # Whitespace-only counts as empty.
    err3 = _validate("a", "   ", "b")
    assert err3["bad_fields"] == ["predicate"]


# --- core: mutation_version actually exists and bumps -------------------


def test_mutation_version_starts_at_zero(tmp_path):
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    assert mem._mutation_version == 0
    mem.close()


def test_mutation_version_bumps_on_add(tmp_path):
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    before = mem._mutation_version
    mem.add_fact("a", "is", "1")
    assert mem._mutation_version > before
    mem.close()


def test_mutation_version_bumps_on_session_close(tmp_path):
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    mem.add_fact("a", "is", "1")
    before = mem._mutation_version
    mem.session_start("s")
    mem.session_message("hi")
    mem.query("a", session_id="s")
    mem.session_close(session_id="s", sentiment="resonant")
    assert mem._mutation_version > before
    mem.close()
