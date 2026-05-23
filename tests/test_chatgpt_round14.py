"""ChatGPT round-14 punch-list regressions.

Round 14 was another partial-stale-snapshot round — 5 of 7 findings
were already shipped in rounds 12 and 13 (set_fact wrap, record_session
abort, session_open partial flag, non-404 HTTPError wrap, README
compactor wording). 2 genuinely new findings shipped:

  1. _touch_existing (the dedupe path of add_fact, and the touch
     on every query hit) updates access_count/last_accessed which
     feed gravity → galaxy → forecast_stability. Without a
     mutation_version bump, repeat queries / duplicate add_fact
     calls would not invalidate the forecast cache.
  2. _reload (cross-process sync) rebuilds in-memory state from
     disk; defensive cache reset removes any subtle window where
     a multi-process race could leave the same data_version value
     observable on both sides momentarily.

The shared _bump_mutation_locked helper makes future write paths
unable to forget the bump+drop pair (which is exactly the trap
round 12's scatter fell into).
"""
from __future__ import annotations

from birch.memory_store import MemoryStore

# --- P1: _touch_existing bumps mutation_version -------------------------


def test_repeat_add_fact_invalidates_forecast_cache(tmp_path):
    """Adding an existing SPO touches the existing fact and updates
    access_count + last_accessed. Those feed gravity → galaxy →
    forecast_stability — the cache must invalidate."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    mem.add_fact("api", "runs on", "Go")
    first = mem.run_forecast(horizon_ticks=5)
    assert first["cached"] is False
    cached = mem.run_forecast(horizon_ticks=5)
    assert cached["cached"] is True

    # Repeat add_fact with the same SPO — _touch_existing fires,
    # mutation_version bumps, cache invalidates.
    mem.add_fact("api", "runs on", "Go")
    after_touch = mem.run_forecast(horizon_ticks=5)
    assert after_touch["cached"] is False
    mem.close()


def test_query_hit_invalidates_forecast_cache(tmp_path):
    """A query that hits a fact also touches it (round-1 attribution
    contract). That mutation needs to invalidate the cache too."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    mem.add_fact("api", "runs on", "Go")
    mem.run_forecast(horizon_ticks=5)
    cached = mem.run_forecast(horizon_ticks=5)
    assert cached["cached"] is True

    mem.session_start("s")
    mem.query("api", session_id="s")
    # query touch -> _touch_existing -> bump
    after = mem.run_forecast(horizon_ticks=5)
    assert after["cached"] is False
    mem.close()


# --- P2: _reload defensively clears forecast cache ----------------------


def test_reload_clears_forecast_cache(tmp_path):
    """_reload rebuilds in-memory state. Forecast cache should be
    cleared even though the cache key would normally catch this via
    data_version mismatch — defensive invariant."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    mem.add_fact("api", "runs on", "Go")
    mem.run_forecast(horizon_ticks=5)
    assert mem._forecast_cache is not None

    # Simulate cross-process sync.
    mem._reload()
    assert mem._forecast_cache is None
    mem.close()


# --- helper: _bump_mutation_locked is the shared invariant --------------


def test_bump_mutation_helper_increments_and_drops_cache(tmp_path):
    """Direct test of the helper contract."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    mem.add_fact("api", "runs on", "Go")
    mem.run_forecast(horizon_ticks=5)
    assert mem._forecast_cache is not None
    before = mem._mutation_version

    mem._bump_mutation_locked()
    assert mem._mutation_version == before + 1
    assert mem._forecast_cache is None
    mem.close()


# --- meta: round 14 stale-snapshot artifact (5/7 already shipped) ------


def test_round14_stale_items_actually_already_shipped(tmp_path):
    """5 of the 7 round-14 findings were closed in rounds 12 and 13.
    This test pins each as a regression — if any of them ever
    actually regresses, the assertion fires."""

    # Round 12: set_fact wraps EmbeddingError. Inline marker — the
    # server module imports the mcp SDK, so we just confirm the
    # error_response helper exists.
    import pathlib
    server_src = (
        pathlib.Path(__file__).resolve().parents[1]
        / "src" / "birch" / "server.py"
    ).read_text()
    assert "_embedding_error_response" in server_src
    # set_fact does wrap.
    assert (
        "return _store.set_fact(\n"
        "            subject, predicate, object, session_id=session_id,\n"
        "        )\n"
        "    except EmbeddingError as exc:\n"
        "        return _embedding_error_response(exc)"
    ) in server_src

    # Round 13: record_session aborts on embed failure.
    assert "_store.abort_session(session_id)" in server_src
    # Round 13: session_open carries first_message_recorded.
    assert "first_message_recorded" in server_src

    # Round 12: _post wraps non-404 HTTPError.
    embeddings_src = (
        pathlib.Path(__file__).resolve().parents[1]
        / "src" / "birch" / "resonance" / "embeddings.py"
    ).read_text()
    assert "if exc.code == 404:" in embeddings_src
    assert "Ollama HTTP {exc.code}" in embeddings_src

    # Round 13: README per-dim compactor note.
    readme = (
        pathlib.Path(__file__).resolve().parents[1] / "README.md"
    ).read_text()
    assert "partitioned by vector dimension" in readme
