"""DeepSeek round-1 punch-list regressions.

First DeepSeek review (after 9 ChatGPT rounds + Codex + Cursor x2 +
chemist/professor) found multi-process AdaptiveWeights race, missing
echo metrics, and the absence of a forecast_memory cache. The first
is a real correctness bug (concurrent processes silently overwrite
each other's SGD steps); the other two are operational hardening.
"""
from __future__ import annotations

from birch.adaptive_gravity import AdaptiveWeights
from birch.memory_store import MemoryStore
from birch.storage.sqlite import SQLiteBackend

# --- P1: AdaptiveWeights reload before SGD step --------------------------


def test_session_close_picks_up_concurrent_weight_writes(tmp_path):
    """Process A boots, process B boots, B trains 5 steps and persists.
    A closes a resonant session — its SGD step must compose on top of
    B's learning (train_count == 6), not overwrite it back to 1."""
    db = str(tmp_path / "m.db")

    # Process A: open MemoryStore but do not let it train yet.
    mem_a = MemoryStore(db_path=db)
    mem_a.add_fact("api", "runs on", "Go")
    mem_a.add_fact("db", "is", "Postgres")

    # Process B: independently trains weights 5 times and persists.
    backend_b = SQLiteBackend(db)
    weights_b = AdaptiveWeights.from_prior()
    for _ in range(5):
        weights_b.update(
            freshness=1.0, access=0.0, graph=0.0, utility=0.0,
            stability=0.0, target=1.0,
        )
    backend_b.save_adaptive_weights(weights_b)
    backend_b.close()

    # A's in-memory weights are still at train_count=0; the next
    # session_close must reload before stepping.
    assert mem_a._engine.weights.train_count == 0

    mem_a.session_start("s")
    mem_a.session_message("how do I connect api to postgres")
    mem_a.query("api Go", session_id="s")
    mem_a.session_message("perfect, exactly what i needed, thanks")
    summary = mem_a.session_close(session_id="s")
    assert summary.get("label") == "resonant"

    # After reload+step: train_count must be at least B's 5 + 1.
    # Without the fix it would be 1 (A overwrote B with its own boot
    # copy plus one step).
    weights_after = mem_a.stats["adaptive_weights"]
    assert weights_after["train_count"] >= 6
    mem_a.close()


# --- P2: echo metrics in memory_stats ------------------------------------


def test_echo_metrics_surface_in_memory_stats(tmp_path):
    """memory_stats exposes total_echoes_{detected,applied,ignored}."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    stats = mem.stats
    assert "total_echoes_detected" in stats
    assert "total_echoes_applied" in stats
    assert "total_echoes_ignored" in stats
    assert "echo_sessions" in stats
    assert stats["total_echoes_detected"] == 0
    assert stats["total_echoes_applied"] == 0
    assert stats["total_echoes_ignored"] == 0
    mem.close()


def test_echo_detected_and_applied_counters_increment(tmp_path):
    """A toxic session followed by an echo bumps detected + applied;
    a second echo on the same matched session bumps ignored."""
    from birch.resonance.echo import EchoStore

    store = EchoStore()
    # Record one past session with positive r_score, fact_weights non-empty.
    store.record(
        "past",
        all_vectors=[[1.0, 0.0, 0.0]],
        r_score=0.5,
        fact_weights={"f1": 1.0},
    )
    # First echo — detected + applied.
    res1 = store.detect_echo([1.0, 0.0, 0.0])
    assert res1.label == "echo"
    assert res1.penalty != 0.0
    assert store.total_echoes_detected == 1
    assert store.total_echoes_applied == 1
    assert store.total_echoes_ignored == 0

    # Second echo — same match, already penalised — ignored.
    res2 = store.detect_echo([1.0, 0.0, 0.0])
    assert res2.label == "echo"
    assert res2.penalty == 0.0
    assert store.total_echoes_detected == 2
    assert store.total_echoes_applied == 1
    assert store.total_echoes_ignored == 1


def test_echo_clean_does_not_bump_detected(tmp_path):
    """A miss below threshold returns label='clean' without bumping
    any of the detected/applied/ignored counters."""
    from birch.resonance.echo import EchoStore

    store = EchoStore()
    store.record("past", [[1.0, 0.0, 0.0]], 0.5, fact_weights={"f1": 1.0})
    # Orthogonal vector → similarity 0, below threshold.
    res = store.detect_echo([0.0, 1.0, 0.0])
    assert res.label == "clean"
    assert store.total_echoes_detected == 0
    assert store.total_echoes_applied == 0
    assert store.total_echoes_ignored == 0


# --- P2: forecast_memory cache by data_version ---------------------------


def test_forecast_memory_cache_hits_on_repeat_call(tmp_path):
    """Two back-to-back run_forecast calls with no intervening write
    return the same payload; the second is marked cached=True."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    mem.add_fact("api", "runs on", "Go")
    mem.add_fact("db", "is", "Postgres")

    first = mem.run_forecast(horizon_ticks=5)
    assert first["cached"] is False
    second = mem.run_forecast(horizon_ticks=5)
    assert second["cached"] is True
    # Same forecasted body count + horizon → same shape.
    assert second["bodies_forecasted"] == first["bodies_forecasted"]
    assert second["horizon_ticks"] == first["horizon_ticks"]
    mem.close()


def test_forecast_memory_cache_invalidates_on_write(tmp_path):
    """Adding a fact bumps data_version → cache miss on next call."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    mem.add_fact("api", "runs on", "Go")
    first = mem.run_forecast(horizon_ticks=5)
    assert first["cached"] is False
    # New fact mutates state — cache must invalidate.
    mem.add_fact("cache", "is", "redis")
    second = mem.run_forecast(horizon_ticks=5)
    assert second["cached"] is False
    assert second["bodies_forecasted"] == first["bodies_forecasted"] + 1
    mem.close()


def test_forecast_memory_cache_invalidates_on_horizon_change(tmp_path):
    """Different horizon_ticks → cache miss (different prediction)."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    mem.add_fact("api", "runs on", "Go")
    first = mem.run_forecast(horizon_ticks=5)
    assert first["cached"] is False
    second = mem.run_forecast(horizon_ticks=10)
    assert second["cached"] is False
    mem.close()
