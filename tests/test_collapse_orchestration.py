"""Counter-triggered collapse orchestration on MemoryStore."""
from __future__ import annotations

import time

from birch.fact import FactPassport
from birch.memory_store import MemoryStore


def _stuff_singularity(mem: MemoryStore, n: int, near: list[float]) -> None:
    """Push n near-duplicates of `near` directly into the black hole."""
    for i in range(n):
        f = FactPassport(subject=f"s{i}", predicate="p", object=f"o{i}",
                         fact_id=f"f-{i}")
        # Tiny perturbation so the cosine stays above 0.95.
        f.vector = [near[0] + i * 1e-5, near[1] - i * 1e-5, *near[2:]]
        mem._hole.absorb(f)


def test_collapse_singularity_synchronous_returns_report():
    mem = MemoryStore()
    _stuff_singularity(mem, n=10, near=[1.0, 0.0, 0.0])
    report = mem.collapse_singularity(threshold=0.95, persist=False)
    assert report.groups == 1
    assert report.absorbed_facts == 10
    assert mem._hole.fact_mass == 0
    assert mem._hole.meta_mass == 1
    assert mem._total_collapses == 1
    assert mem._last_collapse_at is not None
    assert mem._collapse_counter == 0
    mem.close()


def test_collapse_persists_metafact_and_drops_originals(tmp_path):
    db = tmp_path / "collapse.db"
    mem = MemoryStore(db_path=str(db))
    _stuff_singularity(mem, n=5, near=[1.0, 0.0, 0.0])
    # Storage didn't see the absorbed facts (they were synthetic — not
    # persisted on save_fact), so this exercises the new MetaFact saves.
    report = mem.collapse_singularity(threshold=0.95)
    assert report.groups == 1
    metas = mem._storage.load_meta_facts()
    assert len(metas) == 1
    assert metas[0].weight == 5
    assert len(metas[0].source_fact_ids) == 5
    mem.close()


def test_counter_trigger_fires_sync_when_thresholds_met():
    """With async disabled, exceeding both thresholds runs collapse inline."""
    mem = MemoryStore(collapse_async=False)
    # Lower the thresholds for the test.
    mem.COLLAPSE_FACT_MASS_TRIGGER = 5
    mem.COLLAPSE_DELTA_TRIGGER = 5
    _stuff_singularity(mem, n=10, near=[1.0, 0.0, 0.0])

    # Simulate absorption deltas accumulating: _maybe_trigger_collapse_locked
    # is the production path called from session_close.
    with mem._lock:
        mem._maybe_trigger_collapse_locked(absorbed_count=10)

    assert mem._total_collapses == 1
    assert mem._hole.fact_mass == 0
    assert mem._hole.meta_mass == 1
    mem.close()


def test_counter_trigger_does_not_fire_below_thresholds():
    mem = MemoryStore(collapse_async=False)
    mem.COLLAPSE_FACT_MASS_TRIGGER = 100
    mem.COLLAPSE_DELTA_TRIGGER = 50
    _stuff_singularity(mem, n=10, near=[1.0, 0.0, 0.0])
    with mem._lock:
        mem._maybe_trigger_collapse_locked(absorbed_count=5)
    assert mem._total_collapses == 0
    assert mem._collapse_counter == 5
    assert mem._hole.fact_mass == 10
    mem.close()


def test_counter_trigger_async_runs_collapse_in_background():
    """With async on, collapse runs on a worker thread; we wait for it."""
    mem = MemoryStore(collapse_async=True)
    mem.COLLAPSE_FACT_MASS_TRIGGER = 5
    mem.COLLAPSE_DELTA_TRIGGER = 5
    _stuff_singularity(mem, n=10, near=[1.0, 0.0, 0.0])

    with mem._lock:
        mem._maybe_trigger_collapse_locked(absorbed_count=10)

    # Wait on the future the executor was given.
    assert mem._inflight_collapse is not None
    mem._inflight_collapse.result(timeout=5.0)

    assert mem._total_collapses == 1
    assert mem._hole.meta_mass == 1
    mem.close()


def test_concurrent_trigger_does_not_double_schedule():
    """Calling the trigger twice while one collapse is inflight queues at most one."""
    mem = MemoryStore(collapse_async=True)
    mem.COLLAPSE_FACT_MASS_TRIGGER = 5
    mem.COLLAPSE_DELTA_TRIGGER = 5
    _stuff_singularity(mem, n=10, near=[1.0, 0.0, 0.0])

    with mem._lock:
        mem._maybe_trigger_collapse_locked(absorbed_count=10)
        first_future = mem._inflight_collapse
        # Second trigger immediately: must not replace the first future.
        mem._maybe_trigger_collapse_locked(absorbed_count=10)
        assert mem._inflight_collapse is first_future

    first_future.result(timeout=5.0)
    mem.close()


def test_stats_exposes_collapse_state():
    mem = MemoryStore(collapse_async=False)
    s = mem.stats
    assert "collapse_counter" in s
    assert "total_collapses" in s
    assert "last_collapse_at" in s
    assert s["total_collapses"] == 0
    assert s["last_collapse_at"] is None
    mem.close()


def test_close_shuts_down_executor_safely():
    mem = MemoryStore(collapse_async=True)
    mem.COLLAPSE_FACT_MASS_TRIGGER = 5
    mem.COLLAPSE_DELTA_TRIGGER = 5
    _stuff_singularity(mem, n=8, near=[1.0, 0.0, 0.0])
    with mem._lock:
        mem._maybe_trigger_collapse_locked(absorbed_count=8)

    # Should not hang or raise.
    mem.close()
    assert mem._collapse_executor is None
