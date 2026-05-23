"""run_forecast snapshot revalidation contract.

The forecast simulation is heavy (O(n²) per tick), so it runs
OUTSIDE the lock — that's a deliberate choice so concurrent agents
can keep querying. The race: another thread (or another process via
SQLite) can mutate body state between snapshot and writeback. The
writeback used to write scores from the stale snapshot into the
surviving subset of bodies, silently feeding the next tick's
adaptive gravity with values computed from a phantom past.

Fix: recompute the cache key (data_version + mutation_version +
body_count + horizon) inside the writeback's lock. If it differs
from the snapshot key, return forecast_snapshot_stale so the agent
can retry — the next call sees the post-mutation state.
"""
from __future__ import annotations

from birch.memory_store import MemoryStore

# --- Happy path: cache key matches → writeback proceeds ----------------


def test_forecast_writeback_happy_path(tmp_path):
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    mem.add_fact("api", "uses", "Postgres")
    mem.add_fact("api", "uses", "Redis")
    result = mem.run_forecast(horizon_ticks=5)
    # Sanity: forecast actually wrote scores back.
    assert result.get("bodies_updated", 0) >= 1
    assert result.get("ok") is not False
    mem.close()


# --- Race: mutation between snapshot and writeback ---------------------


def test_forecast_aborts_on_concurrent_mutation(tmp_path, monkeypatch):
    """Force the writeback to see a mutation that happened between
    snapshot and lock re-acquisition. Simulate by monkey-patching
    forecast_stability so that during the simulation step (which
    runs outside the lock) we add a new fact, bumping
    _mutation_version. The writeback's cache_key revalidation must
    detect the drift and return forecast_snapshot_stale."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    mem.add_fact("api", "uses", "Postgres")
    mem.add_fact("api", "uses", "Redis")

    # Patch the pure simulation function in _singularity to mutate
    # the store during compute — that's exactly what a concurrent
    # thread would do.
    # forecast_stability is imported lazily inside run_forecast via
    # `from ..galaxy.forecast import forecast_stability`, so the
    # patchable symbol lives in the galaxy.forecast module.
    import birch.galaxy.forecast as singmod
    original = singmod.forecast_stability

    def racing_forecast(bodies_snapshot, horizon_ticks):
        scores = original(bodies_snapshot, horizon_ticks=horizon_ticks)
        # Simulate a concurrent write happening between snapshot
        # and writeback. add_fact bumps mutation_version, which
        # the writeback's cache_key check will catch.
        mem.add_fact("svc", "added", "during-forecast")
        return scores

    monkeypatch.setattr(singmod, "forecast_stability", racing_forecast)

    result = mem.run_forecast(horizon_ticks=3)
    assert result.get("ok") is False, (
        "writeback ignored a concurrent mutation and wrote stale "
        "scores into post-mutation bodies"
    )
    assert result["error"] == "forecast_snapshot_stale"
    assert "snapshot_body_count" in result
    assert "writeback_body_count" in result
    assert result["writeback_body_count"] > result["snapshot_body_count"]
    # forecast_stability values on the seed facts must NOT have been
    # written — the writeback aborted before persisting anything.
    survivors = [f for f in mem.list_facts(subject="api")]
    # All survivors are still at the default neutral 0.5 (or whatever
    # was there before the call started) — definitely no score from
    # this aborted call landed.
    for f in survivors:
        assert f.forecast_stability == 0.5, (
            f"forecast_stability mutated despite aborted writeback: "
            f"{f.subject}/{f.predicate} = {f.forecast_stability}"
        )
    mem.close()


def test_forecast_retry_after_stale_returns_fresh_result(tmp_path, monkeypatch):
    """After a forecast_snapshot_stale response, the agent retries
    and gets a valid result reflecting the post-mutation state.
    Proves the abort doesn't leave the store in a broken cache state."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    mem.add_fact("api", "uses", "Postgres")

    # forecast_stability is imported lazily inside run_forecast via
    # `from ..galaxy.forecast import forecast_stability`, so the
    # patchable symbol lives in the galaxy.forecast module.
    import birch.galaxy.forecast as singmod
    original = singmod.forecast_stability
    fire_once = {"done": False}

    def racing_forecast_once(bodies_snapshot, horizon_ticks):
        scores = original(bodies_snapshot, horizon_ticks=horizon_ticks)
        if not fire_once["done"]:
            fire_once["done"] = True
            mem.add_fact("svc", "added", "during-first-forecast")
        return scores

    monkeypatch.setattr(singmod, "forecast_stability", racing_forecast_once)

    first = mem.run_forecast(horizon_ticks=3)
    assert first.get("error") == "forecast_snapshot_stale"
    # Retry: no concurrent mutation this time → clean writeback.
    second = mem.run_forecast(horizon_ticks=3)
    assert second.get("ok") is not False
    assert second["bodies_updated"] >= 1
    mem.close()
