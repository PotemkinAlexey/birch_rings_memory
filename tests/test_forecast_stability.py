"""forecast_stability — galaxy forecast wired in as the 5th adaptive feature.

The galaxy was the telescope; this test pins it as a producer too. Each
fact gets a forecast in [0, 1] that lands on FactPassport.forecast_stability
and is consumed by compute_gravity via w_stability.
"""
from __future__ import annotations

from birch.fact import FactPassport
from birch.galaxy.forecast import forecast_stability
from birch.memory_store import MemoryStore
from birch.storage.sqlite import SQLiteBackend


def _toy_facts(n: int) -> list[FactPassport]:
    """A handful of facts with deterministic embeddings — enough for the
    galaxy loader to place bodies. We use 8-D so PCA has something to work with.
    """
    out: list[FactPassport] = []
    for i in range(n):
        f = FactPassport(subject=f"s{i}", predicate="rel", object=f"o{i}")
        f.vector = [
            float(((i * j) % 7) - 3) for j in range(8)
        ]
        out.append(f)
    return out


def test_default_forecast_is_neutral_prior():
    f = FactPassport("api", "runs on", "Go")
    assert f.forecast_stability == 0.5


def test_forecast_stability_returns_value_per_fact():
    facts = _toy_facts(20)
    scores = forecast_stability(facts, horizon_ticks=20)
    # Every fact got a score (the loader places everyone, even no-vector).
    assert set(scores.keys()) == {f.fact_id for f in facts}
    for s in scores.values():
        assert 0.0 <= s <= 1.0


def test_forecast_empty_input_returns_empty():
    assert forecast_stability([], horizon_ticks=10) == {}


def test_sqlite_roundtrips_forecast_stability(tmp_path):
    backend = SQLiteBackend(str(tmp_path / "m.db"))
    f = FactPassport("api", "runs on", "Go")
    f.forecast_stability = 0.73
    backend.save_fact(f)
    loaded = {x.fact_id: x for x in backend.load_facts()}[f.fact_id]
    backend.close()
    assert abs(loaded.forecast_stability - 0.73) < 1e-9


def test_run_forecast_writes_back_to_facts_and_persists(tmp_path):
    """End-to-end: MemoryStore.run_forecast updates facts and survives restart."""
    db = str(tmp_path / "m.db")
    mem = MemoryStore(db_path=db)
    triples = [(f"subject{i}", "rel", f"object{i}") for i in range(15)]
    mem.add_facts(triples)
    # All facts start at the neutral prior.
    for f in mem.list_facts(limit=100):
        assert f.forecast_stability == 0.5

    summary = mem.run_forecast(horizon_ticks=20)
    assert summary["facts_forecasted"] == 15
    assert summary["facts_updated"] == 15
    # Distribution buckets sum to total.
    dist = summary["distribution"]
    assert sum(dist.values()) == 15

    # At least one fact must have moved off the 0.5 prior — the galaxy
    # always disperses bodies by radius.
    moved = [
        f for f in mem.list_facts(limit=100)
        if abs(f.forecast_stability - 0.5) > 1e-6
    ]
    assert moved, "expected the forecast to move at least one fact"

    # Persists across a reopen.
    mem.close()
    again = MemoryStore(db_path=db)
    persisted = {
        f.fact_id: f.forecast_stability for f in again.list_facts(limit=100)
    }
    for f in moved:
        assert abs(persisted[f.fact_id] - f.forecast_stability) < 1e-9
