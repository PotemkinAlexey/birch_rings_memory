"""Per-layer race fixes (server + storage symmetry) regressions.

Catches the same race-prone already_existed bug in record_fact
(server layer) that set_fact fixed in the core, plus a couple of API
clarity items (deprecate result, record_facts created field).
"""
from __future__ import annotations

from birch.fact import FactPassport
from birch.memory_store import MemoryStore

# --- P1: record_fact response carries transaction-honest signal ----------


def test_record_fact_already_existed_transaction_honest_via_add_fact(tmp_path):
    """The MCP record_fact tool reads already_existed from add_fact's
    return_status, not a race-prone pre-check. We exercise the
    underlying contract: add_fact(return_status=True) returns
    created=True iff this call inserted, created=False if another
    branch (race winner or pre-existing) returned an existing fact.
    server.record_fact derives already_existed = not created so the
    same SPO sent twice gets created=False on the second call without
    any race window between probe and insert.
    """
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    f1, created1 = mem.add_fact(
        "api", "runs on", "Go", return_status=True,
    )
    assert created1 is True
    f2, created2 = mem.add_fact(
        "api", "runs on", "Go", return_status=True,
    )
    assert created2 is False
    assert f1.fact_id == f2.fact_id


# --- P1: deprecate returns supersede_fact's dict -------------------------


def test_deprecate_returns_supersede_result(tmp_path):
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    old = mem.add_fact("HEAD", "is", "abc")
    new = mem.add_fact("HEAD", "is", "def")

    result = mem.deprecate(old.fact_id, new.fact_id)
    assert isinstance(result, dict)
    assert result.get("superseded") is True
    assert result.get("old_id") == old.fact_id
    assert result.get("new_id") == new.fact_id


def test_deprecate_returns_dict_on_unknown_id(tmp_path):
    """Caller can now tell whether the legacy deprecate actually worked,
    instead of getting None and assuming success."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    out = mem.deprecate("nonexistent", "also-nonexistent")
    assert isinstance(out, dict)
    assert out.get("superseded") is False


# --- P2: add_facts return_status carries created semantics ---------------


def test_add_facts_return_status_includes_full_lifecycle_signal(tmp_path):
    """add_facts(return_status=True) returns per-item
    {fact, already_existed, duplicate_in_batch} from which the server
    derives the explicit `created` field on record_facts response.
    Verify the underlying signals are precise."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    mem.add_fact("api", "uses", "Redis")
    statuses = mem.add_facts(
        [
            ("api", "uses", "Redis"),      # pre-existing
            ("api", "uses", "Postgres"),   # genuinely new
            ("api", "uses", "Postgres"),   # in-batch duplicate
        ],
        return_status=True,
    )
    assert statuses[0]["already_existed"] is True
    assert statuses[0]["duplicate_in_batch"] is False
    assert statuses[1]["already_existed"] is False
    assert statuses[1]["duplicate_in_batch"] is False
    assert statuses[2]["already_existed"] is False
    assert statuses[2]["duplicate_in_batch"] is True

    # The server's `created` derivation:
    #   created = not already_existed AND not duplicate_in_batch
    created_flags = [
        (not s["already_existed"] and not s["duplicate_in_batch"])
        for s in statuses
    ]
    # Pre-existing → False (existed before call)
    # Genuinely new → True
    # In-batch dup → False (the same fact, the batch already created it)
    assert created_flags == [False, True, False]


# --- P3: forecast_memory docstring sanity --------------------------------


def test_forecast_memory_runtime_covers_metafacts():
    """Indirect sanity: run_forecast covers MetaFacts (already pinned
    by the forecast-polymorphism tests). This test confirms the
    runtime contract didn't regress while the docstring was updated."""
    from birch.meta_fact import MetaFact

    mem = MemoryStore()
    f = FactPassport("api", "runs on", "Go")
    f.vector = [0.1] * 64
    mem._facts[f.fact_id] = f
    mem._engine.register(f)
    mem._index.add(f.fact_id, f.vector)

    meta = MetaFact(weight=2, source_texts=["alpha"],
                    gravity_score=0.5, layer=1)
    meta.vector = [0.2] * 64
    mem._meta_facts[meta.meta_id] = meta
    mem._meta_index.add(meta.meta_id, meta.vector)
    mem._engine.register(meta)

    summary = mem.run_forecast(horizon_ticks=5)
    assert summary["facts_updated_count"] >= 1
    assert summary["metas_updated_count"] >= 1
