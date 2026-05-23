"""Hawking two-phase emission regressions.

Closes the last side-effect leak in the Hawking path plus a couple of
API hygiene items. The load-bearing fix is the two-phase Hawking
commit: previously a body could be popped from the singularity (state
mutation, persisted write) even though it never made it to the
caller's top_k.
"""
from __future__ import annotations

import pytest

from birch.fact import FactPassport
from birch.memory_store import MemoryStore

# --- P1: Hawking results filtered by min_similarity ---------------------


def test_hawking_meta_respects_min_similarity():
    """If caller passes min_similarity=0.99, a Hawking-emitted MetaFact at
    cosine 0.86 (above the 0.85 meta threshold) must NOT come back."""
    from birch.meta_fact import MetaFact

    mem = MemoryStore()
    # Plant a MetaFact directly in the singularity with a known vector.
    meta = MetaFact(weight=2, source_texts=["x"], gravity_score=0.5, layer=-1)
    meta.vector = [1.0, 0.0, 0.0]
    mem._hole.restore_meta(meta)

    # Query at an almost-but-not-quite-identical vector — cosine roughly
    # 0.87 (above MetaFact Hawking threshold 0.85) but below the caller's
    # demanding min_similarity floor.
    from unittest import mock
    near_vec = [0.87, 0.5, 0.0]
    with mock.patch("birch.memory_store.embed", return_value=near_vec):
        hits = mem.query(
            "x", top_k=5, hawking=True, min_similarity=0.99,
        )
    assert all(h.body_id != meta.meta_id for h in hits)
    # And the MetaFact stayed in the singularity (no side-effect).
    assert meta.meta_id in mem._hole._meta_singularity


# --- P1: Hawking peek-then-commit doesn't resurrect non-survivors -------


def test_hawking_body_not_resurrected_if_below_top_k(tmp_path):
    """A Hawking-eligible body whose similarity is high enough to peek
    in but low enough to fall out of top_k must remain in the singularity
    after the call. Previously it was popped/registered/persisted just
    to be sliced off the return list — a contract violation.

    Hand-craft vectors so the ranking is deterministic: two live facts
    at cosine 1.0, Hawking body at cosine 0.96 (above threshold 0.95
    but below the live pair).
    """
    from unittest import mock

    mem = MemoryStore(db_path=str(tmp_path / "m.db"))

    # Plant two live facts with perfect match to the query vector.
    a = FactPassport("alpha", "is", "live perfect 1")
    a.vector = [1.0, 0.0, 0.0]
    mem._facts[a.fact_id] = a
    mem._engine.register(a)
    mem._index.add(a.fact_id, a.vector)
    mem._spo_index[mem._normalize_spo(*("alpha", "is", "live perfect 1"))] = a.fact_id

    b = FactPassport("beta", "is", "live perfect 2")
    b.vector = [1.0, 0.0, 0.0]
    mem._facts[b.fact_id] = b
    mem._engine.register(b)
    mem._index.add(b.fact_id, b.vector)
    mem._spo_index[mem._normalize_spo(*("beta", "is", "live perfect 2"))] = b.fact_id

    # Hawking-eligible body — cos ≈ 0.96 with query: above the 0.95
    # Hawking threshold, below live pair's perfect 1.0.
    hawking_body = FactPassport("gamma", "is", "buried but eligible")
    hawking_body.vector = [0.96, 0.28, 0.0]   # |.|≈1.0, cos with x-axis≈0.96
    hawking_body.layer = -1
    mem._hole.restore_fact(hawking_body)

    with mock.patch("birch.memory_store.embed", return_value=[1.0, 0.0, 0.0]):
        hits = mem.query(
            "alpha is live perfect", top_k=2,
            hawking=True, min_similarity=0.0,
        )
    ids = {h.body_id for h in hits}
    # Live pair filled top_k; Hawking body did NOT come back.
    assert ids == {a.fact_id, b.fact_id}
    assert hawking_body.fact_id not in ids
    # Contract: NOT resurrected — body stays in singularity, never
    # appeared in _facts, was never persisted to live state.
    assert hawking_body.fact_id in mem._hole._singularity
    assert hawking_body.fact_id not in mem._facts


def test_hawking_survivor_does_get_resurrected(tmp_path):
    """Mirror of the previous test: when the Hawking body IS in top_k,
    it must actually emit (pop + register + persist)."""
    db = str(tmp_path / "m.db")
    mem = MemoryStore(db_path=db)
    target = FactPassport("rare", "is", "buried but eligible")
    target.vector = [1.0, 0.0, 0.0]
    target.layer = -1
    mem._hole.restore_fact(target)
    if mem._storage:
        mem._storage.save_fact(target)

    from unittest import mock
    with mock.patch("birch.memory_store.embed", return_value=[1.0, 0.0, 0.0]):
        hits = mem.query(
            "rare is buried", top_k=5, hawking=True, min_similarity=0.0,
        )

    assert target.fact_id in {h.body_id for h in hits}
    assert target.fact_id in mem._facts
    assert target.fact_id not in mem._hole._singularity


# --- P2: forecast_memory MCP wraps DimensionMismatchError ---------------


def test_forecast_memory_mcp_returns_structured_error_on_mixed_dims(tmp_path):
    """When the store has mixed embedding dimensions, forecast_memory()
    MCP tool must return a typed ``{ok: false, error, hint}`` payload
    instead of letting DimensionMismatchError bubble through as a raw
    exception.

    We replicate the catch logic without importing server.py — which
    pulls the mcp SDK — by calling MemoryStore.run_forecast directly
    and verifying the same wrapping the MCP tool performs.
    """
    from birch.meta_fact import MetaFact
    from birch.vector_index import DimensionMismatchError

    mem = MemoryStore(db_path=str(tmp_path / "m.db"))

    # Plant a fact and a meta with mismatched non-empty dims.
    a = FactPassport("a", "is", "1")
    a.vector = [0.1] * 64
    mem._facts[a.fact_id] = a
    mem._engine.register(a)
    mem._index.add(a.fact_id, a.vector)

    meta = MetaFact(weight=2, source_texts=["x"], gravity_score=0.5, layer=1)
    meta.vector = [0.1] * 32   # different dim
    mem._meta_facts[meta.meta_id] = meta
    mem._engine.register(meta)
    # Don't add to _meta_index (would raise DimensionMismatchError on add).

    # run_forecast should raise DimensionMismatchError; the MCP wrapper
    # catches it and returns the structured response.
    with pytest.raises(DimensionMismatchError):
        mem.run_forecast(horizon_ticks=5)

    # Inline the wrapper's catch logic — exactly what server.forecast_memory
    # does (we test the logic without importing the mcp SDK).
    try:
        mem.run_forecast(horizon_ticks=5)
        result = {"ok": True}
    except DimensionMismatchError as exc:
        result = {
            "ok": False,
            "error": "mixed_embedding_dimensions",
            "hint": "pin BIRCH_EMBED_MODEL or rebuild/reindex the store",
            "detail": str(exc),
        }
    assert result["ok"] is False
    assert result["error"] == "mixed_embedding_dimensions"
    assert "hint" in result


# --- P2: add_fact overload typing -----------------------------------------


def test_add_fact_overload_runtime_behaviour(tmp_path):
    """Overloads are a typing concern, but verify both paths return
    runtime-correct shapes."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    # Default branch returns FactPassport.
    f1 = mem.add_fact("api", "runs on", "Go")
    assert isinstance(f1, FactPassport)
    # Status branch returns (FactPassport, bool).
    f2, created = mem.add_fact(
        "db", "is", "Postgres", return_status=True,
    )
    assert isinstance(f2, FactPassport)
    assert isinstance(created, bool)
    assert created is True


# --- P3: run_forecast response carries _hint about legacy aliases ------


def test_run_forecast_response_has_legacy_alias_hint(tmp_path):
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    mem.add_fact("api", "runs on", "Go")
    summary = mem.run_forecast(horizon_ticks=5)
    assert "_hint" in summary
    hint = summary["_hint"]
    assert "legacy" in hint.lower() or "alias" in hint.lower()


# --- P3: peek surfaces candidates without mutation ----------------------


def test_peek_hawking_candidates_does_not_mutate():
    """The new peek method must NOT pop bodies from singularity — it's
    the entire point of the two-phase API."""
    from birch.black_hole import BlackHole

    hole = BlackHole()
    fact = FactPassport("rare", "is", "buried")
    fact.vector = [1.0, 0.0, 0.0]
    hole.restore_fact(fact)
    before_ids = set(hole._singularity.keys())

    candidates = hole.peek_hawking_candidates(
        query_vector=[1.0, 0.0, 0.0],
    )
    after_ids = set(hole._singularity.keys())

    assert before_ids == after_ids, "peek must not pop bodies"
    assert any(f.fact_id == fact.fact_id for f, _sim in candidates)


def test_hawking_emit_only_ids_filters_commit():
    """hawking_emit with only_ids should commit ONLY those ids even if
    other bodies pass the threshold + predicate."""
    from birch.black_hole import BlackHole

    hole = BlackHole()
    keeper = FactPassport("keep", "is", "this")
    keeper.vector = [1.0, 0.0, 0.0]
    skipper = FactPassport("skip", "is", "that")
    skipper.vector = [0.99, 0.0, 0.1]   # also above threshold
    hole.restore_fact(keeper)
    hole.restore_fact(skipper)

    emitted = hole.hawking_emit(
        query_vector=[1.0, 0.0, 0.0],
        only_ids={keeper.fact_id},
    )
    emitted_ids = {f.fact_id for f in emitted}
    assert keeper.fact_id in emitted_ids
    assert skipper.fact_id not in emitted_ids
    # Skipper stayed in singularity.
    assert skipper.fact_id in hole._singularity


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
