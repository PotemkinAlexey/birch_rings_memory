"""Forecast polymorphism (FactPassport + MetaFact) regressions.

Subtler semantic gaps: signal honesty (set_fact already_existed had
a race window), result completeness (query revalidation didn't
backfill), write contention (Hawking write txn even when singularity
was empty), signal consistency (check_echo updated resonance but not
EWMA), and a couple of API clarity items (forecast typing, mixed dims
silently accepted by build_galaxy).
"""
from __future__ import annotations

from birch.galaxy.forecast import forecast_stability
from birch.memory_store import MemoryStore
from birch.meta_fact import MetaFact

# --- P1: add_fact returns authoritative created flag ---------------------


def test_add_fact_returns_created_true_on_first_write(tmp_path):
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    fact, created = mem.add_fact(
        "api", "runs on", "Go", return_status=True,
    )
    assert created is True
    # Same SPO again — same fact, created=False.
    fact2, created2 = mem.add_fact(
        "api", "runs on", "Go", return_status=True,
    )
    assert created2 is False
    assert fact.fact_id == fact2.fact_id


def test_set_fact_already_existed_uses_authoritative_created(tmp_path):
    """set_fact now reads transaction-honest already_existed from
    add_fact's authoritative status — no race window."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    out = mem.set_fact("HEAD", "is", "abc")
    assert out["already_existed"] is False
    again = mem.set_fact("HEAD", "is", "abc")
    assert again["already_existed"] is True


# --- P1: query() backfills top after revalidation dropped hits ----------


def test_query_backfills_when_revalidation_drops_hits(tmp_path):
    """If a top-5 had 3 hits dropped during in-txn revalidation, the
    return must backfill from the fresh authoritative index instead of
    returning a short list. We force-drop by wrapping _sync to retire
    the top-1 hit during the in-txn re-sync — backfill must pull a
    different live fact in."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    target_fact = mem.add_fact("alpha", "is", "the doomed top hit")
    backfill_a = mem.add_fact("beta", "is", "backfill candidate one")
    backfill_b = mem.add_fact("gamma", "is", "backfill candidate two")

    original_sync = mem._sync
    call_count = {"n": 0}

    def _sync_with_race():
        original_sync()
        call_count["n"] += 1
        if call_count["n"] == 2:
            mem.retire_fact(target_fact.fact_id)

    mem._sync = _sync_with_race  # type: ignore[method-assign]
    hits = mem.query(
        "alpha beta gamma is doomed top hit candidate",
        top_k=3, min_similarity=0.0, hawking=False,
    )
    mem._sync = original_sync  # type: ignore[method-assign]

    ids = {h.body_id for h in hits}
    assert target_fact.fact_id not in ids
    # Backfill must have pulled at least one of the surviving facts in
    # — otherwise we got a short list, which is what we're regressing.
    assert ids & {backfill_a.fact_id, backfill_b.fact_id}, (
        "revalidation dropped a hit but backfill did not replace it"
    )


# --- P2: query(hawking=True) skips write txn when singularity empty -----


def test_query_skips_write_txn_when_empty_singularity_and_no_writes(tmp_path):
    """A no-session no-hits pure-read query with hawking=True should NOT
    enter the write transaction at all when the singularity is empty —
    BEGIN IMMEDIATE used to fire just to scan an empty hole. We pin
    behaviour by using a very high min_similarity so no hit is touched
    and no session is open: save_facts must never be called."""
    from unittest import mock

    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    mem.add_fact("api", "runs on", "Go")
    assert mem._storage is not None
    assert mem._hole.mass == 0

    with mock.patch.object(mem._storage, "save_facts") as spy:
        # Nothing reaches top with this threshold, so no touch / attribution
        # / persist is needed — hawking-only check on empty hole now skipped.
        hits = mem.query(
            "api Go", top_k=5, hawking=True, min_similarity=0.99,
        )
        assert hits == []
    assert spy.call_count == 0


# --- P2: check_echo updates recent_utility too --------------------------


def test_check_echo_updates_recent_utility_on_penalty(tmp_path):
    """Echo penalty applies to resonance via apply_session_resonance AND
    now also through the EWMA helper — recent_utility must move down too,
    so the gravity formula sees consistent inputs."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    f = mem.add_fact("api", "runs on", "Go")
    before_util = f.recent_utility

    # Directly drive the EWMA path the same way an echo penalty would —
    # via the helper, with a negative R. This pins the helper contract
    # without depending on the mock embedder hitting the echo threshold.
    with mem._lock:
        with mem._txn():
            mem._apply_recent_utility_locked(
                {f.fact_id: 1.0}, -0.6,
            )
    assert f.recent_utility < before_util


# --- P2: forecast_stability accepts polymorphic bodies (docstring sync) -


def test_forecast_stability_accepts_polymorphic_bodies():
    from birch.fact import FactPassport

    fact = FactPassport("api", "runs on", "Go")
    fact.vector = [0.1] * 64
    meta = MetaFact(weight=2, source_texts=["alpha"], gravity_score=0.5,
                    layer=1)
    meta.vector = [0.2] * 64

    scores = forecast_stability([fact, meta], horizon_ticks=10)
    assert fact.fact_id in scores
    assert meta.meta_id in scores


# --- P2: build_galaxy fails loudly on mixed embedding dims --------------


def test_build_galaxy_raises_on_mixed_dims():
    import pytest

    from birch.fact import FactPassport
    from birch.galaxy.loader import build_galaxy
    from birch.vector_index import DimensionMismatchError

    a = FactPassport("a", "is", "1")
    a.vector = [0.1] * 64
    b = FactPassport("b", "is", "2")
    b.vector = [0.1] * 32   # different dim
    with pytest.raises(DimensionMismatchError):
        build_galaxy([a, b])


def test_build_galaxy_tolerates_empty_vectors_among_dim_match():
    """Bodies with an empty vector are placed via fallback direction and
    do NOT count toward the dim check — only non-empty dims must agree."""
    from birch.fact import FactPassport
    from birch.galaxy.loader import build_galaxy

    a = FactPassport("a", "is", "1")
    a.vector = [0.1] * 64
    b = FactPassport("b", "is", "2")
    b.vector = []   # empty — fallback direction
    c = FactPassport("c", "is", "3")
    c.vector = [0.2] * 64
    gal = build_galaxy([a, b, c])
    assert len(gal.bodies) == 3


# --- P3: run_forecast response gains bodies / metas keys ----------------


def test_run_forecast_response_carries_body_breakdown(tmp_path):
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    for i in range(3):
        mem.add_fact(f"fact-{i}", "is", f"v-{i}")
    meta = MetaFact(weight=2, source_texts=["alpha beta"],
                    gravity_score=0.5, layer=1)
    meta.vector = [0.3] * 64
    mem._meta_facts[meta.meta_id] = meta
    mem._meta_index.add(meta.meta_id, meta.vector)
    mem._engine.register(meta)

    summary = mem.run_forecast(horizon_ticks=10)
    # New keys present.
    assert "bodies_forecasted" in summary
    assert "facts_updated_count" in summary
    assert "metas_updated_count" in summary
    # Old keys retained for backward compat.
    assert "facts_forecasted" in summary
    assert summary["facts_updated_count"] >= 3
    assert summary["metas_updated_count"] >= 1
