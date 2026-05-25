"""Four findings from the professor / Vader round on reactor leaks:

  1. BlackHole.absorb is now atomic. Previously: set layer=-1, put
     in _singularity dict, then add to vector index. A
     DimensionMismatchError on the third step left the body
     half-absorbed: ``layer == -1`` and present in singularity
     dict but invisible to Hawking, while ``_absorb_dead``'s
     caller had not yet deleted it from live ``_facts``. With
     ``storage == None`` (in-memory mode), ``_reload()`` cannot
     recover. Three-phase fix: pre-flight the dim check, then
     mutate, then commit the index; rollback dict insert + layer
     on any failure. Also: ``_absorb_dead`` now catches per-fact
     so a single mismatched-dim body doesn't abort the whole
     sweep.

  2. session_close late-message race. session_close snapshots ctx
     state then releases the lock for compute_resonance. A push
     that landed during that window persisted to disk but was
     silently dropped when session_close popped the ctx — the
     agent saw push succeed but the message never influenced R
     / echo / future sessions. Fix: track ``_closing_sessions``
     set, session_message rejects pushes to a closing sid with
     a structured ``session_closing`` error.

  3. run_forecast missing rollback-recovery. Forecast wrote
     ``forecast_stability`` into live bodies under a txn — if
     storage write raised, SQLite rolled back disk truth but
     in-memory bodies stayed mutated. forecast_stability feeds
     gravity via pre_resonance_features, so the drift propagated
     into layer migration. Standard ``except: self._reload();
     raise`` now wraps the writeback.

  4. query() filtered live results by rounded-to-4dp similarity
     while the Hawking branch filtered by raw similarity. Caller
     setting ``min_similarity=0.95005`` would receive Hawking hits
     at raw 0.95004 but not live ones at the same raw score (the
     live path's rounded 0.9500 fell below the threshold). Fix:
     decide on raw, round only on output. Belt-and-suspenders:
     the post-loop filter stays in place as a safety net.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from birch.black_hole import BlackHole
from birch.fact import FactPassport
from birch.memory_store import MemoryStore
from birch.meta_fact import MetaFact
from birch.vector_index import DimensionMismatchError

# --- I1: BlackHole.absorb atomic ---------------------------------------


def test_blackhole_absorb_rejects_dim_mismatch_atomically():
    """Pre-flight check raises BEFORE any mutation. After the raise,
    fact.layer is unchanged AND fact is NOT in _singularity dict."""
    hole = BlackHole()
    # First absorb sets the dim.
    f1 = FactPassport(
        subject="a", predicate="b", object="c", fact_id="f1",
        vector=[0.1, 0.2, 0.3],
    )
    hole.absorb(f1)
    assert hole._index._dim == 3
    # Second absorb with different dim must raise atomically.
    f2 = FactPassport(
        subject="x", predicate="y", object="z", fact_id="f2",
        vector=[0.5] * 5,   # different dim
        layer=1,            # live layer before absorption
    )
    with pytest.raises(DimensionMismatchError):
        hole.absorb(f2)
    # Atomic rollback: f2 not in singularity, layer not flipped.
    assert "f2" not in hole._singularity
    assert f2.layer == 1


def test_blackhole_absorb_meta_atomic_dim_check():
    hole = BlackHole()
    m1 = MetaFact(meta_id="m1", vector=[0.1, 0.2, 0.3], layer=0)
    hole.absorb_meta(m1)
    assert hole._meta_index._dim == 3
    m2 = MetaFact(meta_id="m2", vector=[0.5] * 5, layer=0)
    with pytest.raises(DimensionMismatchError):
        hole.absorb_meta(m2)
    assert "m2" not in hole._meta_singularity
    # Rolled back to live layer (we hard-coded 0 above).
    assert m2.layer == 0


def test_absorb_dead_in_memory_mode_skips_mismatched_dim_body(tmp_path):
    """In-memory store (storage=None) + mixed-dim singularity: a live
    fact whose gravity falls below the absorption floor must NOT
    leave the store in a half-state. Either it lands cleanly in
    the singularity, or it stays live and visible — never both,
    never neither.
    """
    # Use a file-backed store but verify the in-memory consistency
    # property directly. We force the dim mismatch by seeding the
    # singularity with a body of one dim, then trying to absorb
    # a live body of a different dim.
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    # Seed singularity with a dim=3 body via direct hole API.
    seed = FactPassport(
        subject="seed", predicate="lives", object="here",
        fact_id="seed",
        vector=[0.1, 0.2, 0.3],
        gravity_score=0.05,    # below absorption threshold
    )
    mem._hole.absorb(seed)
    # Add a live fact with a different vector dim (mock embed to a
    # different size) and drop its gravity below threshold.
    import birch.memory_store as pkg
    orig_embed = pkg.embed
    pkg.embed = lambda text: [0.5] * 5   # dim=5 — mismatch with singularity
    try:
        f = mem.add_fact("api", "uses", "redis")
        f.gravity_score = 0.05   # below threshold to trigger absorb
    finally:
        pkg.embed = orig_embed
    # Run absorb sweep — body cannot enter singularity (dim mismatch).
    # Contract: must NOT leave the body half-absorbed.
    absorbed = mem._absorb_dead()
    # The mismatched body should be SKIPPED, not absorbed.
    assert f.fact_id not in absorbed
    # And it must still be live (not deleted) AND not in singularity.
    assert f.fact_id in mem._facts
    assert f.fact_id not in mem._hole._singularity
    # Layer untouched (no half-absorption to -1).
    assert mem._facts[f.fact_id].layer != -1
    mem.close()


# --- I2: session_close closing_sessions gate ---------------------------


def test_session_message_rejects_push_to_closing_session(tmp_path):
    """Simulate the race: snapshot is taken, sid marked as closing,
    a push arrives — must raise instead of silently landing in a
    ctx that's about to be popped."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    mem.session_start("s1")
    mem.session_message("hello", session_id="s1")
    # Mark as closing directly to exercise the gate.
    with mem._lock:
        mem._closing_sessions.add("s1")
    with pytest.raises(RuntimeError, match="session_closing"):
        mem.session_message("late message", session_id="s1")
    # Cleanup so close doesn't trip the flag.
    with mem._lock:
        mem._closing_sessions.discard("s1")
    mem.close()


def test_closing_flag_clears_after_successful_close(tmp_path):
    """After session_close completes, the sid is no longer flagged
    as closing — but session_message would now raise
    'unknown session' (the honest result), not 'session_closing'."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    mem.session_start("s1")
    mem.session_message("hello", session_id="s1")
    mem.session_close(session_id="s1", sentiment="resonant")
    assert "s1" not in mem._closing_sessions
    # Pushing to a now-popped session raises "unknown session", NOT
    # the closing-gate error.
    with pytest.raises(KeyError, match="unknown session"):
        mem.session_message("after-close", session_id="s1")
    mem.close()


def test_closing_flag_clears_on_failed_close(tmp_path):
    """A storage failure during session_close must NOT permanently
    brick the sid — flag clears in the except path."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    mem.session_start("s1")
    mem.session_message("hello", session_id="s1")
    original_save = mem._storage.save_facts

    def boom(*a, **kw):
        raise RuntimeError("simulated mid-close write failure")

    mem._storage.save_facts = boom
    try:
        with pytest.raises(RuntimeError, match="simulated mid-close"):
            mem.session_close(session_id="s1", sentiment="resonant")
    finally:
        mem._storage.save_facts = original_save
    # Flag cleared so the sid isn't permanently stuck.
    assert "s1" not in mem._closing_sessions
    mem.close()


# --- I3: run_forecast rollback recovery -------------------------------


def test_run_forecast_rolls_back_in_memory_on_storage_failure(tmp_path):
    """If save_facts raises mid-forecast, in-memory forecast_stability
    must NOT diverge from disk truth — _reload re-anchors."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    f = mem.add_fact("api", "uses", "postgres")
    fid = f.fact_id
    pre_stability = mem._facts[fid].forecast_stability

    # add_fact uses save_fact (singular); run_forecast uses save_facts
    # (plural). Patching save_facts is therefore safe — won't affect
    # the seed write.
    original_save_facts = mem._storage.save_facts

    def boom(facts_list):
        raise RuntimeError("simulated forecast writeback failure")

    mem._storage.save_facts = boom
    try:
        with pytest.raises(RuntimeError, match="forecast writeback"):
            mem.run_forecast(horizon_ticks=10)
    finally:
        mem._storage.save_facts = original_save_facts

    # After _reload, in-memory forecast_stability matches what's on
    # disk (= the original pre-forecast value).
    assert mem._facts[fid].forecast_stability == pre_stability
    mem.close()


# --- I4: query() filters on raw similarity ----------------------------


def test_query_filters_live_on_raw_similarity_not_rounded(tmp_path):
    """Construct a scenario where the 4th-decimal rounding could
    differ from the raw value — patch find to force the boundary."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    f = mem.add_fact("alpha", "beta", "gamma")
    # Patch all_similarities to return a sim that rounds DOWN past
    # the threshold but is raw-above.
    fid = f.fact_id
    with patch.object(
        mem._index, "all_similarities",
        return_value={fid: 0.950049},  # rounds to 0.9500
    ):
        # Threshold ABOVE the rounded value, BELOW the raw value.
        # Old behaviour: live filter on rounded → 0.9500 < 0.950045
        #   → fact dropped.
        # New behaviour: live filter on raw → 0.950049 >= 0.950045
        #   → fact kept.
        results = mem.query(
            "anything", top_k=5, min_similarity=0.950045,
        )
    matching = [r for r in results if r.fact and r.fact.fact_id == fid]
    assert len(matching) == 1, (
        "live filter must decide on raw sim, not the 4-decimal "
        "display value — asymmetric with Hawking otherwise"
    )
    mem.close()


def test_query_still_drops_low_raw_similarity(tmp_path):
    """Sanity: a genuinely below-threshold raw sim is still dropped."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    f = mem.add_fact("alpha", "beta", "gamma")
    fid = f.fact_id
    with patch.object(
        mem._index, "all_similarities",
        return_value={fid: 0.5},
    ):
        results = mem.query(
            "anything", top_k=5, min_similarity=0.9,
        )
    matching = [r for r in results if r.fact and r.fact.fact_id == fid]
    assert len(matching) == 0
    mem.close()


def test_query_meta_path_filters_on_raw_similarity(tmp_path):
    """Symmetric assertion for the live-MetaFact branch."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    # Promote a meta directly to live (layer=0).
    m = MetaFact(
        meta_id="m1",
        vector=[0.1, 0.2, 0.3],
        source_texts=["x"], source_fact_ids=["f"],
        layer=0,
        gravity_score=0.6,
    )
    mem._meta_facts["m1"] = m
    mem._meta_index.add("m1", m.vector)
    with patch.object(
        mem._meta_index, "all_similarities",
        return_value={"m1": 0.950049},
    ):
        results = mem.query(
            "anything", top_k=5, min_similarity=0.950045,
        )
    meta_hits = [r for r in results if r.meta and r.meta.meta_id == "m1"]
    assert len(meta_hits) == 1
    mem.close()
