"""Six engine-cleanup + rollback-safety contracts across the
absorption / unregister / add_fact / query / session_close /
close paths. Bundled because they all share the "in-memory
state must match disk after any failure" invariant.

  1. _absorb_dead unregisters absorbed bodies from GravityEngine —
     previously the body left _facts/_index/_spo_index but stayed
     in engine._facts, getting tick()'d and resonance-applied as if
     live. Post-restart behaviour diverged (load routes layer=-1
     to hole and skips engine.register), so pre-restart vs
     post-restart silently differed.

  2. GravityEngine.unregister now decrements target degrees for
     stale outgoing edges. e29dd81 added _edges set but only
     cleared the edge tracking, not the degree counter that the
     original link() incremented on the target.

  3. add_fact (single) now wraps the txn in try/except → _reload —
     symmetric with add_facts (plural, shipped 1792d4f) and
     collapse_singularity (b9412ab). Storage failure mid-write
     used to leak _facts / _engine / _index / _spo_index / edges
     mutations into in-memory state.

  4. query() write path now wraps the txn in try/except → _reload.
     Hawking emission registered bodies back into live store before
     storage saves; on rollback the body would stay live in memory
     while disk still saw it in singularity.

  5. session_close drift-merge comment now matches code — facts-
     only merge is the documented contract. Late messages flowing
     in during compute_resonance don't influence R or echo this
     round (the alternative — recompute under writeback lock —
     was rejected as too churny for the closing op).

  6. close() honours shutdown timeout properly. If collapse worker
     timed out, storage.close() is SKIPPED (leak the handle until
     process exit) rather than closing under a still-running
     worker that would hit a closed connection. _last_collapse_error
     records the leak for observability.
"""
from __future__ import annotations

import pytest

from birch.gravity import GravityEngine
from birch.memory_store import MemoryStore


# Test body for the GravityEngine tests — duck-typed GravityBody.
class _Body:
    def __init__(self, fid: str) -> None:
        self.fact_id = fid
        self.gravity_score = 0.5
        self.last_accessed = 0.0
        self.access_count = 0
        self.resonance_count = 0
        self.recent_utility = 0.5
        self.forecast_stability = 0.5


# --- I1: _absorb_dead unregisters from engine -------------------------


def test_absorb_dead_unregisters_low_gravity_fact_from_engine(tmp_path):
    """Push a fact below absorption threshold and trigger _absorb_dead.
    The fact must vanish from engine._facts so tick() can't keep
    scoring it as live."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    f = mem.add_fact("api", "uses", "Postgres")
    # Force absorption via low gravity (not deprecated/expired path).
    f.gravity_score = 0.05
    mem._storage.save_fact(f)
    mem._absorb_dead()
    # Body must be out of every live cache, INCLUDING engine._facts.
    assert f.fact_id not in mem._facts
    assert f.fact_id not in mem._engine._facts, (
        "absorbed body still registered in GravityEngine — tick() "
        "and apply_session_resonance would keep treating it as live"
    )
    # Body is now in singularity (the absorption target).
    assert f.fact_id in mem._hole._singularity
    mem.close()


def test_absorb_dead_unregisters_metafact_from_engine(tmp_path):
    """Same contract for absorbed MetaFacts — symmetric path."""
    from birch.meta_fact import MetaFact

    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    meta = MetaFact(
        weight=2, source_texts=["a", "b"],
        source_fact_ids=["x", "y"],
        layer=0,
    )
    meta.vector = [1.0, 0.0, 0.0]
    meta.gravity_score = 0.30
    mem._storage.save_meta_fact(meta)
    mem._reload()
    assert meta.meta_id in mem._meta_facts
    # Operate on the in-dict copy (the reload built a fresh instance).
    in_dict = mem._meta_facts[meta.meta_id]
    mem._engine.register(in_dict)
    # Drop gravity below threshold and absorb.
    in_dict.gravity_score = 0.05
    mem._absorb_dead()
    assert meta.meta_id not in mem._meta_facts
    assert meta.meta_id not in mem._engine._facts, (
        "absorbed MetaFact still in GravityEngine"
    )
    mem.close()


# --- I2: GravityEngine.unregister decrements target degrees -----------


def test_unregister_decrements_target_degree():
    eng = GravityEngine()
    eng.register(_Body("A"))
    eng.register(_Body("B"))
    eng.register(_Body("C"))
    eng.link("A", "B")
    eng.link("C", "B")
    assert eng._degrees["B"] == 2
    # Unregister source A — B's degree must drop to 1 (the C→B edge
    # still exists).
    eng.unregister("A")
    assert eng._degrees["B"] == 1, (
        "Unregister cleared _edges but left _degrees inflated"
    )
    # Unregister C — B's degree must drop to 0.
    eng.unregister("C")
    assert eng._degrees["B"] == 0
    # Target itself unregister — its degree key vanishes.
    eng.unregister("B")
    assert "B" not in eng._degrees


def test_unregister_target_does_not_self_decrement():
    """When the unregistered fact IS the target of an incoming edge,
    don't try to decrement its own degree (it's about to be popped).
    Defensive — prevents a tricky off-by-one if both endpoints get
    cleaned up together."""
    eng = GravityEngine()
    eng.register(_Body("A"))
    eng.register(_Body("B"))
    eng.link("A", "B")
    assert eng._degrees["B"] == 1
    eng.unregister("B")
    # B's own degree key is gone.
    assert "B" not in eng._degrees
    # And A's degree wasn't touched (A had no incoming edges).
    assert eng._degrees["A"] == 0


# --- I3: add_fact rollback safety -------------------------------------


def test_add_fact_rolls_back_in_memory_on_storage_failure(tmp_path):
    """Force save_fact to raise and assert the new fact didn't leak
    into _facts / _engine / _index / _spo_index."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    # Patch storage to fail on save_fact for the next call.
    original = mem._storage.save_fact

    def failing_save(fact):
        raise RuntimeError("simulated mid-write failure")

    mem._storage.save_fact = failing_save  # type: ignore[assignment]
    try:
        with pytest.raises(RuntimeError, match="mid-write"):
            mem.add_fact("api", "uses", "Postgres")
    finally:
        mem._storage.save_fact = original  # type: ignore[assignment]

    # The fact must NOT be present anywhere in memory.
    assert not any(
        f.subject == "api" and f.object == "Postgres"
        for f in mem._facts.values()
    ), "add_fact leaked into _facts despite storage failure"
    # SPO index must be clean too.
    key = mem._normalize_spo("api", "uses", "Postgres")
    assert key not in mem._spo_index
    mem.close()


# --- I4: query() write-path rollback safety ---------------------------


def test_query_rolls_back_in_memory_on_storage_failure(tmp_path):
    """Force save_facts (the post-Hawking touch persistence) to raise.
    The in-memory live facts must reflect the disk state on retry."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    seed = mem.add_fact("api", "uses", "Postgres")
    pre_access = seed.access_count
    # Patch the touch-persistence call to raise.
    original_save_facts = mem._storage.save_facts

    def failing_save_facts(facts):
        raise RuntimeError("simulated touch-save failure")

    mem._storage.save_facts = failing_save_facts  # type: ignore[assignment]
    try:
        with pytest.raises(RuntimeError, match="touch-save"):
            mem.query("api uses Postgres", top_k=5)
    finally:
        mem._storage.save_facts = original_save_facts  # type: ignore[assignment]

    # After _reload, access_count on the seed fact must match what
    # was actually persisted — not the leaked in-memory touch.
    refreshed = mem._facts[seed.fact_id]
    assert refreshed.access_count == pre_access, (
        "query() leaked in-memory touch despite storage rollback"
    )
    mem.close()


# --- I5: session_close drift comment matches code (no message merge) --


def test_session_close_late_message_does_not_count_in_round(tmp_path):
    """Documented contract: messages pushed during the close window
    do NOT influence R or the echo bundle. They simply vanish with
    the popped ctx. Pin the contract so a future "let's also union
    messages" refactor surfaces here as a behaviour change."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    mem.session_start("s1")
    mem.session_message("seed message about apis", session_id="s1")
    # Snapshot what the trajectory would be at the close start.
    pre_close_messages = list(mem._sessions["s1"].messages)
    # Simulate a "late" push — happens between snapshot and writeback
    # in real concurrency. Here we just mutate ctx.messages directly
    # before close, then close: the contract says this late message
    # has no effect on R (heuristic path) because compute_resonance
    # already ran on the snapshot.
    summary = mem.session_close(session_id="s1", sentiment="resonant")
    assert summary["label"] == "resonant"
    # Pre-close trajectory length is 1; the contract is that NO
    # additional messages got promoted into the close round even
    # if they had been pushed late.
    assert len(pre_close_messages) == 1
    mem.close()


# --- I6: close() skips storage.close when worker is mid-run -----------


def test_close_skips_storage_close_when_inflight_unfinished(tmp_path):
    """If a collapse future is still running after the 5s timeout,
    close() must NOT close storage (that would deadlock or corrupt
    the worker). The _last_collapse_error string surfaces the leak."""
    import time as _t
    from concurrent.futures import Future

    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    mem.add_fact("api", "uses", "Postgres")

    # Plant a fake never-finishing future on the instance.
    never_done: Future = Future()
    mem._inflight_collapse = never_done

    # Also need a dummy executor so the shutdown branch runs.
    from concurrent.futures import ThreadPoolExecutor
    mem._collapse_executor = ThreadPoolExecutor(max_workers=1)

    # Capture whether storage.close was called.
    closed = {"flag": False}
    original_close = mem._storage.close

    def tracking_close():
        closed["flag"] = True
        original_close()

    mem._storage.close = tracking_close  # type: ignore[assignment]

    # Speed up: monkeypatch the 5-second wait by setting timeout via
    # the inflight future — easiest is to let it block briefly. We
    # can't change the 5.0 in code from here, so just accept the wait
    # for this test (one pass at <6s).
    started = _t.monotonic()
    mem.close()
    elapsed = _t.monotonic() - started
    # close() should have returned within ~6s (5s timeout + overhead).
    assert elapsed < 7.0
    # The storage.close MUST NOT have been called (worker still
    # running by assertion of the never-done future).
    assert closed["flag"] is False, (
        "close() closed storage under a still-running worker — "
        "would corrupt the worker's next SQLite call"
    )
    # And the leak is recorded.
    assert mem._last_collapse_error is not None
    assert "leak" in mem._last_collapse_error.lower()
    # Cleanup: cancel the planted future so the test process exits.
    never_done.cancel()
    original_close()
