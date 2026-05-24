"""Seven write paths now wrapped in try/except + self._reload().

After this round, EVERY method that mutates in-memory state inside a
write transaction rolls back to disk truth on storage failure. The
five previously protected paths (add_fact, add_facts, query,
collapse_singularity, run_forecast) are joined by:

  1. session_close — the heaviest writeback in the codebase
     (apply_session_resonance, body.touch, EWMA, echo bundle,
     engine.tick, _absorb_dead, weights.update, _pop_session)
  2. check_echo — retroactive penalty applied via apply_session_resonance
     + EWMA before storage saves
  3. delete_fact — pops live fact, removes from index/engine before
     storage.delete_fact
  4. delete_body — same as delete_fact across all 4 body kinds
  5. supersede_fact — sets deprecated_by + drops from spo_index
     before storage.save_fact, then _absorb_dead
  6. retire_fact — sets ttl=now before storage.save_fact, then
     _absorb_dead
  7. link — engine.link increments in-memory degree before
     storage.save_edge

Pin the contract for each: force a storage failure mid-write, assert
in-memory state matches disk truth after the raise propagates.
"""
from __future__ import annotations

import pytest

from birch.memory_store import MemoryStore


def _install_failing_storage(mem, attr: str, msg: str):
    """Monkey-patch a single storage method to raise. Returns the
    original so the test can restore it."""
    original = getattr(mem._storage, attr)

    def boom(*args, **kwargs):
        raise RuntimeError(msg)

    setattr(mem._storage, attr, boom)
    return original


# --- I1: session_close --------------------------------------------------


def test_session_close_rolls_back_on_storage_failure(tmp_path):
    """Force save_facts mid-writeback. The session's body.touch
    mutations should be undone via _reload — access_count on disk
    must match the access_count we see after the raise."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    f = mem.add_fact("api", "uses", "Postgres")
    pre_access = f.access_count
    mem.session_start("s1")
    mem.session_message("about the api postgres", session_id="s1")
    # Run a query so the session has an attributed fact to touch.
    mem.query("api uses Postgres", top_k=2, session_id="s1")
    # Snapshot access_count on disk via a fresh reload.
    persisted_access = mem._facts[f.fact_id].access_count

    original = _install_failing_storage(
        mem, "save_facts", "simulated mid-close storage failure",
    )
    try:
        with pytest.raises(RuntimeError, match="mid-close"):
            mem.session_close(session_id="s1", sentiment="resonant")
    finally:
        mem._storage.save_facts = original  # type: ignore[assignment]

    # After _reload, access_count must equal the disk-persisted value
    # — NOT the in-memory touched value that close was about to apply.
    refreshed = mem._facts[f.fact_id]
    assert refreshed.access_count == persisted_access, (
        "session_close leaked body.touch despite storage rollback"
    )
    assert refreshed.access_count >= pre_access  # sanity
    mem.close()


# --- I2: check_echo -----------------------------------------------------


def test_check_echo_rolls_back_on_storage_failure(tmp_path):
    """check_echo applies retroactive penalty via
    apply_session_resonance + EWMA before storage saves. Force a
    save_facts failure and assert resonance state didn't leak."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    f = mem.add_fact(
        "outage", "root cause",
        "stale connection pool drains during long-running query",
    )
    # Seed an echo session so check_echo has a match.
    mem.session_start("s_seed")
    mem.session_message(
        "outage stale connection pool drain", session_id="s_seed",
    )
    mem.query(
        "outage stale connection pool drain", top_k=3,
        session_id="s_seed",
    )
    mem.session_close(session_id="s_seed", sentiment="toxic")
    pre_resonance_count = mem._facts[f.fact_id].resonance_count

    original = _install_failing_storage(
        mem, "save_facts", "simulated check_echo storage failure",
    )
    try:
        with pytest.raises(RuntimeError, match="check_echo"):
            mem.check_echo(
                "outage stale connection pool drain",
                session_id=None,
            )
    finally:
        mem._storage.save_facts = original  # type: ignore[assignment]

    # resonance_count must equal disk truth, not the leaked penalty
    # application.
    assert mem._facts[f.fact_id].resonance_count == pre_resonance_count, (
        "check_echo retroactive penalty leaked despite rollback"
    )
    mem.close()


# --- I3: delete_fact ----------------------------------------------------


def test_delete_fact_rolls_back_on_storage_failure(tmp_path):
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    f = mem.add_fact("api", "uses", "Postgres")
    original = _install_failing_storage(
        mem, "delete_fact", "simulated delete failure",
    )
    try:
        with pytest.raises(RuntimeError, match="delete failure"):
            mem.delete_fact(f.fact_id)
    finally:
        mem._storage.delete_fact = original  # type: ignore[assignment]

    # Fact must still be in _facts after rollback — disk still has it.
    assert f.fact_id in mem._facts, (
        "delete_fact leaked pop despite storage rollback"
    )
    # Engine and SPO index also restored.
    assert f.fact_id in mem._engine._facts
    key = mem._normalize_spo("api", "uses", "Postgres")
    assert mem._spo_index.get(key) == f.fact_id
    mem.close()


# --- I4: delete_body ----------------------------------------------------


def test_delete_body_rolls_back_on_storage_failure(tmp_path):
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    f = mem.add_fact("api", "uses", "Redis")
    original = _install_failing_storage(
        mem, "delete_fact", "simulated delete_body failure",
    )
    try:
        with pytest.raises(RuntimeError, match="delete_body failure"):
            mem.delete_body(f.fact_id)
    finally:
        mem._storage.delete_fact = original  # type: ignore[assignment]

    assert f.fact_id in mem._facts
    assert f.fact_id in mem._engine._facts
    mem.close()


# --- I5: supersede_fact -------------------------------------------------


def test_supersede_fact_rolls_back_on_storage_failure(tmp_path):
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    old = mem.add_fact("svc", "version", "1.0")
    new = mem.add_fact("svc", "version", "2.0")
    # Force save_fact (called inside _supersede_fact_locked) to fail.
    original = _install_failing_storage(
        mem, "save_fact", "simulated supersede failure",
    )
    try:
        with pytest.raises(RuntimeError, match="supersede failure"):
            mem.supersede_fact(old.fact_id, new.fact_id)
    finally:
        mem._storage.save_fact = original  # type: ignore[assignment]

    # After rollback, old must NOT be deprecated.
    refreshed_old = mem._facts.get(old.fact_id)
    assert refreshed_old is not None
    assert refreshed_old.deprecated_by is None, (
        "supersede leaked deprecated_by despite storage rollback"
    )
    mem.close()


# --- I6: retire_fact ----------------------------------------------------


def test_retire_fact_rolls_back_on_storage_failure(tmp_path):
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    f = mem.add_fact("topic", "is", "active")
    original_ttl = f.ttl
    original = _install_failing_storage(
        mem, "save_fact", "simulated retire failure",
    )
    try:
        with pytest.raises(RuntimeError, match="retire failure"):
            mem.retire_fact(f.fact_id)
    finally:
        mem._storage.save_fact = original  # type: ignore[assignment]

    refreshed = mem._facts.get(f.fact_id)
    assert refreshed is not None
    # ttl must NOT have been advanced to now().
    assert refreshed.ttl == original_ttl, (
        "retire_fact leaked ttl mutation despite storage rollback"
    )
    mem.close()


# --- I7: link -----------------------------------------------------------


def test_link_rolls_back_on_storage_failure(tmp_path):
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    f1 = mem.add_fact("a", "is", "x")
    f2 = mem.add_fact("b", "is", "y")
    pre_degree = mem._engine._degrees.get(f2.fact_id, 0)
    original = _install_failing_storage(
        mem, "save_edge", "simulated link failure",
    )
    try:
        with pytest.raises(RuntimeError, match="link failure"):
            mem.link(f1.fact_id, f2.fact_id)
    finally:
        mem._storage.save_edge = original  # type: ignore[assignment]

    # In-memory degree counter must equal pre-link value after _reload.
    post_degree = mem._engine._degrees.get(f2.fact_id, 0)
    assert post_degree == pre_degree, (
        f"link leaked in-memory degree (pre={pre_degree}, "
        f"post={post_degree}) despite storage rollback"
    )
    mem.close()
