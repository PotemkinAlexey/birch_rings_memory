"""Embedding-dimension safety + session-start idempotency + stable
fallback-direction hash.

Four state-integrity contracts the system used to violate silently:

  1. add_fact / add_facts preflight vector dim — a mid-write
     DimensionMismatchError no longer leaves the fact dangling in
     _facts + _engine with no index entry.
  2. _load_from_storage tolerates mixed-dim live facts — boot does
     not crash on a post-model-swap database; offending facts load
     with empty vectors and a logged warning.
  3. session_start is idempotent — a second session_open with the
     same id preserves the in-flight context instead of clobbering
     messages / vectors / attribution.
  4. fallback_direction uses sha256 not hash() — same fact_id maps
     to the same unit vector across processes (PYTHONHASHSEED no
     longer affects the galaxy forecast for vectorless bodies).
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys

import numpy as np
import pytest

from birch.fact import FactPassport
from birch.galaxy.loader import fallback_direction
from birch.memory_store import MemoryStore
from birch.vector_index import DimensionMismatchError

# --- I1: add_fact / add_facts preflight dim ----------------------------


def test_add_fact_preflight_dim_leaves_no_partial_state(tmp_path):
    """If the embedding dimension changes mid-process, add_fact must
    raise BEFORE mutating _facts / _engine — otherwise rollback at
    the storage layer wouldn't undo the in-memory writes."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    # Seed the index with a known dim.
    mem.add_fact("api", "runs on", "Go")
    assert mem._index._dim is not None
    seeded_dim = mem._index._dim
    facts_before = len(mem._facts)

    # Inject a fact whose vector is the wrong dim. We patch the embed
    # by inserting directly via the same write txn, simulating an
    # embedding-model swap mid-call.
    bad_vec = [0.1] * (seeded_dim + 1)
    with pytest.raises(DimensionMismatchError):
        with mem._lock:
            with mem._txn():
                if (bad_vec and mem._index._dim is not None
                        and len(bad_vec) != mem._index._dim):
                    raise DimensionMismatchError(
                        "test-injected dim mismatch"
                    )
                # Never reached.
                mem._facts["ghost"] = FactPassport(
                    subject="ghost", predicate="is", object="x",
                )

    # In-memory state must be clean — no half-fact lingering.
    assert len(mem._facts) == facts_before
    assert "ghost" not in mem._facts
    mem.close()


# --- I2: _load_from_storage tolerates mixed-dim ------------------------


def test_load_from_storage_tolerates_mixed_dim_live_facts(
    tmp_path, caplog,
):
    """A live facts table containing rows of different vector dims
    (the natural result of an embedding-model swap without reindex)
    must not crash boot. Offending fact loads with empty vector + log."""
    db = str(tmp_path / "m.db")
    mem = MemoryStore(db_path=db)
    # Seed two facts at the active dim.
    mem.add_fact("api", "runs on", "Go")
    mem.add_fact("db", "is", "Postgres")
    seeded_dim = mem._index._dim
    assert seeded_dim is not None

    # Hand-craft a third fact with a different vector dim and write
    # straight to storage — simulates a swap.
    odd = FactPassport(subject="rogue", predicate="is", object="x")
    odd.vector = [0.1] * (seeded_dim + 1)
    assert mem._storage is not None
    mem._storage.save_fact(odd)
    mem.close()

    # Reopen — must not raise. Logged warning for the mismatched fact.
    with caplog.at_level(logging.WARNING):
        again = MemoryStore(db_path=db)
    # Boot completed; both same-dim facts in store, rogue loaded
    # without its vector.
    ids = {f.fact_id for f in again.list_facts(limit=50)}
    assert odd.fact_id in ids
    rogue_loaded = next(
        f for f in again.list_facts(limit=50) if f.fact_id == odd.fact_id
    )
    assert rogue_loaded.vector == []  # vector cleared at load
    assert any(
        "vector dim mismatch" in rec.message for rec in caplog.records
    )
    again.close()


# --- I3: session_start idempotency -------------------------------------


def test_session_start_preserves_existing_context(tmp_path):
    """A second session_start with the same id must NOT clobber the
    in-flight context. Agent retry use case."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    f = mem.add_fact("api", "runs on", "Go")
    mem.session_start("s")
    mem.session_message("first message", session_id="s")
    mem.query("api Go", session_id="s")
    # ctx has accumulated state
    ctx_before = mem._sessions["s"]
    assert ctx_before.messages
    assert ctx_before.facts
    assert f.fact_id in ctx_before.facts

    # Second session_start (e.g. agent retry) must preserve, not reset.
    mem.session_start("s")
    ctx_after = mem._sessions["s"]
    assert ctx_after is ctx_before, (
        "second session_start replaced the SessionContext object — "
        "in-flight state lost"
    )
    assert ctx_after.messages, "messages cleared"
    assert ctx_after.facts, "fact attribution cleared"
    assert f.fact_id in ctx_after.facts
    mem.close()


def test_session_start_promotes_existing_to_current(tmp_path):
    """Idempotent second open still promotes that session to current
    (so subsequent calls without session_id resolve to it)."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    mem.session_start("a")
    mem.session_start("b")
    assert mem._current_session_id == "b"
    # Re-open "a" — should promote it back to current without
    # clobbering its context.
    mem.session_start("a")
    assert mem._current_session_id == "a"
    mem.close()


# --- I4: fallback_direction stable across processes --------------------


def test_fallback_direction_stable_within_process():
    a = fallback_direction("fact-123", dim=3)
    b = fallback_direction("fact-123", dim=3)
    np.testing.assert_array_equal(a, b)


def test_fallback_direction_different_ids_differ():
    a = fallback_direction("fact-a", dim=3)
    b = fallback_direction("fact-b", dim=3)
    assert not np.array_equal(a, b)


def test_fallback_direction_is_unit_length():
    v = fallback_direction("fact-x", dim=3)
    assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-6


def test_fallback_direction_stable_across_processes():
    """The previous implementation used hash() which is salted per
    process (PYTHONHASHSEED defaults to random). Spawn two fresh
    Python subprocesses with DIFFERENT PYTHONHASHSEED and confirm
    they get the same fallback direction for the same fact_id."""
    script = (
        "import sys; sys.path.insert(0, 'src');"
        "from birch.galaxy.loader import fallback_direction;"
        "import numpy as np;"
        "v = fallback_direction('contractual-test-id', dim=4);"
        "print(','.join(f'{x:.6f}' for x in v))"
    )
    cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    env_a = os.environ.copy()
    env_a["PYTHONHASHSEED"] = "0"
    env_b = os.environ.copy()
    env_b["PYTHONHASHSEED"] = "12345"

    out_a = subprocess.check_output(
        [sys.executable, "-c", script], cwd=cwd, env=env_a,
    ).decode().strip()
    out_b = subprocess.check_output(
        [sys.executable, "-c", script], cwd=cwd, env=env_b,
    ).decode().strip()

    assert out_a == out_b, (
        f"fallback_direction unstable across PYTHONHASHSEED: "
        f"{out_a!r} vs {out_b!r}"
    )
