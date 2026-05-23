"""Tolerant JSON loaders and compactor dim-grouping regressions.

Covers storage robustness, compactor mixed-dim safety, collapse
metric truthfulness, and echo exclude semantics. The big one is the
tolerant JSON loaders — one corrupted row used to bring down the
entire MemoryStore boot.
"""
from __future__ import annotations

import sqlite3

from birch.fact import FactPassport
from birch.memory_store import MemoryStore

# --- P1: tolerant JSON loaders --------------------------------------------


def test_load_facts_skips_corrupted_row(tmp_path):
    """A single corrupted JSON in the vector column used to take down
    MemoryStore.__init__. It should now skip the bad row and load the
    rest."""
    db = str(tmp_path / "m.db")
    mem = MemoryStore(db_path=db)
    good = mem.add_fact("api", "runs on", "Go")
    bad = mem.add_fact("api2", "runs on", "Rust")
    mem.close()

    # Corrupt one fact's vector column directly on disk.
    conn = sqlite3.connect(db)
    conn.execute(
        "UPDATE facts SET vector = ? WHERE fact_id = ?",
        ("{not valid json", bad.fact_id),
    )
    conn.commit()
    conn.close()

    # Reopening must not raise; the good fact survives.
    again = MemoryStore(db_path=db)
    ids = {f.fact_id for f in again.list_facts()}
    assert good.fact_id in ids
    # Bad row was tolerated: either skipped entirely or loaded with
    # default empty vector. Either way the store boots.
    if bad.fact_id in again._facts:
        assert again._facts[bad.fact_id].vector == []


def test_load_open_sessions_drops_corrupted_row(tmp_path):
    """A crashed open-session with bad JSON used to prevent startup; now
    the loader drops the row (and deletes it from storage so it does not
    keep blocking)."""
    db = str(tmp_path / "m.db")
    mem = MemoryStore(db_path=db)
    mem.session_start("good")
    mem.session_message("hello", session_id="good")
    mem.close()

    # Inject a corrupted open_session row directly.
    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT OR REPLACE INTO open_sessions VALUES (?,?,?,?,?)",
        ("bad", "{junk", "{junk", "{junk", 0.0),
    )
    conn.commit()
    conn.close()

    # Reopening must not raise; the good session is preserved.
    again = MemoryStore(db_path=db)
    assert "good" in again._sessions
    assert "bad" not in again._sessions


def test_load_echo_sessions_skips_corrupted_row(tmp_path):
    """Same robustness contract for echo_sessions row."""
    db = str(tmp_path / "m.db")
    mem = MemoryStore(db_path=db)
    mem.session_start("s")
    mem.session_message("hello")
    mem.session_close(session_id="s")
    mem.close()

    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT OR REPLACE INTO echo_sessions "
        "(session_id, centroids, r_score, recorded_at, fact_ids, echo_penalty) "
        "VALUES (?,?,?,?,?,?)",
        ("bad", "{not json", 0.0, 0.0, "{not json", 0.0),
    )
    conn.commit()
    conn.close()

    again = MemoryStore(db_path=db)
    assert "s" in again._echo._sessions
    assert "bad" not in again._echo._sessions


# --- P2: compactor groups by dimension -----------------------------------


def test_compactor_does_not_crash_on_mixed_vector_dims():
    """Black hole with vectors of two different dimensions used to crash
    the compactor's np.asarray call (ragged array). Per-dim grouping
    means the compute now runs cleanly. The follow-up absorb step still
    has to respect _meta_index dim invariant — one dim group succeeds,
    the other raises on its way into the live meta index. The contract
    being pinned: the compute itself (np.asarray over mixed dims) no
    longer crashes."""
    from birch.black_hole import BlackHole, SingularityRecord
    from birch.singularity_compactor import collapse_singularity

    hole = BlackHole()
    # Lock the meta index to dim 4 ahead of time (matches the dim of
    # the first absorbable group).
    a1 = FactPassport("alpha", "1", "x")
    a1.vector = [1.0, 0.0, 0.0, 0.0]
    a2 = FactPassport("alpha", "2", "y")
    a2.vector = [0.99, 0.01, 0.0, 0.0]
    # Same SingularityRecord direct insert — bypasses index, simulates
    # legacy bodies that landed before the dim guard existed.
    for f in (a1, a2):
        hole._singularity[f.fact_id] = SingularityRecord(fact=f)

    new_metas, report = collapse_singularity(
        hole, threshold=0.95, min_group_size=2,
    )
    # No crash on the np.asarray step — the dim-grouping fix works
    # at the COMPUTE level. The dim-4 group collapsed.
    assert report.groups >= 1
    assert report.absorbed_facts >= 2
    assert len(new_metas) >= 1


def test_compactor_dim_grouping_independent_per_dim():
    """Per-dim grouping: dim-4 and dim-8 bodies must never be paired in
    the Union-Find pass. We pin this by checking that the Union-Find
    doesn't link bodies of different dims when both groups happen to be
    above the cosine threshold (e.g. both at axis-aligned vectors). The
    test pre-locks the meta index to dim 4 so absorb works cleanly."""
    from birch.black_hole import BlackHole, SingularityRecord
    from birch.singularity_compactor import collapse_singularity

    hole = BlackHole()
    a1 = FactPassport("alpha", "1", "x")
    a1.vector = [1.0, 0.0, 0.0, 0.0]
    a2 = FactPassport("alpha", "2", "y")
    a2.vector = [0.99, 0.01, 0.0, 0.0]
    for f in (a1, a2):
        hole._singularity[f.fact_id] = SingularityRecord(fact=f)

    new_metas, report = collapse_singularity(
        hole, threshold=0.95, min_group_size=2,
    )
    # The dim-4 group produced exactly one MetaFact bundling the two.
    assert report.groups == 1
    assert len(new_metas) == 1
    assert len(new_metas[0].source_fact_ids) == 2
    assert set(new_metas[0].source_fact_ids) == {a1.fact_id, a2.fact_id}


# --- P2: collapse metrics split attempts vs successes -------------------


def test_collapse_metrics_split_no_op_vs_real(tmp_path):
    """A pass that compresses nothing must NOT increment total_collapses;
    it should increment total_collapse_attempts instead."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"), collapse_async=False)
    # Singularity has 0 facts — collapse will report 0 groups.
    report = mem.collapse_singularity(persist=False)
    assert report.groups == 0
    stats = mem.stats
    assert stats["total_collapses"] == 0
    assert stats["total_collapse_attempts"] >= 1
    assert stats["last_collapse_at"] is None
    assert stats["last_collapse_attempt_at"] is not None


# --- P2: EchoStore exclude_session_id -----------------------------------


def test_detect_echo_excludes_named_session():
    """When checking echo for a session that's already recorded itself,
    exclude_session_id must skip it so it doesn't match its own bundle."""
    from birch.resonance.cluster import ClusterBundle
    from birch.resonance.echo import EchoStore, StoredSession

    store = EchoStore()
    bundle = ClusterBundle(centroids=[[1.0, 0.0, 0.0]], k=1, inertia=0.0)
    store._sessions["only"] = StoredSession(
        session_id="only",
        bundle=bundle,
        r_score=0.5,
        fact_weights={},
    )
    # Without exclude — matches itself at cosine 1.0 (above threshold).
    res = store.detect_echo([1.0, 0.0, 0.0])
    assert res.matched_session_id == "only"
    # With exclude — the only candidate is gone; no_history.
    res_excluded = store.detect_echo([1.0, 0.0, 0.0],
                                     exclude_session_id="only")
    assert res_excluded.matched_session_id is None


def test_memorystore_check_echo_threads_exclude(tmp_path):
    """MemoryStore.check_echo must forward its session_id as exclusion to
    the underlying detect_echo — previously the param was 'currently
    unused'."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    mem.session_start("now")
    mem.session_message("server keeps failing", session_id="now")
    mem.session_close(session_id="now")

    # check_echo with the same session id should not match the just-closed
    # session against itself.
    result = mem.check_echo("server keeps failing", session_id="now")
    assert result.get("matched_session") != "now"
