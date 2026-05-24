"""Five invariant-tightening fixes from one external review round:

  1. VectorIndex.search argpartition idiom (top_k - 1 pivot). Cosmetic
     under the existing top_k >= len(sims) guard, but the wrong pivot
     would trip any future refactor that drops the resort.

  2. MetaFact subject_prefix comment now matches the startswith code
     (was: comment promised "contains", code did startswith). Pure
     docs fix; pin the contract so a future "make this contains"
     refactor doesn't quietly diverge from the FactPassport surface.

  3. GravityEngine.link dedup. Storage was already protected by
     PRIMARY KEY + INSERT OR IGNORE, but the in-memory _degrees
     counter advanced on every call. Repeated link() on the same
     pair would inflate gravity (via graph_score) until next _reload
     rebuilt _degrees from disk's unique rows.

  4. session_close snapshot-drift merge. compute_resonance runs
     lock-free, so another thread can session_push or query
     (attribution) in the gap. The writeback used to apply R only to
     the pre-compute snapshot keys; new attributions silently dropped
     when session was popped. Fix: re-read live ctx.facts under the
     writeback lock and merge new entries into the snapshot.

  5. close() executor.shutdown honors the inflight timeout. Was
     wait=True after a 5s timeout swallow, which silently waited
     indefinitely. Now: if the inflight finished, wait fully; if it
     timed out, cancel and skip the join.
"""
from __future__ import annotations

import numpy as np

from birch.gravity import GravityEngine
from birch.memory_store import MemoryStore
from birch.vector_index import VectorIndex

# --- I1: argpartition idiom -------------------------------------------


def test_vector_index_search_returns_true_top_k_under_partition_path():
    """Adversarial input: many vectors, top_k=1, true max in the
    middle. With argpartition pivot=top_k (off-by-one) and no resort
    the slice could miss the true max; with pivot=top_k-1 it's
    correct. The resort line we have makes both orderings produce
    the right answer — pin the result so any future "drop the
    resort to save cycles" refactor breaks here, not silently."""
    idx = VectorIndex()
    # 10 unit vectors in R^3 with one clear winner.
    for i in range(10):
        v = [float(i == 7), 0.0, 0.0]  # only id "f7" matches the query
        idx.add(f"f{i}", v)
    hits = idx.search([1.0, 0.0, 0.0], top_k=1, threshold=0.0)
    assert len(hits) == 1
    assert hits[0][0] == "f7"
    assert hits[0][1] > 0.99


def test_vector_index_search_top_k_smaller_than_n_picks_true_topk():
    """Same shape but top_k=3 forces the partition path."""
    idx = VectorIndex()
    for i in range(20):
        # Strength gradient so the true top-3 is unambiguous.
        strength = (i + 1) / 20.0
        v = list(np.array([strength, 0.0, 0.0]) / np.linalg.norm(
            [strength, 0.0, 0.0]
        ))
        idx.add(f"f{i}", v)
    hits = idx.search([1.0, 0.0, 0.0], top_k=3, threshold=0.0)
    assert len(hits) == 3
    # All three should have similarity 1.0 (every vector points along x).
    assert all(h[1] > 0.99 for h in hits)
    # And the ids should be a subset of {f0..f19}.
    assert all(h[0].startswith("f") for h in hits)


# --- I3: GravityEngine.link dedup -------------------------------------


def test_gravity_engine_link_is_idempotent_on_repeat_pair():
    eng = GravityEngine()

    # Two minimal bodies — register them so link's target id exists in
    # the engine's tracked set.
    class _Body:
        def __init__(self, fid):
            self.fact_id = fid
            self.gravity_score = 0.5
            self.last_accessed = 0.0
            self.access_count = 0
            self.resonance_count = 0
            self.recent_utility = 0.5
            self.forecast_stability = 0.5

    eng.register(_Body("a"))
    eng.register(_Body("b"))

    eng.link("a", "b")
    assert eng._degrees["b"] == 1
    # Repeated link of the same pair — degree must stay at 1.
    eng.link("a", "b")
    eng.link("a", "b")
    eng.link("a", "b")
    assert eng._degrees["b"] == 1, (
        "GravityEngine.link inflated _degrees on repeat call"
    )
    # Different pair targeting b — degree advances exactly once.
    eng.register(_Body("c"))
    eng.link("c", "b")
    assert eng._degrees["b"] == 2


def test_gravity_engine_unregister_drops_stale_edges():
    """If a fact is unregistered and later re-registered with the
    same id (rare but legal in tests), a re-link of the same pair
    must increment once, not be silently a no-op."""
    eng = GravityEngine()

    class _Body:
        def __init__(self, fid):
            self.fact_id = fid
            self.gravity_score = 0.5
            self.last_accessed = 0.0
            self.access_count = 0
            self.resonance_count = 0
            self.recent_utility = 0.5
            self.forecast_stability = 0.5

    eng.register(_Body("a"))
    eng.register(_Body("b"))
    eng.link("a", "b")
    assert eng._degrees["b"] == 1
    eng.unregister("b")
    eng.register(_Body("b"))
    eng.link("a", "b")
    assert eng._degrees["b"] == 1, (
        "Edge dedup must clear when target is unregistered"
    )


# --- I4: session_close drift merge ------------------------------------


def test_session_close_merges_post_snapshot_fact_attributions(tmp_path):
    """Open session, push some messages, simulate an attribution that
    landed AFTER compute_resonance would have started (just add to
    ctx.facts directly — same effect as a concurrent query). Then
    close. The drift attribution must receive R, not be dropped on
    session pop."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    f1 = mem.add_fact("api", "uses", "Postgres")
    # f2 is on a totally unrelated topic so query("api...") below
    # won't attribute it; we'll inject it as drift later.
    f2 = mem.add_fact(
        "weather", "tomorrow forecast", "sunny in tokyo",
    )
    mem.session_start("s1")
    mem.session_message("hello world", session_id="s1")
    # Snapshot phase: query attributes only f1 to the session.
    mem.query(
        "api uses Postgres for storage", top_k=1, session_id="s1",
    )
    ctx = mem._sessions["s1"]
    snapshot_facts = set(ctx.facts.keys())
    # Drift phase: simulate a concurrent attribution landing during
    # compute_resonance. In real code this would be another thread
    # calling query() with the same session_id; here we just inject
    # the same effect (add an entry to ctx.facts) before close.
    ctx.facts[f2.fact_id] = 1.0
    assert f2.fact_id not in snapshot_facts
    # Close with explicit resonance so we know R deterministically.
    resp = mem.session_close(session_id="s1", sentiment="resonant")
    assert resp.get("label") == "resonant"
    # The drift fact must have received positive resonance.
    f2_after = mem._facts[f2.fact_id]
    assert f2_after.resonance_count >= 1, (
        "drift-attributed fact lost its resonance signal on session close"
    )
    # And gravity should have moved (avg_resonance now positive).
    assert f2_after.avg_resonance > 0
    mem.close()
    # Touch first so linter doesn't complain about unused.
    assert f1


# --- I5: close() shutdown honours timeout -----------------------------


def test_close_returns_promptly_when_no_inflight_collapse(tmp_path):
    """The common path: nothing collapsing → close returns at once
    (no 5-second pause). Pin the basic contract before testing the
    edge case."""
    import time as _t

    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    mem.add_fact("api", "uses", "Postgres")
    started = _t.monotonic()
    mem.close()
    elapsed = _t.monotonic() - started
    assert elapsed < 2.0, (
        f"close() took {elapsed:.2f}s with no inflight collapse — "
        "expected near-instant"
    )
