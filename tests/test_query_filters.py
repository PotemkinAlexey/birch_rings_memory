"""query_memory filters apply BEFORE top_k slice (the filter-before-topK fix).

Cursor's review surfaced: previously subject_prefix and min_gravity were
applied AFTER ``MemoryStore.query`` had already sliced its top_k results.
A narrow scope matching only the 9th best hit at top_k=5 returned empty
when the user expected the match. Now filters live inside MemoryStore.query
and act on the full candidate pool before slicing.
"""
from __future__ import annotations

from birch.memory_store import MemoryStore


def _store_with_one_target_among_many(tmp_path) -> tuple[MemoryStore, str]:
    """Build a store where the matching fact is NOT in the unfiltered top_k.

    Twenty unrelated facts under subject "noise"; one target fact under
    subject "needle". The mock embedder ranks by token overlap, so a query
    that includes "noise" tokens will rank the 20 noise facts higher than
    the single needle. Without filter-before-topK, top_k=3 would return
    nothing for subject_prefix="needle".
    """
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    for i in range(20):
        mem.add_fact(f"noise {i}", "is", "filler")
    target = mem.add_fact("needle", "is", "the one we want")
    return mem, target.fact_id


def test_subject_prefix_filter_finds_facts_outside_unfiltered_topk(tmp_path):
    mem, target_id = _store_with_one_target_among_many(tmp_path)
    # The query string is biased toward the noise corpus.
    hits = mem.query(
        "noise filler is the one",
        top_k=3,
        subject_prefix="needle",
        min_similarity=0.0,
    )
    ids = [h.body_id for h in hits]
    assert target_id in ids, (
        "subject_prefix must filter before top_k slice; target was beyond top_k"
    )


def test_min_gravity_filter_drops_low_gravity_hits(tmp_path):
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    weak = mem.add_fact("weak fact", "uses", "thing")
    strong = mem.add_fact("strong fact", "uses", "thing")
    # Manually pull weak's gravity down to below the floor.
    weak.gravity_score = 0.05
    if mem._storage:
        mem._storage.save_fact(weak)

    hits = mem.query(
        "fact uses thing",
        top_k=10,
        min_gravity=0.20,
        min_similarity=0.0,
    )
    ids = {h.body_id for h in hits}
    assert weak.fact_id not in ids
    assert strong.fact_id in ids


def test_filter_compose_with_layer_and_min_similarity(tmp_path):
    """Filters stack — nothing leaks through that violates any single one."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    a = mem.add_fact("needle X", "is", "target")
    b = mem.add_fact("needle Y", "is", "also target")
    mem.add_fact("noise filler", "is", "junk")
    a.gravity_score = 0.05  # below min_gravity
    if mem._storage:
        mem._storage.save_fact(a)

    hits = mem.query(
        "needle X Y is target",
        top_k=10,
        subject_prefix="needle",
        min_gravity=0.20,
        min_similarity=0.0,
    )
    ids = {h.body_id for h in hits}
    # a is excluded by min_gravity, noise is excluded by subject_prefix.
    assert ids == {b.fact_id}


def test_scoped_query_does_not_resurrect_out_of_scope_hawking_facts(tmp_path):
    """A scoped query must NOT side-effect-resurrect a black-hole body that
    fails the scope predicate. The Hawking emission path used to ignore
    subject_prefix / min_gravity and resurrect any sufficiently-cosine match,
    so an agent doing query(..., subject_prefix="needle") would silently
    promote a "wrong-scope" body back to kinetic.
    """
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    # Plant a fact and force it into the singularity by retiring it.
    out_of_scope = mem.add_fact("wrong-scope", "is", "in singularity")
    mem.retire_fact(out_of_scope.fact_id)
    assert out_of_scope.fact_id in mem._hole._singularity

    # Live target inside scope so the query has something legitimate to find.
    target = mem.add_fact("needle inside scope", "is", "live target")

    hits = mem.query(
        "wrong-scope needle inside scope is",
        top_k=5,
        subject_prefix="needle",
        hawking=True,
        min_similarity=0.0,
    )
    ids = {h.body_id for h in hits}

    # The out-of-scope body must NOT appear in results …
    assert out_of_scope.fact_id not in ids
    # … must NOT be silently resurrected into the live store …
    assert out_of_scope.fact_id not in mem._facts
    # … and must remain in the singularity, untouched.
    assert out_of_scope.fact_id in mem._hole._singularity

    # The legitimately-scoped target is still served.
    assert target.fact_id in ids
