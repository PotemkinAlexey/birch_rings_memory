"""Per-dimension singularity indices — the architectural refactor that
deferred Vader finding #5.

Background. The earlier rounds defended a real hazard: a single shared
``_index`` per body type in ``BlackHole`` meant that a fact whose
vector dim differed from existing singularity bodies couldn't be
absorbed without crashing the index. Round 4 shipped atomic
three-phase ``absorb`` + rollback so the failure mode was at least
clean (body stays live, no half-state). But the root cause — one
matrix for all dims — stayed open.

This module pins the per-dim refactor contract:

  1. ``BlackHole`` partitions its vector indices by dim:
     ``_indices: dict[int, VectorIndex]`` for facts,
     ``_meta_indices: dict[int, VectorIndex]`` for metas. Each
     bucket is dim-pure; new dims create new buckets lazily.

  2. Buckets auto-prune when their last body emits or is forgotten —
     long-running stores after a model swap don't accumulate empty
     dim slots forever.

  3. ``hawking_emit`` / ``peek_hawking_candidates`` scan ONLY the
     bucket matching the query vector's dim. Cross-dim cosine is
     undefined; bodies in other dim buckets are correctly invisible.

  4. ``forget_fact`` / ``forget_meta`` are the public helpers
     consumers use to drop a body cleanly (atomic singularity-pop +
     bucket-remove + lazy prune). The compactor and
     ``MemoryStore.delete_body`` no longer reach into private
     ``_index`` / ``_meta_index`` directly.

  5. The loader path (``restore_fact`` / ``restore_meta``) routes by
     dim too — a database with mixed-dim absorbed bodies (post-model-
     swap) rehydrates without warnings, retaining every vector.
"""
from __future__ import annotations

from birch.black_hole import BlackHole, SingularityRecord
from birch.fact import FactPassport
from birch.memory_store import MemoryStore
from birch.meta_fact import MetaFact

# --- I1: dim partitioning -------------------------------------------


def test_blackhole_fact_indices_dict_starts_empty():
    hole = BlackHole()
    assert hole._indices == {}
    assert hole._meta_indices == {}
    assert hole.fact_dims == []
    assert hole.meta_dims == []


def test_first_fact_absorb_creates_dim_bucket():
    hole = BlackHole()
    f = FactPassport(
        subject="a", predicate="b", object="c", fact_id="f",
        vector=[0.1] * 7,
    )
    hole.absorb(f)
    assert 7 in hole._indices
    assert hole.fact_dims == [7]
    assert len(hole._indices[7]) == 1


def test_second_fact_same_dim_lands_in_same_bucket():
    hole = BlackHole()
    f1 = FactPassport(
        subject="a", predicate="b", object="c", fact_id="f1",
        vector=[0.1] * 7,
    )
    f2 = FactPassport(
        subject="d", predicate="e", object="f", fact_id="f2",
        vector=[0.2] * 7,
    )
    hole.absorb(f1)
    hole.absorb(f2)
    assert hole.fact_dims == [7]
    assert len(hole._indices[7]) == 2


def test_different_dim_facts_get_separate_buckets():
    """The cross-dim hazard that round 4 had to defend against is
    architecturally impossible now — both absorbs succeed."""
    hole = BlackHole()
    f3 = FactPassport(
        subject="a", predicate="b", object="c", fact_id="f3",
        vector=[1.0, 0.0, 0.0],
    )
    f5 = FactPassport(
        subject="d", predicate="e", object="g", fact_id="f5",
        vector=[1.0, 0.0, 0.0, 0.0, 0.0],
    )
    hole.absorb(f3)
    hole.absorb(f5)   # used to raise DimensionMismatchError
    assert sorted(hole.fact_dims) == [3, 5]
    assert len(hole._indices[3]) == 1
    assert len(hole._indices[5]) == 1
    # Both bodies fully absorbed.
    assert f3.layer == -1
    assert f5.layer == -1


def test_meta_indices_also_per_dim():
    hole = BlackHole()
    m3 = MetaFact(meta_id="m3", vector=[0.1] * 3)
    m5 = MetaFact(meta_id="m5", vector=[0.1] * 5)
    hole.absorb_meta(m3)
    hole.absorb_meta(m5)
    assert sorted(hole.meta_dims) == [3, 5]
    assert len(hole._meta_indices[3]) == 1
    assert len(hole._meta_indices[5]) == 1


# --- I2: bucket auto-prune ------------------------------------------


def test_bucket_drops_when_last_body_forgotten():
    hole = BlackHole()
    f = FactPassport(
        subject="a", predicate="b", object="c", fact_id="f",
        vector=[0.1, 0.2, 0.3],
    )
    hole.absorb(f)
    assert 3 in hole._indices
    hole.forget_fact("f")
    # Bucket pruned — long-running stores don't accumulate empty
    # dim slots after mass-deletes.
    assert 3 not in hole._indices
    assert hole.fact_dims == []


def test_bucket_drops_when_last_body_emits():
    hole = BlackHole()
    f = FactPassport(
        subject="a", predicate="b", object="c", fact_id="f",
        vector=[1.0, 0.0, 0.0],
    )
    hole.absorb(f)
    emitted = hole.hawking_emit([1.0, 0.0, 0.0])
    assert len(emitted) == 1
    # Bucket pruned after the body left.
    assert 3 not in hole._indices


def test_bucket_persists_while_other_bodies_remain():
    hole = BlackHole()
    f1 = FactPassport(
        subject="a", predicate="b", object="c", fact_id="f1",
        vector=[0.1, 0.2, 0.3],
    )
    f2 = FactPassport(
        subject="d", predicate="e", object="f", fact_id="f2",
        vector=[0.4, 0.5, 0.6],
    )
    hole.absorb(f1)
    hole.absorb(f2)
    hole.forget_fact("f1")
    # f2 still in the bucket — must not be dropped.
    assert 3 in hole._indices
    assert len(hole._indices[3]) == 1


# --- I3: cross-dim isolation in Hawking -----------------------------


def test_hawking_emit_isolated_to_query_dim():
    """A query of dim D never resurrects a body of dim D' ≠ D."""
    hole = BlackHole()
    fact_3d = FactPassport(
        subject="three", predicate="d", object="body",
        fact_id="f3d", vector=[1.0, 0.0, 0.0],
    )
    fact_5d = FactPassport(
        subject="five", predicate="d", object="body",
        fact_id="f5d", vector=[1.0, 0.0, 0.0, 0.0, 0.0],
    )
    hole.absorb(fact_3d)
    hole.absorb(fact_5d)
    # Query at dim=3 → only the 3d body returns.
    emitted = hole.hawking_emit([1.0, 0.0, 0.0])
    assert [f.fact_id for f in emitted] == ["f3d"]
    # The 5d body is still absorbed — it was not scanned.
    assert "f5d" in hole._singularity


def test_peek_hawking_candidates_dim_isolated():
    """Same isolation for the peek (non-mutating) variant."""
    hole = BlackHole()
    fact_3d = FactPassport(
        subject="three", predicate="d", object="body",
        fact_id="f3d", vector=[1.0, 0.0, 0.0],
    )
    fact_5d = FactPassport(
        subject="five", predicate="d", object="body",
        fact_id="f5d", vector=[1.0, 0.0, 0.0, 0.0, 0.0],
    )
    hole.absorb(fact_3d)
    hole.absorb(fact_5d)
    candidates = hole.peek_hawking_candidates([1.0, 0.0, 0.0])
    assert len(candidates) == 1
    assert candidates[0][0].fact_id == "f3d"


def test_hawking_emit_empty_query_returns_empty():
    hole = BlackHole()
    hole.absorb(FactPassport(
        subject="a", predicate="b", object="c", fact_id="f",
        vector=[1.0, 0.0],
    ))
    assert hole.hawking_emit([]) == []


def test_hawking_emit_unknown_dim_returns_empty():
    """Query at a dim with no bucket → empty result, no crash."""
    hole = BlackHole()
    hole.absorb(FactPassport(
        subject="a", predicate="b", object="c", fact_id="f",
        vector=[1.0, 0.0],
    ))
    assert hole.hawking_emit([1.0, 0.0, 0.0, 0.0, 0.0]) == []


# --- I4: forget_fact / forget_meta public helpers -------------------


def test_forget_fact_removes_and_returns_true():
    hole = BlackHole()
    f = FactPassport(
        subject="a", predicate="b", object="c", fact_id="f",
        vector=[0.1, 0.2],
    )
    hole.absorb(f)
    assert hole.forget_fact("f") is True
    assert "f" not in hole._singularity
    assert 2 not in hole._indices   # bucket pruned


def test_forget_fact_on_unknown_id_returns_false():
    hole = BlackHole()
    assert hole.forget_fact("ghost") is False


def test_forget_fact_works_on_vectorless_body():
    """Bodies without vectors live in the singularity dict only —
    forget should still remove the dict entry cleanly (no bucket
    work, since there is no vector to remove)."""
    hole = BlackHole()
    f = FactPassport(
        subject="a", predicate="b", object="c", fact_id="f",
        # No vector — body lives in dict only.
    )
    # Stash directly (absorb would skip the dict entry for empty
    # vector in some paths; this is the artificial "no vector"
    # scenario specifically).
    hole._singularity[f.fact_id] = SingularityRecord(fact=f)
    assert hole.forget_fact("f") is True
    assert "f" not in hole._singularity


def test_forget_meta_symmetric():
    hole = BlackHole()
    m = MetaFact(meta_id="m", vector=[0.1, 0.2, 0.3])
    hole.absorb_meta(m)
    assert hole.forget_meta("m") is True
    assert "m" not in hole._meta_singularity
    assert 3 not in hole._meta_indices


# --- I5: loader path round-trips mixed-dim singularity --------------


def test_mixed_dim_singularity_round_trips_through_storage(tmp_path):
    """A store with absorbed bodies at different dims survives a
    restart with every vector intact — no clear, no warning."""
    db = str(tmp_path / "m.db")
    # Seed two facts at different dims directly into storage via the
    # backend's API (live add_fact would lock us into one dim
    # through the embedding mock).
    mem = MemoryStore(db_path=db)
    f3 = FactPassport(
        subject="three", predicate="dim", object="body",
        fact_id="f3", vector=[0.1, 0.2, 0.3],
        layer=-1, gravity_score=0.05,
    )
    f7 = FactPassport(
        subject="seven", predicate="dim", object="body",
        fact_id="f7", vector=[0.5] * 7,
        layer=-1, gravity_score=0.05,
    )
    mem._storage.save_fact(f3)
    mem._storage.save_fact(f7)
    mem.close()

    # Restart — both bodies rehydrate into their dim buckets.
    again = MemoryStore(db_path=db)
    assert "f3" in again._hole._singularity
    assert "f7" in again._hole._singularity
    assert sorted(again._hole.fact_dims) == [3, 7]
    # Vectors retained — neither was cleared as "dim mismatch".
    assert again._hole._singularity["f3"].fact.vector == [0.1, 0.2, 0.3]
    assert again._hole._singularity["f7"].fact.vector == [0.5] * 7
    again.close()


def test_mixed_dim_meta_singularity_round_trips(tmp_path):
    """MetaFact symmetric — mixed-dim collapsed bundles rehydrate
    without losing vectors."""
    db = str(tmp_path / "m.db")
    mem = MemoryStore(db_path=db)
    m3 = MetaFact(
        meta_id="m3", vector=[0.1, 0.2, 0.3],
        source_texts=["a b c"], source_fact_ids=["x"], layer=-1,
    )
    m7 = MetaFact(
        meta_id="m7", vector=[0.5] * 7,
        source_texts=["d e f"], source_fact_ids=["y"], layer=-1,
    )
    mem._storage.save_meta_fact(m3)
    mem._storage.save_meta_fact(m7)
    mem.close()

    again = MemoryStore(db_path=db)
    assert "m3" in again._hole._meta_singularity
    assert "m7" in again._hole._meta_singularity
    assert sorted(again._hole.meta_dims) == [3, 7]
    again.close()


# --- I6: total_emissions counter survives bucket churn --------------


def test_total_emissions_accumulates_across_buckets():
    hole = BlackHole()
    f3 = FactPassport(
        subject="a", predicate="b", object="c", fact_id="f3",
        vector=[1.0, 0.0, 0.0],
    )
    f5 = FactPassport(
        subject="d", predicate="e", object="f", fact_id="f5",
        vector=[1.0, 0.0, 0.0, 0.0, 0.0],
    )
    hole.absorb(f3)
    hole.absorb(f5)
    assert hole.total_emissions == 0
    hole.hawking_emit([1.0, 0.0, 0.0])
    assert hole.total_emissions == 1
    hole.hawking_emit([1.0, 0.0, 0.0, 0.0, 0.0])
    assert hole.total_emissions == 2


# --- I7: status surface still consistent ----------------------------


def test_mass_sums_across_dim_buckets():
    hole = BlackHole()
    hole.absorb(FactPassport(
        subject="a", predicate="b", object="c", fact_id="f3",
        vector=[0.1] * 3,
    ))
    hole.absorb(FactPassport(
        subject="d", predicate="e", object="f", fact_id="f5",
        vector=[0.2] * 5,
    ))
    hole.absorb_meta(MetaFact(meta_id="m4", vector=[0.3] * 4))
    assert hole.fact_mass == 2
    assert hole.meta_mass == 1
    assert hole.mass == 3


def test_contains_works_across_dim_buckets():
    hole = BlackHole()
    hole.absorb(FactPassport(
        subject="a", predicate="b", object="c", fact_id="f3",
        vector=[0.1] * 3,
    ))
    hole.absorb_meta(MetaFact(meta_id="m5", vector=[0.2] * 5))
    assert "f3" in hole
    assert "m5" in hole
    assert "ghost" not in hole


# --- I8: compactor still works -------------------------------------


def test_compactor_uses_forget_fact_helper(tmp_path):
    """The compactor was rewritten to use hole.forget_fact instead of
    reaching into _singularity / _index directly. Sanity: collapse
    still works end-to-end after the refactor."""
    from birch.singularity_compactor import collapse_singularity

    hole = BlackHole()
    # Seed three near-identical bodies that should collapse.
    for i in range(3):
        v = [1.0, 0.0, 0.0]
        v[1] = 0.001 * i   # tiny perturbation; still > 0.92 cosine
        hole.absorb(FactPassport(
            subject=f"s{i}", predicate="p", object=f"o{i}",
            fact_id=f"f{i}", vector=v,
        ))
    new_metas, report = collapse_singularity(
        hole, threshold=0.92, min_group_size=2,
    )
    assert report.groups == 1
    assert report.absorbed_facts == 3
    # Original facts are gone from the singularity AND the bucket.
    for fid in ("f0", "f1", "f2"):
        assert fid not in hole._singularity
    # The bucket that held them auto-pruned when it went empty —
    # the new MetaFact lives in the meta-singularity, not the fact
    # singularity.
    assert hole.fact_dims == []
    # MetaFact landed.
    assert len(new_metas) == 1
    assert hole.meta_mass == 1
