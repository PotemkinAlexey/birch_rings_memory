"""SingularityCompactor — Union-Find collapse inside the black hole."""
from __future__ import annotations

import math

from birch.black_hole import BlackHole
from birch.fact import FactPassport
from birch.meta_fact import MetaFact
from birch.singularity_compactor import (
    _center_of_mass,
    collapse_singularity,
)


def _absorb_with_vector(hole: BlackHole, fact_id: str, vector: list[float]) -> FactPassport:
    f = FactPassport(subject=f"s-{fact_id}", predicate="is", object=fact_id,
                     fact_id=fact_id)
    f.vector = vector
    hole.absorb(f)
    return f


# ── center of mass ───────────────────────────────────────────────────────────

def test_center_of_mass_uniform_weights_is_normalised_mean():
    c = _center_of_mass([[1.0, 0.0], [1.0, 0.0]], [1.0, 1.0])
    assert abs(c[0] - 1.0) < 1e-6
    assert abs(c[1]) < 1e-6


def test_center_of_mass_weighted_pulls_toward_heavier_side():
    c = _center_of_mass([[1.0, 0.0], [0.0, 1.0]], [3.0, 1.0])
    # Without normalisation: [0.75, 0.25]. After L2 norm: roughly [0.949, 0.316].
    assert c[0] > c[1]
    norm = math.sqrt(c[0] ** 2 + c[1] ** 2)
    assert abs(norm - 1.0) < 1e-6


def test_center_of_mass_zero_vectors_returns_zeros_safely():
    c = _center_of_mass([[0.0, 0.0], [0.0, 0.0]], [1.0, 1.0])
    assert c == [0.0, 0.0]


# ── collapse — happy paths ───────────────────────────────────────────────────

def test_collapses_near_duplicates_into_one_meta():
    hole = BlackHole()
    _absorb_with_vector(hole, "a", [1.0, 0.0, 0.0])
    _absorb_with_vector(hole, "b", [0.99, 0.01, 0.0])
    _absorb_with_vector(hole, "c", [0.98, 0.02, 0.0])

    new_metas, report = collapse_singularity(hole, threshold=0.95)

    assert len(new_metas) == 1
    meta = new_metas[0]
    assert meta.weight == 3
    assert set(meta.source_fact_ids) == {"a", "b", "c"}
    assert len(meta.source_texts) == 3

    assert report.absorbed_facts == 3
    assert report.fact_mass_before == 3
    assert report.fact_mass_after == 0
    assert report.meta_mass_after == 1
    assert hole.fact_mass == 0
    assert hole.meta_mass == 1


def test_does_not_collapse_dissimilar_facts():
    hole = BlackHole()
    _absorb_with_vector(hole, "north", [1.0, 0.0])
    _absorb_with_vector(hole, "east",  [0.0, 1.0])
    _absorb_with_vector(hole, "south", [-1.0, 0.0])

    new_metas, report = collapse_singularity(hole, threshold=0.92)
    assert new_metas == []
    assert report.groups == 0
    assert report.absorbed_facts == 0
    assert hole.fact_mass == 3


def test_two_clusters_collapse_independently():
    hole = BlackHole()
    # Cluster A (near east)
    _absorb_with_vector(hole, "a1", [1.0, 0.0])
    _absorb_with_vector(hole, "a2", [0.99, 0.05])
    # Cluster B (near north)
    _absorb_with_vector(hole, "b1", [0.0, 1.0])
    _absorb_with_vector(hole, "b2", [0.05, 0.99])

    new_metas, report = collapse_singularity(hole, threshold=0.95)
    assert len(new_metas) == 2
    assert report.absorbed_facts == 4
    assert hole.fact_mass == 0
    assert hole.meta_mass == 2

    # Each new MetaFact must have weight=2 and disjoint sources.
    weights = sorted(m.weight for m in new_metas)
    assert weights == [2, 2]
    union = set()
    for m in new_metas:
        union.update(m.source_fact_ids)
    assert union == {"a1", "a2", "b1", "b2"}


def test_centroid_vector_is_unit_normalised():
    hole = BlackHole()
    _absorb_with_vector(hole, "a", [1.0, 0.0])
    _absorb_with_vector(hole, "b", [0.99, 0.01])
    new_metas, _ = collapse_singularity(hole, threshold=0.95)
    centre = new_metas[0].vector
    norm = math.sqrt(sum(x * x for x in centre))
    assert abs(norm - 1.0) < 1e-5


def test_collapse_is_idempotent_after_first_pass():
    hole = BlackHole()
    _absorb_with_vector(hole, "a", [1.0, 0.0])
    _absorb_with_vector(hole, "b", [0.99, 0.01])

    first, _ = collapse_singularity(hole, threshold=0.95)
    assert len(first) == 1

    second, report = collapse_singularity(hole, threshold=0.95)
    assert second == []
    assert report.groups == 0
    # The MetaFact created in pass 1 is left alone.
    assert hole.meta_mass == 1


def test_existing_metafacts_are_not_re_collapsed():
    hole = BlackHole()
    # Put a MetaFact in the singularity ahead of time.
    pre_existing = MetaFact(meta_id="pre", vector=[1.0, 0.0], weight=5)
    hole.absorb_meta(pre_existing)
    # Add two near-duplicate facts that would otherwise collapse.
    _absorb_with_vector(hole, "x1", [0.0, 1.0])
    _absorb_with_vector(hole, "x2", [0.01, 0.99])

    new_metas, _ = collapse_singularity(hole, threshold=0.95)
    assert len(new_metas) == 1
    assert hole.meta_mass == 2, "pre-existing MetaFact must survive"
    assert "pre" in hole


def _absorb_ns(hole, fact_id, vector, namespace):
    f = FactPassport(subject="api", predicate="runs on", object="Go",
                     fact_id=fact_id, namespace=namespace)
    f.vector = vector
    hole.absorb(f)
    return f


def test_collapse_partitions_by_namespace_and_meta_inherits_it():
    """MemoryBricks: same-SPO facts from different namespaces have near-identical
    vectors but must NOT merge into one MetaFact. Each namespace collapses on its
    own and the MetaFact inherits the group's namespace (never the global root)."""
    hole = BlackHole()
    _absorb_ns(hole, "work-1", [1.0, 0.0], "WORK/A")
    _absorb_ns(hole, "work-2", [0.99, 0.01], "WORK/A")
    _absorb_ns(hole, "pers-1", [1.0, 0.0], "PERSONAL")
    _absorb_ns(hole, "pers-2", [0.995, 0.005], "PERSONAL")

    new_metas, _ = collapse_singularity(hole, threshold=0.95)

    assert len(new_metas) == 2, "namespaces must collapse independently"
    assert {m.namespace for m in new_metas} == {"WORK/A", "PERSONAL"}
    assert "" not in {m.namespace for m in new_metas}


def test_collapse_skips_empty_vectors():
    hole = BlackHole()
    _absorb_with_vector(hole, "a", [1.0, 0.0])
    # b has no vector at all
    f_b = FactPassport(subject="x", predicate="is", object="y", fact_id="b")
    hole.absorb(f_b)
    _absorb_with_vector(hole, "c", [0.99, 0.01])

    new_metas, report = collapse_singularity(hole, threshold=0.95)
    assert len(new_metas) == 1
    meta = new_metas[0]
    assert set(meta.source_fact_ids) == {"a", "c"}
    # "b" was untouched and remains in the singularity.
    assert "b" in hole
    assert report.fact_mass_after == 1


def test_collapse_with_too_few_facts_is_noop():
    hole = BlackHole()
    _absorb_with_vector(hole, "lonely", [1.0, 0.0])

    new_metas, report = collapse_singularity(hole, threshold=0.95)
    assert new_metas == []
    assert report.groups == 0
    assert report.absorbed_facts == 0
    assert hole.fact_mass == 1


def test_min_group_size_filters_small_clusters():
    """A cluster smaller than min_group_size must not become a MetaFact."""
    hole = BlackHole()
    # One isolated pair that satisfies the threshold but only has 2 members.
    _absorb_with_vector(hole, "p1", [1.0, 0.0])
    _absorb_with_vector(hole, "p2", [0.99, 0.01])

    new_metas, _ = collapse_singularity(hole, threshold=0.95, min_group_size=3)
    assert new_metas == []
    assert hole.fact_mass == 2, "facts must remain because the cluster was too small"
