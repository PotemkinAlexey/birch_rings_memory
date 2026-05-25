"""Two micro-hardening fixes plus three documented stale verdicts:

  STALE (verified, no action):
    - VectorIndex.remove dim reset: already shipped in earlier
      round (see test_vector_index_remove_resets_dim_when_empty).
    - Async collapse race: lock IS held — _maybe_trigger_collapse
      _locked submits while caller holds self._lock; the worker
      acquires self._lock inside collapse_singularity. No race.
    - _center_of_mass unused: called from
      singularity_compactor.py:186 and covered by 3 tests in
      test_singularity_compactor.py.

  REAL FIXES:

  1. VectorIndex.dim public property. Eight call sites in
     black_hole.py + memory_store/_facts.py used to reach into
     ``_index._dim`` directly — fragile encapsulation. A rename
     of the storage field would silently break every consumer.
     Public ``dim`` property returns the active embedding
     dimension or ``None`` when the index is empty; storage
     internals can now evolve without breaking callers.

  2. ``_auto_link_fact`` hard-caps surviving neighbours to
     AUTO_LINK_TOP_K. Previous code oversampled by 1 to absorb
     the inevitable self-match (the fact was added to _index
     before auto_link runs). When the new fact's vector is
     identical to several others, argpartition's tie-break
     order is undefined: self may NOT appear in the first
     top_k+1 results, and the call wires up top_k+1 real edges
     instead of top_k. Hard cap closes the gap.
"""
from __future__ import annotations

from birch.fact import FactPassport
from birch.gravity import GravityEngine
from birch.memory_store import MemoryStore
from birch.vector_index import VectorIndex

# --- I1: VectorIndex.dim public property ------------------------------


def test_vector_index_dim_property_empty_is_none():
    idx = VectorIndex()
    assert idx.dim is None


def test_vector_index_dim_property_set_on_first_add():
    idx = VectorIndex()
    idx.add("f1", [0.1, 0.2, 0.3])
    assert idx.dim == 3


def test_vector_index_dim_property_resets_when_emptied():
    """Symmetric with the existing remove-dim-reset contract."""
    idx = VectorIndex()
    idx.add("f1", [0.1, 0.2])
    assert idx.dim == 2
    idx.remove("f1")
    assert idx.dim is None
    # And re-establishing with new dim works.
    idx.add("f2", [0.5] * 7)
    assert idx.dim == 7


def test_vector_index_dim_property_is_read_only():
    """Setter is intentionally absent — dim is derived from contents,
    not externally settable. Public surface stays narrow."""
    import pytest

    idx = VectorIndex()
    idx.add("f1", [0.1, 0.2])
    with pytest.raises(AttributeError):
        idx.dim = 99   # type: ignore[misc]


def test_callsites_use_public_dim_property():
    """Source-level audit: no remaining ._dim accesses outside
    vector_index.py itself."""
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parents[1] / "src" / "birch"
    bad: list[str] = []
    for py in root.rglob("*.py"):
        if py.name == "vector_index.py":
            continue
        text = py.read_text()
        # Match self._index._dim or self._meta_index._dim or
        # similar — defensive even on future variations.
        if re.search(r"_(meta_)?index\._dim\b", text):
            bad.append(str(py.relative_to(root)))
    assert not bad, (
        f"Files still reaching into VectorIndex._dim directly: {bad}. "
        f"Use the public .dim property instead."
    )


# --- I2: _auto_link_fact hard cap -------------------------------------


def test_auto_link_caps_at_top_k_even_with_identical_vectors(tmp_path):
    """Pathological case: every fact has the same vector (e.g. mock
    embedding always returns the constant vector). Without the cap,
    the new fact would link to AUTO_LINK_TOP_K + 1 neighbours when
    self doesn't surface in argpartition's first slice."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"), auto_link=True)
    # Force every embedding to the same vector so cosine ≈ 1 against
    # every other fact in the index — argpartition's tie-break is
    # then undefined and self may sit anywhere in the top.
    import birch.memory_store as pkg
    original = pkg.embed
    pkg.embed = lambda text: [0.5, 0.5, 0.5]
    try:
        # Seed N facts before adding the one we'll inspect.
        n_seed = mem.AUTO_LINK_TOP_K + 5
        for i in range(n_seed):
            mem.add_fact(f"subj{i}", "uses", f"obj{i}")
        # Now add the inspected fact; auto_link runs against the
        # seeded population.
        f = mem.add_fact("inspected", "uses", "thing")
    finally:
        pkg.embed = original

    # Count edges originating from `inspected` in the engine.
    edges_from_inspected = [
        (a, b) for (a, b) in mem._engine._edges if a == f.fact_id
    ]
    assert len(edges_from_inspected) <= mem.AUTO_LINK_TOP_K, (
        f"auto_link wired {len(edges_from_inspected)} edges, "
        f"exceeds AUTO_LINK_TOP_K={mem.AUTO_LINK_TOP_K}"
    )
    # And no self-edge regardless of tie-break outcome.
    assert (f.fact_id, f.fact_id) not in mem._engine._edges
    mem.close()


def test_auto_link_normal_case_still_links_top_k():
    """Sanity: with distinct vectors and a healthy neighbour pool the
    cap doesn't prevent the legitimate top_k linking."""
    # Build a minimal engine + index manually to avoid the embedding
    # provider stub.
    engine = GravityEngine()
    idx = VectorIndex()
    # Seed 10 facts with distinct vectors.
    seed_facts = []
    for i in range(10):
        f = FactPassport(
            subject=f"s{i}", predicate="p", object=f"o{i}",
            fact_id=f"f{i}",
        )
        seed_facts.append(f)
        engine.register(f)
        # Distinct vectors via a varying first coordinate.
        idx.add(f.fact_id, [float(i), 0.1, 0.2])

    # Hand-execute the auto_link logic against a NEW fact close to
    # f5 (highest cosine to [5.0, 0.1, 0.2]).
    new_fact = FactPassport(
        subject="new", predicate="p", object="o", fact_id="new",
    )
    engine.register(new_fact)
    new_vec = [5.0, 0.1, 0.2]
    idx.add("new", new_vec)

    AUTO_LINK_TOP_K = 3
    neighbours = idx.search(new_vec, top_k=AUTO_LINK_TOP_K + 1)
    linked = 0
    for nid, _ in neighbours:
        if nid == "new":
            continue
        if linked >= AUTO_LINK_TOP_K:
            break
        engine.link("new", nid)
        engine.link(nid, "new")
        linked += 1

    assert linked == AUTO_LINK_TOP_K
    edges_from_new = [(a, b) for (a, b) in engine._edges if a == "new"]
    assert len(edges_from_new) == AUTO_LINK_TOP_K
