"""BlackHole — polymorphic absorption and emission for FactPassport + MetaFact."""
from birch.black_hole import BlackHole
from birch.fact import FactPassport
from birch.meta_fact import MetaFact


def test_absorb_meta_lands_in_meta_singularity():
    hole = BlackHole()
    m = MetaFact(meta_id="m-1", vector=[1.0, 0.0, 0.0], weight=5)
    hole.absorb_meta(m)
    assert hole.meta_mass == 1
    assert hole.fact_mass == 0
    assert hole.mass == 1
    assert "m-1" in hole
    assert m.layer == -1, "absorb_meta must mark layer beyond core"


def test_absorb_keeps_fact_and_meta_indices_separate():
    hole = BlackHole()
    f = FactPassport("x", "is", "y", fact_id="f-1")
    f.vector = [1.0, 0.0, 0.0]
    m = MetaFact(meta_id="m-1", vector=[0.0, 1.0, 0.0])
    hole.absorb(f)
    hole.absorb_meta(m)

    # A query that hits only the FactPassport must not emit the MetaFact.
    emitted_facts = hole.hawking_emit([1.0, 0.0, 0.0])
    assert [e.fact_id for e in emitted_facts] == ["f-1"]
    assert hole.fact_mass == 0
    assert hole.meta_mass == 1, "MetaFact must remain — separate index"


def test_hawking_emit_metas_returns_meta_with_log_gravity_bonus():
    hole = BlackHole()
    m = MetaFact(meta_id="m-h", vector=[1.0, 0.0], weight=100)
    hole.absorb_meta(m)

    emitted = hole.hawking_emit_metas([1.0, 0.0])
    assert len(emitted) == 1
    assert emitted[0].meta_id == "m-h"
    # weight=100 → bonus = 0.10 * log10(100) = 0.20, gravity = 0.50
    assert abs(emitted[0].gravity_score - 0.50) < 1e-6
    assert emitted[0].layer == 1, "emitted MetaFact returns to kinetic"
    assert hole.meta_mass == 0, "emitted MetaFact must leave the singularity"


def test_hawking_emit_metas_accepts_loose_threshold():
    """A MetaFact centroid drifts away from any single original; the
    caller can pass a looser threshold so emission actually fires."""
    hole = BlackHole()
    m = MetaFact(meta_id="m-loose", vector=[1.0, 0.0])
    hole.absorb_meta(m)

    # Query offset by 30° from the centroid — cosine ≈ 0.866.
    off = [0.8660254, 0.5]
    strict = hole.hawking_emit_metas(off, threshold=0.95)
    assert strict == [], "0.95 should not fire on a 0.86 match"
    loose = hole.hawking_emit_metas(off, threshold=0.80)
    assert [e.meta_id for e in loose] == ["m-loose"]


def test_mass_counter_includes_metas():
    hole = BlackHole()
    f = FactPassport("x", "is", "y")
    f.vector = [1.0, 0.0]
    hole.absorb(f)
    hole.absorb_meta(MetaFact(vector=[0.0, 1.0]))
    hole.absorb_meta(MetaFact(vector=[1.0, 1.0]))
    assert hole.mass == 3
    assert hole.fact_mass == 1
    assert hole.meta_mass == 2


def test_total_emissions_counts_both_kinds():
    hole = BlackHole()
    f = FactPassport("x", "is", "y", fact_id="f1")
    f.vector = [1.0, 0.0]
    hole.absorb(f)
    hole.absorb_meta(MetaFact(meta_id="m1", vector=[1.0, 0.0]))

    hole.hawking_emit([1.0, 0.0])
    hole.hawking_emit_metas([1.0, 0.0])
    assert hole.total_emissions == 2


def test_memory_store_restores_meta_from_storage_on_open(tmp_path):
    """Saved MetaFacts must come back into the BlackHole on next open."""
    from birch.memory_store import MemoryStore

    db = tmp_path / "with-metas.db"
    mem = MemoryStore(db_path=str(db))
    meta = MetaFact(meta_id="m-rt", vector=[1.0, 0.0, 0.0], weight=4)
    mem._hole.absorb_meta(meta)
    mem._storage.save_meta_fact(meta)
    mem._storage.close()

    reopened = MemoryStore(db_path=str(db))
    assert reopened._hole.meta_mass == 1
    assert "m-rt" in reopened._hole
