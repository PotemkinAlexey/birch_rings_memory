"""Salience / irreplaceability — retention orthogonal to frequency.

Frequency-weighted memory systematically forgets the rare-but-critical: a fact
used once a year decays (freshness drops, access stays low) and is absorbed
before its next use. Salience is the counter-signal: a fact with no semantic
substitute in its namespace is irreplaceable — losing it loses knowledge no
neighbour carries — so it earns retention against disuse-absorption regardless
of how rarely it is touched. A redundant fact (many near-duplicates) keeps the
flat floor; the knowledge survives in its neighbours.

See ``_irreplaceability`` / ``_absorption_floor`` in _singularity.py.
Tuned by BIRCH_SALIENCE_NEIGHBOR_THRESHOLD / BIRCH_SALIENCE_PROTECTION.
"""
from __future__ import annotations

import birch.memory_store._singularity as sing
from birch.fact import FactPassport
from birch.memory_store import MemoryStore


def _put(mem, fid, vector, namespace=""):
    f = FactPassport(subject=fid, predicate="p", object="o",
                     fact_id=fid, namespace=namespace)
    f.vector = vector
    mem._facts[fid] = f
    mem._index.add(fid, vector)
    mem._engine.register(f)
    return f


def test_irreplaceability_unique_vs_redundant():
    mem = MemoryStore()
    u = _put(mem, "u", [1.0, 0.0, 0.0])
    r1 = _put(mem, "r1", [0.0, 1.0, 0.0])
    _put(mem, "r2", [0.0, 1.0, 0.0])
    _put(mem, "r3", [0.0, 1.0, 0.0])

    assert mem._irreplaceability(u) == 1.0, "no neighbours → fully irreplaceable"
    # r1 has two identical-vector neighbours (r2, r3) → 1 / (1 + 2).
    assert abs(mem._irreplaceability(r1) - 1.0 / 3.0) < 1e-9


def test_irreplaceability_is_namespace_scoped():
    mem = MemoryStore()
    w = _put(mem, "w", [0.0, 1.0, 0.0], namespace="WORK")
    _put(mem, "p", [0.0, 1.0, 0.0], namespace="PERSONAL")
    # A near-duplicate in another namespace does NOT make it replaceable in WORK.
    assert mem._irreplaceability(w) == 1.0


def test_unique_proven_fact_survives_disuse_while_redundant_is_absorbed():
    mem = MemoryStore()
    u = _put(mem, "u", [1.0, 0.0, 0.0])
    redundant = [_put(mem, f"r{i}", [0.0, 1.0, 0.0]) for i in range(4)]
    # All proved useful (positive resonance) and then decayed below the flat
    # floor through disuse.
    for f in (u, *redundant):
        f.apply_resonance(0.9)
        f.gravity_score = 0.05

    mem._absorb_dead()

    assert "u" in mem._facts, "unique + proven fact must survive disuse"
    assert "u" in mem._salience_retained_ids
    kept = [r.fact_id for r in redundant if r.fact_id in mem._facts]
    # The redundant cluster is pruned, but not annihilated — the last member,
    # once unique, is itself irreplaceable, so the knowledge is never fully lost.
    assert 1 <= len(kept) < 4, f"redundant cluster should be pruned, kept={kept}"


def test_unique_but_unproven_fact_is_not_protected():
    """Uniqueness alone is NOT salience: a unique fact that never proved useful
    (no resonance) decays and absorbs like any other — otherwise a store of
    unique-by-default facts would never absorb anything and hoard junk."""
    mem = MemoryStore()
    u = _put(mem, "u", [1.0, 0.0, 0.0])  # unique, but resonance_count == 0
    u.gravity_score = 0.05
    mem._absorb_dead()
    assert "u" not in mem._facts, "unproven unique fact must absorb normally"
    assert "u" not in mem._salience_retained_ids


def test_salience_protection_zero_reverts_to_flat_floor(monkeypatch):
    """BIRCH_SALIENCE_PROTECTION=0 ⇒ even a unique + proven fact gets the flat
    floor and absorbs — the knob fully disables retention."""
    monkeypatch.setattr(sing, "_SALIENCE_PROTECTION", 0.0)
    mem = MemoryStore()
    u = _put(mem, "u", [1.0, 0.0, 0.0])
    u.apply_resonance(0.9)
    u.gravity_score = 0.05
    mem._absorb_dead()
    assert "u" not in mem._facts, "with protection disabled, unique fact absorbs"
    assert "u" not in mem._salience_retained_ids
