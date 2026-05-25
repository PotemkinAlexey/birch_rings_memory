"""BlackHole.absorb / absorb_meta rollback must restore the body's
ORIGINAL layer on failure, not a hardcoded constant.

Earlier defensive code restored the layer to ``2`` (core) for facts and
``0`` (surface) for metas — both wrong: a fact entering at ``layer=0``
silently landed in core after a transient index error, and a MetaFact
that had been Hawking-emitted to ``layer=1`` silently landed at
``layer=0``. The caller had no way to detect or undo the drift.

True atomic rollback returns the body to the exact state it was in
before absorb was called. These tests pin the contract by monkey-
patching the per-dim VectorIndex to raise on the next add() — the
rollback path is the only branch under test.
"""
from __future__ import annotations

import pytest

from birch.black_hole import BlackHole
from birch.fact import FactPassport
from birch.meta_fact import MetaFact
from birch.vector_index import VectorIndex


def _bucket_add_raises_once(idx: VectorIndex, exc: Exception) -> None:
    """Replace ``idx.add`` with a one-shot raiser. After the raise the
    method is restored to its bound original so subsequent calls work."""
    original = idx.add
    fired = {"done": False}

    def boom(fact_id, vector):
        if not fired["done"]:
            fired["done"] = True
            idx.add = original   # type: ignore[method-assign]
            raise exc
        return original(fact_id, vector)

    idx.add = boom   # type: ignore[method-assign]


# --- absorb (FactPassport) -------------------------------------------


@pytest.mark.parametrize("entry_layer", [0, 1, 2])
def test_absorb_rollback_restores_original_fact_layer(entry_layer):
    """Whichever layer the fact entered at, that's what it goes back
    to on rollback. Earlier code hardcoded layer=2."""
    hole = BlackHole()
    f = FactPassport(
        subject="a", predicate="b", object="c", fact_id="f",
        vector=[0.1, 0.2, 0.3], layer=entry_layer,
    )
    # Pre-create the bucket and arm a one-shot index failure so the
    # raise comes from inside absorb's three-phase commit, not from
    # the bucket-creation path.
    idx = hole._index_for(3)
    _bucket_add_raises_once(idx, RuntimeError("simulated numpy alloc"))

    with pytest.raises(RuntimeError, match="simulated"):
        hole.absorb(f)
    # Atomic rollback:
    assert f.layer == entry_layer, (
        f"fact.layer rolled to {f.layer!r}, expected original "
        f"{entry_layer} — rollback must restore on-entry state, not "
        "hardcode a constant"
    )
    assert "f" not in hole._singularity
    # Bucket should still exist but be empty (we created it, the
    # add failed) — prune fires.
    assert 3 not in hole._indices


def test_absorb_success_does_not_touch_layer_restore_path():
    """Sanity: the happy path still sets layer=-1 (no rollback fires)."""
    hole = BlackHole()
    f = FactPassport(
        subject="a", predicate="b", object="c", fact_id="f",
        vector=[0.1, 0.2, 0.3], layer=1,
    )
    hole.absorb(f)
    assert f.layer == -1
    assert "f" in hole._singularity


# --- absorb_meta (MetaFact) ------------------------------------------


@pytest.mark.parametrize("entry_layer", [0, 1, 2])
def test_absorb_meta_rollback_restores_original_meta_layer(entry_layer):
    """Symmetric with absorb — earlier code hardcoded layer=0."""
    hole = BlackHole()
    m = MetaFact(
        meta_id="m", vector=[0.1, 0.2, 0.3], layer=entry_layer,
    )
    idx = hole._meta_index_for(3)
    _bucket_add_raises_once(idx, RuntimeError("simulated"))

    with pytest.raises(RuntimeError, match="simulated"):
        hole.absorb_meta(m)
    assert m.layer == entry_layer
    assert "m" not in hole._meta_singularity
    assert 3 not in hole._meta_indices


def test_absorb_meta_success_path():
    hole = BlackHole()
    m = MetaFact(meta_id="m", vector=[0.1, 0.2, 0.3], layer=1)
    hole.absorb_meta(m)
    assert m.layer == -1
    assert "m" in hole._meta_singularity


# --- Integration with _absorb_dead -----------------------------------


def test_absorb_dead_leaves_fact_layer_unchanged_on_failure(tmp_path):
    """Round-trip: live fact's layer must survive a sweep where the
    bucket index raises. The fact stays live AND keeps its original
    layer — _absorb_dead's catch was leaking a hardcoded layer=2
    into a live fact before this fix."""
    from birch.memory_store import MemoryStore

    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    f = mem.add_fact("api", "uses", "redis")
    # Force live fact below the absorption threshold.
    f.gravity_score = 0.05
    entry_layer = f.layer
    assert entry_layer != -1   # sanity: live before sweep

    # Arm the singularity bucket to raise. The bucket lives on the
    # hole instance, not on the live _index, so live ops are unaffected.
    dim = len(f.vector)
    idx = mem._hole._index_for(dim)
    _bucket_add_raises_once(idx, RuntimeError("simulated"))

    absorbed = mem._absorb_dead()
    assert f.fact_id not in absorbed
    # The fact is STILL LIVE in the store.
    assert f.fact_id in mem._facts
    # And its layer is UNCHANGED from before the sweep — the failure
    # path no longer silently shifts a live fact to "core".
    assert mem._facts[f.fact_id].layer == entry_layer
    # And it's NOT in the singularity.
    assert f.fact_id not in mem._hole._singularity
    mem.close()
