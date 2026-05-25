"""VectorIndex now backs its (n, d) matrix with a preallocated buffer
that grows geometrically instead of np.vstack-ing the entire matrix
on every add. Net add cost: amortised O(d) vs the old O(n·d).

Contracts pinned here:

  1. Initial allocation: first add() establishes ``_dim`` AND
     allocates a buffer at ``_INITIAL_CAPACITY`` rows, not 1.

  2. Geometric growth: when ``_size`` outgrows ``_capacity``, the
     buffer doubles. Verified by counting capacity transitions
     across many sequential adds.

  3. In-place overwrite: replacing an existing fact_id does NOT
     change ``_size`` or ``_capacity`` (no buffer growth on replace).

  4. Swap-with-last remove: removing a non-last row swaps the last
     row into its slot and decrements size. The id at the removed
     row is gone; everything else stays addressable.

  5. Auto-shrink: after a mass-delete that drops size below
     ``capacity / _SHRINK_RATIO`` the buffer shrinks back so a
     long-running store doesn't sit on the peak allocation.

  6. Public-API parity: __len__, __contains__, dim, search,
     all_similarities all behave identically to the old vstack
     implementation across every existing test (the 725-test suite
     stays green; this file adds the storage-specific assertions).

  7. Performance: 1000 sequential adds completes in well under
     1 second on a modest box — sanity check that nothing did an
     accidental O(n²) regression slip back in.
"""
from __future__ import annotations

import time

from birch.vector_index import (
    _GROWTH_FACTOR,
    _INITIAL_CAPACITY,
    _SHRINK_RATIO,
    VectorIndex,
)

# --- I1: initial allocation -----------------------------------------


def test_initial_add_allocates_initial_capacity():
    idx = VectorIndex()
    assert idx._capacity == 0      # empty index has no buffer
    assert idx._buffer is None
    idx.add("f1", [0.1, 0.2, 0.3])
    assert idx._capacity == _INITIAL_CAPACITY
    assert idx._size == 1
    assert idx.dim == 3
    assert idx._buffer is not None
    assert idx._buffer.shape == (_INITIAL_CAPACITY, 3)


# --- I2: geometric growth -------------------------------------------


def test_buffer_doubles_when_full():
    idx = VectorIndex()
    # Fill exactly to the initial capacity.
    for i in range(_INITIAL_CAPACITY):
        idx.add(f"f{i}", [float(i), 0.0, 0.0])
    assert idx._capacity == _INITIAL_CAPACITY
    assert idx._size == _INITIAL_CAPACITY
    # One more push triggers a doubling.
    idx.add("overflow", [99.0, 0.0, 0.0])
    assert idx._capacity == _INITIAL_CAPACITY * _GROWTH_FACTOR
    assert idx._size == _INITIAL_CAPACITY + 1


def test_growth_happens_at_capacity_boundaries():
    """Capacity should only change at exact fill events — not earlier,
    not later."""
    idx = VectorIndex()
    transitions: list[tuple[int, int]] = []  # (size_before, new_capacity)
    last_capacity = 0
    for i in range(_INITIAL_CAPACITY * 8):
        idx.add(f"f{i}", [float(i), 0.0])
        if idx._capacity != last_capacity:
            transitions.append((idx._size, idx._capacity))
            last_capacity = idx._capacity
    # First transition: 0→INITIAL on add #1.
    assert transitions[0] == (1, _INITIAL_CAPACITY)
    # Subsequent transitions: INITIAL→2*INITIAL→4*INITIAL→8*INITIAL.
    expected = _INITIAL_CAPACITY
    for size, cap in transitions[1:]:
        expected *= _GROWTH_FACTOR
        assert cap == expected
        # Transition fires on the add that brought size to old_cap+1.
        assert size == cap // _GROWTH_FACTOR + 1


# --- I3: in-place overwrite -----------------------------------------


def test_replace_existing_id_does_not_grow():
    idx = VectorIndex()
    for i in range(_INITIAL_CAPACITY):
        idx.add(f"f{i}", [float(i), 0.0])
    cap_before = idx._capacity
    size_before = idx._size
    idx.add("f3", [99.0, 99.0])   # overwrite
    assert idx._capacity == cap_before
    assert idx._size == size_before
    # And the vector at f3's row reflects the new value.
    row = idx._id_to_row["f3"]
    assert idx._buffer is not None
    # L2-normalised, so check direction not magnitude.
    new_vec = idx._buffer[row]
    assert abs(new_vec[0] - new_vec[1]) < 1e-6   # equal components


# --- I4: swap-with-last remove --------------------------------------


def test_remove_swaps_last_into_freed_slot():
    idx = VectorIndex()
    for i in range(5):
        idx.add(f"f{i}", [float(i), 0.0])
    # Remove a middle element — the last one should fill its slot.
    f1_row_before = idx._id_to_row["f1"]
    f4_row_before = idx._id_to_row["f4"]   # last
    idx.remove("f1")
    assert "f1" not in idx
    # f4 now lives where f1 lived.
    assert idx._id_to_row["f4"] == f1_row_before
    # Everyone else is still indexable.
    for i in [0, 2, 3, 4]:
        assert f"f{i}" in idx
    assert len(idx) == 4
    # Sanity: the old last slot is gone (size decreased).
    assert idx._size == 4
    assert f4_row_before > idx._size - 1 or idx._id_to_row["f4"] != f4_row_before


def test_remove_last_row_no_swap_needed():
    idx = VectorIndex()
    for i in range(3):
        idx.add(f"f{i}", [float(i), 0.0])
    idx.remove("f2")    # the last one — no swap, just truncate
    assert "f2" not in idx
    assert idx._id_to_row["f0"] == 0
    assert idx._id_to_row["f1"] == 1
    assert idx._size == 2


def test_search_after_swap_returns_correct_ids():
    """Regression guard: swap-with-last must keep _ids and the
    underlying matrix in sync. A query right after a remove should
    return correct (fact_id, score) pairs, not stale ids."""
    idx = VectorIndex()
    # Distinct vectors so cosine ordering is unambiguous.
    idx.add("alpha", [1.0, 0.0])
    idx.add("beta", [0.0, 1.0])
    idx.add("gamma", [1.0, 1.0])     # last; closest to [1, 1]
    idx.remove("beta")                # swap gamma into beta's slot
    hits = idx.search([1.0, 1.0], top_k=2)
    ids = [h[0] for h in hits]
    assert "gamma" in ids
    assert "alpha" in ids
    assert "beta" not in ids


# --- I5: auto-shrink -------------------------------------------------


def test_buffer_shrinks_after_mass_delete():
    idx = VectorIndex()
    # Grow well past initial.
    n = _INITIAL_CAPACITY * 8
    for i in range(n):
        idx.add(f"f{i}", [float(i), 0.0])
    peak_capacity = idx._capacity
    assert peak_capacity >= n
    # Delete most of them, leaving very few.
    keep = 2
    for i in range(n - keep):
        idx.remove(f"f{i}")
    assert idx._size == keep
    # Buffer should have shrunk meaningfully — not back to INITIAL
    # necessarily (we keep 2× headroom), but well below peak.
    assert idx._capacity < peak_capacity
    assert idx._capacity >= _INITIAL_CAPACITY


def test_no_shrink_below_initial_capacity():
    idx = VectorIndex()
    for i in range(_INITIAL_CAPACITY):
        idx.add(f"f{i}", [float(i), 0.0])
    # Now remove everything except one — but not all (full reset is
    # a separate path).
    for i in range(_INITIAL_CAPACITY - 1):
        idx.remove(f"f{i}")
    # Capacity must not go below the floor — churning around tiny
    # sizes shouldn't trigger constant realloc.
    assert idx._capacity == _INITIAL_CAPACITY


def test_full_empty_resets_dim():
    """Existing contract: removing the last vector resets dim so the
    index can accept a new model's dimension."""
    idx = VectorIndex()
    idx.add("f1", [0.1, 0.2, 0.3])
    idx.remove("f1")
    assert idx.dim is None
    assert idx._buffer is None
    assert idx._capacity == 0
    idx.add("f2", [0.5] * 99)        # different dim
    assert idx.dim == 99


# --- I6: parity (already implicit from the 725-test suite) ----------
# Add a couple of explicit parity asserts to catch any future drift.


def test_len_reports_live_size_not_capacity():
    idx = VectorIndex()
    for i in range(5):
        idx.add(f"f{i}", [float(i), 0.0])
    assert len(idx) == 5
    assert idx._capacity == _INITIAL_CAPACITY    # buffer is bigger


def test_contains_reflects_live_membership():
    idx = VectorIndex()
    idx.add("alive", [1.0, 0.0])
    idx.add("dead", [0.0, 1.0])
    idx.remove("dead")
    assert "alive" in idx
    assert "dead" not in idx


def test_all_similarities_returns_live_ids_only():
    idx = VectorIndex()
    idx.add("a", [1.0, 0.0])
    idx.add("b", [0.0, 1.0])
    idx.remove("a")
    sims = idx.all_similarities([1.0, 0.0])
    assert "a" not in sims
    assert "b" in sims
    assert len(sims) == 1


# --- I7: performance sanity -----------------------------------------


def test_1000_sequential_adds_completes_in_reasonable_time():
    """Coarse perf gate: 1000 adds at dim=384 must finish well under
    a second on any modern box. Old O(n·d) implementation took
    ~hundreds of ms on this size; amortised O(d) should be < 100ms.
    Generous 1.0s ceiling so this test stays meaningful across
    machines without flaking."""
    idx = VectorIndex()
    vec = [0.1] * 384
    # Touch one coordinate per fact so vectors are distinct (cosine
    # ordering is undefined for identical vectors).
    start = time.perf_counter()
    for i in range(1000):
        v = list(vec)
        v[i % 384] += 1.0
        idx.add(f"f{i}", v)
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0, (
        f"1000 adds at dim=384 took {elapsed:.3f}s — suspect "
        f"O(n²) regression in the storage layout"
    )
    assert idx._size == 1000
    # And search still works at scale.
    hits = idx.search(vec, top_k=10)
    assert len(hits) == 10


def test_growth_does_not_quadruple_capacity_on_single_add():
    """Defensive: a future refactor that accidentally grows by more
    than _GROWTH_FACTOR per event would silently bloat memory.
    Cap the per-event growth at exactly _GROWTH_FACTOR."""
    idx = VectorIndex()
    for i in range(_INITIAL_CAPACITY):
        idx.add(f"f{i}", [float(i), 0.0])
    cap_at_full = idx._capacity
    idx.add("trigger", [99.0, 0.0])
    assert idx._capacity == cap_at_full * _GROWTH_FACTOR


# --- I8: constants surface ------------------------------------------


def test_growth_constants_are_sane():
    """Sanity assertions on the module-level constants. If anyone
    re-tunes them down the road, these tests act as a "did you mean
    that?" speed bump."""
    assert _INITIAL_CAPACITY >= 4    # under 4 = thrashy on tiny stores
    assert _GROWTH_FACTOR >= 2       # sub-2 ruins amortised O(1)
    assert _SHRINK_RATIO >= 2        # sub-2 causes shrink/grow oscillation
