"""Vector index — numpy-backed cosine search over fact embeddings.

Pure-Python cosine over every fact on every query is O(n·d) in a
slow interpreter loop. At a few thousand facts × 768 dimensions that
turns into hundreds of milliseconds per query.

VectorIndex keeps an L2-normalised (n, d) matrix in sync with insert
and delete calls, so a query reduces to a single matrix–vector dot
product. For an unknown vector dimension we delay matrix allocation
until the first add().

Capacity strategy
-----------------
The matrix is stored in a preallocated buffer ``_buffer`` of shape
``(_capacity, _dim)`` with ``_size`` live rows. ``add()`` writes
into the next free slot in O(d); when ``_size == _capacity`` we
grow the buffer geometrically (double, then numpy copy once). Net
``add()`` cost is amortised O(d) instead of O(n·d) under the old
``np.vstack`` strategy — at 10k facts × 768 dim that's the
difference between ~30 MB copy per insert and a single 3 KB
overwrite. ``remove()`` uses swap-with-last to stay O(d) too; the
caller-visible ``_ids`` order changes, but the public API never
exposed ordering as a contract (search returns by score, not
insertion order).

The buffer shrinks back when ``_size`` drops far below ``_capacity``
so a long-running store that grew then mass-deleted doesn't sit on
gigabytes of unused float32 forever.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

# Initial allocation size for a fresh index. Small enough to avoid
# wasting RAM in tiny test stores, large enough that the typical
# session (a handful of facts) never triggers a regrow.
_INITIAL_CAPACITY = 16

# Geometric growth factor. Doubling gives amortised O(1) append at
# the cost of up to 2× memory overhead at the worst point in the
# growth cycle. Standard CS textbook trade-off.
_GROWTH_FACTOR = 2

# When ``_size`` drops below capacity / _SHRINK_RATIO we reallocate
# down to the closest power-of-two-multiple of _INITIAL_CAPACITY
# above _size. Prevents long-running stores from sitting on a giant
# buffer after a mass-delete + small-reinsert sequence. 4 is a
# conservative threshold — we only shrink when the buffer is at
# most 25% full so churning near the boundary doesn't oscillate.
_SHRINK_RATIO = 4


class DimensionMismatchError(ValueError):
    """Raised when an incoming vector's dimension does not match the index.

    Silently dropping mismatched dimensions is dangerous: the fact still
    lives in ``MemoryStore._facts`` and on disk, but is unsearchable,
    which usually means the embedding model name (``BIRCH_EMBED_MODEL``)
    changed without a reindex. Raising loudly forces the caller to either
    rebuild the index or pin the model.
    """


class VectorIndex:
    """L2-normalised cosine index keyed by fact_id.

    Storage: ``_buffer[:_size]`` is the live matrix (n=``_size``,
    d=``_dim``); ``_buffer[_size:_capacity]`` is preallocated headroom.
    All public APIs operate on the live view; capacity management is
    private.
    """

    def __init__(self) -> None:
        self._ids: list[str] = []
        self._id_to_row: dict[str, int] = {}
        self._buffer: Optional[np.ndarray] = None
        self._size: int = 0
        self._capacity: int = 0
        self._dim: Optional[int] = None

    def __len__(self) -> int:
        return self._size

    def __contains__(self, fact_id: str) -> bool:
        return fact_id in self._id_to_row

    @property
    def dim(self) -> Optional[int]:
        """Active embedding dimension, or ``None`` if the index is empty.

        Public read-only view of the internal ``_dim`` so callers (live
        write paths in ``_facts``, BlackHole.absorb / absorb_meta) can
        preflight dim compatibility without reaching into a name-
        mangled private attribute. Keeps the encapsulation contract:
        renaming the storage field tomorrow does not break every
        consumer. Returns the active dim of an established index, or
        ``None`` when the index is empty (a future ``add`` will
        establish a new dim, see remove() docstring).
        """
        return self._dim

    # ── Internal storage management ─────────────────────────────────────────

    def _live_matrix(self) -> np.ndarray:
        """Return the live (n, d) slice of the underlying buffer.

        This is a numpy *view* into ``_buffer[:_size]``, not a copy —
        the dot product in ``search`` / ``all_similarities`` operates
        on contiguous memory either way. Helper exists so search /
        all_similarities don't both have to know about the
        ``_buffer`` layout.
        """
        assert self._buffer is not None
        return self._buffer[:self._size]

    def _allocate(self, dim: int, capacity: int) -> None:
        """First-time buffer allocation. Caller must have verified
        the index was empty (``_buffer is None``)."""
        self._dim = dim
        self._capacity = capacity
        # np.empty is fine — we write every row before it is
        # observable through self._size, so reading uninitialised
        # bytes is impossible.
        self._buffer = np.empty((capacity, dim), dtype=np.float32)

    def _grow_if_needed(self) -> None:
        """Geometric doubling when the live count is about to outgrow
        the buffer. Called from ``add()`` immediately before writing
        a new row."""
        if self._size < self._capacity:
            return
        assert self._buffer is not None and self._dim is not None
        new_capacity = self._capacity * _GROWTH_FACTOR
        new_buffer = np.empty(
            (new_capacity, self._dim), dtype=np.float32,
        )
        new_buffer[:self._size] = self._buffer[:self._size]
        self._buffer = new_buffer
        self._capacity = new_capacity

    def _maybe_shrink(self) -> None:
        """Shrink the buffer when usage drops below 1/_SHRINK_RATIO of
        capacity, so a long-running store that grew big and then
        mass-deleted doesn't hold the peak allocation forever. Never
        shrinks below ``_INITIAL_CAPACITY`` — churning around tiny
        sizes is not worth the realloc cost."""
        if self._buffer is None or self._dim is None:
            return
        if self._capacity <= _INITIAL_CAPACITY:
            return
        if self._size > self._capacity // _SHRINK_RATIO:
            return
        # Target capacity: smallest power-of-2 multiple of
        # _INITIAL_CAPACITY that fits the current size with some
        # headroom (2× growth still possible without reallocating
        # immediately on the next add).
        new_capacity = _INITIAL_CAPACITY
        while new_capacity < self._size * _GROWTH_FACTOR:
            new_capacity *= _GROWTH_FACTOR
        if new_capacity >= self._capacity:
            return  # nothing useful to free
        new_buffer = np.empty(
            (new_capacity, self._dim), dtype=np.float32,
        )
        new_buffer[:self._size] = self._buffer[:self._size]
        self._buffer = new_buffer
        self._capacity = new_capacity

    @staticmethod
    def _normalise(vec: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vec))
        if norm == 0.0:
            return vec
        return vec / norm

    # ── Public API ─────────────────────────────────────────────────────────

    def add(self, fact_id: str, vector: list[float]) -> None:
        """Insert or replace a fact's vector. No-op for empty vectors.

        Amortised O(d): the common case writes into a preallocated
        buffer slot. The rare grow-event copies n*d floats and resets
        the buffer to 2× capacity.
        """
        if not vector:
            return
        v = self._normalise(np.asarray(vector, dtype=np.float32))
        if self._dim is None:
            self._allocate(v.shape[0], _INITIAL_CAPACITY)
            # _allocate set _dim + buffer + capacity; size still 0.
            assert self._buffer is not None
            self._buffer[0] = v
            self._size = 1
            self._ids = [fact_id]
            self._id_to_row = {fact_id: 0}
            return
        if v.shape[0] != self._dim:
            raise DimensionMismatchError(
                f"Embedding dimension mismatch: index has dim={self._dim}, "
                f"incoming vector has dim={v.shape[0]} for fact_id={fact_id!r}. "
                "The embedding model probably changed under the store. "
                "Either pin BIRCH_EMBED_MODEL or rebuild the store."
            )
        assert self._buffer is not None
        # Replace-in-place when the id already exists. No size /
        # capacity change.
        if fact_id in self._id_to_row:
            self._buffer[self._id_to_row[fact_id]] = v
            return
        # Append: grow the buffer if needed, then write into the
        # next free row.
        self._grow_if_needed()
        row = self._size
        self._buffer[row] = v
        self._ids.append(fact_id)
        self._id_to_row[fact_id] = row
        self._size += 1

    def remove(self, fact_id: str) -> None:
        """Drop a fact. O(d) via swap-with-last; resets dim when empty.

        Swap-with-last is safe because the public surface never
        promised insertion-order: ``search`` returns by score and
        ``all_similarities`` returns a dict (no order). The internal
        ``_ids`` list is the source of truth for "which fact_id sits
        at this row"; rewriting one entry on swap keeps the
        id↔row map intact.
        """
        row = self._id_to_row.pop(fact_id, None)
        if row is None or self._buffer is None:
            return
        last = self._size - 1
        # If we're removing the last row, no swap needed.
        if row != last:
            self._buffer[row] = self._buffer[last]
            moved_id = self._ids[last]
            self._ids[row] = moved_id
            self._id_to_row[moved_id] = row
        # Truncate logical view. The bytes at _buffer[last] are now
        # unreachable through _size; do not need to zero them (they
        # were already unit-norm garbage from a prior live row).
        self._ids.pop()
        self._size -= 1
        # If the index just became empty, reset _dim + buffer so a
        # future add is free to set a new dimension. Without this
        # reset, an emptied index would still report dim=old_dim and
        # reject a different-dim re-add as a mismatch despite the
        # index being empty. The intent of the dim guard is "don't
        # mix dims in a populated index", not "freeze the first dim
        # ever seen for the lifetime of the object".
        if self._size == 0:
            self._buffer = None
            self._capacity = 0
            self._dim = None
            self._id_to_row = {}
            return
        # Reclaim memory if the buffer is now wastefully oversized.
        self._maybe_shrink()

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        threshold: float = -1.0,
    ) -> list[tuple[str, float]]:
        """Return (fact_id, similarity) sorted by similarity desc.

        ``top_k <= 0`` returns an empty list — guards against callers
        passing 0 or a negative value (e.g. a misclamped MCP input);
        numpy's argpartition is undefined / surprising on these edge
        cases.
        """
        if top_k <= 0:
            return []
        if self._buffer is None or self._size == 0 or not query_vector:
            return []
        q = self._normalise(np.asarray(query_vector, dtype=np.float32))
        if q.shape[0] != self._dim:
            return []
        sims = self._live_matrix() @ q
        if top_k >= len(sims):
            order = np.argsort(-sims)
        else:
            # argpartition's `kth` is a 0-indexed pivot position — to
            # extract the top_k elements correctly the pivot must be
            # at index top_k-1 (NOT top_k). With kth=top_k the slice
            # [:top_k] would include indices [0..top_k-1], i.e. the
            # top_k *smallest* in the negated array — correct positions
            # but the boundary element may be swapped between slots
            # top_k-1 and top_k. The downstream argsort over `sims[part]`
            # re-sorts the slice so the final order is right either
            # way; this fix is for idiom hygiene and to avoid the
            # off-by-one trap if a future refactor drops the resort.
            part = np.argpartition(-sims, top_k - 1)[:top_k]
            order = part[np.argsort(-sims[part])]
        out: list[tuple[str, float]] = []
        for idx in order:
            score = float(sims[idx])
            if score < threshold:
                continue
            out.append((self._ids[int(idx)], score))
        return out

    @staticmethod
    def similarity(a: list[float], b: list[float]) -> float:
        """Cosine similarity of two raw vectors; safe on empty inputs."""
        if not a or not b:
            return 0.0
        va = np.asarray(a, dtype=np.float32)
        vb = np.asarray(b, dtype=np.float32)
        if va.shape != vb.shape:
            return 0.0
        na = float(np.linalg.norm(va))
        nb = float(np.linalg.norm(vb))
        if na == 0.0 or nb == 0.0:
            return 0.0
        return float((va @ vb) / (na * nb))

    def all_similarities(self, query_vector: list[float]) -> dict[str, float]:
        """Cosine similarity for every indexed fact_id; empty when index is."""
        if self._buffer is None or self._size == 0 or not query_vector:
            return {}
        q = self._normalise(np.asarray(query_vector, dtype=np.float32))
        if q.shape[0] != self._dim:
            return {}
        sims = self._live_matrix() @ q
        return {self._ids[i]: float(sims[i]) for i in range(self._size)}
