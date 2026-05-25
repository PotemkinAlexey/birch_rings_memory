"""Vector index — numpy-backed cosine search over fact embeddings.

Pure-Python cosine over every fact on every query is O(n·d) in a
slow interpreter loop. At a few thousand facts × 768 dimensions that
turns into hundreds of milliseconds per query.

VectorIndex keeps an L2-normalised (n, d) matrix in sync with insert
and delete calls, so a query reduces to a single matrix–vector dot
product. For an unknown vector dimension we delay matrix allocation
until the first add().
"""
from __future__ import annotations

from typing import Optional

import numpy as np


class DimensionMismatchError(ValueError):
    """Raised when an incoming vector's dimension does not match the index.

    Silently dropping mismatched dimensions is dangerous: the fact still
    lives in ``MemoryStore._facts`` and on disk, but is unsearchable,
    which usually means the embedding model name (``BIRCH_EMBED_MODEL``)
    changed without a reindex. Raising loudly forces the caller to either
    rebuild the index or pin the model.
    """


class VectorIndex:
    """L2-normalised cosine index keyed by fact_id."""

    def __init__(self) -> None:
        self._ids: list[str] = []
        self._id_to_row: dict[str, int] = {}
        self._matrix: Optional[np.ndarray] = None   # (n, d), unit-normalised
        self._dim: Optional[int] = None

    def __len__(self) -> int:
        return len(self._ids)

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

    @staticmethod
    def _normalise(vec: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vec))
        if norm == 0.0:
            return vec
        return vec / norm

    def add(self, fact_id: str, vector: list[float]) -> None:
        """Insert or replace a fact's vector. No-op for empty vectors."""
        if not vector:
            return
        v = self._normalise(np.asarray(vector, dtype=np.float32))
        if self._dim is None:
            self._dim = v.shape[0]
            self._matrix = v.reshape(1, -1).copy()
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
        # _matrix is allocated together with _dim above — both set or both None.
        assert self._matrix is not None
        if fact_id in self._id_to_row:
            self._matrix[self._id_to_row[fact_id]] = v
            return
        self._matrix = np.vstack([self._matrix, v.reshape(1, -1)])
        self._id_to_row[fact_id] = len(self._ids)
        self._ids.append(fact_id)

    def remove(self, fact_id: str) -> None:
        row = self._id_to_row.pop(fact_id, None)
        if row is None or self._matrix is None:
            return
        self._matrix = np.delete(self._matrix, row, axis=0)
        self._ids.pop(row)
        # Rebuild id→row for everything after the deleted row.
        for i in range(row, len(self._ids)):
            self._id_to_row[self._ids[i]] = i
        # If the index just became empty, reset _dim so a future add
        # is free to set a new dimension. Without this reset, after
        # all vectors are removed the matrix is shape (0, old_dim)
        # and _dim still equals old_dim — a subsequent add() with a
        # new model's dim would raise DimensionMismatchError despite
        # the index being empty. The intent of the dim guard is
        # "don't mix dims in a populated index", not "freeze the
        # first dim ever seen for the lifetime of the object".
        if not self._ids:
            self._matrix = None
            self._dim = None
            self._id_to_row = {}

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
        if self._matrix is None or not query_vector:
            return []
        q = self._normalise(np.asarray(query_vector, dtype=np.float32))
        if q.shape[0] != self._dim:
            return []
        sims = self._matrix @ q
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
        if self._matrix is None or not query_vector:
            return {}
        q = self._normalise(np.asarray(query_vector, dtype=np.float32))
        if q.shape[0] != self._dim:
            return {}
        sims = self._matrix @ q
        return {self._ids[i]: float(sims[i]) for i in range(len(self._ids))}
