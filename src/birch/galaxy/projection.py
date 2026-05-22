"""Projection — one fixed PCA basis shared across the galaxy.

Fact embeddings and session centroids must land in the *same* 2D plane,
or the attention mass would not point where the facts actually are. The
basis is fit once on the facts and reused for everything else.
"""
from __future__ import annotations

import numpy as np


class Projector:
    """A 2D PCA projection learned from a set of embedding vectors."""

    def __init__(self, mean: np.ndarray, basis: np.ndarray) -> None:
        self._mean = mean        # (d,)
        self._basis = basis      # (2, d) — the top-2 principal axes

    @property
    def dim(self) -> int:
        return int(self._mean.shape[0])

    @classmethod
    def fit(cls, vectors: list[list[float]]) -> Projector | None:
        """Learn a projection from embeddings; None if there is too little."""
        dim = max((len(v) for v in vectors), default=0)
        usable = [v for v in vectors if len(v) == dim]
        if dim == 0 or len(usable) < 2:
            return None
        mat = np.array(usable, dtype=float)
        mean = mat.mean(axis=0)
        _, _, vt = np.linalg.svd(mat - mean, full_matrices=False)
        return cls(mean, vt[:2])

    def coords(self, vector: list[float]) -> np.ndarray:
        """Project one embedding to a 2D point; zero if the dims mismatch."""
        if len(vector) != self.dim:
            return np.zeros(2)
        return (np.array(vector, dtype=float) - self._mean) @ self._basis.T

    def angle(self, vector: list[float]) -> float:
        """Polar angle of a projected embedding, in (-pi, pi]."""
        xy = self.coords(vector)
        return float(np.arctan2(xy[1], xy[0]))
