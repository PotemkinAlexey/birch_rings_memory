"""Projection — one fixed PCA basis shared across the galaxy.

Fact embeddings and session centroids must land in the *same* low-D plane
(or volume), or the attention mass would not point where the facts are.
The basis is fit once on the facts and reused for everything else. The
output dimension is 2 for the flat galaxy, 3 for the volumetric one.
"""
from __future__ import annotations

import numpy as np


class Projector:
    """A PCA projection from embedding space down to ``out_dim`` dimensions."""

    def __init__(self, mean: np.ndarray, basis: np.ndarray) -> None:
        self._mean = mean        # (D,)
        self._basis = basis      # (out_dim, D) — the top principal axes

    @property
    def out_dim(self) -> int:
        return int(self._basis.shape[0])

    @classmethod
    def fit(cls, vectors: list[list[float]], dim: int = 2) -> Projector | None:
        """Learn a ``dim``-D projection from embeddings; None if too little data."""
        embed_dim = max((len(v) for v in vectors), default=0)
        usable = [v for v in vectors if len(v) == embed_dim]
        if embed_dim < dim or len(usable) <= dim:
            return None
        mat = np.array(usable, dtype=float)
        mean = mat.mean(axis=0)
        _, _, vt = np.linalg.svd(mat - mean, full_matrices=False)
        return cls(mean, vt[:dim])

    def coords(self, vector: list[float]) -> np.ndarray:
        """Project one embedding to an ``out_dim``-D point; zero on dim mismatch."""
        if len(vector) != len(self._mean):
            return np.zeros(self.out_dim)
        return (np.asarray(vector, dtype=float) - self._mean) @ self._basis.T

    def direction(self, vector: list[float]) -> np.ndarray:
        """Unit ``out_dim``-D direction of a projected embedding."""
        coords = self.coords(vector)
        norm = float(np.linalg.norm(coords))
        if norm < 1e-9:
            unit = np.zeros(self.out_dim)
            unit[0] = 1.0
            return unit
        return coords / norm
