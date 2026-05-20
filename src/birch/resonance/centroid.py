"""Centroid utilities — compress N session vectors into one representative vector."""
from __future__ import annotations

import math


def centroid(vectors: list[list[float]]) -> list[float]:
    """Average of all vectors. O(N*dim) time, O(dim) space."""
    if not vectors:
        return []
    dim = len(vectors[0])
    return [sum(v[i] for v in vectors) / len(vectors) for i in range(dim)]


def dispersion(vectors: list[list[float]], center: list[float]) -> float:
    """
    Mean cosine distance from each vector to the centroid.

    0.0 = all messages identical (tight loop)
    1.0 = messages point in completely different directions
    """
    if len(vectors) < 2:
        return 0.0

    def cosine_distance(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        if na == 0 or nb == 0:
            return 1.0
        return 1.0 - dot / (na * nb)

    return sum(cosine_distance(v, center) for v in vectors) / len(vectors)
