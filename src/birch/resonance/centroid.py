"""Centroid utilities — compress N session vectors into one representative vector."""
from __future__ import annotations

import math


def _validate_same_dim(vectors: list[list[float]]) -> int:
    """Assert every vector in the list shares a single dim.

    Loaders already filter ragged sessions at the storage boundary,
    but direct callers (tests, in-memory migrations, embedded mode)
    can still pass mixed-dim lists. ``centroid`` and ``dispersion``
    both index every vector by ``dim = len(vectors[0])`` — a shorter
    vector would index out of bounds, a longer one would silently
    drop tail dimensions. Raise explicitly so the caller sees the
    contract violation instead of either a crash deep in the stack
    or a wrong-but-finite numeric answer.
    """
    if not vectors:
        return 0
    dim = len(vectors[0])
    if dim == 0:
        raise ValueError("empty vector in centroid input")
    for v in vectors:
        if len(v) != dim:
            raise ValueError(
                f"mixed vector dimensions: expected {dim}, "
                f"got {len(v)}"
            )
    return dim


def centroid(vectors: list[list[float]]) -> list[float]:
    """Average of all vectors. O(N*dim) time, O(dim) space.

    Raises ``ValueError`` if vectors have mixed dimensions — caller
    must validate shape (loaders do; in-memory callers should use
    ``_validate_same_dim`` directly).
    """
    if not vectors:
        return []
    dim = _validate_same_dim(vectors)
    return [sum(v[i] for v in vectors) / len(vectors) for i in range(dim)]


def dispersion(vectors: list[list[float]], center: list[float]) -> float:
    """
    Mean cosine distance from each vector to the centroid.

    0.0 = all messages identical (tight loop)
    1.0 = messages point in completely different directions

    Raises ``ValueError`` if input vectors have mixed dimensions.
    """
    if len(vectors) < 2:
        return 0.0
    _validate_same_dim(vectors)
    if center and len(center) != len(vectors[0]):
        raise ValueError(
            f"center dim {len(center)} does not match vectors "
            f"dim {len(vectors[0])}"
        )

    def cosine_distance(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        if na == 0 or nb == 0:
            return 1.0
        return 1.0 - dot / (na * nb)

    return sum(cosine_distance(v, center) for v in vectors) / len(vectors)
