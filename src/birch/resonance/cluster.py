"""K-means clustering for session vectors — produces a bundle of centroids."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .centroid import centroid, dispersion


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _nearest(vec: list[float], centers: list[list[float]]) -> int:
    return max(range(len(centers)), key=lambda i: _cosine(vec, centers[i]))


@dataclass
class ClusterBundle:
    centroids: list[list[float]]    # K centroids
    k: int
    inertia: float                  # avg dispersion across all clusters


def bundle(
    vectors: list[list[float]],
    k: int = 2,
    max_iter: int = 20,
    seed: int = 42,
) -> ClusterBundle:
    """
    Fit K-means (cosine) on session vectors, return bundle of K centroids.

    Falls back to single centroid when len(vectors) <= k.
    k is auto-reduced to len(vectors) if necessary.
    """
    n = len(vectors)
    if n == 0:
        return ClusterBundle([], 0, 0.0)

    k = min(k, n)
    if k == 1:
        c = centroid(vectors)
        return ClusterBundle([c], 1, dispersion(vectors, c))

    # K-means++ initialisation
    rng = random.Random(seed)
    centers: list[list[float]] = [vectors[rng.randrange(n)]]
    while len(centers) < k:
        # Pick next center with probability proportional to (1 - max_similarity)
        distances = [
            1.0 - max(_cosine(v, c) for c in centers)
            for v in vectors
        ]
        total = sum(distances)
        if total == 0:
            break
        r = rng.random() * total
        cumulative = 0.0
        for i, d in enumerate(distances):
            cumulative += d
            if cumulative >= r:
                centers.append(vectors[i])
                break

    # Iterate
    for _ in range(max_iter):
        clusters: list[list[list[float]]] = [[] for _ in range(k)]
        for v in vectors:
            clusters[_nearest(v, centers)].append(v)

        new_centers = [
            centroid(cl) if cl else centers[i]
            for i, cl in enumerate(clusters)
        ]

        # Convergence check — all centers moved < 1e-6
        if all(
            1.0 - _cosine(a, b) < 1e-6
            for a, b in zip(centers, new_centers)
        ):
            break
        centers = new_centers

    # Compute inertia
    clusters = [[] for _ in range(k)]
    for v in vectors:
        clusters[_nearest(v, centers)].append(v)
    inertia = sum(
        dispersion(cl, centers[i])
        for i, cl in enumerate(clusters) if cl
    ) / k

    return ClusterBundle(centroids=centers, k=k, inertia=round(inertia, 6))


def nearest_similarity(query: list[float], bundle_obj: ClusterBundle) -> float:
    """Max cosine similarity between query and any centroid in the bundle."""
    if not bundle_obj.centroids:
        return 0.0
    return max(_cosine(query, c) for c in bundle_obj.centroids)
