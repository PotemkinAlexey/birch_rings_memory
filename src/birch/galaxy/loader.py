"""Loader — turn BirchKM facts into the bodies of a Galaxy.

The embedding sets a body's *angle* — its place in the topical sky. Freshness
and earned value set its *radius* — which ring it starts in. Mass is the value
a fact has accumulated. From there engine.py takes over and the orbits evolve.

Pure: reads facts, writes nothing.
"""
from __future__ import annotations

import math
import time

import numpy as np

from ..fact import FactPassport
from .engine import Galaxy

# Freshness half-life — matches the live gravity formula's grace period.
_FRESHNESS_HALFLIFE_HOURS = 336.0
_LN2 = math.log(2)


def project_to_angles(vectors: list[list[float]]) -> np.ndarray:
    """PCA every embedding to 2D and return each one's polar angle.

    Only the angle is kept: it carries the semantic direction. Magnitude is
    discarded — a body's radius comes from vitality, not from how far its
    embedding sits from the centroid. Missing vectors get angle 0.0.
    """
    dim = max((len(v) for v in vectors), default=0)
    if dim == 0:
        return np.zeros(len(vectors))
    mat = np.array([v if len(v) == dim else [0.0] * dim for v in vectors])
    centered = mat - mat.mean(axis=0)
    # The top-2 right singular vectors are the principal axes.
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ vt[:2].T
    return np.arctan2(coords[:, 1], coords[:, 0])


def _vitality(fact: FactPassport, value: float, now: float) -> float:
    """How buoyant a fact starts — fresh or valued facts ride a high orbit."""
    age_hours = max(0.0, (now - fact.created_at) / 3600)
    freshness = math.exp(-age_hours * _LN2 / _FRESHNESS_HALFLIFE_HOURS)
    return min(1.0, freshness + 0.15 * value)


def build_galaxy(
    facts: list[FactPassport],
    *,
    now: float | None = None,
    galaxy: Galaxy | None = None,
) -> Galaxy:
    """Build a Galaxy from BirchKM facts."""
    now = now if now is not None else time.time()
    gal = galaxy if galaxy is not None else Galaxy()

    angles = project_to_angles([f.vector for f in facts])
    for fact, angle in zip(facts, angles):
        value = math.log1p(fact.access_count) + max(0.0, fact.resonance_sum)
        mass = 1.0 + 1.5 * value
        vitality = _vitality(fact, value, now)
        radius = gal.horizon + (gal.r_surface * 1.15 - gal.horizon) * vitality
        if not fact.vector:
            # No embedding — scatter deterministically by id.
            angle = (hash(fact.fact_id) % 360) * math.pi / 180.0
        label = f"{fact.subject} {fact.predicate} {fact.object}"
        gal.place_in_orbit(fact.fact_id, radius, float(angle), mass, label[:60])
    return gal
