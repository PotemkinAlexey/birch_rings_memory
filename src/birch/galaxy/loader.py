"""Loader — turn BirchKM facts into the bodies of a Galaxy.

A fact's embedding, projected through a shared PCA basis, becomes a unit
*direction* (2-D or 3-D) — its placement in the topical sky. Freshness and
earned value set the orbit radius and mass. The dynamics in engine.py take
over from there. Pure: reads facts, writes nothing.
"""
from __future__ import annotations

import math
import time

import numpy as np

from ..fact import FactPassport
from .engine import Galaxy
from .projection import Projector

_FRESHNESS_HALFLIFE_HOURS = 336.0
_LN2 = math.log(2)


def fallback_direction(fact_id: str, dim: int) -> np.ndarray:
    """A deterministic unit direction for a fact with no usable embedding."""
    rng = np.random.default_rng(abs(hash(fact_id)) & 0xFFFFFFFF)
    vec = rng.normal(size=dim)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-9:
        vec = np.zeros(dim)
        vec[0] = 1.0
        return vec
    return vec / norm


def fact_direction(
    fact: FactPassport, projector: Projector | None, dim: int
) -> np.ndarray:
    """Unit placement direction for a fact — the PCA of its embedding, or a
    deterministic fallback when it has no usable vector."""
    if projector is not None and fact.vector:
        return projector.direction(fact.vector)
    return fallback_direction(fact.fact_id, dim)


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
    """Build a Galaxy from BirchKM facts — a static snapshot, all placed at once."""
    now = now if now is not None else time.time()
    gal = galaxy if galaxy is not None else Galaxy()

    projector = Projector.fit([f.vector for f in facts], dim=gal.dim)
    for fact in facts:
        value = math.log1p(fact.access_count) + max(0.0, fact.resonance_sum)
        mass = 1.0 + 1.5 * value
        vitality = _vitality(fact, value, now)
        radius = gal.horizon + (gal.r_surface * 1.15 - gal.horizon) * vitality
        direction = fact_direction(fact, projector, gal.dim)
        label = f"{fact.subject} {fact.predicate} {fact.object}"
        gal.place_in_orbit(fact.fact_id, radius, direction, mass, label[:60])
    return gal
