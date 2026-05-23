"""Loader — turn BirchKM facts into the bodies of a Galaxy.

A fact's embedding, projected through a shared PCA basis, becomes a unit
*direction* (2-D or 3-D) — its placement in the topical sky. Freshness and
earned value set the orbit radius and mass. The dynamics in engine.py take
over from there. Pure: reads facts, writes nothing.
"""
from __future__ import annotations

import hashlib
import math
import time

import numpy as np

from ..fact import FactPassport
from .engine import Galaxy
from .projection import Projector

_FRESHNESS_HALFLIFE_HOURS = 336.0
_LN2 = math.log(2)


def fallback_direction(fact_id: str, dim: int) -> np.ndarray:
    """A deterministic unit direction for a fact with no usable embedding.

    Uses sha256 instead of Python's built-in hash() because hash() is
    salted per-process for strings (PYTHONHASHSEED defaults to random),
    which would give the same fact a different fallback direction on
    every restart. The forecast for vectorless bodies would change
    across reboots with no actual data change. sha256 is stable across
    processes and across Python versions.
    """
    digest = hashlib.sha256(fact_id.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "big")
    rng = np.random.default_rng(seed)
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


def _vitality(body, value: float, now: float) -> float:
    """How buoyant a body starts — fresh or valued bodies ride a high orbit.

    Polymorphic over FactPassport and MetaFact: both carry ``created_at``.
    """
    age_hours = max(0.0, (now - body.created_at) / 3600)
    freshness = math.exp(-age_hours * _LN2 / _FRESHNESS_HALFLIFE_HOURS)
    return min(1.0, freshness + 0.15 * value)


def _body_label(body) -> str:
    """Short human label for rendering — works for facts and MetaFacts."""
    if hasattr(body, "subject"):
        return f"{body.subject} {body.predicate} {body.object}"
    sources = getattr(body, "source_texts", None) or []
    if sources:
        return f"meta[{getattr(body, 'weight', '?')}] {sources[0]}"
    return f"meta:{getattr(body, 'fact_id', '?')}"


def build_galaxy(
    facts: list,
    *,
    now: float | None = None,
    galaxy: Galaxy | None = None,
) -> Galaxy:
    """Build a Galaxy from BirchKM bodies — a static snapshot, all placed at once.

    Accepts a polymorphic list of FactPassport and MetaFact bodies. Both
    expose the surface the placement needs (``fact_id``, ``vector``,
    ``access_count``, ``resonance_sum``, ``created_at``).

    Raises ``DimensionMismatchError`` if the non-empty input vectors do
    not all share one dimensionality — that usually means BIRCH_EMBED_MODEL
    changed (or singularity holds rows from an older model) and the
    forecast would silently produce wrong geometry. Bodies with empty
    vectors get a deterministic fallback direction and are not counted
    toward the dimension check.
    """
    now = now if now is not None else time.time()
    gal = galaxy if galaxy is not None else Galaxy()

    non_empty_dims = {len(b.vector) for b in facts if b.vector}
    if len(non_empty_dims) > 1:
        from ..vector_index import DimensionMismatchError
        raise DimensionMismatchError(
            "build_galaxy received bodies with mixed embedding dimensions "
            f"{sorted(non_empty_dims)}. The forecast would silently use "
            "the max-dim subset and project the rest to zero coords — fix "
            "the store (pin BIRCH_EMBED_MODEL or reindex) before retrying."
        )

    projector = Projector.fit([f.vector for f in facts], dim=gal.dim)
    for body in facts:
        value = math.log1p(body.access_count) + max(0.0, body.resonance_sum)
        mass = 1.0 + 1.5 * value
        vitality = _vitality(body, value, now)
        radius = gal.horizon + (gal.r_surface * 1.15 - gal.horizon) * vitality
        direction = fact_direction(body, projector, gal.dim)
        label = _body_label(body)
        gal.place_in_orbit(body.fact_id, radius, direction, mass, label[:60])
    return gal
