"""Gravity engine — computes and updates gravity_score for FactPassports."""
from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .fact import FactPassport


# Layer thresholds — facts migrate when gravity crosses these boundaries
_LAYER_UP = 0.70    # gravity > 0.70 → promote to faster layer
_LAYER_DOWN = 0.30  # gravity < 0.30 → demote to slower layer

# Weights for the three components
_W_ACCESS = 0.35
_W_RESONANCE = 0.45
_W_GRAPH = 0.20


def compute_gravity(
    fact: "FactPassport",
    graph_degree: int = 0,
    max_degree: int = 1,
    now: float | None = None,
) -> float:
    """
    Compute gravity_score for a fact.

    Components:
      access_score   — recency-weighted access frequency
      resonance_score — avg R of sessions that used this fact, normalized to [0,1]
      graph_score    — relative connectivity in the knowledge graph

    Returns float in [0.0, 1.0].
    """
    now = now or time.time()

    # Access score: log-scaled count, decayed by time since last access
    age_hours = max(1.0, (now - fact.last_accessed) / 3600)
    access_raw = math.log1p(fact.access_count) / math.log1p(100)  # saturates at 100 hits
    decay = math.exp(-0.05 * age_hours)                            # half-life ~14h
    access_score = min(1.0, access_raw * decay)

    # Resonance score: avg_resonance lives in [-1, +1], map to [0, 1]
    resonance_score = (fact.avg_resonance + 1.0) / 2.0

    # Graph score: degree relative to max in the store
    graph_score = min(1.0, graph_degree / max(1, max_degree))

    gravity = (
        _W_ACCESS * access_score
        + _W_RESONANCE * resonance_score
        + _W_GRAPH * graph_score
    )
    return round(min(1.0, max(0.0, gravity)), 4)


def update_gravity(
    fact: "FactPassport",
    graph_degree: int = 0,
    max_degree: int = 1,
    now: float | None = None,
) -> int | None:
    """
    Recompute gravity_score in place. Returns new layer if migration triggered.

    Layer migration:
      gravity > _LAYER_UP   → promote (layer - 1, min 0)
      gravity < _LAYER_DOWN → demote  (layer + 1, max 2)
    """
    fact.gravity_score = compute_gravity(fact, graph_degree, max_degree, now)

    new_layer = fact.layer
    if fact.gravity_score > _LAYER_UP and fact.layer > 0:
        new_layer = fact.layer - 1
    elif fact.gravity_score < _LAYER_DOWN and fact.layer < 2:
        new_layer = fact.layer + 1

    if new_layer != fact.layer:
        fact.layer = new_layer
        return new_layer
    return None


class GravityEngine:
    """Manages gravity computation across a collection of facts."""

    def __init__(self) -> None:
        self._facts: dict[str, "FactPassport"] = {}
        self._degrees: dict[str, int] = {}     # fact_id → graph degree

    def register(self, fact: "FactPassport") -> None:
        self._facts[fact.fact_id] = fact
        self._degrees.setdefault(fact.fact_id, 0)

    def link(self, from_id: str, to_id: str) -> None:
        """Record a dependency edge — increases graph degree of to_id."""
        self._degrees[to_id] = self._degrees.get(to_id, 0) + 1

    def apply_session_resonance(self, facts, r: float) -> None:
        """Propagate a session's R to the facts it touched.

        ``facts`` may be either:
          - a list[str] of fact_ids (legacy, uniform weight 1.0)
          - a dict[str, float] of fact_id → relevance weight ∈ [0, 1]

        Per-fact weighting is the right primitive: an irrelevant
        low-similarity fact that happened to be returned by query
        gets a tiny resonance bump, while a high-similarity fact
        the agent actually leaned on gets the full session R.
        """
        if isinstance(facts, dict):
            for fid, weight in facts.items():
                if fid in self._facts:
                    self._facts[fid].apply_resonance(r * float(weight))
        else:
            for fid in facts:
                if fid in self._facts:
                    self._facts[fid].apply_resonance(r)

    def tick(self, now: float | None = None) -> list[tuple[str, int]]:
        """
        Recompute gravity for all facts. Returns list of (fact_id, new_layer)
        for facts that migrated.
        """
        max_deg = max(self._degrees.values(), default=1)
        migrations = []
        for fid, fact in self._facts.items():
            if fact.is_deprecated or fact.is_expired:
                continue
            new_layer = update_gravity(
                fact,
                graph_degree=self._degrees.get(fid, 0),
                max_degree=max_deg,
                now=now,
            )
            if new_layer is not None:
                migrations.append((fid, new_layer))
        return migrations
