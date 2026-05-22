"""Gravity engine — computes and updates gravity_score for memory bodies."""
from __future__ import annotations

import math
import time
from typing import Protocol


class GravityBody(Protocol):
    """The gravity-relevant surface shared by FactPassport and MetaFact.

    Both types expose this duck-typed interface (see MetaFact's module
    docstring), so GravityEngine treats live facts and emitted MetaFacts
    uniformly without caring which concrete class it holds.
    """

    gravity_score: float
    layer: int
    access_count: int
    last_accessed: float
    created_at: float
    resonance_count: int

    @property
    def fact_id(self) -> str: ...
    @property
    def avg_resonance(self) -> float: ...
    @property
    def is_deprecated(self) -> bool: ...
    @property
    def is_expired(self) -> bool: ...
    def apply_resonance(self, r: float) -> None: ...


# Layer thresholds — facts migrate when gravity crosses these boundaries
_LAYER_UP = 0.70    # gravity > 0.70 → promote to faster layer
_LAYER_DOWN = 0.30  # gravity < 0.30 → demote to slower layer

# Component weights — sum to 1.0
_W_FRESHNESS = 0.35   # how recently the fact was created
_W_ACCESS = 0.20      # recency-weighted access frequency
_W_RESONANCE = 0.35   # avg resonance of sessions that used it
_W_GRAPH = 0.10       # connectivity in the knowledge graph

# Freshness half-life — a new fact is presumed relevant and rides high,
# then sinks as it ages untouched. This is the grace period: a fresh fact
# is not archived before it has had a chance to prove itself.
_FRESHNESS_HALFLIFE_HOURS = 336.0   # ~2 weeks
# Access half-life — how fast an un-revisited fact loses its access boost.
_ACCESS_HALFLIFE_HOURS = 72.0       # ~3 days

_LN2 = math.log(2)


def compute_gravity(
    fact: GravityBody,
    graph_degree: int = 0,
    max_degree: int = 1,
    now: float | None = None,
) -> float:
    """
    Compute gravity_score for a memory body. Returns a float in [0.0, 1.0].

    Four components:
      freshness  — decays from 1.0 by age since creation. A new fact starts
                   buoyant; this is the grace period that keeps a just-created
                   fact out of the cold core before it can prove itself.
      access     — log-scaled access count, decayed by time since last touch.
      resonance  — avg R of sessions that used it; 0 until a session scores it,
                   so an un-resonated fact is not propped up by a neutral 0.5.
      graph      — connectivity relative to the most-connected fact.

    A fact rides high while fresh, climbs to the surface when used and
    resonant, and sinks through the core toward the black hole as it ages
    untouched.
    """
    now = now or time.time()

    # Freshness: exponential decay from creation time.
    age_hours = max(0.0, (now - fact.created_at) / 3600)
    freshness = math.exp(-age_hours * _LN2 / _FRESHNESS_HALFLIFE_HOURS)

    # Access: log-scaled count, decayed by time since last access.
    idle_hours = max(0.0, (now - fact.last_accessed) / 3600)
    access_raw = math.log1p(fact.access_count) / math.log1p(100)  # saturates at 100
    access_decay = math.exp(-idle_hours * _LN2 / _ACCESS_HALFLIFE_HOURS)
    access_score = min(1.0, access_raw * access_decay)

    # Resonance: avg_resonance lives in [-1, +1] → [0, 1]. Only counts once a
    # session has actually scored the fact.
    if fact.resonance_count > 0:
        resonance_score = (fact.avg_resonance + 1.0) / 2.0
    else:
        resonance_score = 0.0

    # Graph: degree relative to the most-connected fact in the store.
    graph_score = min(1.0, graph_degree / max(1, max_degree))

    gravity = (
        _W_FRESHNESS * freshness
        + _W_ACCESS * access_score
        + _W_RESONANCE * resonance_score
        + _W_GRAPH * graph_score
    )
    return round(min(1.0, max(0.0, gravity)), 4)


def update_gravity(
    fact: GravityBody,
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
        self._facts: dict[str, GravityBody] = {}
        self._degrees: dict[str, int] = {}     # fact_id → graph degree

    def register(self, fact: GravityBody) -> None:
        self._facts[fact.fact_id] = fact
        self._degrees.setdefault(fact.fact_id, 0)

    def unregister(self, fact_id: str) -> None:
        """Remove a fact from the engine — called on explicit deletion."""
        self._facts.pop(fact_id, None)
        self._degrees.pop(fact_id, None)

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
