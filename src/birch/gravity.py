"""Gravity engine — computes and updates gravity_score for memory bodies."""
from __future__ import annotations

import math
import time
from typing import Protocol

from .adaptive_gravity import AdaptiveWeights


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
    recent_utility: float
    forecast_stability: float

    @property
    def fact_id(self) -> str: ...
    @property
    def avg_resonance(self) -> float: ...
    @property
    def is_deprecated(self) -> bool: ...
    @property
    def is_expired(self) -> bool: ...
    def apply_resonance(self, r: float) -> None: ...


def _finite_clamped(
    value, lo: float, hi: float,
) -> float | None:
    """Coerce to a finite float clamped to [lo, hi]; ``None`` on failure.

    Engine-boundary helper used by ``apply_session_resonance`` to
    reject NaN / Infinity / non-numeric inputs before they reach
    ``apply_resonance`` on a body. The body methods now self-defend
    too, but rejecting up front lets the engine distinguish
    "whole-session no-op" (bad ``r``) from "skip this pair" (bad
    weight) — better debuggability than silent body-level skips.
    """
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return max(lo, min(hi, out))


# Layer thresholds — facts migrate when gravity crosses these boundaries.
_LAYER_UP = 0.70    # gravity > 0.70 → promote
_LAYER_DOWN = 0.30  # gravity < 0.30 → demote

# Resonance weight is fixed: resonance is observed value, not predicted.
# The five pre-resonance weights (freshness, access, graph, recent_utility,
# forecast_stability) live in AdaptiveWeights and are learned from the
# user's actual session feedback.
_W_RESONANCE = 0.35

# Freshness half-life — a new fact rides high and sinks as it ages
# untouched. This is the grace period.
_FRESHNESS_HALFLIFE_HOURS = 336.0   # ~2 weeks
# Access half-life — how fast an un-revisited fact loses its access boost.
_ACCESS_HALFLIFE_HOURS = 72.0       # ~3 days

_LN2 = math.log(2)


def pre_resonance_features(
    fact: GravityBody,
    graph_degree: int = 0,
    max_degree: int = 1,
    now: float | None = None,
) -> tuple[float, float, float, float, float]:
    """Compute (freshness, access, graph, utility, stability) — the features
    whose weights are learned. Resonance is excluded on purpose: it is the
    *target* of the learning, not a predictor of it.

    ``utility`` is the fact's stored ``recent_utility`` EWMA — a slow-
    moving prior on how useful this fact has been in recent sessions,
    independent of how often it was touched.

    ``stability`` is the fact's stored ``forecast_stability`` — the
    galaxy's prediction of how far this body will be from the horizon
    after a short forward simulation. 1.0 = safely on surface, 0.0 =
    predicted to fall, 0.5 = no forecast available (neutral prior).
    """
    now = now or time.time()

    age_hours = max(0.0, (now - fact.created_at) / 3600)
    freshness = math.exp(-age_hours * _LN2 / _FRESHNESS_HALFLIFE_HOURS)

    idle_hours = max(0.0, (now - fact.last_accessed) / 3600)
    access_raw = math.log1p(fact.access_count) / math.log1p(100)
    access_decay = math.exp(-idle_hours * _LN2 / _ACCESS_HALFLIFE_HOURS)
    access_score = min(1.0, access_raw * access_decay)

    graph_score = min(1.0, graph_degree / max(1, max_degree))

    utility = max(0.0, min(1.0, float(getattr(fact, "recent_utility", 0.5))))
    stability = max(0.0, min(1.0,
                             float(getattr(fact, "forecast_stability", 0.5))))

    return freshness, access_score, graph_score, utility, stability


def compute_gravity(
    fact: GravityBody,
    weights: AdaptiveWeights | None = None,
    graph_degree: int = 0,
    max_degree: int = 1,
    now: float | None = None,
) -> float:
    """Compute gravity_score for a memory body in [0.0, 1.0].

    The five pre-resonance weights are read from ``weights`` (learned),
    the resonance weight stays fixed. When ``weights`` is None we fall back
    to the hand-tuned prior — identical to the previous static formula.
    """
    if weights is None:
        weights = AdaptiveWeights.from_prior()

    (freshness, access_score, graph_score,
     utility, stability) = pre_resonance_features(
        fact, graph_degree, max_degree, now)

    if fact.resonance_count > 0:
        resonance_score = (fact.avg_resonance + 1.0) / 2.0
    else:
        resonance_score = 0.0

    gravity = (
        weights.w_freshness * freshness
        + weights.w_access * access_score
        + weights.w_graph * graph_score
        + weights.w_utility * utility
        + weights.w_stability * stability
        + _W_RESONANCE * resonance_score
    )
    return round(min(1.0, max(0.0, gravity)), 4)


def _target_layer(gravity: float) -> int:
    """The layer a gravity score belongs in: 0 surface, 1 kinetic, 2 core."""
    if gravity > _LAYER_UP:
        return 0
    if gravity < _LAYER_DOWN:
        return 2
    return 1


def update_gravity(
    fact: GravityBody,
    weights: AdaptiveWeights | None = None,
    graph_degree: int = 0,
    max_degree: int = 1,
    now: float | None = None,
) -> int | None:
    """Recompute gravity_score in place. Returns the new layer if it migrated.

    Migration steps one layer per tick toward the layer the gravity score
    belongs in — so a fact climbs out of core once its gravity recovers into
    the kinetic band, and a cooled surface fact settles back to kinetic.
    """
    fact.gravity_score = compute_gravity(
        fact, weights, graph_degree, max_degree, now)

    target = _target_layer(fact.gravity_score)
    if target < fact.layer:
        new_layer = fact.layer - 1
    elif target > fact.layer:
        new_layer = fact.layer + 1
    else:
        new_layer = fact.layer

    if new_layer != fact.layer:
        fact.layer = new_layer
        return new_layer
    return None


class GravityEngine:
    """Manages gravity computation across a collection of facts.

    Holds the live ``AdaptiveWeights`` so a tick uses the learned weights;
    callers that don't care get the hand-tuned prior automatically.
    """

    def __init__(self, weights: AdaptiveWeights | None = None) -> None:
        self.weights = weights if weights is not None else AdaptiveWeights.from_prior()
        self._facts: dict[str, GravityBody] = {}
        self._degrees: dict[str, int] = {}     # fact_id → graph degree
        # Track which (from, to) pairs already contributed to _degrees
        # so a repeated link() call doesn't double-count. Storage
        # de-dups via PRIMARY KEY (from_id, to_id) + INSERT OR IGNORE,
        # but the in-memory counter was advancing on every call —
        # which could inflate gravity (graph_score) until the next
        # _reload rebuilt _degrees from disk's unique rows.
        self._edges: set[tuple[str, str]] = set()

    def register(self, fact: GravityBody) -> None:
        self._facts[fact.fact_id] = fact
        self._degrees.setdefault(fact.fact_id, 0)

    def unregister(self, fact_id: str) -> None:
        """Remove a fact from the engine — called on explicit deletion."""
        # Identify edges that need cleanup BEFORE dropping the body.
        stale = [
            edge for edge in self._edges
            if edge[0] == fact_id or edge[1] == fact_id
        ]
        # For each outgoing edge A→B where A is being unregistered,
        # decrement degree of the surviving target B. Previously the
        # _edges set was cleared but B's degree (incremented by the
        # original link call at line 220) stayed inflated until next
        # _reload rebuilt _degrees from disk's unique edge rows.
        # Same accounting that storage's delete_edges_for_fact does
        # on disk; mirror in-memory so graph_score stays honest
        # without depending on reload.
        for from_id, to_id in stale:
            if to_id == fact_id:
                continue  # B == removed body; _degrees[B] dropped below
            if to_id in self._degrees:
                self._degrees[to_id] = max(0, self._degrees[to_id] - 1)
        self._edges.difference_update(stale)
        self._facts.pop(fact_id, None)
        self._degrees.pop(fact_id, None)

    def link(self, from_id: str, to_id: str) -> None:
        """Record a dependency edge — increases graph degree of ``to_id``
        the first time this exact ``(from_id, to_id)`` pair is seen.

        Repeated calls with the same pair are no-ops in the in-memory
        counter (mirrors the on-disk PRIMARY KEY + INSERT OR IGNORE
        semantics in SQLiteBackend.save_edge).
        """
        pair = (from_id, to_id)
        if pair in self._edges:
            return
        self._edges.add(pair)
        self._degrees[to_id] = self._degrees.get(to_id, 0) + 1

    def apply_session_resonance(self, facts, r: float) -> None:
        """Propagate a session's R to the facts it touched.

        ``facts`` may be either:
          - a list[str] of fact_ids (legacy, uniform weight 1.0)
          - a dict[str, float] of fact_id → relevance weight ∈ [0, 1]

        Engine-boundary sanitisation: a NaN / Infinity in either ``r``
        or any per-fact weight would call into ``apply_resonance``
        which now self-defends, but propagating the bad value here
        means *every* fact in the session gets silently skipped.
        Cleaner to reject the bad call up front (whole-session no-op
        on bad ``r``) and per-pair no-op on bad weight, so the agent
        loop continues with a structured outcome.
        """
        r_clean = _finite_clamped(r, -1.0, 1.0)
        if r_clean is None:
            # Bad session resonance — nothing to propagate. Sessions
            # with malformed R get skipped entirely rather than
            # poisoning every touched fact's resonance_sum.
            return
        if isinstance(facts, dict):
            for fid, weight in facts.items():
                if fid not in self._facts:
                    continue
                w_clean = _finite_clamped(weight, 0.0, 1.0)
                if w_clean is None:
                    # Bad weight on one fact does not abort the whole
                    # propagation. Skip the pair, keep going.
                    continue
                self._facts[fid].apply_resonance(r_clean * w_clean)
        else:
            for fid in facts:
                if fid in self._facts:
                    self._facts[fid].apply_resonance(r_clean)

    def tick(self, now: float | None = None) -> list[tuple[str, int]]:
        """Recompute gravity for all facts using the engine's adaptive weights.

        Returns the list of (fact_id, new_layer) for facts that migrated.
        """
        max_deg = max(self._degrees.values(), default=1)
        migrations = []
        for fid, fact in self._facts.items():
            if fact.is_deprecated or fact.is_expired:
                continue
            new_layer = update_gravity(
                fact,
                self.weights,
                graph_degree=self._degrees.get(fid, 0),
                max_degree=max_deg,
                now=now,
            )
            if new_layer is not None:
                migrations.append((fid, new_layer))
        return migrations
