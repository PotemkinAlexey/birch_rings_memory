"""Replay — run the galaxy through the store's real history.

build_galaxy() drops every fact in at once: a static snapshot. Replay
instead schedules each fact's *birth* at the sim-step matching its
created_at, each closed session's resonance as orbital *kicks*, and each
session's topic as a move of the *attention mass*.

The galaxy then grows and breathes along the store's real timeline: a
birth lifts a fact to the surface, drag sinks the untouched, resonance
kicks fight the decay, and the attention mass — parked where the latest
session's topic sits — tugs the facts you are working with toward it.
"""
from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from ..fact import FactPassport
from .collapse import collapse_step
from .engine import Galaxy
from .projection import Projector

# Session R (in [-1, 1]) x per-fact weight -> orbital impulse.
_KICK_SCALE = 9.0
# A positive resonance hit also accretes a little mass onto the fact.
_ACCRETION = 0.5
# How fast the attention mass glides toward a new topic (fraction per step).
_ATTENTION_GLIDE = 0.03


@dataclass
class _Birth:
    step: int
    fact_id: str
    angle: float
    label: str


@dataclass
class _Kick:
    step: int
    fact_id: str
    strength: float


@dataclass
class _Attention:
    step: int
    angle: float          # topic direction of the session


@dataclass
class History:
    """A schedule of births, kicks and attention moves across sim steps."""

    steps: int
    births: list[_Birth] = field(default_factory=list)
    kicks: list[_Kick] = field(default_factory=list)
    attention: list[_Attention] = field(default_factory=list)


def build_history(
    facts: list[FactPassport],
    sessions: list[dict],
    *,
    steps: int = 1400,
    now: float | None = None,
) -> History:
    """Schedule births, kicks and attention moves across ``steps`` sim steps.

    Pure: reads facts and session rows (the shape returned by
    ``SQLiteBackend.load_echo_sessions``), writes nothing.
    """
    now = now if now is not None else time.time()
    history = History(steps=steps)
    if not facts:
        return history

    t_start = min(f.created_at for f in facts)
    recorded = [s["recorded_at"] for s in sessions]
    t_end = max([now, *recorded]) if recorded else now
    span = max(t_end - t_start, 1.0)

    def step_of(t: float) -> int:
        frac = (t - t_start) / span
        return min(steps - 1, max(0, round(frac * (steps - 1))))

    projector = Projector.fit([f.vector for f in facts])

    for fact in facts:
        if projector is not None and fact.vector:
            angle = projector.angle(fact.vector)
        else:
            angle = (hash(fact.fact_id) % 360) * math.pi / 180.0
        label = f"{fact.subject} {fact.predicate} {fact.object}"
        history.births.append(_Birth(
            step=step_of(fact.created_at),
            fact_id=fact.fact_id,
            angle=angle,
            label=label[:60],
        ))

    for session in sessions:
        r = float(session.get("r_score", 0.0))
        for fact_id, weight in session.get("fact_weights", {}).items():
            history.kicks.append(_Kick(
                step=step_of(session["recorded_at"]),
                fact_id=fact_id,
                strength=_KICK_SCALE * r * float(weight),
            ))
        # The session's topic — the mean of its centroids — moves attention.
        centroids = session.get("centroids") or []
        if projector is not None and centroids:
            topic = np.array(centroids, dtype=float).mean(axis=0).tolist()
            history.attention.append(_Attention(
                step=step_of(session["recorded_at"]),
                angle=projector.angle(topic),
            ))
    return history


def replay(
    galaxy: Galaxy,
    history: History,
    on_step: Callable[[int, Galaxy], None] | None = None,
    collapse_every: int = 80,
    hawking_every: int = 240,
) -> list[str]:
    """Run ``history`` against ``galaxy``. Returns every fact_id absorbed.

    Every ``collapse_every`` steps, cold bound clumps are checked for Jeans
    collapse into MetaFacts; every ``hawking_every`` steps the black hole
    leaks one swallowed body back out. ``on_step(step, galaxy)`` is invoked
    after each step — used by the renderer to capture animation frames.
    """
    births: dict[int, list[_Birth]] = {}
    for b in history.births:
        births.setdefault(b.step, []).append(b)
    kicks: dict[int, list[_Kick]] = {}
    for k in history.kicks:
        kicks.setdefault(k.step, []).append(k)
    # The latest session in a step sets where attention should drift to.
    attention_target: dict[int, float] = {}
    for a in history.attention:
        attention_target[a.step] = a.angle

    birth_radius = galaxy.r_surface * 1.05
    attention_radius = 0.5 * (galaxy.r_core + galaxy.r_surface)
    current: float | None = None      # attention angle, glides toward target
    target: float | None = None
    absorbed: list[str] = []

    for step in range(history.steps):
        for b in births.get(step, []):
            galaxy.place_in_orbit(b.fact_id, birth_radius, b.angle, 1.0, b.label)
        for k in kicks.get(step, []):
            if galaxy.kick(k.fact_id, k.strength) and k.strength > 0:
                body = galaxy.find(k.fact_id)
                if body is not None:
                    body.mass += _ACCRETION
        if step in attention_target:
            target = attention_target[step]
            if current is None:       # the first focus simply appears in place
                current = target
        if current is not None and target is not None:
            # Glide the shortest angular way toward the latest topic — a
            # gliding mass perturbs the disk gently; a teleporting one shocks it.
            delta = math.atan2(math.sin(target - current),
                               math.cos(target - current))
            current += delta * _ATTENTION_GLIDE
            galaxy.attention_pos = attention_radius * np.array(
                [math.cos(current), math.sin(current)]
            )
        absorbed.extend(galaxy.step())
        if collapse_every > 0 and step > 0 and step % collapse_every == 0:
            collapse_step(galaxy)
        if hawking_every > 0 and step > 0 and step % hawking_every == 0:
            galaxy.hawking_emit()
        if on_step is not None:
            on_step(step, galaxy)
    return absorbed
