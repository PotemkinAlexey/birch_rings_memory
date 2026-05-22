"""Replay — run the galaxy through the store's real history.

build_galaxy() drops every fact in at once: a static snapshot. Replay
instead schedules each fact's *birth* at the sim-step matching its
created_at, and each closed session's resonance as orbital *kicks* at
the step matching when it happened.

The galaxy then grows and breathes along the store's real timeline: a
birth lifts a fact to the surface, drag sinks the untouched, and
resonance kicks fight the decay — a positive session boosts a fact
outward and accretes mass onto it, a toxic one drags it down.
"""
from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from ..fact import FactPassport
from .collapse import collapse_step
from .engine import Galaxy
from .loader import project_to_angles

# Session R (in [-1, 1]) x per-fact weight -> orbital impulse.
_KICK_SCALE = 9.0
# A positive resonance hit also accretes a little mass onto the fact.
_ACCRETION = 0.5


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
class History:
    """A schedule of births and kicks across a fixed number of sim steps."""

    steps: int
    births: list[_Birth] = field(default_factory=list)
    kicks: list[_Kick] = field(default_factory=list)


def build_history(
    facts: list[FactPassport],
    sessions: list[dict],
    *,
    steps: int = 1400,
    now: float | None = None,
) -> History:
    """Schedule fact births and session kicks across ``steps`` sim steps.

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

    angles = project_to_angles([f.vector for f in facts])
    for fact, angle in zip(facts, angles):
        if not fact.vector:
            angle = (hash(fact.fact_id) % 360) * math.pi / 180.0
        label = f"{fact.subject} {fact.predicate} {fact.object}"
        history.births.append(_Birth(
            step=step_of(fact.created_at),
            fact_id=fact.fact_id,
            angle=float(angle),
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
    return history


def replay(
    galaxy: Galaxy,
    history: History,
    on_step: Callable[[int, Galaxy], None] | None = None,
    collapse_every: int = 80,
) -> list[str]:
    """Run ``history`` against ``galaxy``. Returns every fact_id absorbed.

    Every ``collapse_every`` steps, cold bound clumps are checked for Jeans
    collapse into MetaFacts. ``on_step(step, galaxy)`` is invoked after each
    step — used by the renderer to capture animation frames.
    """
    births: dict[int, list[_Birth]] = {}
    for b in history.births:
        births.setdefault(b.step, []).append(b)
    kicks: dict[int, list[_Kick]] = {}
    for k in history.kicks:
        kicks.setdefault(k.step, []).append(k)

    birth_radius = galaxy.r_surface * 1.05
    absorbed: list[str] = []
    for step in range(history.steps):
        for b in births.get(step, []):
            galaxy.place_in_orbit(b.fact_id, birth_radius, b.angle, 1.0, b.label)
        for k in kicks.get(step, []):
            if galaxy.kick(k.fact_id, k.strength) and k.strength > 0:
                body = galaxy.find(k.fact_id)
                if body is not None:
                    body.mass += _ACCRETION
        absorbed.extend(galaxy.step())
        if collapse_every > 0 and step > 0 and step % collapse_every == 0:
            collapse_step(galaxy)
        if on_step is not None:
            on_step(step, galaxy)
    return absorbed
