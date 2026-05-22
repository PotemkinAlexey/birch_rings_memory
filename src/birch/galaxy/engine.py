"""N-body engine — facts as satellites orbiting the memory's black hole.

Pure 2D physics, numpy only. No notion of facts, embeddings or storage —
a body is just an id, a position, a velocity and a mass. The loader turns
BirchKM facts into bodies; this module only integrates the orbits.

The picture:

  - A central black hole of fixed mass sits at the origin and dominates.
  - Each body orbits it. Orbital radius is the body's *ring*:
    surface (far, safe) -> kinetic -> core (near) -> absorbed (horizon).
  - Dynamical friction (Chandrasekhar drag) bleeds orbital energy, so an
    untouched body slowly spirals inward — the natural decay of memory.
  - A resonance/access event is a *kick*: an impulse that boosts the
    orbit back outward. Use is the thrust that keeps a fact alive.
  - Bodies also attract each other, so valued, semantically near facts
    clump into co-orbiting groups — topics, formed by gravity alone.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Body:
    """A fact rendered as an orbiting body.

    ``kind`` is "fact" for an ordinary body or "meta" for a MetaFact formed
    by a Jeans collapse; ``source_ids`` lists the facts a MetaFact swallowed.
    ``depth`` is 0 for a fact, 1 for a MetaFact, 2+ for a MetaFact that
    collapsed out of other MetaFacts — recursive structure formation.
    """

    fact_id: str
    pos: np.ndarray          # shape (2,) — position in the galactic plane
    vel: np.ndarray          # shape (2,) — velocity
    mass: float              # accumulated value — heavier = more valued
    label: str = ""          # short human label, for rendering only
    kind: str = "fact"       # "fact" | "meta"
    source_ids: list[str] = field(default_factory=list)
    depth: int = 0           # 0 fact, 1 meta, 2+ meta-of-metas

    @property
    def radius(self) -> float:
        """Distance from the central black hole."""
        return float(math.hypot(self.pos[0], self.pos[1]))

    @property
    def speed(self) -> float:
        return float(math.hypot(self.vel[0], self.vel[1]))


# Ring names, ordered from the black hole outward.
ABSORBED = "absorbed"
CORE = "core"
KINETIC = "kinetic"
SURFACE = "surface"


class Galaxy:
    """A 2D gravitational system: bodies orbiting one central black hole.

    All tunables are constructor arguments — this is a research lab, the
    point is to turn the knobs and watch. Defaults give stable orbits at
    a few hundred bodies with a visible inward drift over ~1000 steps.
    """

    def __init__(
        self,
        *,
        g: float = 1.0,
        central_mass: float = 1200.0,
        softening: float = 0.6,
        drag: float = 0.015,
        dt: float = 0.05,
        horizon: float = 1.5,
        r_core: float = 6.0,
        r_surface: float = 14.0,
        attention_mass: float = 0.0,
        attention_softening: float = 4.0,
    ) -> None:
        self.g = g
        self.central_mass = central_mass
        self.softening = softening
        self.drag = drag
        self.dt = dt
        self.horizon = horizon
        self.r_core = r_core
        self.r_surface = r_surface
        self.attention_mass = attention_mass
        # Attention is softened heavily on purpose: a broad, gentle bias
        # toward the current focus, never a slingshot at close range.
        self.attention_softening = attention_softening

        self.bodies: list[Body] = []
        self.absorbed: list[str] = []
        self.steps = 0
        # Bodies that crossed the horizon, kept so Hawking emission can
        # leak them back out; absorbed holds their ids for the record.
        self._swallowed: list[Body] = []
        self.hawking_count = 0
        # A second, externally-driven attractor: the user's current focus.
        # Set by the replay; None means no attention pull is active.
        self.attention_pos: np.ndarray | None = None

    # ── Construction ────────────────────────────────────────────────────────

    def add_body(self, body: Body) -> None:
        self.bodies.append(body)

    def circular_speed(self, radius: float) -> float:
        """Speed of a circular orbit at ``radius`` around the black hole."""
        return math.sqrt(self.g * self.central_mass / max(radius, self.softening))

    def place_in_orbit(
        self,
        fact_id: str,
        radius: float,
        angle: float,
        mass: float,
        label: str = "",
    ) -> Body:
        """Add a body on a circular orbit at (radius, angle). Returns it."""
        direction = np.array([math.cos(angle), math.sin(angle)])
        tangent = np.array([-direction[1], direction[0]])
        body = Body(
            fact_id=fact_id,
            pos=direction * radius,
            vel=tangent * self.circular_speed(radius),
            mass=mass,
            label=label,
        )
        self.add_body(body)
        return body

    # ── Dynamics ────────────────────────────────────────────────────────────

    def _accelerations(self) -> np.ndarray:
        """Gravitational acceleration on every body — central + mutual.

        Vectorised O(n^2); fine for the few hundred to few thousand facts a
        personal store holds. Plummer softening keeps close pairs finite.
        """
        n = len(self.bodies)
        if n == 0:
            return np.zeros((0, 2))

        pos = np.array([b.pos for b in self.bodies])       # (n, 2)
        mass = np.array([b.mass for b in self.bodies])     # (n,)
        eps2 = self.softening * self.softening

        # Pull of the central black hole at the origin.
        r2_central = (pos * pos).sum(axis=1) + eps2        # (n,)
        acc = self.g * self.central_mass * (-pos) / (r2_central[:, None] ** 1.5)

        # Mutual pull between bodies.
        diff = pos[None, :, :] - pos[:, None, :]           # (n, n, 2): j - i
        dist2 = (diff * diff).sum(axis=2) + eps2           # (n, n)
        inv = dist2 ** -1.5
        np.fill_diagonal(inv, 0.0)                         # no self-force
        acc = acc + self.g * (inv[:, :, None] * diff * mass[None, :, None]).sum(axis=1)

        # Pull of the attention mass — the user's current focus — if active.
        if self.attention_pos is not None and self.attention_mass > 0.0:
            to_attn = self.attention_pos[None, :] - pos    # (n, 2)
            soft = self.attention_softening * self.attention_softening
            r2_attn = (to_attn * to_attn).sum(axis=1) + soft
            acc = acc + self.g * self.attention_mass * to_attn / (r2_attn[:, None] ** 1.5)
        return acc

    def step(self) -> list[str]:
        """Advance one timestep. Returns fact_ids that crossed the horizon.

        Leapfrog kick-drift-kick for the conservative gravity, with linear
        dynamical friction folded into the velocity each step.
        """
        if not self.bodies:
            self.steps += 1
            return []

        half = self.dt * 0.5
        damp = 1.0 - self.drag * self.dt

        acc = self._accelerations()
        for i, b in enumerate(self.bodies):
            b.vel = (b.vel + acc[i] * half) * damp
            b.pos = b.pos + b.vel * self.dt

        acc = self._accelerations()
        for i, b in enumerate(self.bodies):
            b.vel = b.vel + acc[i] * half

        self.steps += 1
        return self._absorb()

    def run(self, steps: int) -> list[str]:
        """Advance many steps. Returns every fact_id absorbed along the way."""
        fell_in: list[str] = []
        for _ in range(steps):
            fell_in.extend(self.step())
        return fell_in

    def _absorb(self) -> list[str]:
        """Remove bodies that crossed the event horizon."""
        survivors: list[Body] = []
        fell_in: list[str] = []
        for b in self.bodies:
            if b.radius < self.horizon:
                fell_in.append(b.fact_id)
                self._swallowed.append(b)
            else:
                survivors.append(b)
        self.bodies = survivors
        self.absorbed.extend(fell_in)
        return fell_in

    def hawking_emit(self) -> Body | None:
        """Spontaneously leak the oldest swallowed body back onto an orbit.

        The galaxy's analogue of Hawking radiation: the black hole is not
        perfectly final. A long-forgotten body occasionally returns to a
        precarious inner-kinetic orbit, where it must earn a kick or sink
        straight back in. Returns the emitted body, or None if empty.
        """
        if not self._swallowed:
            return None
        body = self._swallowed.pop(0)
        while body.fact_id in self.absorbed:
            self.absorbed.remove(body.fact_id)
        angle = (math.atan2(body.pos[1], body.pos[0])
                 if body.radius > 1e-6 else 0.0)
        direction = np.array([math.cos(angle), math.sin(angle)])
        tangent = np.array([-direction[1], direction[0]])
        radius = self.r_core + 1.0
        body.pos = direction * radius
        body.vel = tangent * self.circular_speed(radius)
        self.bodies.append(body)
        self.hawking_count += 1
        return body

    def kick(self, fact_id: str, strength: float) -> bool:
        """Apply an orbital impulse to a body — the thrust of a resonance hit.

        Positive ``strength`` boosts the orbit outward (toward surface),
        negative drops it inward. The impulse is split tangential/radial so
        a boost both speeds the body up and lifts it. Returns False if no
        such body is live.
        """
        for b in self.bodies:
            if b.fact_id != fact_id:
                continue
            r = max(b.radius, self.softening)
            radial = b.pos / r
            tangent = np.array([-radial[1], radial[0]])
            # Align the tangential push with current travel direction.
            if float(b.vel @ tangent) < 0:
                tangent = -tangent
            b.vel = b.vel + strength * (0.7 * tangent + 0.3 * radial)
            return True
        return False

    def find(self, fact_id: str) -> Body | None:
        """Return the live body for a fact, or None if it is gone."""
        for b in self.bodies:
            if b.fact_id == fact_id:
                return b
        return None

    # ── Observation ─────────────────────────────────────────────────────────

    def ring_of(self, body: Body) -> str:
        """Which ring a body currently occupies, by orbital radius."""
        r = body.radius
        if r < self.horizon:
            return ABSORBED
        if r < self.r_core:
            return CORE
        if r < self.r_surface:
            return KINETIC
        return SURFACE

    def ring_counts(self) -> dict[str, int]:
        """Population of each live ring, plus the absorbed total."""
        counts = {SURFACE: 0, KINETIC: 0, CORE: 0, ABSORBED: len(self.absorbed)}
        for b in self.bodies:
            counts[self.ring_of(b)] += 1
        return counts

    def total_energy(self) -> float:
        """Kinetic + potential energy of the system. Without drag it is

        conserved; with drag it decays — a direct readout of the galaxy
        cooling toward the black hole.
        """
        energy = 0.0
        for b in self.bodies:
            energy += 0.5 * b.mass * b.speed * b.speed
            r = max(b.radius, self.softening)
            energy -= self.g * self.central_mass * b.mass / r
        return energy
