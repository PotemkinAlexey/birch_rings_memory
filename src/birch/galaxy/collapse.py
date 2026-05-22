"""Jeans collapse — a cold, bound clump of facts fuses into a MetaFact.

Friends-of-friends — the standard N-body halo finder — groups bodies that
sit within a linking length of one another. A group that is gravitationally
bound and sub-virial (colder than the virial theorem allows, ``2*KE < |PE|``)
is unstable: gravity beats its internal pressure and it collapses. The whole
group is replaced by one heavier MetaFact body at its centre of mass, with
total mass and momentum conserved.

This is the live engine's SingularityCompactor, but as real physics —
proximity and coldness, not a cosine-similarity Union-Find.
"""
from __future__ import annotations

import uuid

import numpy as np

from .engine import Body, Galaxy


def friends_of_friends(bodies: list[Body], linking_length: float) -> list[list[Body]]:
    """Group bodies so any two within ``linking_length`` share a group."""
    n = len(bodies)
    if n == 0:
        return []
    pos = np.array([b.pos for b in bodies])
    ll2 = linking_length * linking_length
    seen = [False] * n
    groups: list[list[Body]] = []
    for start in range(n):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        group: list[Body] = []
        while stack:
            i = stack.pop()
            group.append(bodies[i])
            delta = pos - pos[i]
            d2 = (delta * delta).sum(axis=1)
            for j in range(n):
                if not seen[j] and d2[j] <= ll2:
                    seen[j] = True
                    stack.append(j)
        groups.append(group)
    return groups


def _internal_energy(group: list[Body], g: float) -> tuple[float, float]:
    """Kinetic energy in the group's COM frame, and its gravitational PE."""
    mass = np.array([b.mass for b in group])
    vel = np.array([b.vel for b in group])
    pos = np.array([b.pos for b in group])
    total = float(mass.sum())

    v_com = (mass[:, None] * vel).sum(axis=0) / total
    rel = vel - v_com
    ke = float((0.5 * mass * (rel * rel).sum(axis=1)).sum())

    pe = 0.0
    for i in range(len(group)):
        for j in range(i + 1, len(group)):
            dist = float(np.hypot(*(pos[i] - pos[j]))) + 1e-6
            pe -= g * mass[i] * mass[j] / dist
    return ke, pe


def is_jeans_unstable(group: list[Body], g: float, min_group: int) -> bool:
    """True when a group is big enough and sub-virial (``2*KE < |PE|``)."""
    if len(group) < min_group:
        return False
    ke, pe = _internal_energy(group, g)
    return 2.0 * ke < abs(pe)


def collapse_group(group: list[Body]) -> Body:
    """Fuse a group into one MetaFact body — mass and momentum conserved."""
    mass = np.array([b.mass for b in group])
    pos = np.array([b.pos for b in group])
    vel = np.array([b.vel for b in group])
    total = float(mass.sum())

    sources: list[str] = []
    for b in group:
        sources.extend(b.source_ids if b.kind == "meta" else [b.fact_id])

    return Body(
        fact_id=f"meta-{uuid.uuid4().hex[:8]}",
        pos=(mass[:, None] * pos).sum(axis=0) / total,
        vel=(mass[:, None] * vel).sum(axis=0) / total,
        mass=total,
        label=f"meta · {len(sources)} facts",
        kind="meta",
        source_ids=sources,
    )


def collapse_step(
    galaxy: Galaxy,
    *,
    linking_length: float = 2.0,
    min_group: int = 4,
) -> list[Body]:
    """Find Jeans-unstable clumps and collapse each into a MetaFact.

    Returns the new MetaFact bodies. The galaxy's body list is rewritten
    in place: collapsed members out, MetaFacts in.
    """
    groups = friends_of_friends(galaxy.bodies, linking_length)
    survivors: list[Body] = []
    metas: list[Body] = []
    for group in groups:
        if is_jeans_unstable(group, galaxy.g, min_group):
            metas.append(collapse_group(group))
        else:
            survivors.extend(group)
    galaxy.bodies = survivors + metas
    return metas
