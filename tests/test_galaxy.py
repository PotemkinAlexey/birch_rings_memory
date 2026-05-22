"""Galaxy N-body engine — orbital physics."""
import math

import numpy as np

from birch.fact import FactPassport
from birch.galaxy.engine import CORE, KINETIC, SURFACE, Body, Galaxy
from birch.galaxy.loader import build_galaxy, project_to_angles


def test_circular_orbit_is_stable():
    """A body placed on a circular orbit stays near its radius (no drag)."""
    gal = Galaxy(drag=0.0)
    body = gal.place_in_orbit("f", radius=10.0, angle=0.0, mass=1.0)
    gal.run(400)
    assert 8.5 < body.radius < 11.5, body.radius


def test_drag_decays_the_orbit():
    """Dynamical friction spirals an untouched body inward."""
    gal = Galaxy(drag=0.05)
    body = gal.place_in_orbit("f", radius=12.0, angle=0.0, mass=1.0)
    start = body.radius
    gal.run(400)
    assert body.radius < start, (start, body.radius)


def test_body_spiralling_past_the_horizon_is_absorbed():
    """Heavy friction drags a body down across the event horizon."""
    gal = Galaxy(drag=0.4)
    gal.place_in_orbit("doomed", radius=4.0, angle=0.0, mass=1.0)
    absorbed = gal.run(2000)
    assert "doomed" in absorbed
    assert gal.bodies == []
    assert "doomed" in gal.absorbed


def test_kick_raises_the_orbit():
    """A resonance kick boosts a body to a higher orbit."""
    gal = Galaxy(drag=0.0)
    gal.place_in_orbit("f", radius=8.0, angle=0.0, mass=1.0)
    before = _mean_radius(gal, "f", 120)
    gal.kick("f", strength=4.0)
    after = _mean_radius(gal, "f", 120)
    assert after > before, (before, after)


def _mean_radius(gal: Galaxy, fact_id: str, steps: int) -> float:
    samples = []
    for _ in range(steps):
        gal.step()
        for b in gal.bodies:
            if b.fact_id == fact_id:
                samples.append(b.radius)
    return sum(samples) / len(samples)


def test_ring_assignment_by_radius():
    gal = Galaxy()
    far = Body("a", np.array([20.0, 0.0]), np.zeros(2), 1.0)
    mid = Body("b", np.array([9.0, 0.0]), np.zeros(2), 1.0)
    near = Body("c", np.array([3.0, 0.0]), np.zeros(2), 1.0)
    assert gal.ring_of(far) == SURFACE
    assert gal.ring_of(mid) == KINETIC
    assert gal.ring_of(near) == CORE


def test_total_energy_decays_under_drag():
    gal = Galaxy(drag=0.05)
    gal.place_in_orbit("f", radius=10.0, angle=0.0, mass=1.0)
    e_start = gal.total_energy()
    gal.run(300)
    assert gal.total_energy() < e_start


def test_build_galaxy_from_facts():
    facts = [
        FactPassport("api", "runs on", "Go"),
        FactPassport("db", "is", "Postgres"),
        FactPassport("cache", "is", "Redis"),
    ]
    for i, f in enumerate(facts):
        f.vector = list(np.random.default_rng(i).normal(size=8))
    gal = build_galaxy(facts)
    assert len(gal.bodies) == 3
    # Every body sits on a sane orbit, clear of the event horizon.
    assert all(b.radius > gal.horizon for b in gal.bodies)


def test_project_to_angles_shape():
    vecs = [list(np.random.default_rng(i).normal(size=6)) for i in range(5)]
    angles = project_to_angles(vecs)
    assert angles.shape == (5,)
    assert np.all(np.abs(angles) <= math.pi)
