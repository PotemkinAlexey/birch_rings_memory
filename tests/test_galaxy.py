"""Galaxy N-body engine — orbital physics and history replay."""
import math
import time

import numpy as np

from birch.fact import FactPassport
from birch.galaxy.collapse import collapse_step, friends_of_friends
from birch.galaxy.engine import CORE, KINETIC, SURFACE, Body, Galaxy
from birch.galaxy.loader import build_galaxy, project_to_angles
from birch.galaxy.projection import Projector
from birch.galaxy.replay import build_history, replay
from birch.galaxy.report import GalaxyReport, diagnose, format_report, run_diagnosis


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


def test_replay_births_facts_across_the_timeline():
    """Facts created at different times are born at different sim steps."""
    now = time.time()
    facts = []
    for i in range(5):
        f = FactPassport(f"s{i}", "p", "o")
        f.created_at = now - (10 - i) * 86400      # 10, 9, 8, 7, 6 days old
        facts.append(f)
    history = build_history(facts, [], steps=200, now=now)
    steps = sorted(b.step for b in history.births)
    assert steps[0] < steps[-1], steps


def test_replay_resonance_keeps_a_fact_alive():
    """A fact a positive session resonates survives higher than an identical
    fact that is never touched."""
    now = time.time()
    kept = FactPassport("kept", "is", "useful")
    lost = FactPassport("lost", "is", "ignored")
    for f in (kept, lost):
        f.created_at = now - 20 * 86400
    sessions = [{
        "recorded_at": now - 9 * 86400,
        "r_score": 0.85,
        "fact_weights": {kept.fact_id: 1.0},
    }]
    history = build_history([kept, lost], sessions, steps=900, now=now)
    galaxy = Galaxy()
    replay(galaxy, history)

    kept_body = galaxy.find(kept.fact_id)
    lost_body = galaxy.find(lost.fact_id)
    assert kept_body is not None, "the resonated fact should survive"
    if lost_body is not None:
        assert kept_body.radius > lost_body.radius


def test_replay_on_empty_store():
    galaxy = Galaxy()
    assert replay(galaxy, build_history([], [], steps=50)) == []


def test_replay_on_step_callback_fires_each_step():
    now = time.time()
    fact = FactPassport("a", "b", "c")
    fact.created_at = now - 86400
    history = build_history([fact], [], steps=30, now=now)
    seen: list[int] = []
    replay(Galaxy(), history, on_step=lambda step, gal: seen.append(step))
    assert seen == list(range(30))


def test_friends_of_friends_separates_distant_clumps():
    bodies = [
        Body("a", np.array([0.0, 0.0]), np.zeros(2), 1.0),
        Body("b", np.array([0.5, 0.0]), np.zeros(2), 1.0),
        Body("c", np.array([20.0, 0.0]), np.zeros(2), 1.0),
    ]
    groups = friends_of_friends(bodies, linking_length=1.0)
    assert sorted(len(g) for g in groups) == [1, 2]


def test_jeans_collapse_fuses_a_cold_clump():
    """A tight, near-co-moving clump collapses into one MetaFact, conserving
    total mass and momentum."""
    gal = Galaxy()
    for i in range(4):
        gal.add_body(Body(
            f"f{i}",
            np.array([5.0 + 0.2 * i, 0.0]),
            np.array([0.0, 0.05 * i - 0.075]),   # tiny dispersion — cold
            2.0,
        ))
    mass_before = sum(b.mass for b in gal.bodies)
    momentum_before = sum((b.mass * b.vel for b in gal.bodies), np.zeros(2))

    metas = collapse_step(gal, linking_length=2.0, min_group=4)
    assert len(metas) == 1
    meta = metas[0]
    assert meta.kind == "meta"
    assert sorted(meta.source_ids) == ["f0", "f1", "f2", "f3"]
    assert abs(meta.mass - mass_before) < 1e-9
    np.testing.assert_allclose(meta.mass * meta.vel, momentum_before, atol=1e-9)
    assert gal.bodies == metas


def test_hot_group_resists_collapse():
    """A clump with large velocity dispersion is too hot to collapse."""
    gal = Galaxy()
    for i in range(4):
        gal.add_body(Body(
            f"f{i}",
            np.array([5.0 + 0.2 * i, 0.0]),
            np.array([9.0 * (i - 1.5), 7.0 * (i % 2 - 0.5)]),   # hot
            1.0,
        ))
    metas = collapse_step(gal, linking_length=2.0, min_group=4)
    assert metas == []
    assert len(gal.bodies) == 4


def test_projector_is_consistent():
    rng = np.random.default_rng(7)
    vecs = [list(rng.normal(size=10)) for _ in range(8)]
    projector = Projector.fit(vecs)
    assert projector is not None
    assert projector.dim == 10
    assert projector.angle(vecs[0]) == projector.angle(vecs[0])
    assert -math.pi <= projector.angle(vecs[3]) <= math.pi
    # Too little data to fit a 2D basis.
    assert Projector.fit([[1.0, 2.0]]) is None


def test_attention_mass_attracts_a_body():
    """A body at rest is dragged toward the attention mass."""
    pos0 = np.array([9.0, 0.0])

    plain = Galaxy(drag=0.0)
    plain.add_body(Body("f", pos0.copy(), np.zeros(2), 1.0))
    plain.run(15)

    pulled = Galaxy(drag=0.0, attention_mass=800.0)
    pulled.add_body(Body("f", pos0.copy(), np.zeros(2), 1.0))
    pulled.attention_pos = np.array([12.0, 0.0])
    pulled.run(15)

    plain_body = plain.find("f")
    pulled_body = pulled.find("f")
    assert plain_body is not None and pulled_body is not None
    # Without attention the body just falls toward the hole (-x);
    # the attention mass at x=12 drags it the other way.
    assert pulled_body.pos[0] > plain_body.pos[0]


def test_build_history_schedules_attention_from_sessions():
    now = time.time()
    facts = []
    for i in range(4):
        f = FactPassport(f"s{i}", "p", "o")
        f.created_at = now - 5 * 86400
        f.vector = list(np.random.default_rng(i).normal(size=6))
        facts.append(f)
    sessions = [{
        "recorded_at": now - 2 * 86400,
        "r_score": 0.5,
        "fact_weights": {},
        "centroids": [list(np.random.default_rng(99).normal(size=6))],
    }]
    history = build_history(facts, sessions, steps=300, now=now)
    assert len(history.attention) == 1
    assert 0 <= history.attention[0].step < 300


def test_diagnose_flags_core_facts_as_at_risk():
    """A fact decayed into the core ring is reported at risk; a far one is not."""
    gal = Galaxy()
    gal.add_body(Body("a", np.array([3.0, 0.0]), np.zeros(2), 1.0, label="risky"))
    gal.add_body(Body("b", np.array([20.0, 0.0]), np.zeros(2), 1.0, label="safe"))
    report = diagnose(gal, absorbed_ids=[], fact_labels={})
    risky = [label for label, _radius in report.at_risk]
    assert "risky" in risky
    assert "safe" not in risky


def test_run_diagnosis_produces_a_report():
    now = time.time()
    facts = []
    for i in range(20):
        f = FactPassport(f"fact{i}", "is", f"thing{i}")
        f.created_at = now - 15 * 86400
        f.vector = list(np.random.default_rng(i).normal(size=8))
        facts.append(f)
    report = run_diagnosis(facts, [], steps=600, now=now)
    assert isinstance(report, GalaxyReport)
    assert report.total == 20
    assert isinstance(report.topics, list)


def test_format_report_is_text():
    report = GalaxyReport(
        at_risk=[("x is y", 2.4)],
        topics=[["a", "b"]],
        metafacts=[["c", "d", "e"]],
        absorbed=["f"],
        live=3,
        total=6,
    )
    text = format_report(report)
    assert isinstance(text, str)
    assert "At risk" in text
    assert "Emergent topics" in text
