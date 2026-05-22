"""Gravity engine — fact migration across the surface / kinetic / core rings."""
import time

from birch.fact import FactPassport
from birch.gravity import GravityEngine

_DAY = 86400.0


def _aged(fact: FactPassport, days: float) -> FactPassport:
    """Backdate a fact's creation and last-access by ``days``."""
    past = time.time() - days * _DAY
    fact.created_at = past
    fact.last_accessed = past
    return fact


def test_hot_fact_promotes_to_surface():
    """Fresh, frequently used and positively resonant → surface (layer 0)."""
    engine = GravityEngine()
    f_hot = FactPassport("mailer service", "runs on", "Go")
    f_other = FactPassport("legacy script", "written in", "Python")
    engine.register(f_hot)
    engine.register(f_other)
    engine.link(f_other.fact_id, f_hot.fact_id)  # f_hot gains graph degree

    for _ in range(15):
        f_hot.touch()
        engine.apply_session_resonance([f_hot.fact_id], r=+0.8)

    engine.tick()
    assert f_hot.gravity_score > 0.70, f_hot.gravity_score
    assert f_hot.layer == 0, f"expected surface, got layer {f_hot.layer}"


def test_fresh_fact_survives_a_bad_session():
    """The grace period: a brand-new fact scored by a negative session must
    NOT drop to the cold core — freshness keeps it in kinetic."""
    engine = GravityEngine()
    f = FactPassport("new idea", "scored", "badly")
    engine.register(f)
    for _ in range(2):
        f.touch()
        engine.apply_session_resonance([f.fact_id], r=-0.7)

    engine.tick()
    assert f.layer == 1, f"fresh fact must stay kinetic, got layer {f.layer}"
    assert f.gravity_score >= 0.30, f.gravity_score


def test_aged_unused_fact_demotes_to_core():
    """A fact nobody has touched for weeks sinks out of kinetic into core."""
    engine = GravityEngine()
    f = _aged(FactPassport("stale note", "is", "forgotten"), days=14)
    engine.register(f)

    engine.tick()
    assert f.gravity_score < 0.30, f.gravity_score
    assert f.layer == 2, f"expected core, got layer {f.layer}"


def test_aged_unused_fact_sinks_below_black_hole_threshold():
    """Given enough age with no use or resonance, gravity falls under the
    0.10 absorption floor — the fact becomes black-hole eligible."""
    engine = GravityEngine()
    f = _aged(FactPassport("ancient", "long", "gone"), days=120)
    engine.register(f)

    engine.tick()
    assert f.gravity_score < 0.10, f.gravity_score


def test_graph_degree_lifts_gravity():
    """Connectivity raises gravity: a linked fact outranks an identical
    unlinked one, all else equal."""
    engine = GravityEngine()
    f_linked = FactPassport("Go", "used by", "mailer service")
    f_lonely = FactPassport("perl", "used by", "nobody")
    hub = FactPassport("hub", "node", "x")
    for f in (f_linked, f_lonely, hub):
        engine.register(f)
    engine.link(hub.fact_id, f_linked.fact_id)

    engine.tick()
    assert f_linked.gravity_score > f_lonely.gravity_score
