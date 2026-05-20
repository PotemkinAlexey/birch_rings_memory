"""Gravity engine — fact migration between layers."""
from birch.fact import FactPassport
from birch.gravity import GravityEngine


def _build_engine():
    engine = GravityEngine()
    f_hot       = FactPassport("mailer service", "runs on",    "Go",      source_session="s1")
    f_cold      = FactPassport("legacy script",  "written in", "Python",  source_session="s1")
    f_connected = FactPassport("Go",             "used by",    "mailer service", source_session="s1")

    for f in (f_hot, f_cold, f_connected):
        engine.register(f)

    engine.link(f_hot.fact_id, f_connected.fact_id)
    engine.link(f_connected.fact_id, f_hot.fact_id)

    for _ in range(15):
        f_hot.touch()
        engine.apply_session_resonance([f_hot.fact_id], r=+0.8)

    for _ in range(2):
        f_cold.touch()
        engine.apply_session_resonance([f_cold.fact_id], r=-0.7)

    for _ in range(3):
        f_connected.touch()
        engine.apply_session_resonance([f_connected.fact_id], r=+0.6)

    engine.tick()
    return f_hot, f_cold, f_connected


def test_hot_fact_promotes():
    f_hot, _, _ = _build_engine()
    assert f_hot.gravity_score > 0.70, f"expected gravity > 0.70, got {f_hot.gravity_score}"
    assert f_hot.layer == 0, f"expected layer 0 (surface), got {f_hot.layer}"


def test_cold_fact_demotes():
    _, f_cold, _ = _build_engine()
    assert f_cold.gravity_score < 0.30, f"expected gravity < 0.30, got {f_cold.gravity_score}"
    assert f_cold.layer == 2, f"expected layer 2 (core), got {f_cold.layer}"


def test_graph_degree_helps():
    _, f_cold, f_connected = _build_engine()
    assert f_connected.gravity_score > f_cold.gravity_score, (
        f"f_connected ({f_connected.gravity_score}) should outrank f_cold ({f_cold.gravity_score})"
    )


if __name__ == "__main__":
    f_hot, f_cold, f_connected = _build_engine()
    for f in (f_hot, f_cold, f_connected):
        layer = {0: "surface", 1: "kinetic", 2: "core"}[f.layer]
        print(f"  {f!r}  gravity={f.gravity_score:.3f}  layer={layer}")
