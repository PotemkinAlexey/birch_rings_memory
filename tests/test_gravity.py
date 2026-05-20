"""Gravity engine experiment — do facts migrate correctly under different conditions?"""
import sys
sys.path.insert(0, "/Users/alexpotemkin/IdeaProjects/birch_rings_memory")

from birch.fact import FactPassport
from birch.gravity import GravityEngine


def make_fact(subject, predicate, obj, session_id="s1") -> FactPassport:
    return FactPassport(
        subject=subject,
        predicate=predicate,
        object=obj,
        source_session=session_id,
    )


def label(layer: int) -> str:
    return {0: "surface", 1: "kinetic", 2: "core"}[layer]


def run():
    print(f"\n{'='*62}")
    print("BirchKM — Gravity Engine Experiment")
    print(f"{'='*62}\n")

    engine = GravityEngine()

    # Three facts with different access + resonance profiles
    f_hot       = make_fact("mailer service", "runs on",    "Go")
    f_cold      = make_fact("legacy script",  "written in", "Python")
    f_connected = make_fact("Go",             "used by",    "mailer service")

    for f in (f_hot, f_cold, f_connected):
        engine.register(f)

    # f_hot → f_connected (mailer depends on Go)
    engine.link(f_hot.fact_id, f_connected.fact_id)
    # f_connected → f_hot (Go used by mailer — back-reference)
    engine.link(f_connected.fact_id, f_hot.fact_id)

    print("Initial state:")
    for f in (f_hot, f_cold, f_connected):
        print(f"  {f!r}")

    # Simulate: f_hot used in 15 resonant sessions
    print("\n── Simulating 15 resonant sessions using f_hot ──")
    for _ in range(15):
        f_hot.touch()
        engine.apply_session_resonance([f_hot.fact_id], r=+0.8)

    # f_cold used in 2 toxic sessions
    print("── Simulating 2 toxic sessions using f_cold ──")
    for _ in range(2):
        f_cold.touch()
        engine.apply_session_resonance([f_cold.fact_id], r=-0.7)

    # f_connected: 3 resonant sessions, high graph degree
    print("── Simulating 3 resonant sessions using f_connected ──\n")
    for _ in range(3):
        f_connected.touch()
        engine.apply_session_resonance([f_connected.fact_id], r=+0.6)

    # Tick — recompute gravity
    migrations = engine.tick()

    print("After gravity tick:")
    for f in (f_hot, f_cold, f_connected):
        migrated = any(fid == f.fact_id for fid, _ in migrations)
        tag = " → MIGRATED" if migrated else ""
        print(
            f"  {f!r}"
            f"\n    access={f.access_count}  avg_R={f.avg_resonance:+.2f}"
            f"  gravity={f.gravity_score:.3f}  layer={label(f.layer)}{tag}"
        )
    print()

    # Assertions
    passed = failed = 0
    checks = [
        (f_hot.gravity_score > 0.70,   "f_hot gravity > 0.70 (should promote)"),
        (f_hot.layer == 0,             "f_hot migrated to layer 0 (surface)"),
        (f_cold.gravity_score < 0.30,  "f_cold gravity < 0.30 (should demote)"),
        (f_cold.layer == 2,            "f_cold migrated to layer 2 (core/archive)"),
        (f_connected.gravity_score > f_cold.gravity_score,
                                       "f_connected gravity > f_cold (graph degree helps)"),
    ]

    for ok, desc in checks:
        status = "✓" if ok else "✗"
        if ok:
            passed += 1
        else:
            failed += 1
        print(f"  {status}  {desc}")

    print(f"\n{'='*62}")
    print(f"Result: {passed}/{len(checks)} checks passed")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    run()
