"""Quick experiment: does the resonance detector classify sessions correctly?"""
import sys
sys.path.insert(0, "/Users/alexpotemkin/IdeaProjects/birch_rings_memory")

from birch.resonance.detector import compute_resonance
from tests.fixtures.sessions import SESSIONS


def run():
    print(f"\n{'='*60}")
    print("BirchKM — Resonance Detector Experiment")
    print(f"{'='*60}\n")

    passed = 0
    failed = 0

    for s in SESSIONS:
        result = compute_resonance(s["messages"])
        ok = result.label == s["expected_label"]
        status = "✓" if ok else "✗"
        if ok:
            passed += 1
        else:
            failed += 1

        print(f"{status}  [{s['name']}]")
        print(f"   {s['description']}")
        print(f"   behavioral={result.behavioral_score:+.2f}  "
              f"semantic={result.semantic_score:+.2f}  "
              f"R={result.r:+.3f}  label={result.label!r}")
        print(f"   expected={s['expected_label']!r}")
        print()

    print(f"{'='*60}")
    print(f"Result: {passed}/{len(SESSIONS)} passed, {failed} failed")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run()
