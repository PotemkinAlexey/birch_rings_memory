"""Resonance detector experiment — baseline vs embeddings vs full (+ repetition)."""
import sys
sys.path.insert(0, "/Users/alexpotemkin/IdeaProjects/birch_rings_memory")

from birch.resonance.detector import compute_resonance
from birch.resonance.embeddings import embed_batch
from tests.fixtures.sessions import SESSIONS


def run_session(s, mode):
    messages = s["messages"]
    start_vec = end_vec = None
    all_vecs = None

    if mode in ("embeddings", "full"):
        vecs = embed_batch(messages)
        start_vec = vecs[0]
        end_vec = vecs[-1]
        if mode == "full":
            all_vecs = vecs

    result = compute_resonance(
        messages,
        start_vector=start_vec,
        end_vector=end_vec,
        all_vectors=all_vecs,
    )
    ok = result.label == s["expected_label"]
    return ok, result


def run(mode):
    labels = {
        "baseline": "Baseline (patterns only)",
        "embeddings": "Patterns + semantic embeddings",
        "full": "Full (patterns + semantic + repetition)",
    }
    print(f"\n{'='*62}")
    print(f"BirchKM — [{labels[mode]}]")
    print(f"{'='*62}\n")

    passed = failed = 0
    for s in SESSIONS:
        ok, result = run_session(s, mode)
        status = "✓" if ok else "✗"
        if ok:
            passed += 1
        else:
            failed += 1

        print(f"{status}  [{s['name']}]")
        print(f"   {s['description']}")
        print(
            f"   behavioral={result.behavioral_score:+.2f}  "
            f"semantic={result.semantic_score:+.2f}  "
            f"repetition={result.repetition_score:+.2f}  "
            f"R={result.r:+.3f}  label={result.label!r}"
        )
        print(f"   expected={s['expected_label']!r}")
        print()

    print(f"Result: {passed}/{len(SESSIONS)} passed\n")
    return passed


if __name__ == "__main__":
    n = len(SESSIONS)
    scores = {}
    for mode in ("baseline", "embeddings", "full"):
        scores[mode] = run(mode)

    print(f"{'='*62}")
    print(f"Baseline (patterns only):              {scores['baseline']}/{n}")
    print(f"Patterns + semantic embeddings:        {scores['embeddings']}/{n}")
    print(f"Full (+ repetition detector):          {scores['full']}/{n}")
    print(f"{'='*62}\n")
