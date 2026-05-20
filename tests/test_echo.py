"""Echo Validation experiment — does the system detect returning problems?"""
import sys
sys.path.insert(0, "/Users/alexpotemkin/IdeaProjects/birch_rings_memory")

from birch.resonance.detector import compute_resonance
from birch.resonance.embeddings import embed, embed_batch
from birch.resonance.echo import EchoStore
from tests.fixtures.echo_sessions import ECHO_PAIRS


def run():
    print(f"\n{'='*62}")
    print("BirchKM — Echo Validation Experiment")
    print(f"{'='*62}\n")

    store = EchoStore()
    passed = failed = 0

    for pair in ECHO_PAIRS:
        s1 = pair["session_1"]
        s2 = pair["session_2"]
        print(f"── [{pair['name']}]")
        print(f"   {pair['description']}\n")

        # Run session 1
        msgs1 = s1["messages"]
        vecs1 = embed_batch(msgs1)
        result1 = compute_resonance(
            msgs1,
            start_vector=vecs1[0],
            end_vector=vecs1[-1],
            all_vectors=vecs1,
        )
        r1_ok = result1.label == s1["expected_r_before_echo"]
        print(f"   Session 1: R={result1.r:+.3f} label={result1.label!r} "
              f"{'✓' if r1_ok else '✗'} (expected {s1['expected_r_before_echo']!r})")

        # Store session 1 in echo store — centroid computed internally
        store.record(s1["id"], vecs1, result1.r)

        # Run session 2 — detect echo
        vec2_start = embed(s2["messages"][0])
        echo = store.detect_echo(vec2_start)
        is_echo = echo.label == "echo"
        echo_ok = is_echo == s2["expected_echo"]

        print(f"   Session 2: similarity={echo.similarity:.4f} "
              f"echo={is_echo} {'✓' if echo_ok else '✗'} "
              f"(expected echo={s2['expected_echo']})")

        if is_echo:
            print(f"   Echo penalty={echo.penalty:+.1f} → "
                  f"matched session '{echo.matched_session_id}' retroactively updated")
            stored = store.get(s1["id"])
            new_label = (
                "resonant" if stored.r_score > 0.35
                else "toxic" if stored.r_score < -0.15
                else "neutral"
            )
            r2_ok = new_label == s2["expected_r_after_echo"]
            print(f"   Session 1 after echo: R={stored.r_score:+.3f} "
                  f"label={new_label!r} {'✓' if r2_ok else '✗'} "
                  f"(expected {s2['expected_r_after_echo']!r})")
        else:
            stored = store.get(s1["id"])
            new_label = (
                "resonant" if stored.r_score > 0.35
                else "toxic" if stored.r_score < -0.15
                else "neutral"
            )
            r2_ok = new_label == s2["expected_r_after_echo"]
            print(f"   Session 1 unchanged: R={stored.r_score:+.3f} "
                  f"label={new_label!r} {'✓' if r2_ok else '✗'} "
                  f"(expected {s2['expected_r_after_echo']!r})")

        all_ok = r1_ok and echo_ok and r2_ok
        if all_ok:
            passed += 1
        else:
            failed += 1
        print(f"   Overall: {'✓ PASS' if all_ok else '✗ FAIL'}\n")

    print(f"{'='*62}")
    print(f"Result: {passed}/{len(ECHO_PAIRS)} passed, {failed} failed")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    run()
