"""Echo Validation — cross-session topic matching with retroactive penalty."""
import pytest

from birch.resonance.detector import compute_resonance
from birch.resonance.echo import EchoStore
from birch.resonance.embeddings import embed, embed_batch
from tests.conftest import needs_real_embeddings
from tests.fixtures.echo_sessions import ECHO_PAIRS


def _run_pair(pair):
    store = EchoStore()
    s1, s2 = pair["session_1"], pair["session_2"]

    msgs1 = s1["messages"]
    vecs1 = embed_batch(msgs1)
    result1 = compute_resonance(
        msgs1, start_vector=vecs1[0], end_vector=vecs1[-1], all_vectors=vecs1
    )
    store.record(s1["id"], vecs1, result1.r)

    vec2 = embed(s2["messages"][0])
    echo = store.detect_echo(vec2)
    stored = store.get(s1["id"])
    new_label = (
        "resonant" if stored.r_score > 0.35
        else "toxic" if stored.r_score < -0.15
        else "neutral"
    )
    return result1, echo, new_label


@needs_real_embeddings
@pytest.mark.parametrize("pair", ECHO_PAIRS, ids=[p["name"] for p in ECHO_PAIRS])
def test_echo_pair(pair):
    s1, s2 = pair["session_1"], pair["session_2"]
    result1, echo, new_label = _run_pair(pair)

    assert result1.label == s1["expected_r_before_echo"], (
        f"session_1 R before echo: got {result1.label!r}, expected {s1['expected_r_before_echo']!r}"
    )
    assert (echo.label == "echo") == s2["expected_echo"], (
        f"echo detection: got {echo.label!r} (sim={echo.similarity:.4f}), "
        f"expected echo={s2['expected_echo']}"
    )
    assert new_label == s2["expected_r_after_echo"], (
        f"session_1 R after echo: got {new_label!r}, expected {s2['expected_r_after_echo']!r}"
    )


if __name__ == "__main__":
    for pair in ECHO_PAIRS:
        result1, echo, new_label = _run_pair(pair)
        ok = (
            result1.label == pair["session_1"]["expected_r_before_echo"]
            and (echo.label == "echo") == pair["session_2"]["expected_echo"]
            and new_label == pair["session_2"]["expected_r_after_echo"]
        )
        print(f"{'✓' if ok else '✗'}  [{pair['name']}]  sim={echo.similarity:.4f}")
