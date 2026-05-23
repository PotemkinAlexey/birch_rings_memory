"""Resonance detector — baseline vs embeddings vs full (+ repetition)."""
import pytest

from birch.resonance.detector import compute_resonance
from birch.resonance.embeddings import embed_batch
from tests.conftest import needs_real_embeddings
from tests.fixtures.sessions import SESSIONS


def _run_session(s, mode):
    messages = s["messages"]
    start_vec = end_vec = all_vecs = None
    if mode in ("embeddings", "full"):
        vecs = embed_batch(messages)
        start_vec, end_vec = vecs[0], vecs[-1]
        if mode == "full":
            all_vecs = vecs
    result = compute_resonance(
        messages, start_vector=start_vec, end_vector=end_vec, all_vectors=all_vecs
    )
    return result


@pytest.mark.parametrize("session", SESSIONS, ids=[s["name"] for s in SESSIONS])
def test_baseline(session):
    if session.get("requires_embeddings"):
        pytest.skip("requires embeddings — not meaningful in baseline mode")
    result = _run_session(session, "baseline")
    assert result.label == session["expected_label"], (
        f"[baseline] {session['name']}: R={result.r:+.3f} got {result.label!r}, "
        f"expected {session['expected_label']!r}"
    )


@needs_real_embeddings
@pytest.mark.parametrize("session", SESSIONS, ids=[s["name"] for s in SESSIONS])
def test_full(session):
    result = _run_session(session, "full")
    assert result.label == session["expected_label"], (
        f"[full] {session['name']}: R={result.r:+.3f} got {result.label!r}, "
        f"expected {session['expected_label']!r}"
    )


if __name__ == "__main__":
    # Quick experiment runner (non-pytest)
    for mode in ("baseline", "embeddings", "full"):
        passed = sum(1 for s in SESSIONS if _run_session(s, mode).label == s["expected_label"])
        print(f"{mode:12}: {passed}/{len(SESSIONS)}")
