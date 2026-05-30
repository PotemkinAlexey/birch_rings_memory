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


# ── Confidence (signal agreement) — pure detector math, no embeddings ────────

def test_confidence_in_unit_interval():
    """Confidence is always a valid [0, 1] weight, including the empty case."""
    assert compute_resonance([]).confidence == 1.0  # nothing to disagree about
    for msgs in (["thanks, works"], ["broken again", "still error"], ["hmm ok"]):
        c = compute_resonance(msgs).confidence
        assert 0.0 <= c <= 1.0, (msgs, c)


def test_confidence_full_when_single_signal():
    """One message, no vectors: only the behavioral signal fires, so there is
    nothing to conflict with — confidence must be 1.0 (R fully trusted)."""
    result = compute_resonance(["perfect, that worked, thanks!"])
    assert result.r > 0.0
    assert result.confidence == 1.0


def test_confidence_drops_when_signals_conflict():
    """The 'grumpy declarative summary' case: behavioral reads toxic (matched
    'broken'/'error') while the semantic trajectory reads productive (high
    cosine + rising specificity). The signals cancel, so confidence must fall
    well below 1.0 — which damps the gravity step downstream."""
    start_vec = [1.0, 0.0, 0.0]
    end_vec = [1.0, 0.0, 0.0]  # cosine 1.0 ⇒ "stayed on topic"
    result = compute_resonance(
        ["x x x x x", "broken error subject predicate object detail"],
        start_vector=start_vec,
        end_vector=end_vec,          # specificity rises sharply ⇒ semantic +0.6
        all_vectors=[start_vec, end_vec],
    )
    assert result.behavioral_score < 0.0, "behavioral should read negative"
    assert result.semantic_score > 0.0, "semantic should read positive"
    assert result.confidence < 0.85, (
        f"conflicting signals must lower confidence, got {result.confidence}"
    )


if __name__ == "__main__":
    # Quick experiment runner (non-pytest)
    for mode in ("baseline", "embeddings", "full"):
        passed = sum(1 for s in SESSIONS if _run_session(s, mode).label == s["expected_label"])
        print(f"{mode:12}: {passed}/{len(SESSIONS)}")
