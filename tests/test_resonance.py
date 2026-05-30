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


def test_confidence_reduced_for_lone_signal():
    """One message, no vectors: only the behavioral signal fires. Agreement is
    trivially 1.0 (a signal agrees with itself), but the verdict rests on a
    single uncorroborated signal — confidence must be DAMPED (~0.75 floor), not
    full. This is the single-signal-dominance guard: a lone regex match should
    not move gravity at full strength, whether it is right or wrong."""
    result = compute_resonance(["perfect, that worked, thanks!"])
    assert result.r > 0.0
    assert 0.7 <= result.confidence <= 0.85, result.confidence


def test_confidence_rises_when_a_second_signal_corroborates():
    """A verdict backed by two balanced agreeing signals is trusted more than
    the same verdict from one. Corroboration lifts confidence toward 1.0."""
    # Lone behavioral negative.
    lone = compute_resonance(["still broken, same error again"])
    # Behavioral negative AND repetition negative (tight semantic loop) — two
    # signals agreeing on toxic.
    v = [1.0, 0.0, 0.0]
    corroborated = compute_resonance(
        ["why is it broken", "still broken, same error again"],
        all_vectors=[v, v],  # zero dispersion ⇒ repetition fires negative too
    )
    assert corroborated.repetition_score < 0.0, "repetition should corroborate"
    assert corroborated.confidence > lone.confidence, (
        f"corroborated {corroborated.confidence} should exceed lone "
        f"{lone.confidence}"
    )


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
