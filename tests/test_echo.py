"""Echo Validation — cross-session topic matching with retroactive penalty."""
import pytest

from birch.resonance.detector import compute_resonance
from birch.resonance.echo import EchoStore, echo_penalty_for
from birch.resonance.embeddings import embed, embed_batch
from tests.conftest import needs_real_embeddings
from tests.fixtures.echo_sessions import ECHO_PAIRS

# ── Penalty magnitude (prior_R gate) — pure function, no embeddings ──────────

def test_echo_penalty_suppressed_for_strongly_resonant_prior():
    """A revisit to a strongly-resonant past topic is ambiguous (continued
    use vs. false closure) — penalty must be near zero, never the flat -0.8."""
    p = echo_penalty_for(0.9)
    assert -0.1 < p <= 0.0, f"strong-resonant prior should barely penalise, got {p}"


def test_echo_penalty_full_for_toxic_prior():
    """A revisit to an already-failing topic is unambiguous — full penalty."""
    assert echo_penalty_for(-0.5) == -0.6
    assert echo_penalty_for(0.0) == -0.6


def test_echo_penalty_monotonic_in_prior():
    """Higher prior_r ⇒ weaker (closer-to-zero) penalty. The gate is a
    monotone confidence ramp, not a step function."""
    grid = [-0.5, 0.0, 0.4, 0.7, 0.9, 1.0]
    penalties = [echo_penalty_for(r) for r in grid]
    # Penalties are negative; "weaker" means larger (closer to 0).
    assert all(a <= b for a, b in zip(penalties, penalties[1:])), penalties
    assert penalties[-1] == 0.0  # prior_r == 1.0 ⇒ zero confidence in failure


def test_echo_penalty_continuous_at_old_threshold():
    """No step at prior_r=0.35: a marginally-better prior must not draw a
    harsher penalty than a marginally-worse one (the old -0.6→-0.8 base jump)."""
    below = echo_penalty_for(0.349)
    above = echo_penalty_for(0.351)
    # Higher prior ⇒ weaker (closer to zero) penalty, and the gap is tiny.
    assert above >= below, f"seam reversed: {above} < {below}"
    assert abs(above - below) < 0.01, f"discontinuity at 0.35: {below} vs {above}"


def test_apply_echo_scales_penalty_by_severity():
    """apply_echo(scale=…) attenuates the penalty linearly — the hook
    session_close uses to make a neutral return weaker than a toxic one."""
    dim = 8
    v = [1.0] + [0.0] * (dim - 1)
    store = EchoStore()
    store.record("full", [v, v], r_score=0.5, fact_weights={"f": 1.0})
    store.record("weak", [v, v], r_score=0.5, fact_weights={"f": 1.0})
    full = store.apply_echo("full")             # scale 1.0
    weak = store.apply_echo("weak", scale=0.3)  # neutral-return strength
    assert weak.penalty > full.penalty          # weaker = closer to 0
    assert abs(weak.penalty - full.penalty * 0.3) < 1e-3


def test_echo_penalty_no_forced_toxic_floor():
    """Applying a suppressed penalty to a resonant prior must NOT force the
    score into the toxic zone (regression on the old min(-0.2, …) floor)."""
    store = EchoStore()
    dim = 8
    vec = [1.0] + [0.0] * (dim - 1)
    store.record("past", [vec, vec], r_score=0.9)
    res = store.detect_echo(vec)  # identical topic ⇒ guaranteed echo
    assert res.label == "echo"
    past = store.get("past")
    assert past.r_score > 0.35, (
        f"resonant prior must survive an ambiguous echo, got {past.r_score}"
    )


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
