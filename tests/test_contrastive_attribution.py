"""Contrastive / outlier-robust attribution (proposal #5, v1).

The plain attribution adds ``effective_r · cosine_weight`` to a fact's
resonance every session it is retrieved into. Blame/credit therefore scales
with *topical* relevance — but a fact can be highly on-topic for a session
that fails for reasons unrelated to it. The reviewer's case: a genuinely
useful fact, retrieved at high cosine into one session that went toxic for
unrelated reasons, gets sunk "за компанию".

The fix anchors each fact on its own resonance history (its discriminative
signal: does it ride resonant or toxic sessions on net) and attenuates a
session whose outcome *contradicts* that history, in proportion to how
established the fact is. These tests pin that behaviour at the engine level —
deterministic, no embeddings.

See ``contrastive_impulse`` in gravity.py. Disabled by ``BIRCH_CONTRAST_K<=0``.
"""
from __future__ import annotations

from birch.fact import FactPassport
from birch.gravity import GravityEngine, contrastive_impulse


def _fact(subject: str) -> FactPassport:
    return FactPassport(subject=subject, predicate="p", object="o")


def _resonant_history(engine: GravityEngine, fact: FactPassport, n: int, r: float):
    for _ in range(n):
        engine.apply_session_resonance({fact.fact_id: 0.9}, r)


# ── The headline case: a useful fact survives one incidental toxic session ───

def test_established_good_fact_resists_one_toxic_outlier():
    eng = GravityEngine()
    good = _fact("genuinely useful")
    eng.register(good)
    _resonant_history(eng, good, n=8, r=0.7)
    prior = good.avg_resonance
    assert prior > 0.5

    # One toxic session, fact highly relevant (high cosine) but not the cause.
    eng.apply_session_resonance({good.fact_id: 0.9}, -0.7)

    after = good.avg_resonance
    # The outlier barely moves an established fact — it is not sunk for
    # incidental co-occurrence.
    assert after > prior - 0.12, f"established fact dragged too far: {prior}→{after}"
    assert after > 0.4


def test_young_fact_takes_the_full_hit():
    """No history ⇒ no reason to doubt the session ⇒ full-strength impulse.
    (Contrast with the established fact above, hit by the identical session.)"""
    eng = GravityEngine()
    young = _fact("brand new")
    eng.register(young)
    eng.apply_session_resonance({young.fact_id: 0.9}, -0.7)
    # First-ever session, toxic, applied in full.
    assert young.avg_resonance < 0.0
    assert abs(young.avg_resonance - (-0.7 * 0.9)) < 1e-9


def test_consistently_toxic_fact_still_sinks():
    """Symmetry / no free pass: a fact whose history is toxic keeps taking
    full toxic hits (sessions agree with history), so misleading facts still
    sink fast — the robustness only resists *contradicting* outliers."""
    eng = GravityEngine()
    bad = _fact("misleading")
    eng.register(bad)
    _resonant_history(eng, bad, n=8, r=-0.7)
    assert bad.avg_resonance < -0.5
    # One stray resonant session does NOT redeem it much.
    eng.apply_session_resonance({bad.fact_id: 0.9}, 0.7)
    assert bad.avg_resonance < -0.3, "a single good session over-redeemed a bad fact"


def test_good_and_bad_stay_separated_after_shared_toxic_session():
    """Discrimination end state: a good fact (resonant history) and a bad fact
    (toxic history) both attributed to the SAME final toxic session must stay
    clearly separated — the good one's history protects it, the bad one's
    confirms it."""
    eng = GravityEngine()
    good, bad = _fact("good"), _fact("bad")
    eng.register(good)
    eng.register(bad)
    _resonant_history(eng, good, n=8, r=0.7)
    _resonant_history(eng, bad, n=8, r=-0.7)

    # Both retrieved into one shared toxic session.
    eng.apply_session_resonance({good.fact_id: 0.9, bad.fact_id: 0.9}, -0.7)

    assert good.avg_resonance - bad.avg_resonance > 0.8, (
        f"separation collapsed: good={good.avg_resonance:.3f} "
        f"bad={bad.avg_resonance:.3f}"
    )


# ── The mechanism in isolation ───────────────────────────────────────────────

def test_confirming_session_applies_full_strength():
    """A session agreeing with the fact's history is never attenuated — a real
    shift up or down is still learned."""
    eng = GravityEngine()
    f = _fact("x")
    eng.register(f)
    _resonant_history(eng, f, n=8, r=0.7)
    # Confirming positive session ⇒ full impulse 0.7 * 0.8.
    assert abs(contrastive_impulse(f, 0.7, 0.8) - 0.7 * 0.8) < 1e-9


def test_contradicting_impulse_shrinks_with_history():
    """More agreeing history ⇒ more protection against a contradicting
    session (monotone), but it is always a shrink, never an amplification."""
    eng = GravityEngine()
    young, seasoned = _fact("young"), _fact("seasoned")
    eng.register(young)
    eng.register(seasoned)
    _resonant_history(eng, young, n=2, r=0.7)
    _resonant_history(eng, seasoned, n=20, r=0.7)

    raw = -0.7 * 0.9
    young_imp = contrastive_impulse(young, -0.7, 0.9)
    seasoned_imp = contrastive_impulse(seasoned, -0.7, 0.9)

    # Both shrunk (closer to 0 than raw), seasoned more so.
    assert raw < young_imp < 0.0
    assert young_imp < seasoned_imp < 0.0
    assert seasoned_imp > raw


def test_trust_prior_is_order_independent():
    """The shrink decision reads the RAW track record, which is an
    order-independent mean — so two facts given the same multiset of sessions
    in opposite order end with the identical trust prior. This is the property
    that breaks the old self-reference (trust read from already-shrunk history,
    which was order-dependent)."""
    eng = GravityEngine()
    a, b = _fact("a"), _fact("b")
    eng.register(a)
    eng.register(b)
    seq = [0.7, 0.7, 0.7, 0.7, -0.7, -0.7, -0.7, -0.7]
    for r in seq:
        eng.apply_session_resonance({a.fact_id: 0.9}, r)
    for r in reversed(seq):
        eng.apply_session_resonance({b.fact_id: 0.9}, r)
    assert abs(a.raw_avg_resonance - b.raw_avg_resonance) < 1e-9


def test_turned_bad_fact_eventually_lands_full_and_follows_late():
    """A fact that is resonant then genuinely turns toxic: once enough toxic
    sessions arrive, the RAW prior flips sign, later toxic sessions stop being
    shrunk, and the gravity-side resonance follows the late reality instead of
    freezing on the early reputation (the old self-referential failure)."""
    eng = GravityEngine()
    f = _fact("turned")
    eng.register(f)
    for _ in range(5):
        eng.apply_session_resonance({f.fact_id: 0.9}, 0.7)
    assert f.raw_avg_resonance > 0 and f.avg_resonance > 0
    for _ in range(15):
        eng.apply_session_resonance({f.fact_id: 0.9}, -0.7)
    # True record flipped, and the gravity-side mean followed it negative —
    # not frozen positive by self-protecting trust.
    assert f.raw_avg_resonance < 0, f"raw prior did not flip: {f.raw_avg_resonance}"
    assert f.avg_resonance < 0, f"gravity-side froze on early reputation: {f.avg_resonance}"


def test_armor_follows_consistency_not_just_tenure():
    """Two facts with the SAME tenure (n) but different consistency: a strongly
    one-signed history earns full outlier armor, a mushy near-zero history stays
    responsive to a contradicting session. Armor must come from "how stably
    useful", not "how long seen"."""
    eng = GravityEngine()
    strong, mushy = _fact("strong"), _fact("mushy")
    eng.register(strong)
    eng.register(mushy)
    for _ in range(40):
        eng.apply_session_resonance({strong.fact_id: 0.9}, 0.8)   # raw_avg ≈ 0.72
    for i in range(40):
        # net slightly positive, but hovering near zero → low consistency
        eng.apply_session_resonance({mushy.fact_id: 0.9}, 0.1 if i % 2 == 0 else -0.08)

    assert abs(mushy.raw_avg_resonance) < 0.05, mushy.raw_avg_resonance
    assert strong.raw_avg_resonance > 0.6

    raw = -0.7 * 0.9
    imp_strong = contrastive_impulse(strong, -0.7, 0.9)   # consistent → armored
    imp_mushy = contrastive_impulse(mushy, -0.7, 0.9)     # mushy → responsive

    assert imp_strong > imp_mushy, "consistent history should resist more"
    # Despite identical n=40, the mushy fact takes almost the full hit while the
    # strong one is heavily shrunk — armor tracks consistency, not tenure.
    assert imp_mushy < raw * 0.6, f"mushy should stay responsive, got {imp_mushy}"
    assert imp_strong > raw * 0.4, f"strong should be armored, got {imp_strong}"


def test_attenuation_counter_tracks_only_real_shrinks():
    """The engine counts an attenuation only when an impulse was actually
    shrunk (a contradicting session on an established fact), not on confirming
    or first-ever sessions — so the stat means what it says."""
    eng = GravityEngine()
    f = _fact("x")
    eng.register(f)
    _resonant_history(eng, f, n=6, r=0.7)        # confirming ⇒ no attenuation
    assert eng.contrastive_attenuations == 0
    eng.apply_session_resonance({f.fact_id: 0.9}, -0.7)  # contradicting ⇒ +1
    assert eng.contrastive_attenuations == 1
