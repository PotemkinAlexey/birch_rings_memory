"""Adversarial out-of-loop drift detector (proposal #4).

The resonance loop scores itself: R is inferred from sessions, propagated to
gravity, and the adaptive weights learn from R. A closed loop like that can
drift toward *self-confirmation* — ranking facts by an artefact of when/how
often they were seen rather than by how useful they actually proved. No
assertion inside the loop can catch that, because the loop has no ground truth.

This test supplies one. Each synthetic fact is assigned a FIXED true utility
up front, hidden from the system. The loop is then driven so that each fact's
realised sessions reflect its true utility (useful facts ride resonant
sessions; useless facts ride toxic ones). Crucially, true utility is NOT
monotone in creation order, so "rank by utility" and "rank by appearance
order" are separable. After many rounds we check the thing the loop cannot:

    does final gravity track assigned UTILITY, or has it drifted toward ORDER?

If a future change to weights / thresholds / the resonance or echo formula
starts making gravity self-confirming, the utility correlation collapses
and/or the order correlation takes over, and this fails — long before it would
surface on real traffic.

Runs end-to-end under whatever embedding provider is available: real Ollama
when an endpoint answers (the ``embed_provider`` fixture probes it), mock
otherwise. The driving signal is behavioral (regex on message tone), which is
provider-independent, so the assertion holds either way while the semantic and
repetition signals exercise the real embedder when present.

SCOPE — what this does NOT guard. The first test attributes each fact to its
session deterministically (``mem._session_fact_ids = [...]``), bypassing the
cosine relevance weighting. That isolates the gravity dynamics on purpose, but
it means the test proves "given clean attribution, gravity tracks utility" —
NOT "attribution hands credit to the right facts". The open risk there (a
genuinely useful fact that happened to ride a failed session sinking with it,
or credit smeared across co-retrieved facts) is proposal #5 (contrastive
attribution). It is only partially exercised by the second test below, which
drives attribution through real ``query_memory`` cosine weighting; full
contrastive-attribution coverage is still pending #5.
"""
from __future__ import annotations

import numpy as np

from birch.memory_store import MemoryStore

# True utilities, deliberately shuffled w.r.t. creation order so that a system
# ranking by recency/order would NOT reproduce this ranking.
_UTILITIES = [0.10, 0.90, 0.30, 0.70, 0.00, 1.00, 0.40, 0.60, 0.20, 0.80]
_ROUNDS = 6


def _messages_for(utility: float, topic: str) -> list[str]:
    """Two user messages whose closing tone encodes the fact's true utility.

    Behavioral scoring keys on the final message, so the tier boundaries here
    set the session's resonance: useful → resonant, useless → toxic, middling
    → neutral. First message varies the trajectory so the real embedder sees a
    genuine (non-degenerate) session shape.
    """
    if utility > 0.66:
        return [f"how should I use the {topic} setting", "perfect, that fixed it, thank you"]
    if utility < 0.34:
        return [f"the {topic} setting keeps failing", "still broken, same error again"]
    return [f"a question about the {topic} setting", "I will look into it later"]


def _drive_loop(mem: MemoryStore) -> list:
    facts = [
        mem.add_fact(f"topic{i}", "controls", f"subsystem{i}")
        for i in range(len(_UTILITIES))
    ]
    for rnd in range(_ROUNDS):
        for i, (f, u) in enumerate(zip(facts, _UTILITIES)):
            sid = f"s-{i}-{rnd}"
            mem.session_start(sid)  # becomes the current session
            for msg in _messages_for(u, f"topic{i}"):
                mem.session_message(msg, session_id=sid)
            # Attribute exactly this fact to the session (deterministic, no
            # reliance on retrieval similarity), then close on the heuristic.
            mem._session_fact_ids = [f.fact_id]
            mem.session_close(sid)
    return facts


def test_gravity_tracks_assigned_utility_not_appearance_order(embed_provider):
    mem = MemoryStore()
    facts = _drive_loop(mem)

    # Read final gravity from the held references — the value persists on the
    # object even if a low-utility fact was absorbed into the black hole
    # (which is itself the correct "lowest gravity" outcome).
    gravity = np.array([f.gravity_score for f in facts], dtype=float)
    utility = np.array(_UTILITIES, dtype=float)
    order = np.arange(len(facts), dtype=float)

    corr_utility = float(np.corrcoef(gravity, utility)[0, 1])
    corr_order = float(np.corrcoef(gravity, order)[0, 1])

    msg = (
        f"[{embed_provider}] corr(gravity, utility)={corr_utility:+.3f}  "
        f"corr(gravity, order)={corr_order:+.3f}  gravity={np.round(gravity, 3)}"
    )

    # 1. Gravity must track assigned utility, strongly and positively.
    assert corr_utility > 0.6, f"gravity does not track utility — {msg}"
    # 2. Utility must DOMINATE appearance order by a clear margin. This is the
    #    self-confirmation guard: if the loop started ranking by when-seen,
    #    corr_order would rise to rival corr_utility.
    assert corr_utility - abs(corr_order) > 0.3, (
        f"appearance order rivals utility — possible self-confirming drift — {msg}"
    )


def test_high_utility_facts_outrank_low_utility_facts(embed_provider):
    """Coarser, more legible companion check: the top-utility cohort must end
    with clearly higher mean gravity than the bottom-utility cohort."""
    mem = MemoryStore()
    facts = _drive_loop(mem)
    grav = {f.fact_id: f.gravity_score for f in facts}

    ranked = sorted(zip(_UTILITIES, facts), key=lambda t: t[0])
    low = [grav[f.fact_id] for _, f in ranked[:3]]
    high = [grav[f.fact_id] for _, f in ranked[-3:]]

    assert sum(high) / 3 > sum(low) / 3 + 0.1, (
        f"[{embed_provider}] high-utility mean {sum(high)/3:.3f} did not clear "
        f"low-utility mean {sum(low)/3:.3f} by margin"
    )


# Distinct-vocabulary topics so cosine attribution separates them under both
# the mock (word-hash) and real (semantic) embedders. Utilities shuffled vs
# creation order, same as above.
_DISTINCT = [
    ("postgres", "index strategy", "use a partial index", 0.10),
    ("react", "hydration mismatch", "defer non-critical components", 0.90),
    ("docker", "container networking", "use a user-defined bridge", 0.30),
    ("kafka", "topic partitioning", "key by tenant id", 0.70),
    ("redis", "key eviction", "set allkeys-lru policy", 0.00),
    ("nginx", "tls handshake", "enable ocsp stapling", 1.00),
]


def test_gravity_tracks_utility_through_real_cosine_attribution(embed_provider):
    """Interim attribution guard (partial #5 coverage).

    Unlike the deterministic tests above, this drives attribution through real
    ``query_memory`` cosine weighting — credit lands on whatever the query
    retrieves, co-attribution noise and all. It does NOT isolate the subtle
    "right fact in a failed session" case (that is full #5), but it removes the
    "attribution is entirely unguarded" gap: even with real, noisy attribution,
    gravity must still track assigned utility positively.
    """
    mem = MemoryStore()
    facts, utils = [], []
    for subj, pred, obj, u in _DISTINCT:
        facts.append(mem.add_fact(subj, pred, obj))
        utils.append(u)

    for rnd in range(_ROUNDS):
        for (subj, pred, _obj, u), f in zip(_DISTINCT, facts):
            sid = f"q-{subj}-{rnd}"
            mem.session_start(sid)
            # Real attribution: whatever cosine retrieves gets the credit.
            mem.query(f"{subj} {pred}", top_k=2, session_id=sid)
            for msg in _messages_for(u, f"{subj} {pred}"):
                mem.session_message(msg, session_id=sid)
            mem.session_close(sid)

    gravity = np.array([f.gravity_score for f in facts], dtype=float)
    utility = np.array(utils, dtype=float)
    corr = float(np.corrcoef(gravity, utility)[0, 1])
    # Looser than the clean-attribution test: real attribution smears credit
    # across co-retrieved facts, so we only require a clear positive signal.
    assert corr > 0.4, (
        f"[{embed_provider}] gravity lost the utility signal through real "
        f"attribution: corr={corr:+.3f}, gravity={np.round(gravity, 3)}"
    )


def _close_with(mem, sid, fact, sentiment):
    """One attributed session closed at a known sentiment (sentiment bypasses
    the heuristic, so the outcome is deterministic and embedding-free)."""
    mem.session_start(sid)
    mem.session_message("session", session_id=sid)  # non-empty guard
    mem._session_fact_ids = [fact.fact_id]
    mem.session_close(sid, sentiment=sentiment)


def test_gravity_follows_late_utility_after_sign_flip(embed_provider):
    """The order-dependence guard the sign-consistent tests are blind to.

    A fact resonant for several rounds then genuinely toxic for more must end
    BELOW a fact that stayed resonant — its gravity follows its LATE utility,
    not its early reputation. Under the old self-referential contrastive rule
    the early-good fact would freeze its reputation and resist the toxic turn.
    """
    mem = MemoryStore()
    good = mem.add_fact("stable", "stays", "useful")
    turned = mem.add_fact("turncoat", "was", "useful then not")

    for rnd in range(5):                       # both earn a resonant history
        _close_with(mem, f"r-good-{rnd}", good, "resonant")
        _close_with(mem, f"r-turn-{rnd}", turned, "resonant")
    for rnd in range(15):                      # good stays good, turned goes bad
        _close_with(mem, f"g-{rnd}", good, "resonant")
        _close_with(mem, f"t-{rnd}", turned, "toxic")

    assert turned.raw_avg_resonance < 0, "true record should have flipped toxic"
    assert turned.avg_resonance < good.avg_resonance, "gravity-side did not follow"
    assert good.gravity_score - turned.gravity_score > 0.1, (
        f"[{embed_provider}] turned fact did not follow its late utility: "
        f"good={good.gravity_score:.3f} turned={turned.gravity_score:.3f}"
    )
