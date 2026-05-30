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
