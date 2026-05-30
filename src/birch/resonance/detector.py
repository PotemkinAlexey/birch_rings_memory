"""Session Closure Detector — combines behavioral, semantic, repetition into R score."""
from __future__ import annotations

from dataclasses import dataclass

from .behavioral import score_session_decay
from .repetition import RepetitionScore, score_repetition
from .semantic import SemanticScore, score_semantic_shift


@dataclass
class ResonanceResult:
    behavioral_score: float     # -1.0 … +1.0
    semantic_score: float       # -1.0 … +1.0
    repetition_score: float     # -1.0 … 0.0
    r: float                    # final resonance index -1.0 … +1.0
    label: str                  # "resonant" | "neutral" | "toxic"
    # Agreement among the three signals, in [0, 1]. 1.0 = they all pull the
    # same way (trust R fully); near 0 = they cancel (e.g. behavioral reads
    # toxic on grumpy tech vocabulary while semantic reads productive). The
    # gravity step is scaled by this so conflicted sessions barely move
    # gravity and a noisy signal can't compound through the feedback loop.
    # Defaults to 1.0 so explicit caller signals (r_override / sentiment),
    # which construct ResonanceResult directly, are treated as fully trusted.
    confidence: float = 1.0


# Weights
_W_BEHAVIORAL = 0.55
_W_SEMANTIC = 0.25
_W_REPETITION = 0.20


def compute_resonance(
    messages: list[str],
    start_vector: list[float] | None = None,
    end_vector: list[float] | None = None,
    all_vectors: list[list[float]] | None = None,
) -> ResonanceResult:
    """
    Compute resonance index R for a completed session.

    Args:
        messages:     All user messages in the session (chronological).
        start_vector: Embedding of the first message (optional).
        end_vector:   Embedding of the last message (optional).
        all_vectors:  Embeddings of all messages for repetition detection (optional).

    Returns:
        ResonanceResult with R in [-1.0, +1.0].
    """
    if not messages:
        return ResonanceResult(0.0, 0.0, 0.0, 0.0, "neutral")

    behavioral = score_session_decay(messages)

    semantic_result: SemanticScore = score_semantic_shift(
        start_text=messages[0],
        end_text=messages[-1],
        start_vector=start_vector,
        end_vector=end_vector,
    )
    semantic = semantic_result.score

    repetition_result: RepetitionScore = score_repetition(all_vectors or [])
    repetition = repetition_result.score

    contributions = [
        _W_BEHAVIORAL * behavioral,
        _W_SEMANTIC * semantic,
        _W_REPETITION * repetition,
    ]
    r_raw = sum(contributions)
    r = max(-1.0, min(1.0, r_raw))

    # Confidence has two independent dimensions; we need both.
    #
    # (a) AGREEMENT — do the voting signals point the same way?
    #     |Σ contributions| / Σ|contributions|: 1.0 when every signal pulls the
    #     same direction, → 0 as they cancel. This is the "declarative grumpy
    #     tech summary" guard: behavioral -0.8 ("error"/"broken" in a happy
    #     summary) against semantic +0.6 yields a low ratio, so gravity barely
    #     moves instead of confidently mislabelling.
    #
    # (b) CORROBORATION — how broad is the base the verdict rests on?
    #     Agreement alone is blind to single-signal dominance: behavioral -0.8
    #     with semantic and repetition both silent gives agreement 1.0, because
    #     one signal trivially "agrees" with itself. But a verdict carried by a
    #     lone regex match is exactly where mis-classification hides. We measure
    #     breadth with the participation ratio PR = 1 / Σ pᵢ² (pᵢ = |cᵢ|/Σ|c|):
    #     PR≈1 ⇒ one signal does all the work; PR≈2 ⇒ two balanced signals
    #     corroborate. Floor 0.75 so a lone clear signal still counts for most
    #     of its weight; full trust once a second balanced signal joins in.
    abs_total = sum(abs(c) for c in contributions)
    if abs_total <= 1e-9:
        confidence = 1.0
    else:
        agreement = abs(r_raw) / abs_total
        sq = sum((abs(c) / abs_total) ** 2 for c in contributions)
        participation = 1.0 / sq if sq > 1e-9 else 1.0
        corroboration = min(1.0, 0.75 + 0.25 * (participation - 1.0))
        confidence = max(0.0, min(1.0, agreement * corroboration))

    if r > 0.35:
        label = "resonant"
    elif r < -0.15:
        label = "toxic"
    else:
        label = "neutral"

    return ResonanceResult(
        behavioral_score=behavioral,
        semantic_score=semantic,
        repetition_score=repetition,
        r=round(r, 3),
        label=label,
        confidence=round(confidence, 3),
    )
