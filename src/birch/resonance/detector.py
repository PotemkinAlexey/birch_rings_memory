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

    # Confidence = signal agreement. |Σ contributions| / Σ|contributions| is 1.0
    # when every signal pulls the same direction and falls toward 0 as they
    # cancel. This is exactly the "declarative grumpy tech summary" guard: a
    # behavioral score of -0.8 (matched "error"/"broken" in a happy summary)
    # against a +0.6 semantic score yields a low agreement ratio, so the
    # session moves gravity only weakly instead of confidently mislabelling.
    abs_total = sum(abs(c) for c in contributions)
    confidence = abs(r_raw) / abs_total if abs_total > 1e-9 else 1.0
    confidence = max(0.0, min(1.0, confidence))

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
