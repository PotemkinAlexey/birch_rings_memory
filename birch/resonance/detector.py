"""Session Closure Detector — combines behavioral and semantic signals into R score."""
from __future__ import annotations

from dataclasses import dataclass

from .behavioral import score_session_decay
from .semantic import SemanticScore, score_semantic_shift


@dataclass
class ResonanceResult:
    behavioral_score: float     # -1.0 … +1.0
    semantic_score: float       # -1.0 … +1.0
    r: float                    # final resonance index -1.0 … +1.0
    label: str                  # "resonant" | "neutral" | "toxic"


# Weights — behavioral carries more because it's more reliable without embeddings
_W_BEHAVIORAL = 0.65
_W_SEMANTIC = 0.35


def compute_resonance(
    messages: list[str],
    start_vector: list[float] | None = None,
    end_vector: list[float] | None = None,
) -> ResonanceResult:
    """
    Compute resonance index R for a completed session.

    Args:
        messages:     All user messages in the session (chronological).
        start_vector: Embedding of the first message (optional).
        end_vector:   Embedding of the last message (optional).

    Returns:
        ResonanceResult with R in [-1.0, +1.0].
    """
    if not messages:
        return ResonanceResult(0.0, 0.0, 0.0, "neutral")

    behavioral = score_session_decay(messages)

    semantic_result: SemanticScore = score_semantic_shift(
        start_text=messages[0],
        end_text=messages[-1],
        start_vector=start_vector,
        end_vector=end_vector,
    )
    semantic = semantic_result.score

    r = _W_BEHAVIORAL * behavioral + _W_SEMANTIC * semantic
    r = max(-1.0, min(1.0, r))

    if r > 0.35:
        label = "resonant"
    elif r < -0.2:
        label = "toxic"
    else:
        label = "neutral"

    return ResonanceResult(
        behavioral_score=behavioral,
        semantic_score=semantic,
        r=round(r, 3),
        label=label,
    )
