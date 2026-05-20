"""Semantic Shift metric — measures how the conversation trajectory moved."""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class SemanticScore:
    cosine_similarity: float   # similarity between start and end vectors
    specificity_delta: float   # how much more specific the end message is
    score: float               # combined score in [-1.0, +1.0]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _token_specificity(text: str) -> float:
    """Rough proxy for specificity: unique tokens / total tokens."""
    tokens = text.lower().split()
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def score_semantic_shift(
    start_text: str,
    end_text: str,
    start_vector: list[float] | None = None,
    end_vector: list[float] | None = None,
) -> SemanticScore:
    """
    Score how productively the conversation moved.

    If vectors are provided: use cosine similarity.
    Specificity delta works on raw text always (no embedding needed).

    High similarity + higher specificity at end = good (topic narrowed down).
    High similarity + same/lower specificity = stuck (user rephrasing).
    """
    cosine = 0.0
    if start_vector and end_vector:
        cosine = _cosine(start_vector, end_vector)

    spec_start = _token_specificity(start_text)
    spec_end = _token_specificity(end_text)
    specificity_delta = spec_end - spec_start  # positive = got more specific

    # Scoring logic:
    # - moved to more specific territory (delta > 0) → positive signal
    # - stayed on same topic but got specific → good
    # - stayed on same topic, same vagueness → stuck
    if start_vector and end_vector:
        if cosine > 0.85 and specificity_delta < 0.05:
            # High similarity, no specificity gain = stuck
            combined = -0.4
        elif cosine > 0.85 and specificity_delta >= 0.05:
            # High similarity but specificity grew = productive narrowing
            combined = +0.6
        else:
            # Topic shifted — could be good or bad, use specificity as tiebreak
            combined = 0.2 + specificity_delta * 0.5
    else:
        # No vectors — fall back to specificity delta only
        combined = max(-1.0, min(1.0, specificity_delta * 2.0))

    return SemanticScore(
        cosine_similarity=cosine,
        specificity_delta=specificity_delta,
        score=max(-1.0, min(1.0, combined)),
    )
