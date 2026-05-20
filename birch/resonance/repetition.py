"""Repetition detector — catches circular sessions via pairwise similarity."""
from __future__ import annotations

import math
from dataclasses import dataclass


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


@dataclass
class RepetitionScore:
    avg_pairwise_similarity: float
    score: float    # -1.0 … 0.0 (only penalizes, never rewards)


def score_repetition(vectors: list[list[float]]) -> RepetitionScore:
    """
    Compute pairwise cosine similarity across all message vectors.

    High average similarity = messages are semantically circular = penalty.
    Returns score in [-1.0, 0.0] — this metric only penalizes.
    """
    if len(vectors) < 2:
        return RepetitionScore(0.0, 0.0)

    pairs = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            pairs.append(_cosine(vectors[i], vectors[j]))

    avg = sum(pairs) / len(pairs)

    # avg > 0.80: clear circular loop → strong penalty
    # avg > 0.70: likely rephrasing → moderate penalty
    # below 0.70: conversation moved → no penalty
    if avg > 0.80:
        penalty = -0.8
    elif avg > 0.70:
        penalty = -0.4
    else:
        penalty = 0.0

    return RepetitionScore(avg_pairwise_similarity=round(avg, 4), score=penalty)
