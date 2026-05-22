"""Repetition detector — catches circular sessions via centroid dispersion."""
from __future__ import annotations

from dataclasses import dataclass

from .centroid import centroid, dispersion


@dataclass
class RepetitionScore:
    avg_dispersion: float
    score: float    # -1.0 … 0.0 (only penalizes, never rewards)


def score_repetition(vectors: list[list[float]]) -> RepetitionScore:
    """
    Detect circular sessions via dispersion around the centroid.

    Low dispersion = messages semantically stuck in the same place = penalty.
    High dispersion = conversation moved = no penalty.

    Thresholds (cosine distance from centroid):
      < 0.05 → tight loop   → -0.8
      < 0.12 → rephrasing   → -0.4
      >= 0.12 → moving       → 0.0
    """
    if len(vectors) < 2:
        return RepetitionScore(0.0, 0.0)

    center = centroid(vectors)
    avg_disp = dispersion(vectors, center)

    if avg_disp < 0.05:
        penalty = -0.8
    elif avg_disp < 0.12:
        penalty = -0.4
    else:
        penalty = 0.0

    return RepetitionScore(avg_dispersion=round(avg_disp, 4), score=penalty)
