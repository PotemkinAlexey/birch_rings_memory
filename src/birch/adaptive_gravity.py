"""AdaptiveWeights — the pre-resonance gravity weights, learned not guessed.

The gravity formula has the shape

    gravity = w_freshness · freshness
            + w_access    · access
            + w_graph     · graph
            + w_utility   · recent_utility    (EWMA of closure-weighted R)
            + 0.35        · resonance         (fixed: resonance is observation)

Four of those weights were hand-set magic numbers — the personalised
ones, "what predicts a fact's value *before* the user has reacted to
it this session". This module makes them learned from the user's own
resonance feedback: every closed session contributes one regularised
SGD step toward predicting realised value from
(freshness, access, graph, recent_utility) measured before that
resonance landed.

The learner is deliberately simple — a regularised linear model,
budgeted and clamped — so the weights stay legible (you can always
print them) and bounded (gravity stays in [0, 1]). Behaviour at zero
training data is identical to the hand-tuned prior.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar


@dataclass
class AdaptiveWeights:
    """The four pre-resonance weights — learned from resonance feedback."""

    w_freshness: float
    w_access: float
    w_graph: float
    w_utility: float
    train_count: int = 0

    # Hand-set prior — the weights at zero data.
    PRIOR: ClassVar[tuple[float, float, float, float]] = (0.30, 0.15, 0.10, 0.10)
    # Pre-resonance budget — the four weights sum to BUDGET. Resonance
    # carries the remaining 0.35, fixed by design.
    BUDGET: ClassVar[float] = 0.65
    LR: ClassVar[float] = 0.05       # SGD step size
    REG: ClassVar[float] = 0.02      # pull toward the prior each step

    @classmethod
    def from_prior(cls) -> AdaptiveWeights:
        return cls(*cls.PRIOR)

    def predict(
        self,
        freshness: float,
        access: float,
        graph: float,
        utility: float,
    ) -> float:
        return (
            self.w_freshness * freshness
            + self.w_access * access
            + self.w_graph * graph
            + self.w_utility * utility
        )

    def update(
        self,
        freshness: float,
        access: float,
        graph: float,
        utility: float,
        target: float,
    ) -> None:
        """One regularised SGD step toward ``target`` ∈ [0, 1].

        ``target`` is the realised value of the fact for the user — for
        the first-resonance training event, ``(R + 1) / 2`` of the
        session's R.
        """
        err = target - self.predict(freshness, access, graph, utility)
        self.w_freshness += self.LR * err * freshness
        self.w_access    += self.LR * err * access
        self.w_graph     += self.LR * err * graph
        self.w_utility   += self.LR * err * utility
        # Pull each weight back toward its prior — regularisation.
        self.w_freshness += self.REG * (self.PRIOR[0] - self.w_freshness)
        self.w_access    += self.REG * (self.PRIOR[1] - self.w_access)
        self.w_graph     += self.REG * (self.PRIOR[2] - self.w_graph)
        self.w_utility   += self.REG * (self.PRIOR[3] - self.w_utility)
        # Clamp non-negative.
        self.w_freshness = max(0.0, self.w_freshness)
        self.w_access    = max(0.0, self.w_access)
        self.w_graph     = max(0.0, self.w_graph)
        self.w_utility   = max(0.0, self.w_utility)
        # Renormalise to the budget so gravity stays in [0, 1].
        total = (
            self.w_freshness + self.w_access + self.w_graph + self.w_utility
        )
        if total > 0.0:
            scale = self.BUDGET / total
            self.w_freshness *= scale
            self.w_access *= scale
            self.w_graph *= scale
            self.w_utility *= scale
        else:
            (self.w_freshness, self.w_access,
             self.w_graph, self.w_utility) = self.PRIOR
        self.train_count += 1

    def as_dict(self) -> dict[str, float | int]:
        """Readable snapshot — handy for stats and debugging."""
        return {
            "w_freshness": round(self.w_freshness, 4),
            "w_access": round(self.w_access, 4),
            "w_graph": round(self.w_graph, 4),
            "w_utility": round(self.w_utility, 4),
            "train_count": self.train_count,
        }
