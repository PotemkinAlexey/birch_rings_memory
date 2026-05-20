"""Black Hole — irreversible sink with Hawking emission for extreme retrieval."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .fact import FactPassport

# Similarity threshold for Hawking emission — only the strongest queries pull facts back
_HAWKING_THRESHOLD = 0.95
# Gravity assigned to emitted facts — they return weakened
_HAWKING_GRAVITY = 0.30


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


@dataclass
class SingularityRecord:
    fact: "FactPassport"
    absorbed_at: float = field(default_factory=time.time)
    emission_count: int = 0         # how many times Hawking-emitted


class BlackHole:
    """
    Irreversible sink for facts that have lost all gravity.

    Facts fall in and stay — except when a query is semantically
    extreme enough to trigger Hawking emission.
    """

    def __init__(self, hawking_threshold: float = _HAWKING_THRESHOLD) -> None:
        self._singularity: dict[str, SingularityRecord] = {}
        self._hawking_threshold = hawking_threshold
        self._total_emissions = 0   # cumulative, survives record removal

    def absorb(self, fact: "FactPassport") -> None:
        """Pull a fact across the event horizon. Irreversible."""
        fact.layer = -1     # sentinel: beyond core
        self._singularity[fact.fact_id] = SingularityRecord(fact=fact)

    def hawking_emit(
        self,
        query_vector: list[float],
    ) -> list["FactPassport"]:
        """
        Attempt Hawking emission: return facts similar enough to the query.

        Emitted facts are restored to layer=1 (kinetic) with gravity reset
        to _HAWKING_GRAVITY and removed from the singularity. The caller
        is responsible for re-registering them in the live store.
        """
        to_emit: list[str] = []
        for fid, rec in self._singularity.items():
            if not rec.fact.vector:
                continue
            sim = _cosine(query_vector, rec.fact.vector)
            if sim >= self._hawking_threshold:
                to_emit.append(fid)

        emitted: list["FactPassport"] = []
        for fid in to_emit:
            rec = self._singularity.pop(fid)
            rec.fact.gravity_score = _HAWKING_GRAVITY
            rec.fact.layer = 1
            self._total_emissions += 1
            emitted.append(rec.fact)
        return emitted

    @property
    def mass(self) -> int:
        """Number of facts currently inside the singularity."""
        return len(self._singularity)

    @property
    def total_emissions(self) -> int:
        return self._total_emissions

    def __contains__(self, fact_id: str) -> bool:
        return fact_id in self._singularity
