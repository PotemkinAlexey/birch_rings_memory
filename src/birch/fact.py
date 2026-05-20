"""FactPassport — atomic unit of knowledge in BirchKM."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FactPassport:
    subject: str
    predicate: str
    object: str

    fact_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    vector: list[float] = field(default_factory=list)

    gravity_score: float = 0.5      # starts neutral, drifts with usage
    layer: int = 1                  # 0=surface, 1=kinetic, 2=core

    created_at: float = field(default_factory=time.time)
    ttl: Optional[float] = None     # None = no expiry

    source_session: Optional[str] = None
    deprecated_by: Optional[str] = None   # fact_id that superseded this

    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    resonance_sum: float = 0.0      # cumulative R from sessions that used this
    resonance_count: int = 0        # how many sessions contributed

    @property
    def is_deprecated(self) -> bool:
        return self.deprecated_by is not None

    @property
    def is_expired(self) -> bool:
        return self.ttl is not None and time.time() > self.ttl

    @property
    def avg_resonance(self) -> float:
        if self.resonance_count == 0:
            return 0.0
        return self.resonance_sum / self.resonance_count

    def touch(self) -> None:
        self.access_count += 1
        self.last_accessed = time.time()

    def apply_resonance(self, r: float) -> None:
        """Record that a session with resonance R used this fact."""
        self.resonance_sum += r
        self.resonance_count += 1

    def __repr__(self) -> str:
        return (
            f"Fact({self.fact_id!r}: {self.subject!r} {self.predicate!r} "
            f"{self.object!r} g={self.gravity_score:.2f} layer={self.layer})"
        )
