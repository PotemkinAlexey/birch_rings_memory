"""MetaFact — compressed representative of several absorbed facts.

A MetaFact is what remains after a cluster of facts has fallen into the
black hole: a single centroid vector, the list of original SPO texts it
swallowed, an integer weight (how many facts it represents) and, once
LLM summarisation lands, a natural-language summary.

It lives in the black hole at ``layer = -1`` and starts with a non-zero
``gravity_score`` so it has weight from the moment it is created — even
before Hawking emission can promote it back to the live layers.

A MetaFact implements the same gravity-relevant surface as
``FactPassport`` — ``access_count``, ``last_accessed``, ``resonance_sum``,
``resonance_count``, ``apply_resonance()``, ``touch()`` — so the
``GravityEngine`` and the ``BlackHole`` can treat it polymorphically
through duck typing.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MetaFact:
    meta_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    vector: list[float] = field(default_factory=list)
    weight: int = 1                                  # how many facts it absorbed
    source_texts: list[str] = field(default_factory=list)  # "subject predicate object"
    source_fact_ids: list[str] = field(default_factory=list)  # lineage to originals
    summary: str = ""                                # populated by later LLM pass
    gravity_score: float = 0.30                      # starts non-zero — meta is dense
    created_at: float = field(default_factory=time.time)
    layer: int = -1                                  # lives in the black hole until emitted

    # ── Feedback-loop participation (same surface as FactPassport) ──────────
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    resonance_sum: float = 0.0
    resonance_count: int = 0
    recent_utility: float = 0.5
    forecast_stability: float = 0.5

    # ── Polymorphism shims for code that ducks on FactPassport ──────────────
    # FactPassport carries optional ttl/deprecated_by fields; MetaFacts cannot
    # currently be deprecated or expire, but exposing the same booleans keeps
    # GravityEngine and BlackHole._absorb_dead checks symmetric.
    @property
    def is_deprecated(self) -> bool:
        return False

    @property
    def is_expired(self) -> bool:
        return False

    @property
    def fact_id(self) -> str:
        """Alias so duck-typed callers can use .fact_id on either type."""
        return self.meta_id

    @property
    def avg_resonance(self) -> float:
        if self.resonance_count == 0:
            return 0.0
        return self.resonance_sum / self.resonance_count

    def touch(self) -> None:
        self.access_count += 1
        self.last_accessed = time.time()

    def apply_resonance(self, r: float) -> None:
        """Record that a session with resonance R used this MetaFact."""
        self.resonance_sum += r
        self.resonance_count += 1

    # ── Hawking emission helper ─────────────────────────────────────────────
    def gravity_on_emission(self, base: float = 0.30) -> float:
        """Gravity to assign on Hawking emission.

        A MetaFact carries the weight of every fact it absorbed, but we
        do not want a MetaFact with weight=50 to leap straight into the
        surface layer: each of its constituents was dead. A logarithmic
        bonus on top of the base gives a defensible, bounded scaling.
        """
        import math
        weight = max(1, int(self.weight))
        bonus = 0.10 * math.log10(weight)       # log10(1)=0, log10(10)=0.10
        return float(max(0.0, min(0.70, base + bonus)))

    # ── Serialisation ───────────────────────────────────────────────────────
    #
    # to_dict() returns a row already shaped for SQLite: scalars stay scalar,
    # list-valued fields are JSON-encoded so they fit a TEXT column. from_dict
    # is symmetric and tolerant — it accepts both the JSON-encoded form and
    # raw Python lists, so callers can build a MetaFact from a hand-written
    # dict in tests without having to JSON-dump first.

    def to_dict(self) -> dict[str, Any]:
        return {
            "meta_id": self.meta_id,
            "vector": json.dumps(self.vector),
            "weight": self.weight,
            "source_texts": json.dumps(self.source_texts),
            "source_fact_ids": json.dumps(self.source_fact_ids),
            "summary": self.summary,
            "gravity_score": self.gravity_score,
            "created_at": self.created_at,
            "layer": self.layer,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed,
            "resonance_sum": self.resonance_sum,
            "resonance_count": self.resonance_count,
            "recent_utility": self.recent_utility,
            "forecast_stability": self.forecast_stability,
        }

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "MetaFact":
        return cls(
            meta_id=row["meta_id"],
            vector=_load_list(row.get("vector"), float),
            weight=int(row.get("weight", 1)),
            source_texts=_load_list(row.get("source_texts"), str),
            source_fact_ids=_load_list(row.get("source_fact_ids"), str),
            summary=row.get("summary", "") or "",
            gravity_score=float(row.get("gravity_score", 0.30)),
            created_at=float(row.get("created_at", time.time())),
            layer=int(row.get("layer", -1)),
            access_count=int(row.get("access_count", 0)),
            last_accessed=float(row.get("last_accessed", time.time())),
            resonance_sum=float(row.get("resonance_sum", 0.0)),
            resonance_count=int(row.get("resonance_count", 0)),
            recent_utility=float(row.get("recent_utility", 0.5)),
            forecast_stability=float(row.get("forecast_stability", 0.5)),
        )


def _load_list(value: Any, cast) -> list:
    """Accept a JSON-encoded string, a Python list, or None.

    Tolerant on item-level cast failures too — a list whose JSON
    parsed cleanly but contains values the caster can't accept
    (e.g. ``[1, "oops"]`` with ``float`` cast) returns ``[]``
    rather than raising. The SQLite loader catches the row-level
    case via its pre-validation, but direct callers (tests,
    in-memory migrations) get the same forgiving contract here
    — symmetric with FactPassport loading via ``_safe_vector``."""
    if value is None or value == "":
        return []
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            return []
    else:
        parsed = value
    if not isinstance(parsed, list):
        return []
    try:
        out = [cast(x) for x in parsed]
    except (TypeError, ValueError):
        return []
    # NaN / Infinity sneak through float() — they're valid floats per
    # Python but poison numpy cosine similarity (every comparison
    # becomes NaN, ranking undefined). FactPassport vectors go through
    # _safe_vector in the SQLite backend which checks math.isfinite;
    # MetaFacts used to skip that gate. Symmetric defence here.
    if cast is float:
        import math as _math
        if any(not _math.isfinite(x) for x in out):
            return []
    return out
