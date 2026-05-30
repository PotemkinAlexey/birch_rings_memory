"""Shared dataclasses for the MemoryStore package.

Lives in its own module so the per-area mixins can import it without
introducing an import cycle through ``_base`` (which itself imports
every mixin to assemble ``MemoryStore``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..fact import FactPassport
from ..meta_fact import MetaFact


@dataclass
class QueryResult:
    """Polymorphic query hit — either a FactPassport or a MetaFact.

    Exactly one of ``fact`` and ``meta`` is non-None. Legacy callers that
    read ``r.fact.fact_id`` keep working for fact hits; new callers branch
    on ``r.kind`` (``"fact"`` or ``"meta"``) and read the right field.

    ``similarity`` holds the **raw** cosine score, not rounded. Internal
    consumers (session attribution, gravity weighting, ranking) use the
    full-precision value; only ``to_mcp_dict`` rounds to 4 decimals for
    display. Earlier code rounded at construction, which silently fed a
    truncated weight into the resonance feedback loop — fine for display
    but technically dishonest for "round only on output".
    """
    similarity: float
    source: str     # "surface" | "kinetic" | "core" | "hawking" | "hawking_meta"
    fact: Optional[FactPassport] = None
    meta: Optional[MetaFact] = None

    @property
    def kind(self) -> str:
        return "meta" if self.meta is not None else "fact"

    @property
    def body_id(self) -> str:
        if self.meta is not None:
            return self.meta.meta_id
        if self.fact is not None:
            return self.fact.fact_id
        return ""

    def to_mcp_dict(self) -> dict:
        """JSON-serializable payload for MCP ``query_memory`` consumers."""
        base: dict = {
            "kind": self.kind,
            "body_id": self.body_id,
            "similarity": round(self.similarity, 4),
            "source": self.source,
        }
        if self.meta is not None:
            m = self.meta
            base.update({
                "meta_id": m.meta_id,
                "weight": m.weight,
                "source_texts": list(m.source_texts),
                "source_fact_ids": list(m.source_fact_ids),
                "summary": m.summary or "",
                "layer": m.layer,
                "gravity_score": round(m.gravity_score, 3),
            })
            return base
        if self.fact is not None:
            f = self.fact
            base.update({
                "fact_id": f.fact_id,
                "subject": f.subject,
                "predicate": f.predicate,
                "object": f.object,
                "layer": f.layer,
                "gravity_score": round(f.gravity_score, 3),
            })
            return base
        return base


@dataclass
class SessionContext:
    """Per-session mutable state. Two agents = two independent contexts."""
    session_id: str
    messages: list[str] = field(default_factory=list)
    vectors: list[list[float]] = field(default_factory=list)
    # fact_id → relevance weight in [0, 1] for this session.
    facts: dict[str, float] = field(default_factory=dict)
    # Deferred-echo marker set at open, resolved at close. Shape
    # {"matched_session_id", "similarity"} or None. Not persisted — a restart
    # mid-session just drops it (echo is best-effort, not an invariant).
    pending_echo: Optional[dict] = None
