"""MemoryStore — unified entry point for the BirchKM memory system."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional

from .fact import FactPassport
from .gravity import GravityEngine
from .black_hole import BlackHole
from .resonance.detector import compute_resonance
from .resonance.echo import EchoStore
from .resonance.embeddings import embed, embed_batch
from .resonance.cluster import bundle as _bundle


# Gravity floor — facts below this after tick fall into the black hole
_ABSORPTION_THRESHOLD = 0.10


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


@dataclass
class QueryResult:
    fact: FactPassport
    similarity: float
    source: str     # "surface" | "kinetic" | "core" | "hawking"


class MemoryStore:
    """
    Three-layer memory with black hole sink and Hawking emission.

    Layers:
      0 — surface  (gravity > 0.70, hot facts)
      1 — kinetic  (gravity 0.30–0.70, working memory)
      2 — core     (gravity < 0.30, cold archive)
     -1 — black hole (gravity < 0.10 after tick, absorbed)
    """

    def __init__(self, echo_k: int = 2) -> None:
        self._engine = GravityEngine()
        self._hole = BlackHole()
        self._echo = EchoStore(default_k=echo_k)
        self._facts: dict[str, FactPassport] = {}

        # Active session tracking
        self._session_messages: list[str] = []
        self._session_vectors: list[list[float]] = []
        self._session_fact_ids: list[str] = []
        self._session_id: Optional[str] = None

    # ── Fact management ─────────────────────────────────────────────────────

    def add_fact(
        self,
        subject: str,
        predicate: str,
        obj: str,
        layer: int = 1,
    ) -> FactPassport:
        """Create, embed, and register a new fact."""
        fact = FactPassport(
            subject=subject,
            predicate=predicate,
            object=obj,
            layer=layer,
            source_session=self._session_id,
        )
        fact.vector = embed(f"{subject} {predicate} {obj}")
        self._facts[fact.fact_id] = fact
        self._engine.register(fact)
        if self._session_id:
            self._session_fact_ids.append(fact.fact_id)
        return fact

    def link(self, from_id: str, to_id: str) -> None:
        self._engine.link(from_id, to_id)

    def deprecate(self, old_id: str, new_id: str) -> None:
        if old_id in self._facts:
            self._facts[old_id].deprecated_by = new_id

    # ── Session lifecycle ────────────────────────────────────────────────────

    def session_start(self, session_id: str) -> None:
        self._session_id = session_id
        self._session_messages = []
        self._session_vectors = []
        self._session_fact_ids = []

    def session_message(self, text: str) -> None:
        """Record a user message in the current session."""
        self._session_messages.append(text)
        self._session_vectors.append(embed(text))

    def session_close(self) -> dict:
        """
        Close session: compute resonance, propagate R to facts,
        record echo bundle, tick gravity, absorb dead facts.
        """
        if not self._session_messages:
            return {}

        vecs = self._session_vectors
        result = compute_resonance(
            self._session_messages,
            start_vector=vecs[0],
            end_vector=vecs[-1],
            all_vectors=vecs,
        )

        # Propagate R to all facts used in this session
        self._engine.apply_session_resonance(self._session_fact_ids, result.r)

        # Touch facts that were accessed
        for fid in self._session_fact_ids:
            if fid in self._facts:
                self._facts[fid].touch()

        # Register session in echo store
        if self._session_id and vecs:
            self._echo.record(self._session_id, vecs, result.r)

        # Tick gravity and absorb dead facts
        migrations = self._engine.tick()
        absorbed = self._absorb_dead()

        summary = {
            "session_id": self._session_id,
            "r": result.r,
            "label": result.label,
            "migrations": migrations,
            "absorbed": absorbed,
        }

        self._session_id = None
        self._session_messages = []
        self._session_vectors = []
        self._session_fact_ids = []

        return summary

    def _absorb_dead(self) -> list[str]:
        """Send facts with gravity below threshold into the black hole."""
        absorbed = []
        for fid, fact in list(self._facts.items()):
            if fact.is_deprecated or fact.is_expired:
                self._hole.absorb(fact)
                del self._facts[fid]
                absorbed.append(fid)
            elif fact.gravity_score < _ABSORPTION_THRESHOLD:
                self._hole.absorb(fact)
                del self._facts[fid]
                absorbed.append(fid)
        return absorbed

    # ── Retrieval ────────────────────────────────────────────────────────────

    def query(
        self,
        text: str,
        top_k: int = 5,
        min_layer: int = 0,
        max_layer: int = 2,
        hawking: bool = True,
    ) -> list[QueryResult]:
        """
        Retrieve relevant facts by cosine similarity.

        Searches live layers first. If hawking=True, also attempts
        Hawking emission from the black hole for extreme matches.
        """
        vec = embed(text)
        results: list[QueryResult] = []

        # Search live facts
        layer_labels = {0: "surface", 1: "kinetic", 2: "core"}
        for fact in self._facts.values():
            if not fact.vector:
                continue
            if not (min_layer <= fact.layer <= max_layer):
                continue
            sim = _cosine(vec, fact.vector)
            results.append(QueryResult(
                fact=fact,
                similarity=round(sim, 4),
                source=layer_labels.get(fact.layer, "kinetic"),
            ))

        # Hawking emission from black hole
        if hawking:
            emitted = self._hole.hawking_emit(vec)
            for fact in emitted:
                sim = _cosine(vec, fact.vector)
                # Re-register emitted fact
                self._facts[fact.fact_id] = fact
                self._engine.register(fact)
                results.append(QueryResult(
                    fact=fact,
                    similarity=round(sim, 4),
                    source="hawking",
                ))

        results.sort(key=lambda r: r.similarity, reverse=True)
        return results[:top_k]

    def check_echo(self, first_message: str) -> dict:
        """Check if new session echoes a past unresolved problem."""
        vec = embed(first_message)
        result = self._echo.detect_echo(vec)
        return {
            "echo": result.label == "echo",
            "matched_session": result.matched_session_id,
            "similarity": result.similarity,
            "penalty": result.penalty,
        }

    # ── Status ───────────────────────────────────────────────────────────────

    @property
    def stats(self) -> dict:
        layers = {0: 0, 1: 0, 2: 0}
        for f in self._facts.values():
            layers[f.layer] = layers.get(f.layer, 0) + 1
        return {
            "surface": layers[0],
            "kinetic": layers[1],
            "core": layers[2],
            "black_hole_mass": self._hole.mass,
            "hawking_emissions": self._hole.total_emissions,
            "total_live": len(self._facts),
        }
