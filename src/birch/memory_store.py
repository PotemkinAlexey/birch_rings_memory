"""MemoryStore — unified entry point for the BirchKM memory system."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .fact import FactPassport
from .gravity import GravityEngine
from .black_hole import BlackHole
from .resonance.detector import compute_resonance
from .resonance.echo import EchoStore
from .resonance.embeddings import embed, embed_batch
from .resonance.cluster import bundle as _bundle, ClusterBundle
from .resonance.echo import StoredSession
from .storage import StorageBackend, SQLiteBackend


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

    def __init__(
        self,
        echo_k: int = 2,
        db_path: Optional[str | Path] = None,
        storage: Optional[StorageBackend] = None,
    ) -> None:
        self._engine = GravityEngine()
        self._hole = BlackHole()
        self._echo = EchoStore(default_k=echo_k)
        self._facts: dict[str, FactPassport] = {}
        if storage is not None:
            self._storage: Optional[StorageBackend] = storage
        elif db_path is not None:
            self._storage = SQLiteBackend(db_path)
        else:
            self._storage = None

        # Active session tracking
        self._session_messages: list[str] = []
        self._session_vectors: list[list[float]] = []
        self._session_fact_ids: list[str] = []
        self._session_id: Optional[str] = None

        if self._storage:
            self._load_from_storage()

    # ── Storage bootstrap ────────────────────────────────────────────────────

    def _load_from_storage(self) -> None:
        for fact in self._storage.load_facts():
            self._facts[fact.fact_id] = fact
            self._engine.register(fact)
        for from_id, to_id in self._storage.load_edges():
            self._engine.link(from_id, to_id)
        for row in self._storage.load_echo_sessions():
            centroids = row["centroids"]
            cb = ClusterBundle(centroids=centroids, k=len(centroids), inertia=0.0)
            self._echo._sessions[row["session_id"]] = StoredSession(
                session_id=row["session_id"],
                bundle=cb,
                r_score=row["r_score"],
                fact_ids=list(row.get("fact_ids", [])),
                echo_penalty=row.get("echo_penalty", 0.0),
            )

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
        if self._storage:
            self._storage.save_fact(fact)
        if self._session_id:
            self._session_fact_ids.append(fact.fact_id)
        return fact

    def link(self, from_id: str, to_id: str) -> None:
        self._engine.link(from_id, to_id)
        if self._storage:
            self._storage.save_edge(from_id, to_id)

    def deprecate(self, old_id: str, new_id: str) -> None:
        if old_id in self._facts:
            self._facts[old_id].deprecated_by = new_id
            if self._storage:
                self._storage.save_fact(self._facts[old_id])

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

        # Register session in echo store with the fact_ids it touched —
        # required so future echoes can apply a retroactive gravity penalty.
        if self._session_id and vecs:
            self._echo.record(
                self._session_id,
                vecs,
                result.r,
                fact_ids=list(self._session_fact_ids),
            )
            if self._storage:
                session_obj = self._echo.get(self._session_id)
                if session_obj:
                    self._storage.save_echo_session(
                        self._session_id,
                        session_obj.bundle.centroids,
                        session_obj.r_score,
                        time.time(),
                        fact_ids=session_obj.fact_ids,
                        echo_penalty=session_obj.echo_penalty,
                    )

        # Tick gravity and absorb dead facts
        migrations = self._engine.tick()
        absorbed = self._absorb_dead()

        # Persist updated gravity scores for all live facts
        if self._storage:
            for fact in self._facts.values():
                self._storage.save_fact(fact)

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
                if self._storage:
                    self._storage.delete_fact(fid)
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

        Side effects on every returned fact:
          - access_count is incremented (touch)
          - if a session is active, fact_id is attributed to it so the
            session's resonance later propagates back to its gravity.
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

        # Hawking emission: black hole returns facts AND removes them from
        # the singularity. We must re-register them in the live store and
        # persist the resurrection.
        if hawking:
            emitted = self._hole.hawking_emit(vec)
            for fact in emitted:
                sim = _cosine(vec, fact.vector)
                self._facts[fact.fact_id] = fact
                self._engine.register(fact)
                if self._storage:
                    self._storage.save_fact(fact)
                results.append(QueryResult(
                    fact=fact,
                    similarity=round(sim, 4),
                    source="hawking",
                ))

        results.sort(key=lambda r: r.similarity, reverse=True)
        top = results[:top_k]

        # Attribution + touch — only on what the caller actually receives.
        seen = set(self._session_fact_ids)
        for r in top:
            r.fact.touch()
            if self._session_id and r.fact.fact_id not in seen:
                self._session_fact_ids.append(r.fact.fact_id)
                seen.add(r.fact.fact_id)

        return top

    def check_echo(self, first_message: str) -> dict:
        """
        Check if a new session echoes a past unresolved problem.

        If echo is detected and a non-zero retroactive penalty is applied
        for the first time, the penalty is propagated to the gravity of
        every fact that the matched past session touched. Affected facts
        are re-persisted.
        """
        vec = embed(first_message)
        result = self._echo.detect_echo(vec)

        penalized_fact_ids: list[str] = []
        if result.label == "echo" and result.penalty != 0.0 and result.fact_ids:
            self._engine.apply_session_resonance(result.fact_ids, result.penalty)
            penalized_fact_ids = list(result.fact_ids)

            if self._storage:
                for fid in penalized_fact_ids:
                    if fid in self._facts:
                        self._storage.save_fact(self._facts[fid])
                # Persist the mutated echo session (r_score + echo_penalty).
                past = self._echo.get(result.matched_session_id)
                if past:
                    self._storage.save_echo_session(
                        past.session_id,
                        past.bundle.centroids,
                        past.r_score,
                        time.time(),
                        fact_ids=past.fact_ids,
                        echo_penalty=past.echo_penalty,
                    )

        return {
            "echo": result.label == "echo",
            "matched_session": result.matched_session_id,
            "similarity": result.similarity,
            "penalty": result.penalty,
            "penalized_fact_ids": penalized_fact_ids,
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
