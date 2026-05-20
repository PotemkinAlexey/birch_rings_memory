"""MemoryStore — unified entry point for the BirchKM memory system."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .fact import FactPassport
from .gravity import GravityEngine
from .black_hole import BlackHole
from .resonance.detector import compute_resonance
from .resonance.echo import EchoStore
from .resonance.embeddings import embed
from .resonance.cluster import ClusterBundle
from .resonance.echo import StoredSession
from .storage import StorageBackend, SQLiteBackend
from .vector_index import VectorIndex


# Gravity floor — facts below this after tick fall into the black hole
_ABSORPTION_THRESHOLD = 0.10


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
        # Normalised SPO → fact_id, for cheap duplicate detection in add_fact.
        self._spo_index: dict[tuple[str, str, str], str] = {}
        # Numpy-backed cosine index, kept in sync with live facts.
        self._index = VectorIndex()
        if storage is not None:
            self._storage: Optional[StorageBackend] = storage
        elif db_path is not None:
            self._storage = SQLiteBackend(db_path)
        else:
            self._storage = None

        # Active session tracking
        self._session_messages: list[str] = []
        self._session_vectors: list[list[float]] = []
        # fact_id → relevance weight in [0, 1] for this session.
        # We keep the MAX similarity observed across all queries that
        # returned each fact; explicit add_fact calls pin weight=1.0.
        self._session_facts: dict[str, float] = {}
        self._session_id: Optional[str] = None

        if self._storage:
            self._load_from_storage()

    # ── Storage bootstrap ────────────────────────────────────────────────────

    # Back-compat shim — older tests and callers read/write this as a list.
    @property
    def _session_fact_ids(self) -> list[str]:
        return list(self._session_facts.keys())

    @_session_fact_ids.setter
    def _session_fact_ids(self, value) -> None:
        if isinstance(value, dict):
            self._session_facts = {fid: float(w) for fid, w in value.items()}
        else:
            self._session_facts = {fid: 1.0 for fid in value}

    def _attribute_fact(self, fact_id: str, weight: float) -> None:
        """Tag a fact to the active session, keeping the max weight seen."""
        if not self._session_id:
            return
        clipped = max(0.0, min(1.0, float(weight)))
        prev = self._session_facts.get(fact_id, 0.0)
        if clipped > prev:
            self._session_facts[fact_id] = clipped

    @staticmethod
    def _normalize_spo(subject: str, predicate: str, obj: str) -> tuple[str, str, str]:
        return (
            " ".join(subject.lower().split()),
            " ".join(predicate.lower().split()),
            " ".join(obj.lower().split()),
        )

    def _load_from_storage(self) -> None:
        for fact in self._storage.load_facts():
            self._facts[fact.fact_id] = fact
            self._engine.register(fact)
            self._index.add(fact.fact_id, fact.vector)
            if not fact.is_deprecated:
                key = self._normalize_spo(fact.subject, fact.predicate, fact.object)
                self._spo_index.setdefault(key, fact.fact_id)
        for from_id, to_id in self._storage.load_edges():
            self._engine.link(from_id, to_id)
        for row in self._storage.load_echo_sessions():
            centroids = row["centroids"]
            cb = ClusterBundle(centroids=centroids, k=len(centroids), inertia=0.0)
            self._echo._sessions[row["session_id"]] = StoredSession(
                session_id=row["session_id"],
                bundle=cb,
                r_score=row["r_score"],
                fact_weights=dict(row.get("fact_weights", {})),
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
        """
        Create, embed, and register a new fact.

        If an identical (case-insensitive, whitespace-normalised) SPO triple
        already lives in the store, return the existing fact instead of
        creating a duplicate. The existing fact is touched and attributed
        to the active session, so the caller's intent (it was used here)
        still propagates to gravity.
        """
        key = self._normalize_spo(subject, predicate, obj)
        existing_id = self._spo_index.get(key)
        if existing_id and existing_id in self._facts:
            existing = self._facts[existing_id]
            existing.touch()
            if self._storage:
                self._storage.save_fact(existing)
            self._attribute_fact(existing_id, 1.0)
            return existing

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
        self._index.add(fact.fact_id, fact.vector)
        self._spo_index[key] = fact.fact_id
        if self._storage:
            self._storage.save_fact(fact)
        self._attribute_fact(fact.fact_id, 1.0)
        return fact

    def link(self, from_id: str, to_id: str) -> None:
        self._engine.link(from_id, to_id)
        if self._storage:
            self._storage.save_edge(from_id, to_id)

    def deprecate(self, old_id: str, new_id: str) -> None:
        if old_id in self._facts:
            old = self._facts[old_id]
            old.deprecated_by = new_id
            # A deprecated fact is no longer the canonical bearer of its SPO.
            key = self._normalize_spo(old.subject, old.predicate, old.object)
            if self._spo_index.get(key) == old_id:
                del self._spo_index[key]
            if self._storage:
                self._storage.save_fact(old)

    # ── Session lifecycle ────────────────────────────────────────────────────

    def session_start(self, session_id: str) -> None:
        self._session_id = session_id
        self._session_messages = []
        self._session_vectors = []
        self._session_facts = {}

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

        # Propagate R to facts used in this session, weighted by how
        # relevant each fact was to the session's queries.
        self._engine.apply_session_resonance(self._session_facts, result.r)

        # Touch facts that were accessed
        for fid in self._session_facts:
            if fid in self._facts:
                self._facts[fid].touch()

        # Register session in echo store with the fact weights it touched —
        # required so future echoes can apply a retroactive gravity penalty
        # scaled by how relevant each fact was.
        if self._session_id and vecs:
            self._echo.record(
                self._session_id,
                vecs,
                result.r,
                fact_weights=dict(self._session_facts),
            )
            if self._storage:
                session_obj = self._echo.get(self._session_id)
                if session_obj:
                    self._storage.save_echo_session(
                        self._session_id,
                        session_obj.bundle.centroids,
                        session_obj.r_score,
                        time.time(),
                        fact_weights=session_obj.fact_weights,
                        echo_penalty=session_obj.echo_penalty,
                    )

        # Tick gravity and absorb dead facts
        migrations = self._engine.tick()
        absorbed = self._absorb_dead()

        # Persist updated gravity scores for all live facts in one transaction.
        if self._storage:
            self._storage.save_facts(list(self._facts.values()))

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
        self._session_facts = {}

        return summary

    def _absorb_dead(self) -> list[str]:
        """Send facts with gravity below threshold into the black hole."""
        absorbed = []
        for fid, fact in list(self._facts.items()):
            if fact.is_deprecated or fact.is_expired:
                self._hole.absorb(fact)
                del self._facts[fid]
                self._index.remove(fid)
                self._drop_from_spo_index(fact)
                absorbed.append(fid)
            elif fact.gravity_score < _ABSORPTION_THRESHOLD:
                self._hole.absorb(fact)
                del self._facts[fid]
                self._index.remove(fid)
                self._drop_from_spo_index(fact)
                if self._storage:
                    self._storage.delete_fact(fid)
                absorbed.append(fid)
        return absorbed

    def _drop_from_spo_index(self, fact: FactPassport) -> None:
        key = self._normalize_spo(fact.subject, fact.predicate, fact.object)
        if self._spo_index.get(key) == fact.fact_id:
            del self._spo_index[key]

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

        # Search live facts via the numpy-backed index. Single matmul over
        # the whole live store; layer filter applied afterwards.
        layer_labels = {0: "surface", 1: "kinetic", 2: "core"}
        sims = self._index.all_similarities(vec)
        for fid, sim in sims.items():
            fact = self._facts.get(fid)
            if fact is None:
                continue
            if not (min_layer <= fact.layer <= max_layer):
                continue
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
                self._facts[fact.fact_id] = fact
                self._engine.register(fact)
                self._index.add(fact.fact_id, fact.vector)
                if not fact.is_deprecated:
                    key = self._normalize_spo(fact.subject, fact.predicate, fact.object)
                    self._spo_index.setdefault(key, fact.fact_id)
                if self._storage:
                    self._storage.save_fact(fact)
                sim = VectorIndex.similarity(vec, fact.vector)
                results.append(QueryResult(
                    fact=fact,
                    similarity=round(sim, 4),
                    source="hawking",
                ))

        results.sort(key=lambda r: r.similarity, reverse=True)
        top = results[:top_k]

        # Attribution + touch — only on what the caller actually receives.
        # Weight is the similarity itself (clipped to [0, 1]); a fact
        # returned with cosine 0.95 ends up nine times more sensitive
        # to the session R than one returned at 0.10.
        for r in top:
            r.fact.touch()
            self._attribute_fact(r.fact.fact_id, r.similarity)

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
        if result.label == "echo" and result.penalty != 0.0 and result.fact_weights:
            self._engine.apply_session_resonance(result.fact_weights, result.penalty)
            penalized_fact_ids = list(result.fact_weights.keys())

            if self._storage:
                affected = [
                    self._facts[fid] for fid in penalized_fact_ids if fid in self._facts
                ]
                self._storage.save_facts(affected)
                # Persist the mutated echo session (r_score + echo_penalty).
                past = self._echo.get(result.matched_session_id)
                if past:
                    self._storage.save_echo_session(
                        past.session_id,
                        past.bundle.centroids,
                        past.r_score,
                        time.time(),
                        fact_weights=past.fact_weights,
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
