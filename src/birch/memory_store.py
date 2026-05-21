"""MemoryStore — unified entry point for the BirchKM memory system."""
from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .fact import FactPassport
from .gravity import GravityEngine
from .black_hole import BlackHole
from .meta_fact import MetaFact
from .resonance.detector import compute_resonance
from .resonance.echo import EchoStore
from .resonance.embeddings import embed, embed_batch
from .resonance.cluster import ClusterBundle
from .resonance.echo import StoredSession
from .singularity_compactor import (
    CollapseReport,
    collapse_singularity,
)
from .storage import StorageBackend, SQLiteBackend
from .vector_index import VectorIndex


# Gravity floor — bodies below this after tick fall into the black hole
_ABSORPTION_THRESHOLD = 0.10
# Hawking emission threshold for MetaFacts: a centroid lives between its
# sources, so a strict 0.95 almost never fires. 0.85 is the working default.
_META_HAWKING_THRESHOLD = 0.85


@dataclass
class QueryResult:
    """Polymorphic query hit — either a FactPassport or a MetaFact.

    Exactly one of ``fact`` and ``meta`` is non-None. Legacy callers that
    read ``r.fact.fact_id`` keep working for fact hits; new callers branch
    on ``r.kind`` (``"fact"`` or ``"meta"``) and read the right field.
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


class MemoryStore:
    """
    Three-layer memory with black hole sink and Hawking emission.

    Layers:
      0 — surface  (gravity > 0.70, hot facts)
      1 — kinetic  (gravity 0.30–0.70, working memory)
      2 — core     (gravity < 0.30, cold archive)
     -1 — black hole (gravity < 0.10 after tick, absorbed)
    """

    # Minimum cosine similarity for auto-linking two facts.
    # High enough to avoid false edges; low enough to catch related triples.
    AUTO_LINK_THRESHOLD: float = 0.80
    # Max neighbours considered per new fact to keep startup cost linear.
    AUTO_LINK_TOP_K: int = 5

    # Background collapse triggers — once both are satisfied, queue a
    # gravitational collapse pass on the singularity.
    COLLAPSE_FACT_MASS_TRIGGER: int = 100     # min absolute fact_mass
    COLLAPSE_DELTA_TRIGGER: int = 50          # min absorbed-since-last-collapse

    def __init__(
        self,
        echo_k: int = 2,
        db_path: Optional[str | Path] = None,
        storage: Optional[StorageBackend] = None,
        auto_link: bool = True,
        collapse_threshold: float = 0.92,
        collapse_min_group_size: int = 2,
        collapse_async: bool = True,
    ) -> None:
        self._auto_link = auto_link
        # Collapse configuration — knobs the operator can tune per deployment.
        self._collapse_threshold = collapse_threshold
        self._collapse_min_group_size = collapse_min_group_size
        self._collapse_async = collapse_async
        self._collapse_counter = 0       # facts absorbed since last collapse
        self._last_collapse_at: Optional[float] = None
        self._total_collapses = 0
        self._collapse_executor: Optional[ThreadPoolExecutor] = None
        self._inflight_collapse: Optional[Future] = None
        self._engine = GravityEngine()
        self._hole = BlackHole()
        self._echo = EchoStore(default_k=echo_k)
        self._facts: dict[str, FactPassport] = {}
        # Normalised SPO → fact_id, for cheap duplicate detection in add_fact.
        self._spo_index: dict[tuple[str, str, str], str] = {}
        # Numpy-backed cosine index, kept in sync with live facts.
        self._index = VectorIndex()
        # MetaFacts that have re-entered the live layers via Hawking emission.
        # Kept on a separate index so query() can search and surface them
        # without colliding with FactPassport ids in the SPO machinery.
        self._meta_facts: dict[str, MetaFact] = {}
        self._meta_index = VectorIndex()
        if storage is not None:
            self._storage: Optional[StorageBackend] = storage
        elif db_path is not None:
            self._storage = SQLiteBackend(db_path)
        else:
            self._storage = None

        # Multiple sessions can be open concurrently — one per agent in
        # the typical MCP setup. Public methods accept an explicit
        # session_id; the convenience fallback _current_session_id is
        # only safe under single-user / sequential use.
        self._sessions: dict[str, SessionContext] = {}
        self._current_session_id: Optional[str] = None
        # Re-entrant lock guards _facts, _index, _spo_index, _sessions and
        # gravity/echo internals. Embed() calls are kept OUT of the lock
        # so concurrent agents don't serialize on a slow HTTP roundtrip.
        self._lock = threading.RLock()

        if self._storage:
            self._load_from_storage()

    # ── Storage bootstrap ────────────────────────────────────────────────────

    # ── Back-compat shims for the legacy single-session API ────────────────
    #
    # The public methods now accept an explicit session_id. When omitted
    # they fall back to ``_current_session_id`` so old single-user code
    # (``mem.session_start("s"); mem.add_fact(...); mem.session_close()``)
    # keeps working. These shims expose the current session's state under
    # the original attribute names that some tests still touch directly.

    @property
    def _session_id(self) -> Optional[str]:
        return self._current_session_id

    @property
    def _session_messages(self) -> list[str]:
        ctx = self._sessions.get(self._current_session_id or "")
        return ctx.messages if ctx else []

    @property
    def _session_vectors(self) -> list[list[float]]:
        ctx = self._sessions.get(self._current_session_id or "")
        return ctx.vectors if ctx else []

    @property
    def _session_facts(self) -> dict[str, float]:
        ctx = self._sessions.get(self._current_session_id or "")
        return ctx.facts if ctx else {}

    @_session_facts.setter
    def _session_facts(self, value) -> None:
        ctx = self._sessions.get(self._current_session_id or "")
        if ctx is not None:
            ctx.facts = {fid: float(w) for fid, w in dict(value).items()}

    @property
    def _session_fact_ids(self) -> list[str]:
        return list(self._session_facts.keys())

    @_session_fact_ids.setter
    def _session_fact_ids(self, value) -> None:
        if isinstance(value, dict):
            self._session_facts = value
        else:
            self._session_facts = {fid: 1.0 for fid in value}

    def _resolve_sid(self, session_id: Optional[str]) -> Optional[str]:
        return session_id if session_id is not None else self._current_session_id

    @staticmethod
    def _attribute_to(ctx: SessionContext, fact_id: str, weight: float) -> None:
        clipped = max(0.0, min(1.0, float(weight)))
        if clipped > ctx.facts.get(fact_id, 0.0):
            ctx.facts[fact_id] = clipped

    def _attribute_fact(self, fact_id: str, weight: float) -> None:
        """Tag a fact to the current (legacy) session if any."""
        ctx = self._sessions.get(self._current_session_id or "")
        if ctx is None:
            return
        self._attribute_to(ctx, fact_id, weight)

    @staticmethod
    def _normalize_spo(subject: str, predicate: str, obj: str) -> tuple[str, str, str]:
        return (
            " ".join(subject.lower().split()),
            " ".join(predicate.lower().split()),
            " ".join(obj.lower().split()),
        )

    def _auto_link_fact(self, fact_id: str, vec: list[float]) -> None:
        """Lock must be held. Link new fact to semantically close neighbours."""
        if not self._auto_link or len(self._index) < 2:
            return
        neighbours = self._index.search(
            vec, top_k=self.AUTO_LINK_TOP_K + 1, threshold=self.AUTO_LINK_THRESHOLD
        )
        for neighbour_id, sim in neighbours:
            if neighbour_id == fact_id:
                continue
            self._engine.link(fact_id, neighbour_id)
            self._engine.link(neighbour_id, fact_id)
            if self._storage:
                self._storage.save_edge(fact_id, neighbour_id)
                self._storage.save_edge(neighbour_id, fact_id)

    def fact_exists(self, subject: str, predicate: str, obj: str) -> bool:
        """Return True if an identical SPO triple is already in the live index."""
        key = self._normalize_spo(subject, predicate, obj)
        with self._lock:
            existing_id = self._spo_index.get(key)
            return existing_id is not None and existing_id in self._facts

    def _load_from_storage(self) -> None:
        for fact in self._storage.load_facts():
            self._facts[fact.fact_id] = fact
            self._engine.register(fact)
            self._index.add(fact.fact_id, fact.vector)
            if not fact.is_deprecated:
                key = self._normalize_spo(fact.subject, fact.predicate, fact.object)
                self._spo_index.setdefault(key, fact.fact_id)
        # MetaFacts: layer -1 live in the singularity; promoted ones (after
        # Hawking emission) live in the live meta store with the engine.
        if hasattr(self._storage, "load_meta_facts"):
            for meta in self._storage.load_meta_facts():
                if meta.layer == -1:
                    self._hole.restore_meta(meta)
                else:
                    self._meta_facts[meta.meta_id] = meta
                    self._meta_index.add(meta.meta_id, meta.vector)
                    self._engine.register(meta)
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
        _SESSION_TTL = 86_400  # 24 h — discard crashed/orphaned sessions
        if hasattr(self._storage, "load_open_sessions"):
            now = time.time()
            for row in self._storage.load_open_sessions():
                if now - row["started_at"] > _SESSION_TTL:
                    self._storage.delete_open_session(row["session_id"])
                    continue
                ctx = SessionContext(session_id=row["session_id"])
                ctx.messages = row["messages"]
                ctx.vectors = row["vectors"]
                ctx.facts = {k: float(v) for k, v in row["facts"].items()}
                self._sessions[row["session_id"]] = ctx
                self._current_session_id = row["session_id"]

    # ── Fact management ─────────────────────────────────────────────────────

    def add_fact(
        self,
        subject: str,
        predicate: str,
        obj: str,
        layer: int = 1,
        session_id: Optional[str] = None,
    ) -> FactPassport:
        """
        Create, embed, and register a new fact.

        If an identical (case-insensitive, whitespace-normalised) SPO triple
        already lives in the store, return the existing fact instead of
        creating a duplicate. The existing fact is touched and attributed
        to the named session (or the current one), so the caller's intent
        still propagates to gravity at weight 1.0.

        Embedding happens outside the lock so concurrent agents don't
        serialize on the slow HTTP roundtrip to Ollama.
        """
        key = self._normalize_spo(subject, predicate, obj)

        # Fast path: SPO already present.
        with self._lock:
            existing_id = self._spo_index.get(key)
            if existing_id and existing_id in self._facts:
                return self._touch_existing(existing_id, session_id)

        # Slow path: embed without holding the lock.
        vec = embed(f"{subject} {predicate} {obj}")

        with self._lock:
            # Double-check: another thread may have created the same triple
            # while we were embedding.
            existing_id = self._spo_index.get(key)
            if existing_id and existing_id in self._facts:
                return self._touch_existing(existing_id, session_id)

            sid = self._resolve_sid(session_id)
            fact = FactPassport(
                subject=subject,
                predicate=predicate,
                object=obj,
                layer=layer,
                source_session=sid,
            )
            fact.vector = vec
            self._facts[fact.fact_id] = fact
            self._engine.register(fact)
            self._index.add(fact.fact_id, vec)
            self._spo_index[key] = fact.fact_id
            self._auto_link_fact(fact.fact_id, vec)
            if self._storage:
                self._storage.save_fact(fact)
            if sid is not None:
                ctx = self._sessions.get(sid)
                if ctx is not None:
                    self._attribute_to(ctx, fact.fact_id, 1.0)
            return fact

    def _touch_existing(
        self,
        existing_id: str,
        session_id: Optional[str],
    ) -> FactPassport:
        """Lock-guarded helper for the dedupe path of add_fact."""
        existing = self._facts[existing_id]
        existing.touch()
        if self._storage:
            self._storage.save_fact(existing)
        sid = self._resolve_sid(session_id)
        if sid is not None:
            ctx = self._sessions.get(sid)
            if ctx is not None:
                self._attribute_to(ctx, existing_id, 1.0)
        return existing

    def add_facts(
        self,
        triples: list[tuple[str, str, str]],
        layer: int = 1,
        session_id: Optional[str] = None,
    ) -> list[FactPassport]:
        """
        Batch-insert a list of (subject, predicate, object) triples.

        One Ollama round-trip for all embeddings, one SQLite transaction for
        all inserts. Duplicate SPOs are touched and returned, not duplicated.

        Returns one FactPassport per input triple, in the same order.
        """
        if not triples:
            return []

        # ── Phase 1: SPO dedup under lock ──────────────────────────────────
        # Split into existing (already in store) and new (need embedding).
        results: list[Optional[FactPassport]] = [None] * len(triples)
        need_embed: list[int] = []          # indices into triples

        with self._lock:
            sid = self._resolve_sid(session_id)
            ctx = self._sessions.get(sid) if sid else None
            for i, (s, p, o) in enumerate(triples):
                key = self._normalize_spo(s, p, o)
                existing_id = self._spo_index.get(key)
                if existing_id and existing_id in self._facts:
                    fact = self._facts[existing_id]
                    fact.touch()
                    if ctx is not None:
                        self._attribute_to(ctx, existing_id, 1.0)
                    results[i] = fact
                else:
                    need_embed.append(i)

        if not need_embed:
            # All duplicates — persist touches in one shot and return.
            if self._storage:
                self._storage.save_facts([results[i] for i in range(len(triples))])  # type: ignore[arg-type]
            return results  # type: ignore[return-value]

        # ── Phase 2: batch-embed new triples (outside lock) ────────────────
        texts = [
            f"{triples[i][0]} {triples[i][1]} {triples[i][2]}"
            for i in need_embed
        ]
        vectors = embed_batch(texts)

        # ── Phase 3: register new facts + bulk save ─────────────────────────
        new_facts: list[FactPassport] = []
        with self._lock:
            sid = self._resolve_sid(session_id)
            ctx = self._sessions.get(sid) if sid else None
            for idx, vec in zip(need_embed, vectors):
                s, p, o = triples[idx]
                key = self._normalize_spo(s, p, o)
                # Double-check: concurrent thread may have inserted same SPO.
                existing_id = self._spo_index.get(key)
                if existing_id and existing_id in self._facts:
                    fact = self._facts[existing_id]
                    fact.touch()
                    if ctx is not None:
                        self._attribute_to(ctx, existing_id, 1.0)
                    results[idx] = fact
                    continue

                fact = FactPassport(
                    subject=s,
                    predicate=p,
                    object=o,
                    layer=layer,
                    source_session=sid,
                )
                fact.vector = vec
                self._facts[fact.fact_id] = fact
                self._engine.register(fact)
                self._index.add(fact.fact_id, vec)
                self._spo_index[key] = fact.fact_id
                self._auto_link_fact(fact.fact_id, vec)
                if ctx is not None:
                    self._attribute_to(ctx, fact.fact_id, 1.0)
                results[idx] = fact
                new_facts.append(fact)

            if self._storage and new_facts:
                self._storage.save_facts(new_facts)

        return results  # type: ignore[return-value]

    def delete_fact(self, fact_id: str) -> bool:
        """
        Permanently remove a fact from the live store.

        Cleans up _facts, _index, _spo_index, gravity engine, and storage.
        Returns True if the fact existed and was deleted, False if not found.
        Unlike absorption, this does NOT send the fact to the black hole —
        the data is gone.
        """
        with self._lock:
            fact = self._facts.pop(fact_id, None)
            if fact is None:
                return False
            self._index.remove(fact_id)
            self._drop_from_spo_index(fact)
            self._engine.unregister(fact_id)
            if self._storage:
                self._storage.delete_fact(fact_id)
            return True

    def list_facts(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        limit: int = 50,
    ) -> list[FactPassport]:
        """
        Return live facts, optionally filtered by subject and/or predicate.

        Matching is case-insensitive substring. Results are sorted by
        gravity_score descending so the most relevant facts come first.
        """
        with self._lock:
            facts = list(self._facts.values())
        if subject is not None:
            needle = subject.lower()
            facts = [f for f in facts if needle in f.subject.lower()]
        if predicate is not None:
            needle = predicate.lower()
            facts = [f for f in facts if needle in f.predicate.lower()]
        facts.sort(key=lambda f: f.gravity_score, reverse=True)
        return facts[:limit]

    def link(self, from_id: str, to_id: str) -> None:
        with self._lock:
            self._engine.link(from_id, to_id)
            if self._storage:
                self._storage.save_edge(from_id, to_id)

    def deprecate(self, old_id: str, new_id: str) -> None:
        with self._lock:
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
        """Open a session context. Safe to call concurrently."""
        with self._lock:
            ctx = SessionContext(session_id=session_id)
            self._sessions[session_id] = ctx
            self._current_session_id = session_id
            if self._storage and hasattr(self._storage, "save_open_session"):
                self._storage.save_open_session(
                    session_id, ctx.messages, ctx.vectors, ctx.facts, time.time()
                )

    def session_message(self, text: str, session_id: Optional[str] = None) -> None:
        """Record a user message in the named session (or the current one)."""
        # Embed outside the lock — slow HTTP call, must not serialize agents.
        vec = embed(text)
        with self._lock:
            sid = self._resolve_sid(session_id)
            if sid is None or sid not in self._sessions:
                raise KeyError(f"unknown session: {sid!r}")
            ctx = self._sessions[sid]
            ctx.messages.append(text)
            ctx.vectors.append(vec)
            if self._storage and hasattr(self._storage, "save_open_session"):
                self._storage.save_open_session(
                    sid, ctx.messages, ctx.vectors, ctx.facts, time.time()
                )

    def session_close(self, session_id: Optional[str] = None) -> dict:
        """
        Close session: compute resonance, propagate R to facts,
        record echo bundle, tick gravity, absorb dead facts.

        Operates on the named session if provided; otherwise on the
        most recently opened one.
        """
        with self._lock:
            sid = self._resolve_sid(session_id)
            if sid is None or sid not in self._sessions:
                return {}
            ctx = self._sessions[sid]
            if not ctx.messages:
                self._pop_session_locked(sid)
                return {}
            messages_snapshot = list(ctx.messages)
            vectors_snapshot = list(ctx.vectors)
            facts_snapshot = dict(ctx.facts)

        # Resonance is pure computation on the snapshot — do it outside the
        # lock so other agents can keep querying.
        result = compute_resonance(
            messages_snapshot,
            start_vector=vectors_snapshot[0],
            end_vector=vectors_snapshot[-1],
            all_vectors=vectors_snapshot,
        )

        with self._lock:
            # Propagate R to facts used in this session, weighted by how
            # relevant each fact was to the session's queries.
            self._engine.apply_session_resonance(facts_snapshot, result.r)

            for fid in facts_snapshot:
                if fid in self._facts:
                    self._facts[fid].touch()

            self._echo.record(
                sid,
                vectors_snapshot,
                result.r,
                fact_weights=facts_snapshot,
            )
            # Opportunistic TTL sweep on session close. Drops stale resolved
            # and already-penalised echo sessions so the store stays bounded
            # without a separate cron job.
            expired = self._echo.expire()
            if self._storage:
                session_obj = self._echo.get(sid)
                if session_obj:
                    self._storage.save_echo_session(
                        sid,
                        session_obj.bundle.centroids,
                        session_obj.r_score,
                        time.time(),
                        fact_weights=session_obj.fact_weights,
                        echo_penalty=session_obj.echo_penalty,
                    )
                # Storage layer doesn't yet expose a delete; drop via the
                # tolerated load-time gc instead. We still drop in-memory.
                if expired and hasattr(self._storage, "delete_echo_session"):
                    for stale_sid in expired:
                        self._storage.delete_echo_session(stale_sid)

            migrations = self._engine.tick()
            absorbed = self._absorb_dead()

            if self._storage:
                self._storage.save_facts(list(self._facts.values()))
                if self._meta_facts and hasattr(self._storage, "save_meta_facts"):
                    self._storage.save_meta_facts(list(self._meta_facts.values()))

            # Counter-triggered collapse. Held inside the lock so the
            # collapse counter and the trigger decision are consistent.
            self._maybe_trigger_collapse_locked(len(absorbed))

            summary = {
                "session_id": sid,
                "r": result.r,
                "label": result.label,
                "migrations": migrations,
                "absorbed": absorbed,
            }
            self._pop_session_locked(sid)

        return summary

    def _pop_session_locked(self, sid: str) -> None:
        """Caller must hold self._lock."""
        self._sessions.pop(sid, None)
        if self._current_session_id == sid:
            self._current_session_id = (
                next(reversed(self._sessions)) if self._sessions else None
            )
        if self._storage and hasattr(self._storage, "delete_open_session"):
            self._storage.delete_open_session(sid)

    def _absorb_dead(self) -> list[str]:
        """Send facts and live MetaFacts below the threshold back into the hole."""
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
        # Live MetaFacts use the same gravity floor — they came out of the
        # singularity once, they can fall back in.
        for mid, meta in list(self._meta_facts.items()):
            if meta.gravity_score < _ABSORPTION_THRESHOLD:
                self._hole.absorb_meta(meta)
                del self._meta_facts[mid]
                self._meta_index.remove(mid)
                if self._storage and hasattr(self._storage, "save_meta_fact"):
                    # absorb_meta resets layer to -1, persist that.
                    self._storage.save_meta_fact(meta)
                absorbed.append(mid)
        return absorbed

    # ── Collapse orchestration ──────────────────────────────────────────────

    def collapse_singularity(
        self,
        threshold: Optional[float] = None,
        min_group_size: Optional[int] = None,
        persist: bool = True,
    ) -> CollapseReport:
        """Synchronous compactor pass — usable from tests, jobs, or by hand.

        Holds the store lock for the duration. Returns the CollapseReport
        even if nothing was collapsed, so the caller can log it.
        """
        thr = self._collapse_threshold if threshold is None else threshold
        mgs = self._collapse_min_group_size if min_group_size is None else min_group_size
        with self._lock:
            new_metas, report = collapse_singularity(
                self._hole, threshold=thr, min_group_size=mgs,
            )
            if persist and self._storage and hasattr(self._storage, "save_meta_facts"):
                self._storage.save_meta_facts(new_metas)
                # The absorbed FactPassports are gone from the live store and
                # were never persisted as "absorbed" — but if storage was
                # tracking them as deleted-on-absorb, we should drop them now.
                for meta in new_metas:
                    for fid in meta.source_fact_ids:
                        if hasattr(self._storage, "delete_fact"):
                            self._storage.delete_fact(fid)
            self._last_collapse_at = time.time()
            self._total_collapses += 1
            self._collapse_counter = 0
            return report

    def _maybe_trigger_collapse_locked(self, absorbed_count: int) -> None:
        """Caller must hold self._lock. Schedules collapse if thresholds met."""
        self._collapse_counter += absorbed_count
        if self._hole.fact_mass < self.COLLAPSE_FACT_MASS_TRIGGER:
            return
        if self._collapse_counter < self.COLLAPSE_DELTA_TRIGGER:
            return
        # Skip if a previous collapse is still running.
        if self._inflight_collapse is not None and not self._inflight_collapse.done():
            return
        if not self._collapse_async:
            self.collapse_singularity()
            return
        if self._collapse_executor is None:
            self._collapse_executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="birch-collapse",
            )
        self._inflight_collapse = self._collapse_executor.submit(
            self.collapse_singularity,
        )

    def close(self) -> None:
        """Release the background executor and close the storage layer.

        We must NOT wait for inflight collapse or call ``shutdown(wait=True)``
        while holding ``self._lock`` — the worker thread acquires the same
        lock from inside ``collapse_singularity``, so blocking on it while
        holding it would deadlock. Snapshot the handles under the lock, then
        wait outside it.
        """
        with self._lock:
            inflight = self._inflight_collapse
            executor = self._collapse_executor
            storage = self._storage
            self._inflight_collapse = None
            self._collapse_executor = None

        if inflight is not None:
            try:
                inflight.result(timeout=5.0)
            except Exception:
                pass
        if executor is not None:
            executor.shutdown(wait=True)
        if storage is not None and hasattr(storage, "close"):
            storage.close()

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
        session_id: Optional[str] = None,
        min_similarity: float = 0.0,
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
        # Embed outside the lock.
        vec = embed(text)

        with self._lock:
            results: list[QueryResult] = []
            layer_labels = {0: "surface", 1: "kinetic", 2: "core"}

            # Live FactPassports.
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

            # Live MetaFacts — promoted out of the black hole by past
            # Hawking emissions; share the same layer machinery as facts.
            meta_sims = self._meta_index.all_similarities(vec)
            for mid, sim in meta_sims.items():
                meta = self._meta_facts.get(mid)
                if meta is None:
                    continue
                if not (min_layer <= meta.layer <= max_layer):
                    continue
                results.append(QueryResult(
                    meta=meta,
                    similarity=round(sim, 4),
                    source=layer_labels.get(meta.layer, "kinetic"),
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

                # MetaFact Hawking emission — looser threshold so a centroid
                # actually fires on a topically close query.
                meta_emitted = self._hole.hawking_emit_metas(
                    vec, threshold=_META_HAWKING_THRESHOLD
                )
                for meta in meta_emitted:
                    self._meta_facts[meta.meta_id] = meta
                    self._engine.register(meta)
                    self._meta_index.add(meta.meta_id, meta.vector)
                    if self._storage and hasattr(self._storage, "save_meta_fact"):
                        self._storage.save_meta_fact(meta)
                    sim = VectorIndex.similarity(vec, meta.vector)
                    results.append(QueryResult(
                        meta=meta,
                        similarity=round(sim, 4),
                        source="hawking_meta",
                    ))

            results.sort(key=lambda r: r.similarity, reverse=True)
            if min_similarity > 0.0:
                results = [r for r in results if r.similarity >= min_similarity]
            top = results[:top_k]

            # Attribution + touch — only on what the caller actually
            # receives. Weight is the similarity itself; a body returned at
            # cosine 0.95 ends up nine times more sensitive to session R
            # than one returned at 0.10. Polymorphic over fact / meta.
            sid = self._resolve_sid(session_id)
            ctx = self._sessions.get(sid) if sid else None
            for r in top:
                body = r.fact if r.fact is not None else r.meta
                if body is None:
                    continue
                body.touch()
                if ctx is not None:
                    self._attribute_to(ctx, r.body_id, r.similarity)

        return top

    def check_echo(self, first_message: str, session_id: Optional[str] = None) -> dict:
        """
        Check if a new session echoes a past unresolved problem.

        If echo is detected and a non-zero retroactive penalty is applied
        for the first time, the penalty is propagated to the gravity of
        every fact that the matched past session touched. Affected facts
        are re-persisted.
        """
        # session_id is accepted for symmetry with the other methods but
        # echo penalties apply to the matched PAST session, not to the
        # active one; we keep the param for forward compatibility.
        _ = session_id  # currently unused — explicit silence
        vec = embed(first_message)

        with self._lock:
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
        with self._lock:
            layers = {0: 0, 1: 0, 2: 0}
            for f in self._facts.values():
                layers[f.layer] = layers.get(f.layer, 0) + 1
            meta_layers = {0: 0, 1: 0, 2: 0}
            for m in self._meta_facts.values():
                meta_layers[m.layer] = meta_layers.get(m.layer, 0) + 1
            return {
                "surface": layers[0],
                "kinetic": layers[1],
                "core": layers[2],
                "black_hole_mass": self._hole.mass,
                "black_hole_fact_mass": self._hole.fact_mass,
                "black_hole_meta_mass": self._hole.meta_mass,
                "hawking_emissions": self._hole.total_emissions,
                "total_live": len(self._facts),
                "total_live_metas": len(self._meta_facts),
                "meta_layers": meta_layers,
                "active_sessions": len(self._sessions),
                "collapse_counter": self._collapse_counter,
                "total_collapses": self._total_collapses,
                "last_collapse_at": self._last_collapse_at,
            }
