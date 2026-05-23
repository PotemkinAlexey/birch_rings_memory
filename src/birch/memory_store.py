"""MemoryStore — unified entry point for the BirchKM memory system."""
from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Iterator, Literal, Optional, Union, overload

from .black_hole import BlackHole
from .fact import FactPassport
from .gravity import GravityEngine, pre_resonance_features
from .meta_fact import MetaFact
from .resonance.cluster import ClusterBundle
from .resonance.detector import ResonanceResult, compute_resonance
from .resonance.echo import EchoStore, StoredSession
from .resonance.embeddings import embed, embed_batch
from .singularity_compactor import (
    CollapseReport,
    collapse_singularity,
)
from .storage import SQLiteBackend, StorageBackend
from .thresholds import Thresholds
from .vector_index import VectorIndex

# Module-level aliases kept for the existing call sites; the values
# now come from the centralised env-overridable Thresholds module
# (round 12 / Gemini round 1). Operators can pin every threshold via
# BIRCH_* env vars to match their embedding model's cosine distribution.
_ABSORPTION_THRESHOLD = Thresholds.ABSORPTION
_META_HAWKING_THRESHOLD = Thresholds.HAWKING_META


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

    # Minimum cosine similarity for auto-linking two facts. Sourced
    # from the env-overridable Thresholds module (round 12) so an
    # operator pinning a different embedding model can tune it.
    AUTO_LINK_THRESHOLD: float = Thresholds.AUTO_LINK
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
        collapse_threshold: Optional[float] = None,
        collapse_min_group_size: int = 2,
        collapse_async: bool = True,
    ) -> None:
        if collapse_threshold is None:
            collapse_threshold = Thresholds.COLLAPSE
        self._auto_link = auto_link
        # Collapse configuration — knobs the operator can tune per deployment.
        self._collapse_threshold = collapse_threshold
        self._collapse_min_group_size = collapse_min_group_size
        self._collapse_async = collapse_async
        self._collapse_counter = 0       # facts absorbed since last collapse
        # Distinguish "we ran a collapse pass" from "the pass actually
        # compressed anything". A pass with report.groups == 0 is an
        # attempt; only a non-zero pass is a successful collapse.
        # operators reading memory_stats need both numbers to interpret
        # "no-op collapses keep firing — singularity is too sparse / too
        # heterogeneous".
        self._last_collapse_at: Optional[float] = None
        self._last_collapse_attempt_at: Optional[float] = None
        self._total_collapses = 0
        self._total_collapse_attempts = 0
        self._collapse_executor: Optional[ThreadPoolExecutor] = None
        self._inflight_collapse: Optional[Future] = None
        self._echo_k = echo_k
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

        # data_version seen at the last load. When the backend reports a
        # different value, another process has written and our caches are
        # stale — see _sync().
        self._data_version = 0
        if self._storage:
            self._load_from_storage()
            self._data_version = self._data_version_now()

        # run_forecast cache. Keyed by (data_version, body_count, horizon)
        # so an agent re-calling forecast_memory back-to-back gets the
        # last result without re-running the O(n²) simulation. Invalidated
        # automatically the moment ANY write bumps data_version — the
        # backend's authoritative version counter is the cache key.
        self._forecast_cache: tuple[
            tuple[int, int, int], dict
        ] | None = None

    # ── Cross-process cache coherence ────────────────────────────────────────

    def _data_version_now(self) -> int:
        """Backend data_version, or 0 if the backend does not support it."""
        if self._storage is None or not hasattr(self._storage, "data_version"):
            return 0
        try:
            return self._storage.data_version()
        except Exception:
            return 0

    def _sync(self) -> None:
        """Reload caches if another process wrote since our last load.

        Caller must hold self._lock. Cheap when nothing changed — a single
        PRAGMA, no I/O. When the database moved under us, _reload() rebuilds
        every cache so we never read or write on top of a stale view. With a
        single active process data_version never changes, so this is a no-op
        and the store stays hot.
        """
        if self._storage is None:
            return
        if self._data_version_now() != self._data_version:
            self._reload()

    def _reload(self) -> None:
        """Rebuild every in-memory cache from storage. Caller holds self._lock."""
        if self._storage is None:
            return
        self._facts = {}
        self._spo_index = {}
        self._index = VectorIndex()
        self._meta_facts = {}
        self._meta_index = VectorIndex()
        self._engine = GravityEngine()
        self._hole = BlackHole()
        self._echo = EchoStore(default_k=self._echo_k)
        self._sessions = {}
        self._current_session_id = None
        self._load_from_storage()
        self._data_version = self._data_version_now()

    @contextmanager
    def _txn(self) -> Iterator[None]:
        """Exclusive storage transaction; plain no-op if unsupported.

        Inside the block the caller holds SQLite's write lock, so a _reload()
        as the first statement yields the authoritative state and nothing can
        interleave before commit.
        """
        if self._storage is not None and hasattr(self._storage, "transaction"):
            with self._storage.transaction():
                yield
        else:
            yield

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

    # EWMA half-life for recent_utility: ~7 events to half-life. Slow
    # enough that a single outlier session does not swing the prior, fast
    # enough that a meaningful streak shows up within a week.
    _RECENT_UTILITY_ALPHA: float = 0.15

    def _apply_recent_utility_locked(
        self,
        body_weights: dict[str, float],
        r: float,
    ) -> None:
        """Caller must hold self._lock. Update the EWMA recent_utility of
        every touched body (FactPassport or MetaFact) by the per-body
        realised value ``target = clamp((r·weight + 1) / 2, 0, 1)``.

        Used by both ``session_close`` (positive/negative R from the just-
        closed session) and ``check_echo`` (retroactive negative penalty
        from an echo). Without sharing the EWMA path, the resonance signal
        would change but recent_utility would not — gravity formula would
        see contradictory inputs.
        """
        alpha = self._RECENT_UTILITY_ALPHA
        for bid, attr_weight in body_weights.items():
            body = self._facts.get(bid) or self._meta_facts.get(bid)
            if body is None:
                continue
            per_body_r = r * float(attr_weight)
            target = max(0.0, min(1.0, (per_body_r + 1.0) / 2.0))
            body.recent_utility = (
                (1.0 - alpha) * body.recent_utility + alpha * target
            )

    def _persist_session_locked(self, ctx: Optional[SessionContext]) -> None:
        """Caller must hold self._lock AND be inside a write transaction.

        After mutating ``ctx.facts`` (attribution from a write or read), flush
        the open-session row so the per-fact relevance map survives a crash
        before ``session_close``. Without this, an agent that records facts
        and dies mid-session loses the attribution mapping and the eventual
        resonance signal will not reach those bodies.
        """
        if ctx is None or self._storage is None:
            return
        if not hasattr(self._storage, "save_open_session"):
            return
        self._storage.save_open_session(
            ctx.session_id, ctx.messages, ctx.vectors, ctx.facts, time.time(),
        )

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
            self._sync()
            existing_id = self._spo_index.get(key)
            return existing_id is not None and existing_id in self._facts

    def _load_from_storage(self) -> None:
        # Only ever called when storage is configured (from __init__ / _reload).
        assert self._storage is not None
        # Learned pre-resonance weights, if the user's history has trained any.
        if hasattr(self._storage, "load_adaptive_weights"):
            persisted = self._storage.load_adaptive_weights()
            if persisted is not None:
                self._engine.weights = persisted
        for fact in self._storage.load_facts():
            if fact.layer == -1:
                # Absorbed body — restore into the singularity so Hawking
                # emission and singularity collapse still see it after a
                # process restart. Symmetric with MetaFacts at layer=-1.
                self._hole.restore_fact(fact)
                continue
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
        # Edges between live facts only — orphan endpoints (referenced fact
        # was deleted) used to inflate _degrees and skew max_deg, which
        # depressed graph_score for healthy facts. Drop them here AND in
        # storage so the corruption does not grow.
        live_fact_ids = set(self._facts.keys())
        stale_edges: list[tuple[str, str]] = []
        for from_id, to_id in self._storage.load_edges():
            if from_id in live_fact_ids and to_id in live_fact_ids:
                self._engine.link(from_id, to_id)
            else:
                stale_edges.append((from_id, to_id))
        if (stale_edges and self._storage
                and hasattr(self._storage, "delete_edges_for_fact")):
            # Cheaper to delete by endpoint than to add a per-edge delete;
            # the orphan list shares many endpoints in practice.
            seen: set[str] = set()
            for from_id, to_id in stale_edges:
                for endpoint in (from_id, to_id):
                    if (endpoint not in live_fact_ids
                            and endpoint not in seen):
                        self._storage.delete_edges_for_fact(endpoint)
                        seen.add(endpoint)
        for row in self._storage.load_echo_sessions():
            centroids = row["centroids"]
            cb = ClusterBundle(centroids=centroids, k=len(centroids), inertia=0.0)
            self._echo._sessions[row["session_id"]] = StoredSession(
                session_id=row["session_id"],
                bundle=cb,
                r_score=row["r_score"],
                fact_weights=dict(row.get("fact_weights", {})),
                echo_penalty=row.get("echo_penalty", 0.0),
                # Preserve the original record time so TTL survives restart.
                # Without this every restart resets timestamps to "now" and
                # the penalty / resolved / default tiers stop being TTLs.
                timestamp=float(row.get("recorded_at") or time.time()),
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

    @overload
    def add_fact(
        self,
        subject: str,
        predicate: str,
        obj: str,
        layer: int = ...,
        session_id: Optional[str] = ...,
        *,
        return_status: Literal[False] = False,
    ) -> FactPassport: ...

    @overload
    def add_fact(
        self,
        subject: str,
        predicate: str,
        obj: str,
        layer: int = ...,
        session_id: Optional[str] = ...,
        *,
        return_status: Literal[True],
    ) -> tuple[FactPassport, bool]: ...

    def add_fact(
        self,
        subject: str,
        predicate: str,
        obj: str,
        layer: int = 1,
        session_id: Optional[str] = None,
        return_status: bool = False,
    ) -> Union[FactPassport, tuple[FactPassport, bool]]:
        """
        Create, embed, and register a new fact.

        If an identical (case-insensitive, whitespace-normalised) SPO triple
        already lives in the store, return the existing fact instead of
        creating a duplicate. The existing fact is touched and attributed
        to the named session (or the current one), so the caller's intent
        still propagates to gravity at weight 1.0.

        ``return_status=True`` returns ``(fact, created)`` where ``created``
        is True only when this call actually inserted a new row in its
        authoritative transaction. A race that loses (another process
        inserted the same SPO between our embed and our write lock)
        returns ``created=False`` with the winner's fact. This is what
        ``set_fact`` uses to report ``already_existed`` honestly.

        Embedding happens outside the lock so concurrent agents don't
        serialize on the slow HTTP roundtrip to Ollama.
        """
        key = self._normalize_spo(subject, predicate, obj)

        # Fast path: SPO already present — skip the embed entirely.
        with self._lock:
            self._sync()
            existing_id = self._spo_index.get(key)
            if existing_id and existing_id in self._facts:
                with self._txn():
                    self._sync()
                    eid = self._spo_index.get(key)
                    if eid and eid in self._facts:
                        fact = self._touch_existing(eid, session_id)
                        return (fact, False) if return_status else fact
                # Raced away between sync and the write lock — fall through.

        # Slow path: embed without holding the lock.
        vec = embed(f"{subject} {predicate} {obj}")

        with self._lock:
            with self._txn():
                # Reload under the write lock — the authoritative view.
                self._sync()
                # Double-check: another process may have created the same
                # triple while we were embedding.
                existing_id = self._spo_index.get(key)
                if existing_id and existing_id in self._facts:
                    fact = self._touch_existing(existing_id, session_id)
                    return (fact, False) if return_status else fact

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
                        self._persist_session_locked(ctx)
                return (fact, True) if return_status else fact

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
                self._persist_session_locked(ctx)
        return existing

    def add_facts(
        self,
        triples: list[tuple[str, str, str]],
        layer: int = 1,
        session_id: Optional[str] = None,
        session_ids: Optional[list[Optional[str]]] = None,
        return_status: bool = False,
    ) -> list:
        """
        Batch-insert a list of (subject, predicate, object) triples.

        One Ollama round-trip for all embeddings, one SQLite transaction for
        all inserts. Duplicate SPOs are touched and returned, not duplicated.

        ``session_ids`` (optional) is a parallel list of per-item session_ids
        — if present it must have the same length as ``triples``. Each fact
        is attributed to its own session, falling back to the top-level
        ``session_id`` when the per-item entry is None. This is what makes
        the MCP ``record_facts`` per-item ``session_id`` contract real.

        ``return_status=True`` returns a list of dicts with
        ``{fact, already_existed, duplicate_in_batch}`` instead of bare
        ``FactPassport`` objects. ``already_existed=True`` means the SPO
        triple was in the store BEFORE this batch ran; ``duplicate_in_batch=True``
        means an earlier item in the same batch already created it. Both
        flags can be true (existing fact also duplicated in the batch).

        Returns one ``FactPassport`` per input triple, in the same order
        (or one status dict per input triple if ``return_status=True``).
        """
        if not triples:
            return []
        if session_ids is not None and len(session_ids) != len(triples):
            raise ValueError(
                f"session_ids length ({len(session_ids)}) must match triples "
                f"length ({len(triples)})"
            )

        results: list[Optional[FactPassport]] = [None] * len(triples)
        already_existed: list[bool] = [False] * len(triples)
        duplicate_in_batch: list[bool] = [False] * len(triples)

        # Embed every triple outside the lock — one batch round-trip.
        # embed_batch validates length, so a partial response from the
        # provider becomes a typed EmbeddingError instead of a silent
        # alignment drift through zip().
        texts = [f"{s} {p} {o}" for (s, p, o) in triples]
        vectors = embed_batch(texts)
        if len(vectors) != len(triples):
            from .resonance.embeddings import EmbeddingError
            raise EmbeddingError(
                f"embedding provider returned {len(vectors)} vectors for "
                f"{len(triples)} inputs — refusing to write a partial batch"
            )

        # Touched sessions, persisted ONCE at the end inside the same txn.
        touched_ctxs: set[str] = set()

        with self._lock:
            with self._txn():
                # Reload under the write lock — the authoritative view.
                self._sync()
                # Track per-key first-occurrence within this batch so a
                # duplicate SPO inside the same input list is marked.
                seen_in_batch: dict[tuple[str, str, str], int] = {}

                for idx, (triple, vec) in enumerate(zip(triples, vectors)):
                    s, p, o = triple
                    # Per-item session_id overrides the top-level default.
                    raw_sid = (session_ids[idx] if session_ids is not None
                               else None)
                    sid = self._resolve_sid(raw_sid or session_id)
                    ctx = self._sessions.get(sid) if sid else None
                    key = self._normalize_spo(s, p, o)

                    # 1) Pre-existing in the store (before this batch ran).
                    existing_id = self._spo_index.get(key)
                    if (existing_id and existing_id in self._facts
                            and key not in seen_in_batch):
                        already_existed[idx] = True

                    # 2) Already appeared in this batch above.
                    if key in seen_in_batch:
                        duplicate_in_batch[idx] = True

                    if existing_id and existing_id in self._facts:
                        fact = self._facts[existing_id]
                        fact.touch()
                        if ctx is not None:
                            self._attribute_to(ctx, existing_id, 1.0)
                            touched_ctxs.add(ctx.session_id)
                        results[idx] = fact
                        seen_in_batch.setdefault(key, idx)
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
                        touched_ctxs.add(ctx.session_id)
                    results[idx] = fact
                    seen_in_batch.setdefault(key, idx)

                # Persist new facts and touched duplicates in one shot.
                if self._storage:
                    self._storage.save_facts(
                        [r for r in results if r is not None]
                    )
                # Persist every open session whose attribution changed.
                for sid in touched_ctxs:
                    self._persist_session_locked(self._sessions.get(sid))

        if return_status:
            return [
                {
                    "fact": results[i],
                    "already_existed": already_existed[i],
                    "duplicate_in_batch": duplicate_in_batch[i],
                }
                for i in range(len(triples))
            ]
        return results  # type: ignore[return-value]

    def find_similar(
        self,
        text: str,
        top_k: int = 5,
        min_similarity: float = 0.85,
        subject_prefix: Optional[str] = None,
        exclude_ids: Optional[set[str]] = None,
    ) -> list[dict]:
        """Read-only semantic search — surface paraphrase candidates.

        Returns live (non-deprecated, non-expired) facts whose embedding
        cosine to ``text`` is at or above ``min_similarity``. Use this to
        discover candidates that should be folded together with
        ``supersede_fact`` / ``set_fact`` — write-time hygiene without
        committing to a mutation here.

        ``subject_prefix`` is a case-insensitive ``startswith`` filter on the
        fact's subject; useful for scoping a search to one project.
        ``exclude_ids`` skips known facts (e.g., the one you just wrote).
        """
        if not text.strip():
            return []
        vec = embed(text)
        return self._find_similar_by_vector(
            vec,
            top_k=top_k,
            min_similarity=min_similarity,
            subject_prefix=subject_prefix,
            exclude_ids=exclude_ids,
        )

    def _find_similar_by_vector(
        self,
        vec: list[float],
        top_k: int = 5,
        min_similarity: float = 0.85,
        subject_prefix: Optional[str] = None,
        exclude_ids: Optional[set[str]] = None,
    ) -> list[dict]:
        """Caller-provided embedding variant — used by record_fact's
        similar_existing hint to avoid embedding the same text twice.
        """
        prefix = subject_prefix.lower() if subject_prefix else None
        skip = exclude_ids or set()
        with self._lock:
            self._sync()
            sims = self._index.all_similarities(vec)
        hits: list[dict] = []
        for fid, sim in sims.items():
            if fid in skip:
                continue
            if sim < min_similarity:
                continue
            fact = self._facts.get(fid)
            if fact is None or fact.is_deprecated or fact.is_expired:
                continue
            if prefix and not fact.subject.lower().startswith(prefix):
                continue
            hits.append({
                "fact_id": fid,
                "subject": fact.subject,
                "predicate": fact.predicate,
                "object": fact.object,
                "similarity": round(float(sim), 4),
                "gravity_score": round(fact.gravity_score, 3),
                "layer": fact.layer,
            })
        hits.sort(key=lambda h: h["similarity"], reverse=True)
        return hits[:top_k]

    def explain_fact(self, fact_id: str) -> dict:
        """Decompose a fact's gravity into per-component contributions.

        Returns the live values of every adaptive feature, the weight each
        carries right now, and the actual contribution each makes to the
        current gravity score. Use this when a fact's gravity surprises you
        — you'll see immediately whether the freshness term is high but
        recent_utility is dragging it down, or the forecast says it's about
        to fall, or whatever.
        """
        with self._lock:
            self._sync()
            fact = self._facts.get(fact_id)
            if fact is None:
                return {"found": False, "fact_id": fact_id}
            max_deg = max(self._engine._degrees.values(), default=1)
            degree = self._engine._degrees.get(fact_id, 0)
            features = pre_resonance_features(
                fact, graph_degree=degree, max_degree=max_deg,
            )
            weights = self._engine.weights
            freshness, access, graph, utility, stability = features
            if fact.resonance_count > 0:
                resonance_score = (fact.avg_resonance + 1.0) / 2.0
            else:
                resonance_score = 0.0
            from .gravity import _W_RESONANCE
            contributions = {
                "freshness":  round(weights.w_freshness * freshness, 4),
                "access":     round(weights.w_access * access, 4),
                "graph":      round(weights.w_graph * graph, 4),
                "recent_utility":     round(weights.w_utility * utility, 4),
                "forecast_stability": round(weights.w_stability * stability, 4),
                "resonance":  round(_W_RESONANCE * resonance_score, 4),
            }
            live_gravity = sum(contributions.values())
            return {
                "found": True,
                "fact_id": fact_id,
                "subject": fact.subject,
                "predicate": fact.predicate,
                "object": fact.object,
                "layer": fact.layer,
                "stored_gravity_score": round(fact.gravity_score, 4),
                "live_gravity_score": round(min(1.0, max(0.0, live_gravity)), 4),
                "features": {
                    "freshness": round(freshness, 4),
                    "access": round(access, 4),
                    "graph": round(graph, 4),
                    "recent_utility": round(utility, 4),
                    "forecast_stability": round(stability, 4),
                    "resonance_score": round(resonance_score, 4),
                },
                "weights": weights.as_dict(),
                "contributions": contributions,
                "is_deprecated": fact.is_deprecated,
                "is_expired": fact.is_expired,
                "deprecated_by": fact.deprecated_by,
                "resonance_count": fact.resonance_count,
                "access_count": fact.access_count,
                "last_accessed": fact.last_accessed,
                "created_at": fact.created_at,
            }

    def delete_fact(self, fact_id: str) -> bool:
        """
        Permanently remove a fact from the live store.

        Cleans up _facts, _index, _spo_index, gravity engine, and storage.
        Returns True if the fact existed and was deleted, False if not found.
        Unlike absorption, this does NOT send the fact to the black hole —
        the data is gone.
        """
        with self._lock:
            with self._txn():
                self._sync()
                fact = self._facts.pop(fact_id, None)
                if fact is None:
                    return False
                self._index.remove(fact_id)
                self._drop_from_spo_index(fact)
                self._engine.unregister(fact_id)
                if self._storage:
                    self._storage.delete_fact(fact_id)
                    # Drop every edge incident to this fact — otherwise on
                    # next load the orphan endpoints inflate _degrees and
                    # depress graph_score for healthy facts.
                    if hasattr(self._storage, "delete_edges_for_fact"):
                        self._storage.delete_edges_for_fact(fact_id)
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
            self._sync()
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
            with self._txn():
                self._sync()
                self._engine.link(from_id, to_id)
                if self._storage:
                    self._storage.save_edge(from_id, to_id)

    def deprecate(self, old_id: str, new_id: str) -> dict:
        """Legacy alias for :meth:`supersede_fact`.

        Older callers used to set ``deprecated_by`` directly without
        sending the body to the singularity, leaking deprecated facts
        into live ``query()`` until next tick. Now delegates to
        ``supersede_fact``, which runs ``_absorb_dead`` synchronously
        and keeps the body in the singularity with lineage intact.

        Returns the same dict ``supersede_fact`` does — including
        ``{"superseded": False, "reason": "old_id not found"}`` when
        the id is unknown — so callers can tell whether anything
        actually happened. Legacy callers that ignore the return
        value remain compatible.
        """
        return self.supersede_fact(old_id, new_id)

    def supersede_fact(self, old_id: str, new_id: str) -> dict:
        """Mark ``old_id`` as superseded by ``new_id`` and send it to the singularity.

        This is the canonical path for "we now know better": the old fact's
        ``deprecated_by`` is set (lineage preserved), the SPO slot is freed
        for the new claim, and an immediate ``_absorb_dead`` pass moves the
        body into the black hole on the same call — so the caller sees the
        effect synchronously instead of waiting for the next session_close.

        Unlike ``delete_fact``, the body is **not** removed from storage —
        it stays in the ``facts`` table with ``deprecated_by`` set, and the
        runtime treats it as a singularity resident on every restart. This
        keeps it available for singularity collapse (MetaFact compression)
        and Hawking emission, and preserves the "we used to think X" record.
        """
        with self._lock:
            with self._txn():
                self._sync()
                result = self._supersede_fact_locked(old_id, new_id)
        return result

    def _supersede_fact_locked(self, old_id: str, new_id: str) -> dict:
        """Locked helper for supersede_fact. Caller MUST hold self._lock
        AND be inside a write ``_txn()`` AND have already done ``_sync()``.

        Exists so set_fact can supersede slot occupants inside its own
        transaction without nesting public ``supersede_fact`` calls —
        nesting works via reentrant transaction(), but the chain of
        public-method ↔ public-method calls is harder to reason about
        than a single transactional flow that uses this helper.
        """
        if old_id not in self._facts:
            return {"superseded": False, "reason": "old_id not found"}
        old = self._facts[old_id]
        old.deprecated_by = new_id
        key = self._normalize_spo(old.subject, old.predicate, old.object)
        if self._spo_index.get(key) == old_id:
            del self._spo_index[key]
        if self._storage:
            self._storage.save_fact(old)
        absorbed = self._absorb_dead()
        return {
            "superseded": True,
            "old_id": old_id,
            "new_id": new_id,
            "absorbed": absorbed,
        }

    def _live_slot_occupants(self, subject: str, predicate: str) -> list[str]:
        """Caller must hold self._lock. Live fact_ids with this (s, p) slot.

        A "slot" is the (case-insensitive, whitespace-normalised) subject and
        predicate pair — the unit ``set_fact`` enforces uniqueness on. Only
        non-deprecated, non-expired facts count as occupants.
        """
        s_norm = " ".join(subject.lower().split())
        p_norm = " ".join(predicate.lower().split())
        out: list[str] = []
        for f in self._facts.values():
            if f.is_deprecated or f.is_expired:
                continue
            if (" ".join(f.subject.lower().split()) == s_norm
                    and " ".join(f.predicate.lower().split()) == p_norm):
                out.append(f.fact_id)
        return out

    def set_fact(
        self,
        subject: str,
        predicate: str,
        obj: str,
        session_id: Optional[str] = None,
    ) -> dict:
        """Slot-based upsert: ``(subject, predicate)`` becomes a unique slot.

        Whatever live facts already exist with the same ``(subject, predicate)``
        — regardless of their ``object`` — get superseded by the new one. The
        new fact takes the SPO slot; the old bodies land in the singularity
        with ``deprecated_by`` pointing at the new fact, exactly like
        ``supersede_fact`` does.

        This is the right tool for "mutable scalar" knowledge — version
        strings, HEADs, current counts, settings — where one canonical value
        replaces the previous one. ``record_fact`` stays the append-only
        primitive for atomic relations where multiple objects can coexist
        ("api uses Postgres" + "api uses Redis"). Pick by intent.
        """
        # add_fact returns an authoritative ``created`` flag from inside
        # its own write transaction: True only when this call actually
        # inserted the row, False when a dedup branch (pre-existing SPO
        # or race-winner from another process) returned the existing
        # fact. ``already_existed = not created`` is therefore
        # transaction-honest — no race window between a pre-check and
        # the insert.
        new_fact, created = self.add_fact(
            subject, predicate, obj,
            session_id=session_id, return_status=True,
        )
        already_existed = not created

        # AUTHORITATIVE slot recompute inside a write transaction — the
        # pre-add snapshot is not enough in multi-process: another writer
        # could have inserted a same-(subject, predicate) body between
        # the pre-check and add_fact. Recompute under the write lock so
        # every current occupant other than the new fact is superseded.
        superseded: list[str] = []
        with self._lock:
            with self._txn():
                self._sync()
                occupants = self._live_slot_occupants(subject, predicate)
                for old_id in occupants:
                    if old_id == new_fact.fact_id:
                        continue
                    # Use the locked helper rather than the public
                    # supersede_fact: we are already inside the lock + txn
                    # and have just synced, so re-entering them just to
                    # call the public wrapper would be wasteful and
                    # harder to reason about.
                    result = self._supersede_fact_locked(
                        old_id, new_fact.fact_id)
                    if result.get("superseded"):
                        superseded.append(old_id)

        return {
            "set": True,
            "fact_id": new_fact.fact_id,
            "subject": subject,
            "predicate": predicate,
            "object": obj,
            "already_existed": already_existed,
            "superseded": superseded,
        }

    def retire_fact(self, fact_id: str) -> dict:
        """Mark a fact as no longer relevant and send it to the singularity.

        Use when a fact is stale but has no direct replacement (the topic
        is just over). The fact's ``ttl`` is set to "now" so the next
        absorption pass treats it as expired; an immediate ``_absorb_dead``
        runs in the same call so the caller sees the effect synchronously.

        Like ``supersede_fact``, the row stays in storage so the body can
        feed singularity collapse and be Hawking-emitted if a future query
        wakes it up. Use ``delete_fact`` only when the data must truly
        cease to exist (secrets, accidental insertions).
        """
        with self._lock:
            with self._txn():
                self._sync()
                if fact_id not in self._facts:
                    return {"retired": False, "reason": "fact_id not found"}
                fact = self._facts[fact_id]
                fact.ttl = time.time()
                if self._storage:
                    self._storage.save_fact(fact)
                absorbed = self._absorb_dead()
        return {
            "retired": True,
            "fact_id": fact_id,
            "absorbed": absorbed,
        }

    # ── Session lifecycle ────────────────────────────────────────────────────

    def session_start(self, session_id: str) -> None:
        """Open a session context. Safe to call concurrently."""
        with self._lock:
            with self._txn():
                self._sync()
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
            with self._txn():
                self._sync()
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

    # Sentiment shortcut → R value mapping. Discrete choices spelled
    # out so a calling agent doesn't have to memorise which float means
    # what. 0.7 / -0.7 land cleanly inside the resonant / toxic bands
    # (threshold 0.35 / -0.20 per detector.classify) without saturating.
    _SENTIMENT_MAP: ClassVar[dict[str, float]] = {
        "resonant": 0.7,
        "positive": 0.7,
        "neutral": 0.0,
        "toxic": -0.7,
        "negative": -0.7,
    }

    def session_close(
        self,
        session_id: Optional[str] = None,
        sentiment: Optional[str] = None,
        r_override: Optional[float] = None,
    ) -> dict:
        """
        Close session: compute resonance, propagate R to facts,
        record echo bundle, tick gravity, absorb dead facts.

        Operates on the named session if provided; otherwise on the
        most recently opened one.

        Resonance scoring — three paths, caller's choice:
          - ``r_override`` (float in [-1, 1]): use this exact value.
            Wins over everything else. Right when the caller already
            knows the realised value precisely.
          - ``sentiment`` (``"resonant"`` / ``"neutral"`` / ``"toxic"``,
            or aliases ``"positive"`` / ``"negative"``): discrete
            shortcut. Maps to ±0.7 / 0.0 — lands cleanly inside the
            label bands without saturating. Right when an agent
            summarises declaratively ("round closed, 6 fixes, all
            tests pass") and knows the heuristic will mis-classify
            grumpy-sounding tech vocabulary as toxic.
          - Neither: fall back to ``compute_resonance`` on the message
            text. The original contract, unchanged when sentiment +
            r_override are both None.

        Response carries ``scoring_source`` (``"heuristic"`` /
        ``"sentiment"`` / ``"r_override"``) for transparency.
        """
        with self._lock:
            self._sync()
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

        # Resolve the resonance score. Override paths skip the
        # behavioural / semantic / repetition computation entirely.
        scoring_source = "heuristic"
        if r_override is not None:
            r_value = float(max(-1.0, min(1.0, r_override)))
            label = (
                "resonant" if r_value > 0.35
                else ("toxic" if r_value < -0.20 else "neutral")
            )
            result = ResonanceResult(
                behavioral_score=0.0, semantic_score=0.0,
                repetition_score=0.0, r=r_value, label=label,
            )
            scoring_source = "r_override"
        elif sentiment is not None:
            if sentiment not in self._SENTIMENT_MAP:
                raise ValueError(
                    f"sentiment must be one of "
                    f"{sorted(self._SENTIMENT_MAP)}, got {sentiment!r}"
                )
            r_value = self._SENTIMENT_MAP[sentiment]
            label = (
                "resonant" if r_value > 0.35
                else ("toxic" if r_value < -0.20 else "neutral")
            )
            result = ResonanceResult(
                behavioral_score=0.0, semantic_score=0.0,
                repetition_score=0.0, r=r_value, label=label,
            )
            scoring_source = "sentiment"
        else:
            # Heuristic path — pure computation on the snapshot, run
            # outside the lock so other agents can keep querying.
            result = compute_resonance(
                messages_snapshot,
                start_vector=vectors_snapshot[0],
                end_vector=vectors_snapshot[-1],
                all_vectors=vectors_snapshot,
            )

        with self._lock:
            with self._txn():
                # Reload under the write lock — tick recomputes gravity for
                # every fact, so it must run on the authoritative state.
                self._sync()

                # Snapshot pre-resonance features for facts about to receive
                # their first ever resonance — these are the training events
                # for the adaptive weights. Resonance is the teacher; the
                # weights learn what predicted realised value *before* it.
                now_ts = time.time()
                max_deg = max(self._engine._degrees.values(), default=1)
                training_features: list[
                    tuple[float, float, float, float, float]
                ] = []
                for fid in facts_snapshot:
                    # Body may be a live FactPassport or a live MetaFact
                    # (Hawking-emitted); both share the same gravity surface
                    # and must train the formula symmetrically.
                    body = (self._facts.get(fid)
                            or self._meta_facts.get(fid))
                    if body is not None and body.resonance_count == 0:
                        training_features.append(pre_resonance_features(
                            body,
                            graph_degree=self._engine._degrees.get(fid, 0),
                            max_degree=max_deg,
                            now=now_ts,
                        ))

                # Propagate R to facts used in this session, weighted by how
                # relevant each fact was to the session's queries.
                self._engine.apply_session_resonance(facts_snapshot, result.r)

                # Touch every body the session actually consulted (so
                # access_count / last_accessed reflect the read), then
                # update recent_utility EWMA via the shared helper —
                # symmetric with check_echo which now applies the same
                # EWMA update on retroactive negative penalty.
                for fid in facts_snapshot:
                    body = (self._facts.get(fid)
                            or self._meta_facts.get(fid))
                    if body is not None:
                        body.touch()
                self._apply_recent_utility_locked(facts_snapshot, result.r)

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
                    if expired and hasattr(self._storage, "delete_echo_session"):
                        for stale_sid in expired:
                            self._storage.delete_echo_session(stale_sid)

                migrations = self._engine.tick()
                absorbed = self._absorb_dead()

                if self._storage:
                    self._storage.save_facts(list(self._facts.values()))
                    if self._meta_facts and hasattr(self._storage, "save_meta_facts"):
                        self._storage.save_meta_facts(list(self._meta_facts.values()))

                # Train the adaptive weights: one regularised SGD step per
                # session toward (R + 1) / 2, using the mean of first-resonance
                # features as the predictor. One signal in, one step out — the
                # weights drift slowly toward what predicts value FOR YOU.
                if training_features:
                    n = len(training_features)
                    mean_f = sum(f[0] for f in training_features) / n
                    mean_a = sum(f[1] for f in training_features) / n
                    mean_g = sum(f[2] for f in training_features) / n
                    mean_u = sum(f[3] for f in training_features) / n
                    mean_s = sum(f[4] for f in training_features) / n
                    target = max(0.0, min(1.0, (result.r + 1.0) / 2.0))
                    # Multi-process safety: another process may have stepped
                    # the weights between our boot and this commit. Reload
                    # the authoritative row UNDER the write txn so our SGD
                    # step composes on top of their learning, not on top of
                    # our stale in-memory copy. Without this, the last
                    # writer silently overwrites every concurrent process's
                    # train_count.
                    if (self._storage
                            and hasattr(self._storage, "load_adaptive_weights")):
                        fresh = self._storage.load_adaptive_weights()
                        if (fresh is not None
                                and fresh.train_count
                                    >= self._engine.weights.train_count):
                            self._engine.weights = fresh
                    self._engine.weights.update(
                        mean_f, mean_a, mean_g, mean_u, mean_s, target=target,
                    )
                    if self._storage and hasattr(self._storage, "save_adaptive_weights"):
                        self._storage.save_adaptive_weights(self._engine.weights)

                # Counter-triggered collapse. Held inside the lock so the
                # collapse counter and the trigger decision are consistent.
                self._maybe_trigger_collapse_locked(len(absorbed))

                summary = {
                    "session_id": sid,
                    "r": result.r,
                    "label": result.label,
                    "migrations": migrations,
                    "absorbed": absorbed,
                    # Tells the caller which path resolved R: heuristic
                    # text scan (the original contract) vs an explicit
                    # sentiment label vs a direct numeric override.
                    "scoring_source": scoring_source,
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
        """Send facts and live MetaFacts below the threshold back into the hole.

        Absorbed bodies are NOT deleted from storage — they are persisted
        with ``layer = -1`` so that a restart re-hydrates the singularity
        via ``BlackHole.restore_fact`` (see ``_load_from_storage``) and
        Hawking emission / collapse lineage survive the crash. Only the
        explicit ``delete_fact`` primitive removes a row from storage.
        """
        absorbed = []
        for fid, fact in list(self._facts.items()):
            falls_to_hole = (
                fact.is_deprecated or fact.is_expired
                or fact.gravity_score < _ABSORPTION_THRESHOLD
            )
            if not falls_to_hole:
                continue
            fact.layer = -1
            self._hole.absorb(fact)
            del self._facts[fid]
            self._index.remove(fid)
            self._drop_from_spo_index(fact)
            if self._storage:
                # Persist the layer=-1 transition so the body survives
                # restart inside the singularity (not as a live fact).
                self._storage.save_fact(fact)
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
            with self._txn():
                self._sync()
                new_metas, report = collapse_singularity(
                    self._hole, threshold=thr, min_group_size=mgs,
                )
                if persist and self._storage and hasattr(self._storage, "save_meta_facts"):
                    self._storage.save_meta_facts(new_metas)
                    # Source FactPassports now live as MetaFact lineage
                    # (source_fact_ids / source_texts); their layer=-1 rows
                    # in the facts table are no longer needed and would
                    # otherwise be re-hydrated into the singularity on next
                    # restart. Drop them — and their incident edges — now
                    # that the bundle owns the lineage.
                    for meta in new_metas:
                        for fid in meta.source_fact_ids:
                            if hasattr(self._storage, "delete_fact"):
                                self._storage.delete_fact(fid)
                            if hasattr(self._storage, "delete_edges_for_fact"):
                                self._storage.delete_edges_for_fact(fid)
                now_ts = time.time()
                self._last_collapse_attempt_at = now_ts
                self._total_collapse_attempts += 1
                # Only count as a successful collapse if something actually
                # compressed; otherwise total_collapses would lie ("we
                # collapsed 47 times" when nothing was bundled).
                if report.groups > 0:
                    self._last_collapse_at = now_ts
                    self._total_collapses += 1
                # Reset counter regardless so we don't re-trigger on the
                # same empty conditions in a tight loop.
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

    def run_forecast(self, horizon_ticks: int = 50) -> dict:
        """Run a galaxy forecast and write ``forecast_stability`` back to facts.

        The galaxy module models a fact as a body in orbit around a central
        black hole; running it forward gives a per-fact prediction of how
        close that body will be to the event horizon after ``horizon_ticks``
        steps. Stability ∈ [0, 1]: 1.0 = predicted safely on surface,
        0.0 = predicted to fall, 0.5 = neutral prior (default for facts the
        galaxy could not place).

        The value is stored on FactPassport.forecast_stability and consumed
        by the adaptive gravity formula via ``w_stability`` — so this call
        materially feeds back into how the formula scores facts on the next
        tick. The galaxy build + simulation is O(n²) per step in fact count
        and pure numpy, fine for the few hundred to few thousand facts a
        personal store holds. Returns a small summary; full per-fact values
        are persisted, not returned.
        """
        from .galaxy.forecast import forecast_stability

        with self._lock:
            with self._txn():
                self._sync()
                # Forecast both live FactPassports and live MetaFacts.
                # MetaFacts carry forecast_stability and feed the same
                # adaptive gravity formula, so leaving them at a neutral
                # prior while facts get a learned forecast was an
                # asymmetric contract.
                bodies_snapshot: list = list(self._facts.values())
                bodies_snapshot.extend(self._meta_facts.values())
                # Cache hit: same data_version + body count + horizon
                # means the simulation would produce the same result
                # (forecast_stability is pure over the body snapshot and
                # horizon). Return the previous response verbatim with
                # a cached=True marker so callers can tell.
                cache_key = (
                    self._data_version_now(),
                    len(bodies_snapshot),
                    horizon_ticks,
                )
                if (self._forecast_cache is not None
                        and self._forecast_cache[0] == cache_key):
                    cached = dict(self._forecast_cache[1])
                    cached["cached"] = True
                    return cached

        # The simulation itself is pure numpy and reads no shared state —
        # run it OUTSIDE the lock so other agents can keep querying.
        scores = forecast_stability(bodies_snapshot, horizon_ticks=horizon_ticks)

        with self._lock:
            with self._txn():
                self._sync()
                updated_facts: list = []
                updated_metas: list = []
                for bid, score in scores.items():
                    fact = self._facts.get(bid)
                    if fact is not None:
                        fact.forecast_stability = float(score)
                        updated_facts.append(fact)
                        continue
                    meta = self._meta_facts.get(bid)
                    if meta is not None:
                        meta.forecast_stability = float(score)
                        updated_metas.append(meta)
                if self._storage:
                    if updated_facts:
                        self._storage.save_facts(updated_facts)
                    if (updated_metas
                            and hasattr(self._storage, "save_meta_facts")):
                        self._storage.save_meta_facts(updated_metas)
                facts_updated_n = len(updated_facts)
                metas_updated_n = len(updated_metas)
                updated = facts_updated_n + metas_updated_n

        # Quick distribution snapshot so the caller can see what landed.
        ranges = {"safe": 0, "kinetic": 0, "near_horizon": 0, "predicted_fall": 0}
        for score in scores.values():
            if score >= 0.7:
                ranges["safe"] += 1
            elif score >= 0.3:
                ranges["kinetic"] += 1
            elif score > 0.0:
                ranges["near_horizon"] += 1
            else:
                ranges["predicted_fall"] += 1
        result_payload = {
            "horizon_ticks": horizon_ticks,
            "cached": False,
            # Kept for wire-format stability; aliases of the new keys.
            "facts_forecasted": len(scores),
            "facts_updated": updated,
            # Clearer keys: forecast now covers both FactPassports and
            # MetaFacts (both carry forecast_stability), so the operator
            # can see how the update split across body types.
            "bodies_forecasted": len(scores),
            "bodies_updated": updated,
            "facts_updated_count": facts_updated_n,
            "metas_updated_count": metas_updated_n,
            "distribution": ranges,
            "_hint": (
                "facts_forecasted / facts_updated are legacy aliases — "
                "they actually count BODIES (FactPassport + MetaFact). "
                "Prefer bodies_forecasted / bodies_updated, or read the "
                "per-type split via facts_updated_count and "
                "metas_updated_count."
            ),
        }
        # Cache the response keyed by the snapshot we forecasted against.
        # Subsequent calls with no intervening writes hit the cache.
        self._forecast_cache = (cache_key, dict(result_payload))
        return result_payload

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
        subject_prefix: Optional[str] = None,
        min_gravity: float = 0.0,
        allowed_layers: Optional[set[int]] = None,
    ) -> list[QueryResult]:
        """
        Retrieve relevant facts by cosine similarity.

        Searches live layers first. If hawking=True, also attempts
        Hawking emission from the black hole for extreme matches.

        Filters (``subject_prefix``, ``min_gravity``, ``min_similarity``)
        are applied **before** the top_k slice, so a narrow scope still
        returns its top hits instead of an empty list when the matching
        facts sit beyond top_k in the full ranking.

        Side effects on every returned fact:
          - access_count is incremented (touch)
          - if a session is active, fact_id is attributed to it so the
            session's resonance later propagates back to its gravity.
        """
        # top_k guard at the core boundary. The server layer rejects
        # non-positive top_k with a structured response (round 10), but
        # MemoryStore.query is a public core API used in tests and
        # embedded mode too — a negative top_k would slice
        # results[:top_k] from the right end (returning all-except-last),
        # not return "nothing". That is a Python-list semantics trap;
        # close it at the core. (Also skips a needless embed roundtrip.)
        if top_k <= 0:
            return []
        # Embed outside the lock.
        vec = embed(text)
        prefix = subject_prefix.lower() if subject_prefix else None

        with self._lock:
            self._sync()
            results: list[QueryResult] = []
            layer_labels = {0: "surface", 1: "kinetic", 2: "core"}

            # Live FactPassports.
            sims = self._index.all_similarities(vec)
            for fid, sim in sims.items():
                fact = self._facts.get(fid)
                if fact is None:
                    continue
                # Lifecycle filter — symmetric with the Hawking predicate.
                # A deprecate() call sets deprecated_by without going through
                # _absorb_dead, and TTL may expire between ticks. Either case
                # used to leak the body into live results until next tick.
                if fact.is_deprecated or fact.is_expired:
                    continue
                if not (min_layer <= fact.layer <= max_layer):
                    continue
                if allowed_layers is not None and fact.layer not in allowed_layers:
                    continue
                if fact.gravity_score < min_gravity:
                    continue
                if prefix and not fact.subject.lower().startswith(prefix):
                    continue
                results.append(QueryResult(
                    fact=fact,
                    similarity=round(sim, 4),
                    source=layer_labels.get(fact.layer, "kinetic"),
                ))

            # Live MetaFacts — promoted out of the black hole by past
            # Hawking emissions; share the same layer machinery as facts.
            # MetaFacts have no single "subject" so subject_prefix never
            # filters them in; min_gravity still applies symmetrically.
            meta_sims = self._meta_index.all_similarities(vec)
            for mid, sim in meta_sims.items():
                meta = self._meta_facts.get(mid)
                if meta is None:
                    continue
                if not (min_layer <= meta.layer <= max_layer):
                    continue
                if allowed_layers is not None and meta.layer not in allowed_layers:
                    continue
                if meta.gravity_score < min_gravity:
                    continue
                if prefix:
                    # No single subject on a meta — only include if any
                    # source_text actually contains the prefix.
                    if not any((st or "").lower().startswith(prefix)
                               for st in meta.source_texts):
                        continue
                results.append(QueryResult(
                    meta=meta,
                    similarity=round(sim, 4),
                    source=layer_labels.get(meta.layer, "kinetic"),
                ))

            # Pre-Hawking sort / top_k slice — top selection is pure over
            # the live snapshot; mutation (touch / attribute / Hawking pop /
            # persist) happens together under the write transaction below.
            results.sort(key=lambda r: r.similarity, reverse=True)
            if min_similarity > 0.0:
                results = [r for r in results if r.similarity >= min_similarity]
            top = results[:top_k]
            sid = self._resolve_sid(session_id)

            # Collect intentions (ids + attribution pairs) and the data
            # Hawking needs (predicate closures + query vector). Apply them
            # to authoritative state inside a single write transaction.
            touched_fact_ids = [r.fact.fact_id for r in top if r.fact is not None]
            touched_meta_ids = [r.meta.meta_id for r in top if r.meta is not None]
            attribution_pairs: list[tuple[str, float]] = [
                (r.body_id, r.similarity) for r in top
            ]

            # Storage availability MUST NOT decide whether in-memory state
            # mutates. An in-memory store (no storage backend) still needs
            # touch/attribution to land — otherwise the feedback loop
            # silently breaks for embedded / test usage. Persistence is a
            # later concern, gated separately inside the txn block.
            #
            # Hawking-only triggering of the write path is skipped when
            # the singularity is empty — pointless BEGIN IMMEDIATE that
            # blocks other writers for no work. The hole is empty in the
            # common read-only case, so this is a meaningful perf win.
            needs_hawking_pass = hawking and self._hole.mass > 0
            need_write_path = (
                needs_hawking_pass
                or touched_fact_ids
                or touched_meta_ids
                or sid is not None
            )

            if not need_write_path:
                return top

            # ---- Write path: one transaction, _sync inside, then mutate.
            def _fact_predicate(f) -> bool:
                # Lifecycle: a fact that was superseded by set_fact /
                # supersede_fact, or expired via retire_fact, must NOT
                # come back through Hawking emission as if it were
                # current. The agent thinks it's reading live truth;
                # the body knows it has been retired.
                if f.is_deprecated or f.is_expired:
                    return False
                if f.gravity_score < min_gravity:
                    return False
                if prefix and not f.subject.lower().startswith(prefix):
                    return False
                return True

            def _meta_predicate(m) -> bool:
                if getattr(m, "is_deprecated", False):
                    return False
                if getattr(m, "is_expired", False):
                    return False
                if m.gravity_score < min_gravity:
                    return False
                if prefix:
                    if not any((st or "").lower().startswith(prefix)
                               for st in m.source_texts):
                        return False
                return True

            with self._txn():
                # Re-sync under the write lock; if another process committed
                # we now hold the authoritative state.
                self._sync()

                # Revalidate the pre-sync top: another process may have
                # deprecated / retired / deleted bodies that were in our
                # snapshot. Drop the vanished ones and replace each
                # surviving QueryResult's body with the authoritative
                # post-sync object, so the caller never sees a stale ref.
                revalidated_top: list[QueryResult] = []
                for r in top:
                    if r.fact is not None:
                        live = self._facts.get(r.fact.fact_id)
                        if live is None or live.is_deprecated or live.is_expired:
                            continue
                        r.fact = live
                        revalidated_top.append(r)
                    elif r.meta is not None:
                        live_meta = self._meta_facts.get(r.meta.meta_id)
                        if live_meta is None:
                            continue
                        r.meta = live_meta
                        revalidated_top.append(r)
                # Backfill: if revalidation dropped any hits, re-run the
                # live search on the now-authoritative state so the caller
                # gets up to top_k results instead of a short list. Only
                # runs when we actually lost something — the common case
                # (nothing was racing) costs nothing.
                if len(revalidated_top) < len(top):
                    already_in_top = {r.body_id for r in revalidated_top}
                    backfill_candidates: list[QueryResult] = []
                    fresh_sims = self._index.all_similarities(vec)
                    for fid, sim in fresh_sims.items():
                        if fid in already_in_top:
                            continue
                        fact = self._facts.get(fid)
                        if fact is None or fact.is_deprecated or fact.is_expired:
                            continue
                        if not (min_layer <= fact.layer <= max_layer):
                            continue
                        if (allowed_layers is not None
                                and fact.layer not in allowed_layers):
                            continue
                        if fact.gravity_score < min_gravity:
                            continue
                        if prefix and not fact.subject.lower().startswith(prefix):
                            continue
                        if sim < min_similarity:
                            continue
                        backfill_candidates.append(QueryResult(
                            fact=fact,
                            similarity=round(sim, 4),
                            source=layer_labels.get(fact.layer, "kinetic"),
                        ))
                    fresh_meta_sims = self._meta_index.all_similarities(vec)
                    for mid, sim in fresh_meta_sims.items():
                        if mid in already_in_top:
                            continue
                        meta = self._meta_facts.get(mid)
                        if meta is None:
                            continue
                        if not (min_layer <= meta.layer <= max_layer):
                            continue
                        if (allowed_layers is not None
                                and meta.layer not in allowed_layers):
                            continue
                        if meta.gravity_score < min_gravity:
                            continue
                        if prefix and not any(
                                (st or "").lower().startswith(prefix)
                                for st in meta.source_texts):
                            continue
                        if sim < min_similarity:
                            continue
                        backfill_candidates.append(QueryResult(
                            meta=meta,
                            similarity=round(sim, 4),
                            source=layer_labels.get(meta.layer, "kinetic"),
                        ))
                    backfill_candidates.sort(
                        key=lambda r: r.similarity, reverse=True)
                    needed = len(top) - len(revalidated_top)
                    for picked in backfill_candidates[:needed]:
                        revalidated_top.append(picked)
                        if picked.fact is not None:
                            touched_fact_ids.append(picked.fact.fact_id)
                        elif picked.meta is not None:
                            touched_meta_ids.append(picked.meta.meta_id)
                        attribution_pairs.append(
                            (picked.body_id, picked.similarity))
                top = revalidated_top
                # Re-derive the touched / attribution lists from the
                # surviving top so the persist step does not try to update
                # bodies that disappeared during the race.
                kept_after_revalidate = {r.body_id for r in top}
                touched_fact_ids = [
                    fid for fid in touched_fact_ids
                    if fid in kept_after_revalidate
                ]
                touched_meta_ids = [
                    mid for mid in touched_meta_ids
                    if mid in kept_after_revalidate
                ]
                attribution_pairs = [
                    pair for pair in attribution_pairs
                    if pair[0] in kept_after_revalidate
                ]

                # Hawking emission as a two-phase commit: PEEK candidates
                # without popping, merge into the ranking, take top_k,
                # then COMMIT (actually emit) only those that survived.
                # Previously, hawking_emit popped and re-registered every
                # eligible body even if it later fell below top_k — a
                # state mutation the caller never received. min_similarity
                # is applied to Hawking candidates too, symmetric with
                # live results.
                if hawking:
                    fact_candidates = self._hole.peek_hawking_candidates(
                        vec, predicate=_fact_predicate)
                    meta_candidates = self._hole.peek_hawking_meta_candidates(
                        vec,
                        threshold=_META_HAWKING_THRESHOLD,
                        predicate=_meta_predicate,
                    )
                    for fact, sim in fact_candidates:
                        if sim < min_similarity:
                            continue
                        top.append(QueryResult(
                            fact=fact,
                            similarity=round(sim, 4),
                            source="hawking",
                        ))
                    for meta, sim in meta_candidates:
                        if sim < min_similarity:
                            continue
                        top.append(QueryResult(
                            meta=meta,
                            similarity=round(sim, 4),
                            source="hawking_meta",
                        ))

                # Re-sort after Hawking additions and re-clamp to top_k.
                top.sort(key=lambda r: r.similarity, reverse=True)
                top = top[:top_k]
                kept_ids = {r.body_id for r in top}

                # Commit: actually emit ONLY the Hawking survivors. Bodies
                # that fell out of top_k stay in the singularity, untouched.
                if hawking:
                    hawking_survivor_fact_ids = {
                        r.fact.fact_id for r in top
                        if r.source == "hawking" and r.fact is not None
                    }
                    hawking_survivor_meta_ids = {
                        r.meta.meta_id for r in top
                        if r.source == "hawking_meta" and r.meta is not None
                    }
                    emitted = self._hole.hawking_emit(
                        vec,
                        predicate=_fact_predicate,
                        only_ids=hawking_survivor_fact_ids or None,
                    ) if hawking_survivor_fact_ids else []
                    meta_emitted = self._hole.hawking_emit_metas(
                        vec,
                        threshold=_META_HAWKING_THRESHOLD,
                        predicate=_meta_predicate,
                        only_ids=hawking_survivor_meta_ids or None,
                    ) if hawking_survivor_meta_ids else []

                    for fact in emitted:
                        self._facts[fact.fact_id] = fact
                        self._engine.register(fact)
                        self._index.add(fact.fact_id, fact.vector)
                        if not fact.is_deprecated:
                            key = self._normalize_spo(
                                fact.subject, fact.predicate, fact.object)
                            self._spo_index.setdefault(key, fact.fact_id)
                        if self._storage:
                            self._storage.save_fact(fact)
                        touched_fact_ids.append(fact.fact_id)
                        sim = VectorIndex.similarity(vec, fact.vector)
                        attribution_pairs.append((fact.fact_id, sim))
                    for meta in meta_emitted:
                        self._meta_facts[meta.meta_id] = meta
                        self._engine.register(meta)
                        self._meta_index.add(meta.meta_id, meta.vector)
                        if (self._storage
                                and hasattr(self._storage, "save_meta_fact")):
                            self._storage.save_meta_fact(meta)
                        touched_meta_ids.append(meta.meta_id)
                        sim = VectorIndex.similarity(vec, meta.vector)
                        attribution_pairs.append((meta.meta_id, sim))

                # Restrict downstream work to bodies that survived top_k.
                touched_fact_ids = [
                    fid for fid in touched_fact_ids if fid in kept_ids
                ]
                touched_meta_ids = [
                    mid for mid in touched_meta_ids if mid in kept_ids
                ]
                attribution_pairs = [
                    pair for pair in attribution_pairs if pair[0] in kept_ids
                ]

                # Apply touch + attribution to AUTHORITATIVE objects (the
                # ones currently in self._facts / self._meta_facts after
                # _sync). Previously we touched pre-sync object refs and
                # then saved fresh objects without re-applying the touch,
                # so the bump silently vanished across a multi-process
                # reload.
                fresh_facts: list[FactPassport] = []
                for fid in touched_fact_ids:
                    f = self._facts.get(fid)
                    if f is None:
                        continue
                    f.touch()
                    fresh_facts.append(f)
                fresh_metas: list[MetaFact] = []
                for mid in touched_meta_ids:
                    m = self._meta_facts.get(mid)
                    if m is None:
                        continue
                    m.touch()
                    fresh_metas.append(m)

                fresh_ctx = self._sessions.get(sid) if sid else None
                if fresh_ctx is not None:
                    for body_id, sim in attribution_pairs:
                        self._attribute_to(fresh_ctx, body_id, sim)

                if self._storage:
                    if fresh_facts:
                        self._storage.save_facts(fresh_facts)
                    if (fresh_metas
                            and hasattr(self._storage, "save_meta_facts")):
                        self._storage.save_meta_facts(fresh_metas)
                    if (fresh_ctx is not None
                            and hasattr(self._storage, "save_open_session")):
                        self._storage.save_open_session(
                            fresh_ctx.session_id,
                            fresh_ctx.messages,
                            fresh_ctx.vectors,
                            fresh_ctx.facts,
                            time.time(),
                        )

        return top

    def check_echo(self, first_message: str, session_id: Optional[str] = None) -> dict:
        """
        Check if a new session echoes a past unresolved problem.

        If echo is detected and a non-zero retroactive penalty is applied
        for the first time, the penalty is propagated to the gravity of
        every fact that the matched past session touched. Affected facts
        are re-persisted.
        """
        # session_id, when provided, is excluded from the match pool —
        # so an explicit check_echo() called with a currently-open
        # session id does not match itself. This used to be ignored
        # ("currently unused") which was harmless in the normal
        # session_open(first_message=...) flow but a footgun for any
        # workflow that calls check_echo after the session was
        # already recorded.
        vec = embed(first_message)

        with self._lock:
            with self._txn():
                self._sync()
                result = self._echo.detect_echo(
                    vec, exclude_session_id=session_id)
                penalized_body_ids: list[str] = []
                if result.label == "echo" and result.penalty != 0.0 and result.fact_weights:
                    # engine.apply_session_resonance is polymorphic over
                    # FactPassport and MetaFact — both are registered in
                    # the engine. Previously we only persisted the
                    # FactPassport changes, losing MetaFact penalty
                    # updates on restart.
                    self._engine.apply_session_resonance(
                        result.fact_weights, result.penalty)
                    # Mirror the EWMA update too: an echo penalty is
                    # retroactive realised-negative-utility, so leaving
                    # recent_utility unchanged would make the gravity
                    # formula see contradictory signals (resonance got
                    # worse, utility stayed where it was). Same helper
                    # session_close uses.
                    self._apply_recent_utility_locked(
                        result.fact_weights, result.penalty)
                    penalized_body_ids = list(result.fact_weights.keys())

                    if self._storage:
                        affected_facts = [
                            self._facts[bid] for bid in penalized_body_ids
                            if bid in self._facts
                        ]
                        affected_metas = [
                            self._meta_facts[bid] for bid in penalized_body_ids
                            if bid in self._meta_facts
                        ]
                        if affected_facts:
                            self._storage.save_facts(affected_facts)
                        if (affected_metas
                                and hasattr(self._storage, "save_meta_facts")):
                            self._storage.save_meta_facts(affected_metas)
                        past = (
                            self._echo.get(result.matched_session_id)
                            if result.matched_session_id
                            else None
                        )
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
                    # Kept under the old key for wire-format stability.
                    # The list may include MetaFact ids too — they all
                    # received the echo penalty symmetrically.
                    "penalized_fact_ids": penalized_body_ids,
                }

    # ── Status ───────────────────────────────────────────────────────────────

    @property
    def stats(self) -> dict:
        with self._lock:
            self._sync()
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
                "total_collapse_attempts": self._total_collapse_attempts,
                "last_collapse_at": self._last_collapse_at,
                "last_collapse_attempt_at": self._last_collapse_attempt_at,
                "adaptive_weights": self._engine.weights.as_dict(),
                "echo_sessions": len(self._echo),
                "total_echoes_detected": self._echo.total_echoes_detected,
                "total_echoes_applied": self._echo.total_echoes_applied,
                "total_echoes_ignored": self._echo.total_echoes_ignored,
                # Diagnostics: which thresholds the process actually
                # picked up. Operator can confirm BIRCH_* env vars
                # took effect without reading the process environment.
                "thresholds": Thresholds.as_dict(),
            }
