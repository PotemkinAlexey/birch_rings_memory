"""Composition root for MemoryStore.

Contains the dataclasses (``QueryResult``, ``SessionContext``), the
module-level threshold constants, and the assembled ``MemoryStore``
class that mixes in the per-area mixins. Lifecycle / utility methods
that do not naturally belong to one mixin (``__init__``, ``_sync``,
``_reload``, ``_txn``, ``_load_from_storage``, ``close``) live here.
"""
from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

from ..black_hole import BlackHole
from ..fact import FactPassport
from ..gravity import GravityEngine
from ..meta_fact import MetaFact
from ..resonance.cluster import ClusterBundle
from ..resonance.echo import EchoStore, StoredSession
from ..storage import SQLiteBackend, StorageBackend
from ..thresholds import Thresholds
from ..vector_index import DimensionMismatchError, VectorIndex
from ._facts import FactsMixin
from ._models import SessionContext
from ._query import QueryMixin
from ._sessions import SessionsMixin
from ._singularity import SingularityMixin
from ._stats import StatsMixin

_logger = logging.getLogger(__name__)

# Module-level aliases kept for the existing call sites; the values
# now come from the centralised env-overridable Thresholds module.
# Operators can pin every threshold via BIRCH_* env vars to match
# their embedding model's cosine distribution.
_ABSORPTION_THRESHOLD = Thresholds.ABSORPTION
_META_HAWKING_THRESHOLD = Thresholds.HAWKING_META


class MemoryStore(
    SessionsMixin,
    FactsMixin,
    QueryMixin,
    SingularityMixin,
    StatsMixin,
):
    """
    Three-layer memory with black hole sink and Hawking emission.

    Layers:
      0 — surface  (gravity > 0.70, hot facts)
      1 — kinetic  (gravity 0.30–0.70, working memory)
      2 — core     (gravity < 0.30, cold archive)
     -1 — black hole (gravity < 0.10 after tick, absorbed)
    """

    # Minimum cosine similarity for auto-linking two facts. Sourced
    # from the env-overridable Thresholds module so an operator pinning
    # a different embedding model can tune it.
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
        # Async collapse failures used to be swallowed in close(). Now
        # we capture the last error string and surface it in stats
        # so an operator can see a worker crash without grepping logs.
        # Process-lifetime, reset only on successful collapse.
        self._last_collapse_error: Optional[str] = None
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
        # MemoryBricks Step 1: key is (namespace, s, p, o) — see
        # ``FactsMixin._normalize_spo``. The first element scopes
        # dedup so two facts with the same SPO under different
        # namespaces coexist as independent live rows.
        self._spo_index: dict[tuple[str, str, str, str], str] = {}
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
        # Sessions currently inside session_close after the snapshot
        # phase. session_message must reject pushes to a closing
        # session — otherwise a late message lands in ctx.messages,
        # gets persisted to disk, and is then silently dropped when
        # session_close pops the ctx (it computed R over the snapshot
        # taken BEFORE the late push). The agent saw push succeed
        # but the message never influenced R / echo / future
        # sessions. Set membership is the cheapest possible gate.
        self._closing_sessions: set[str] = set()
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
            tuple[int, int, int, int], dict
        ] | None = None
        # Process-local mutation counter. SQLite's PRAGMA data_version
        # only changes for writes from OTHER connections — same-process
        # writes leave it untouched. Without this counter, run_forecast
        # could serve a stale cached result after the same process
        # added a fact (body count unchanged → cache_key unchanged).
        # Bumped by every method that mutates _facts / _meta_facts /
        # _hole — see _bump_mutation calls below.
        self._mutation_version: int = 0
        # Process-life set of fact_ids that salience (irreplaceability) kept
        # from disuse-absorption at least once. Surfaced as a count in stats.
        self._salience_retained_ids: set[str] = set()
        # Encoding-salience (declarative pin) telemetry — the metric that earns
        # the channel the right to exist: every fact ever pinned, those that
        # later rode a resonant session (declaration predicted criticality),
        # and budget evictions. A near-zero resonated:created ratio over real
        # traffic = people pin noise → bury the channel.
        self._ever_pinned_ids: set[str] = set()
        self._pins_resonated_ids: set[str] = set()
        self._pins_evicted: int = 0

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
        """Rebuild every in-memory cache from storage. Caller holds self._lock.

        Atomic-swap pattern. The previous version cleared every cache
        BEFORE calling ``_load_from_storage`` — if the loader raised
        mid-rebuild (transient ``database is locked``, a row class no
        existing tolerant loader catches, etc.) the store was left
        with empty caches AND ``_data_version`` unchanged from before
        the clear. The next ``_sync`` then saw equal data_version and
        skipped the retry — the store would silently return zero
        results until something else bumped the version. ``_reload``
        is the failsafe for every other invariant in this file; it
        itself has to be crash-safe.

        Build into fresh local instances first, install with whole-
        attribute assignment only when ``_load_from_storage`` returns
        cleanly. On any exception the previous caches stay live; we
        also reset ``_data_version`` to a sentinel so the next call
        retries instead of declaring the empty view authoritative.
        """
        if self._storage is None:
            return
        # Stash references to the live caches so we can swap atomically.
        saved_facts = self._facts
        saved_spo = self._spo_index
        saved_index = self._index
        saved_meta_facts = self._meta_facts
        saved_meta_index = self._meta_index
        saved_engine = self._engine
        saved_hole = self._hole
        saved_echo = self._echo
        saved_sessions = self._sessions
        saved_current = self._current_session_id
        # Install fresh empty caches; _load_from_storage populates
        # these by reading self._facts/etc., so we have to assign
        # before the call. On failure we restore the saved refs.
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
        try:
            # _reload is the rollback-recovery path — must be strictly
            # read-only. _load_from_storage's default (prune=True) does
            # destructive cleanup (orphan edges + TTL'd sessions) which
            # is the right behaviour at fresh process start (__init__)
            # but not under recovery: a destructive write executed
            # mid-rollback could shadow the failing transaction's
            # original error and confuse the caller. Pass prune=False
            # so reload truly only reads.
            self._load_from_storage(prune=False)
        except Exception:
            # Restore the pre-reload view so callers don't see a
            # phantom-empty store. Force data_version to a sentinel
            # so the very next _sync retries instead of trusting the
            # empty load.
            self._facts = saved_facts
            self._spo_index = saved_spo
            self._index = saved_index
            self._meta_facts = saved_meta_facts
            self._meta_index = saved_meta_index
            self._engine = saved_engine
            self._hole = saved_hole
            self._echo = saved_echo
            self._sessions = saved_sessions
            self._current_session_id = saved_current
            self._data_version = -1
            raise
        self._data_version = self._data_version_now()
        # Defensive: cross-process sync rebuilt everything in-memory,
        # so any cached recompute keyed on the old snapshot is now
        # formally invalid. The forecast cache key already includes
        # data_version (which bumps for other-process writes), but
        # explicit invalidation here closes the subtle window where
        # a multi-process race could leave the same data_version
        # value briefly observable on both sides.
        self._forecast_cache = None

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

    def _load_from_storage(self, *, prune: bool = True) -> None:
        """Rebuild every in-memory cache from storage.

        ``prune`` (default True for backwards compat — __init__ path
        wants self-healing) controls whether destructive cleanup
        runs during the load: orphan-edge GC + TTL-expired open
        session sweep. Both write to disk. _reload sets ``prune=False``
        so rollback-recovery rebuilds are strictly read-only — a
        leaky cleanup mid-rollback could interact unexpectedly with
        the failing transaction higher up.
        """
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
                # BlackHole's singularity is now per-dim partitioned
                # (each dim gets its own VectorIndex bucket), so a
                # mixed-dim singularity post-model-swap routes cleanly:
                # the old-dim bodies live in one bucket, the new-dim
                # bodies in another, and Hawking emission scans only
                # the matching dim. The try/except wrap that the
                # earlier rounds needed here is gone with the cause.
                self._hole.restore_fact(fact)
                continue
            self._facts[fact.fact_id] = fact
            self._engine.register(fact)
            # Mixed-dim live facts (after an embedding-model swap
            # without reindex) must not take startup down. The fact
            # stays registered with gravity engine for layer-only ops;
            # its vector is cleared so downstream consumers can't try
            # to use a dim that won't match anything else. The fact
            # remains visible through list_facts and gravity migrations
            # but won't appear in semantic search until reindexed.
            try:
                self._index.add(fact.fact_id, fact.vector)
            except DimensionMismatchError:
                _logger.warning(
                    "fact %r loaded but not indexed (vector dim "
                    "mismatch — embedding model likely changed; "
                    "rebuild the store with a single BIRCH_EMBED_MODEL)",
                    fact.fact_id,
                )
                fact.vector = []
            if not fact.is_deprecated:
                # MemoryBricks Step 1: read the body's namespace so
                # rebuild matches what _drop_from_spo_index will later
                # look up. Symmetric with the same line in the Hawking
                # emission path in _query.py.
                key = self._normalize_spo(
                    fact.subject, fact.predicate, fact.object,
                    fact.namespace,
                )
                self._spo_index.setdefault(key, fact.fact_id)
        # MetaFacts: layer -1 live in the singularity; promoted ones (after
        # Hawking emission) live in the live meta store with the engine.
        if hasattr(self._storage, "load_meta_facts"):
            for meta in self._storage.load_meta_facts(cleanup=prune):
                if meta.layer == -1:
                    # Per-dim singularity routes by dim — no
                    # mismatch possible on restore. The previous
                    # try/except wrap is obsolete.
                    self._hole.restore_meta(meta)
                else:
                    self._meta_facts[meta.meta_id] = meta
                    try:
                        self._meta_index.add(meta.meta_id, meta.vector)
                    except DimensionMismatchError:
                        _logger.warning(
                            "metafact %r loaded but not indexed "
                            "(vector dim mismatch)", meta.meta_id,
                        )
                        meta.vector = []
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
        if (prune and stale_edges and self._storage
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
        for row in self._storage.load_echo_sessions(cleanup=prune):
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
            for row in self._storage.load_open_sessions(cleanup=prune):
                if now - row["started_at"] > _SESSION_TTL:
                    if prune:
                        self._storage.delete_open_session(row["session_id"])
                    # Either way, skip rehydrating the expired session
                    # into memory — restart-from-stale would otherwise
                    # resurrect crashed sessions.
                    continue
                ctx = SessionContext(session_id=row["session_id"])
                ctx.messages = row["messages"]
                ctx.vectors = row["vectors"]
                ctx.facts = {k: float(v) for k, v in row["facts"].items()}
                self._sessions[row["session_id"]] = ctx
                self._current_session_id = row["session_id"]

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
            except Exception as exc:
                # Don't crash close() — but don't lose the error either.
                # Stored on the instance (best effort, since stats may
                # be read shortly after close in tests / shutdown
                # handlers).
                self._last_collapse_error = repr(exc)
        # cancel_futures only stops PENDING futures; a worker already
        # running keeps going. So we have to choose between:
        #  - wait=True: bounded by however long collapse takes;
        #  - wait=False + closing storage: worker hits closed SQLite
        #    connection mid-write, surfaces as opaque background
        #    exception after close().
        # Pick safety: if the inflight finished (common case), shutdown
        # and close cleanly. If it timed out, leave the storage handle
        # open so the still-running worker can finish its writes —
        # leaking the handle until process exit is much better than
        # corrupting an in-flight collapse with a closed connection.
        # The leak is bounded (close() is end-of-life), the corruption
        # would not be.
        inflight_done = inflight is None or inflight.done()
        if executor is not None:
            if inflight_done:
                executor.shutdown(wait=True)
            else:
                executor.shutdown(wait=False, cancel_futures=True)
        if storage is not None and hasattr(storage, "close"):
            if inflight_done:
                storage.close()
            else:
                # Record the leak so tests / observers can see why
                # the handle is dangling.
                self._last_collapse_error = (
                    (self._last_collapse_error or "")
                    + " | storage handle leaked: collapse worker still "
                    "running at close()"
                ).strip(" |")
