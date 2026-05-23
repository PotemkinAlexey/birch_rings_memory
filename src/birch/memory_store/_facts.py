"""FactsMixin — fact CRUD, dedup, SPO index, gravity-engine writes.

Extracted from the historical single-file ``birch.memory_store`` module.
Method bodies are verbatim — only the enclosing class changed.
"""
from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional, Union, overload

from ..fact import FactPassport
from ..vector_index import DimensionMismatchError
from ._embed_proxy import embed, embed_batch

if TYPE_CHECKING:  # pragma: no cover
    from concurrent.futures import Future, ThreadPoolExecutor

    from ..black_hole import BlackHole
    from ..gravity import GravityEngine
    from ..meta_fact import MetaFact
    from ..resonance.echo import EchoStore
    from ..storage import StorageBackend
    from ..vector_index import VectorIndex
    from ._models import SessionContext


class FactsMixin:
    """Fact-management methods. See ``MemoryStore`` for the assembled API."""

    # These attributes are provided by MemoryStore.__init__; declared
    # here so mypy / type checkers can see the contract a mixin relies
    # on without forcing each method to redeclare it.
    _lock: "threading.RLock"
    _storage: "Optional[StorageBackend]"
    _facts: "dict[str, FactPassport]"
    _meta_facts: "dict[str, MetaFact]"
    _spo_index: "dict[tuple[str, str, str], str]"
    _index: "VectorIndex"
    _meta_index: "VectorIndex"
    _engine: "GravityEngine"
    _hole: "BlackHole"
    _echo: "EchoStore"
    _sessions: "dict[str, SessionContext]"
    _current_session_id: "Optional[str]"
    _auto_link: bool
    _mutation_version: int
    _forecast_cache: "Optional[tuple[tuple[int, int, int, int], dict]]"
    _collapse_counter: int
    _collapse_async: bool
    _collapse_threshold: float
    _collapse_min_group_size: int
    _collapse_executor: "Optional[ThreadPoolExecutor]"
    _inflight_collapse: "Optional[Future]"
    _last_collapse_error: "Optional[str]"
    AUTO_LINK_THRESHOLD: float
    AUTO_LINK_TOP_K: int
    COLLAPSE_FACT_MASS_TRIGGER: int
    COLLAPSE_DELTA_TRIGGER: int

    # Cross-mixin helpers provided by sibling mixins / the base class.
    # Declared as ``Callable`` attributes (not method stubs) so the
    # annotation does NOT shadow the real binding at runtime — only
    # mypy sees them; ``MemoryStore`` provides the real implementations
    # via mixin composition.
    if TYPE_CHECKING:
        _sync: Callable[[], None]
        _txn: Callable[[], Any]
        _resolve_sid: Callable[[Optional[str]], Optional[str]]
        _attribute_to: Callable[["SessionContext", str, float], None]
        _persist_session_locked: Callable[["Optional[SessionContext]"], None]
        _absorb_dead: Callable[[], list[str]]

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
                # Preflight: if the embedding model changed under the
                # store, _index.add() will raise DimensionMismatchError.
                # Doing it BEFORE we mutate _facts/_engine means a
                # mismatch leaves no half-state behind (otherwise the
                # fact would live in _facts + _engine but not in the
                # vector index, and storage rollback wouldn't undo
                # the in-memory writes).
                if (vec and self._index._dim is not None
                        and len(vec) != self._index._dim):
                    raise DimensionMismatchError(
                        f"Embedding dimension mismatch: index has "
                        f"dim={self._index._dim}, incoming vector has "
                        f"dim={len(vec)}. The embedding model probably "
                        f"changed under the store. Pin BIRCH_EMBED_MODEL "
                        f"or rebuild the store before writing facts."
                    )
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
                self._mutation_version += 1
                return (fact, True) if return_status else fact

    def _bump_mutation_locked(self) -> None:
        """Caller must hold self._lock. Single source of truth for
        invalidating same-process caches that key on body state.

        Used in two situations:

        1. Any write path that mutates _facts / _meta_facts / _hole
           or persisted fact state. The mutation_version composes
           with SQLite's data_version (which only bumps for OTHER-
           connection writes) so the forecast cache key fully
           captures "something changed, recompute".

        2. After _reload: cross-process sync rebuilds in-memory state
           from disk, and any cached recompute keyed on the old
           snapshot is now formally invalid even if data_version
           happens to look familiar (defensive — cache key already
           captures data_version, but explicit invalidation removes
           any subtle race).

        Centralised as a helper so future write paths can't forget
        the bump+cache-drop pair (the earlier scattered call-sites
        missed _touch_existing, where access_count/last_accessed
        updates change galaxy/forecast inputs).
        """
        self._mutation_version += 1
        self._forecast_cache = None

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
        # access_count and last_accessed feed gravity → galaxy →
        # forecast_stability. Without this bump, a touch on a
        # duplicate add_fact / query hit could serve a stale
        # forecast cache.
        self._bump_mutation_locked()
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
            from ..resonance.embeddings import EmbeddingError
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

                    # Preflight dimension to keep mutations atomic per
                    # item — same pattern as add_fact above.
                    if (vec and self._index._dim is not None
                            and len(vec) != self._index._dim):
                        raise DimensionMismatchError(
                            f"Embedding dimension mismatch in batch: "
                            f"index has dim={self._index._dim}, "
                            f"incoming vector has dim={len(vec)} for "
                            f"({s!r}, {p!r}, {o!r}). The embedding "
                            f"model probably changed under the store."
                        )
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
                self._mutation_version += 1

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

    def explain_fact(self, fact_id: str) -> dict:
        """Decompose a fact's gravity into per-component contributions.

        Returns the live values of every adaptive feature, the weight each
        carries right now, and the actual contribution each makes to the
        current gravity score. Use this when a fact's gravity surprises you
        — you'll see immediately whether the freshness term is high but
        recent_utility is dragging it down, or the forecast says it's about
        to fall, or whatever.
        """
        from ..gravity import pre_resonance_features
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
            from ..gravity import _W_RESONANCE
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
                self._mutation_version += 1
                return True

    def delete_body(self, body_id: str) -> dict:
        """
        Polymorphic hard-delete across all four body locations.

        Queries return polymorphic bodies (live FactPassport, live
        MetaFact, singularity FactPassport, singularity MetaFact) under
        a single ``body_id``. ``delete_fact`` only handles live
        FactPassports — an agent that gets a MetaFact body_id from
        ``query_memory`` and wants to delete it for GDPR / secrets
        reasons used to silently fail. ``delete_body`` checks all four
        locations and deletes wherever the id lives.

        Returns ``{"deleted": True, "kind": "fact"|"meta"|"singularity_fact"|
        "singularity_meta", "body_id": ...}`` on success, or
        ``{"deleted": False, "body_id": ...}`` if the id is not found.

        Same destructive contract as ``delete_fact``: the body is gone,
        no singularity, no lineage, no Hawking emission. Reserved for
        secrets / GDPR / accidental writes; prefer ``supersede_fact``
        or ``retire_fact`` for stale data.
        """
        with self._lock:
            with self._txn():
                self._sync()
                # 1. Live FactPassport.
                if body_id in self._facts:
                    fact = self._facts.pop(body_id)
                    self._index.remove(body_id)
                    self._drop_from_spo_index(fact)
                    self._engine.unregister(body_id)
                    if self._storage:
                        self._storage.delete_fact(body_id)
                        if hasattr(self._storage, "delete_edges_for_fact"):
                            self._storage.delete_edges_for_fact(body_id)
                    self._mutation_version += 1
                    return {"deleted": True, "kind": "fact",
                            "body_id": body_id}
                # 2. Live MetaFact.
                if body_id in self._meta_facts:
                    self._meta_facts.pop(body_id)
                    self._meta_index.remove(body_id)
                    self._engine.unregister(body_id)
                    if (self._storage
                            and hasattr(self._storage, "delete_meta_fact")):
                        self._storage.delete_meta_fact(body_id)
                    self._mutation_version += 1
                    return {"deleted": True, "kind": "meta",
                            "body_id": body_id}
                # 3. Singularity FactPassport.
                if body_id in self._hole._singularity:
                    self._hole._singularity.pop(body_id)
                    self._hole._index.remove(body_id)
                    if self._storage:
                        self._storage.delete_fact(body_id)
                        # Same edge cleanup as the live-fact branch
                        # above — destructive delete must leave no
                        # orphan edge rows on disk that would inflate
                        # _degrees on next load.
                        if hasattr(self._storage, "delete_edges_for_fact"):
                            self._storage.delete_edges_for_fact(body_id)
                    self._mutation_version += 1
                    return {"deleted": True,
                            "kind": "singularity_fact",
                            "body_id": body_id}
                # 4. Singularity MetaFact.
                if body_id in self._hole._meta_singularity:
                    self._hole._meta_singularity.pop(body_id)
                    self._hole._meta_index.remove(body_id)
                    if (self._storage
                            and hasattr(self._storage, "delete_meta_fact")):
                        self._storage.delete_meta_fact(body_id)
                    self._mutation_version += 1
                    return {"deleted": True,
                            "kind": "singularity_meta",
                            "body_id": body_id}
                return {"deleted": False, "body_id": body_id}

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
        self._mutation_version += 1
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
                self._mutation_version += 1
        return {
            "retired": True,
            "fact_id": fact_id,
            "absorbed": absorbed,
        }

    def _drop_from_spo_index(self, fact: FactPassport) -> None:
        key = self._normalize_spo(fact.subject, fact.predicate, fact.object)
        if self._spo_index.get(key) == fact.fact_id:
            del self._spo_index[key]
