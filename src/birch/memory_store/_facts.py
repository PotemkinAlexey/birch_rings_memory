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
        _reload: Callable[[], None]
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
        """Lock must be held. Link new fact to semantically close neighbours.

        Oversample by 1 to absorb the inevitable self-match (the fact
        was added to ``_index`` before this call). Drop self AND cap
        the surviving neighbours to ``AUTO_LINK_TOP_K`` — the cap
        matters in the pathological case where the new fact's vector
        is identical to several other facts: argpartition's tie-break
        is undefined, self may not appear in the first ``top_k+1``
        results, and the call would otherwise wire up ``top_k+1`` real
        edges instead of ``top_k``. Hard cap closes that gap.
        """
        if not self._auto_link or len(self._index) < 2:
            return
        neighbours = self._index.search(
            vec, top_k=self.AUTO_LINK_TOP_K + 1, threshold=self.AUTO_LINK_THRESHOLD
        )
        linked = 0
        for neighbour_id, sim in neighbours:
            if neighbour_id == fact_id:
                continue
            if linked >= self.AUTO_LINK_TOP_K:
                break
            self._engine.link(fact_id, neighbour_id)
            self._engine.link(neighbour_id, fact_id)
            if self._storage:
                self._storage.save_edge(fact_id, neighbour_id)
                self._storage.save_edge(neighbour_id, fact_id)
            linked += 1

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
        # Wrapped in try/except + _reload so a storage failure inside
        # _touch_existing (which mutates access_count / last_accessed
        # on the live fact BEFORE saving) doesn't leak in-memory
        # touches after a SQLite rollback.
        with self._lock:
            self._sync()
            existing_id = self._spo_index.get(key)
            if existing_id and existing_id in self._facts:
                try:
                    with self._txn():
                        self._sync()
                        eid = self._spo_index.get(key)
                        if eid and eid in self._facts:
                            fact = self._touch_existing(eid, session_id)
                            return (fact, False) if return_status else fact
                    # Raced away between sync and write lock — fall through.
                except Exception:
                    # Storage rolled back; in-memory fact.touch() and
                    # any session attribution applied inside
                    # _touch_existing are now divergent from disk.
                    # Re-anchor before propagating.
                    self._reload()
                    raise

        # Slow path: embed without holding the lock.
        vec = embed(f"{subject} {predicate} {obj}")

        try:
            with self._lock:
                with self._txn():
                    # Reload under the write lock — authoritative view.
                    self._sync()
                    # Double-check: another process may have created
                    # the same triple while we were embedding.
                    existing_id = self._spo_index.get(key)
                    if existing_id and existing_id in self._facts:
                        fact = self._touch_existing(existing_id, session_id)
                        return (fact, False) if return_status else fact

                    sid = self._resolve_sid(session_id)
                    # Preflight dim check BEFORE mutating any state
                    # so a mismatch leaves no half-state behind.
                    if (vec and self._index.dim is not None
                            and len(vec) != self._index.dim):
                        raise DimensionMismatchError(
                            f"Embedding dimension mismatch: index has "
                            f"dim={self._index.dim}, incoming vector "
                            f"has dim={len(vec)}. The embedding model "
                            f"probably changed under the store. Pin "
                            f"BIRCH_EMBED_MODEL or rebuild the store "
                            f"before writing facts."
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
                    self._bump_mutation_locked()
                    return (fact, True) if return_status else fact
        except Exception:
            # Storage rolled back; in-memory _facts/_engine/_index/
            # _spo_index/_auto_link edges may be partially populated.
            # Re-anchor every cache to disk truth before propagating.
            self._reload()
            raise

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
            # The apply loop below mutates self._facts / self._engine /
            # self._index / self._spo_index per item, and touches the
            # access_count / last_accessed on existing facts. If anything
            # raises mid-batch (mismatched embedding dim sneaking past
            # preflight, storage I/O failure, etc.) SQLite rolls back
            # but the in-memory dicts and the existing-fact touch
            # mutations stay dirty. Wrap in try/except and _reload() on
            # any failure to re-anchor every cache to disk truth — same
            # pattern as collapse_singularity uses for its rollback.
            try:
                with self._txn():
                    # Reload under the write lock — the authoritative view.
                    self._sync()
                    # ── Preflight pass: classify every item, validate
                    # every dim, build a plan list. NO mutation of
                    # self._facts / _index / _spo_index / fact.touch()
                    # happens in this pass — every raise here aborts the
                    # batch before any state changes.
                    seen_in_batch: dict[tuple[str, str, str], int] = {}
                    # Each plan entry: ("touch", idx, existing_id, sid)
                    # or ("new", idx, fact, key, vec, sid)
                    # or ("dup_in_batch", idx, first_idx, sid).
                    plans: list[tuple] = []
                    for idx, (triple, vec) in enumerate(
                        zip(triples, vectors)
                    ):
                        s, p, o = triple
                        raw_sid = (
                            session_ids[idx] if session_ids is not None
                            else None
                        )
                        sid = self._resolve_sid(raw_sid or session_id)
                        key = self._normalize_spo(s, p, o)

                        # In-batch duplicate (resolves to the first
                        # occurrence after apply pass).
                        if key in seen_in_batch:
                            duplicate_in_batch[idx] = True
                            plans.append((
                                "dup_in_batch", idx,
                                seen_in_batch[key], sid,
                            ))
                            continue

                        # Pre-existing in the store BEFORE this batch.
                        existing_id = self._spo_index.get(key)
                        if existing_id and existing_id in self._facts:
                            already_existed[idx] = True
                            plans.append((
                                "touch", idx, existing_id, sid,
                            ))
                            seen_in_batch[key] = idx
                            continue

                        # New fact — preflight dimension BEFORE we
                        # construct it so a later item's bad dim aborts
                        # the batch cleanly with zero state change.
                        if (vec and self._index.dim is not None
                                and len(vec) != self._index.dim):
                            raise DimensionMismatchError(
                                f"Embedding dimension mismatch in batch: "
                                f"index has dim={self._index.dim}, "
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
                        plans.append(("new", idx, fact, key, vec, sid))
                        seen_in_batch[key] = idx

                    # ── Apply pass: every item passed preflight, now
                    # mutate state. The only things that can raise from
                    # here on are storage I/O failures (caught by the
                    # outer try/except → _reload).
                    for plan in plans:
                        kind = plan[0]
                        if kind == "dup_in_batch":
                            _, idx, first_idx, sid = plan
                            # Result resolves to the same fact the first
                            # occurrence created/touched. A second
                            # `body.touch()` would over-count access for
                            # a single batch's duplicate, so we skip
                            # the touch. BUT attribution is per-session
                            # and the duplicate may carry a DIFFERENT
                            # per-item session_id than the first
                            # occurrence — skipping attribution here
                            # silently drops the second session's
                            # gravity feedback for this fact. Apply
                            # attribution explicitly so per-item
                            # session_id contract holds.
                            results[idx] = results[first_idx]
                            first_fact = results[first_idx]
                            if first_fact is not None and sid is not None:
                                ctx = self._sessions.get(sid)
                                if ctx is not None:
                                    self._attribute_to(
                                        ctx, first_fact.fact_id, 1.0,
                                    )
                                    touched_ctxs.add(ctx.session_id)
                        elif kind == "touch":
                            _, idx, existing_id, sid = plan
                            fact = self._facts[existing_id]
                            fact.touch()
                            ctx = (
                                self._sessions.get(sid) if sid else None
                            )
                            if ctx is not None:
                                self._attribute_to(ctx, existing_id, 1.0)
                                touched_ctxs.add(ctx.session_id)
                            results[idx] = fact
                        else:  # "new"
                            _, idx, fact, key, vec, sid = plan
                            self._facts[fact.fact_id] = fact
                            self._engine.register(fact)
                            self._index.add(fact.fact_id, vec)
                            self._spo_index[key] = fact.fact_id
                            self._auto_link_fact(fact.fact_id, vec)
                            ctx = (
                                self._sessions.get(sid) if sid else None
                            )
                            if ctx is not None:
                                self._attribute_to(ctx, fact.fact_id, 1.0)
                                touched_ctxs.add(ctx.session_id)
                            results[idx] = fact

                    # Persist new facts and touched duplicates in one shot.
                    if self._storage:
                        self._storage.save_facts(
                            [r for r in results if r is not None]
                        )
                    # Persist every open session whose attribution changed.
                    for sid in touched_ctxs:
                        self._persist_session_locked(
                            self._sessions.get(sid)
                        )
                    self._bump_mutation_locked()
            except Exception:
                # SQLite rolled back; in-memory _facts / _index /
                # _spo_index / fact.access_count / ctx.facts may be
                # partially mutated. Re-anchor every cache to disk truth
                # before propagating so callers don't see a phantom
                # mid-batch state.
                self._reload()
                raise

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
        from ..gravity import _W_RESONANCE, pre_resonance_features
        with self._lock:
            self._sync()
            # Polymorphic body lookup — same four locations as
            # delete_body / query_memory so the agent can pipe a
            # query hit's body_id straight in regardless of whether
            # it points at a FactPassport, MetaFact, or singularity
            # body. Returning {"found": False} for a valid query
            # result was misleading.
            fact: Any = self._facts.get(fact_id)
            kind = "fact"
            if fact is None:
                meta = self._meta_facts.get(fact_id)
                if meta is not None:
                    fact = meta
                    kind = "meta"
            if fact is None and fact_id in self._hole._singularity:
                fact = self._hole._singularity[fact_id].fact
                kind = "singularity_fact"
            if fact is None and fact_id in self._hole._meta_singularity:
                fact = self._hole._meta_singularity[fact_id].meta
                kind = "singularity_meta"
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
            contributions = {
                "freshness":  round(weights.w_freshness * freshness, 4),
                "access":     round(weights.w_access * access, 4),
                "graph":      round(weights.w_graph * graph, 4),
                "recent_utility":     round(weights.w_utility * utility, 4),
                "forecast_stability": round(weights.w_stability * stability, 4),
                "resonance":  round(_W_RESONANCE * resonance_score, 4),
            }
            live_gravity = sum(contributions.values())
            response: dict = {
                "found": True,
                "fact_id": fact_id,
                "kind": kind,
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
                "resonance_count": fact.resonance_count,
                "access_count": fact.access_count,
                "last_accessed": fact.last_accessed,
                "created_at": fact.created_at,
            }
            if kind in ("fact", "singularity_fact"):
                # SPO triple + lifecycle flags only meaningful for
                # FactPassports — MetaFacts aren't bound to a single
                # SPO slot.
                response["subject"] = fact.subject
                response["predicate"] = fact.predicate
                response["object"] = fact.object
                response["layer"] = fact.layer
                response["is_deprecated"] = fact.is_deprecated
                response["is_expired"] = fact.is_expired
                response["deprecated_by"] = fact.deprecated_by
            else:
                # MetaFact-shaped fields. weight = number of source
                # facts absorbed; source_fact_ids + source_texts are
                # the lineage so the agent can explain "this bundle
                # represents these N facts".
                response["weight"] = fact.weight
                response["source_fact_ids"] = list(fact.source_fact_ids)
                response["source_texts"] = list(fact.source_texts)
            return response

    def delete_fact(self, fact_id: str) -> bool:
        """
        Permanently remove a fact from the live store.

        Cleans up _facts, _index, _spo_index, gravity engine, and storage.
        Returns True if the fact existed and was deleted, False if not found.
        Unlike absorption, this does NOT send the fact to the black hole —
        the data is gone.
        """
        with self._lock:
            try:
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
                    self._bump_mutation_locked()
                    return True

            except Exception:
                # Storage rolled back mid-write; in-memory caches
                # may be partially mutated (touches / unregisters /
                # pops / engine state). Re-anchor every cache to
                # disk truth before propagating — symmetric with
                # add_fact / add_facts / query / collapse_singularity.
                self._reload()
                raise
    def explain_body(self, body_id: str) -> dict:
        """Polymorphic alias for ``explain_fact``.

        Naming symmetry with ``delete_body`` and ``query_memory`` (both
        of which return polymorphic body_ids spanning FactPassports
        and MetaFacts in live/singularity locations). ``explain_fact``
        was originally FactPassport-only and grew to handle all four
        body kinds — keeping the old name is fine for backward compat,
        but agents that get a ``body_id`` from ``query_memory`` and
        want to explain it should reach for the body-named method.
        """
        return self.explain_fact(body_id)

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
            try:
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
                        self._bump_mutation_locked()
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
                        self._bump_mutation_locked()
                        return {"deleted": True, "kind": "meta",
                                "body_id": body_id}
                    # 3. Singularity FactPassport.
                    if body_id in self._hole._singularity:
                        # forget_fact pops the singularity dict AND
                        # removes from the right dim-bucket, pruning
                        # the bucket if it becomes empty.
                        self._hole.forget_fact(body_id)
                        if self._storage:
                            self._storage.delete_fact(body_id)
                            # Same edge cleanup as the live-fact branch
                            # above — destructive delete must leave no
                            # orphan edge rows on disk that would inflate
                            # _degrees on next load.
                            if hasattr(self._storage, "delete_edges_for_fact"):
                                self._storage.delete_edges_for_fact(body_id)
                        self._bump_mutation_locked()
                        return {"deleted": True,
                                "kind": "singularity_fact",
                                "body_id": body_id}
                    # 4. Singularity MetaFact.
                    if body_id in self._hole._meta_singularity:
                        self._hole.forget_meta(body_id)
                        if (self._storage
                                and hasattr(self._storage, "delete_meta_fact")):
                            self._storage.delete_meta_fact(body_id)
                        self._bump_mutation_locked()
                        return {"deleted": True,
                                "kind": "singularity_meta",
                                "body_id": body_id}
                    return {"deleted": False, "body_id": body_id}

            except Exception:
                # Storage rolled back mid-write; in-memory caches
                # may be partially mutated (touches / unregisters /
                # pops / engine state). Re-anchor every cache to
                # disk truth before propagating — symmetric with
                # add_fact / add_facts / query / collapse_singularity.
                self._reload()
                raise
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
        """Record a graph dependency edge between two live FactPassports.

        Both ids MUST point at currently-live FactPassports — without
        the existence check a caller could create dangling edges
        referencing facts that never existed (or already moved to the
        singularity). Those phantom edges accumulate forever because
        edge cleanup keys on fact deletion, which the never-existent
        endpoint can't trigger. Engine degree counter also gets
        ghost-inflated.
        """
        with self._lock:
            try:
                with self._txn():
                    self._sync()
                    if from_id not in self._facts:
                        raise KeyError(
                            f"link(from_id={from_id!r}, ...): from_id is "
                            f"not a live FactPassport — refusing to "
                            f"create a dangling edge."
                        )
                    if to_id not in self._facts:
                        raise KeyError(
                            f"link(..., to_id={to_id!r}): to_id is not "
                            f"a live FactPassport — refusing to create "
                            f"a dangling edge."
                        )
                    self._engine.link(from_id, to_id)
                    if self._storage:
                        self._storage.save_edge(from_id, to_id)

            except Exception:
                # Storage rolled back mid-write; in-memory caches
                # may be partially mutated (touches / unregisters /
                # pops / engine state). Re-anchor every cache to
                # disk truth before propagating — symmetric with
                # add_fact / add_facts / query / collapse_singularity.
                self._reload()
                raise
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
            try:
                with self._txn():
                    self._sync()
                    result = self._supersede_fact_locked(old_id, new_id)
            except Exception:
                # Storage rolled back mid-write; in-memory caches
                # may be partially mutated (touches / unregisters /
                # pops / engine state). Re-anchor every cache to
                # disk truth before propagating — symmetric with
                # add_fact / add_facts / query / collapse_singularity.
                self._reload()
                raise
        return result

    def _supersede_fact_locked(self, old_id: str, new_id: str) -> dict:
        """Locked helper for supersede_fact. Caller MUST hold self._lock
        AND be inside a write ``_txn()`` AND have already done ``_sync()``.

        Exists so set_fact can supersede slot occupants inside its own
        transaction without nesting public ``supersede_fact`` calls —
        nesting works via reentrant transaction(), but the chain of
        public-method ↔ public-method calls is harder to reason about
        than a single transactional flow that uses this helper.

        FactPassport-only by design. MetaFacts are aggregate bundles
        without a single SPO slot — there is no sane semantics for
        "supersede a cluster with one new fact" (which slot would the
        new fact occupy?). For destructive removal of any body kind
        prefer ``delete_body`` (polymorphic); for stale MetaFact data
        the model contract is: record contradicting facts and let
        next-cycle collapse re-aggregate. Failure responses now echo
        both ids so callers don't KeyError on ``result["old_id"]``.
        """
        if old_id not in self._facts:
            return self._not_a_factpassport_failure(
                "superseded", old_id, new_id=new_id,
            )
        old = self._facts[old_id]
        old.deprecated_by = new_id
        key = self._normalize_spo(old.subject, old.predicate, old.object)
        if self._spo_index.get(key) == old_id:
            del self._spo_index[key]
        if self._storage:
            self._storage.save_fact(old)
        absorbed = self._absorb_dead()
        self._bump_mutation_locked()
        return {
            "superseded": True,
            "old_id": old_id,
            "new_id": new_id,
            "absorbed": absorbed,
        }

    def _not_a_factpassport_failure(
        self,
        action_key: str,
        body_id: str,
        *,
        new_id: Optional[str] = None,
    ) -> dict:
        """Failure response shared by supersede_fact / retire_fact when
        the id is not a live FactPassport. Detects the four other body
        locations so the response can tell the caller exactly what kind
        of body the id points to (or that nothing matches at all).

        ``action_key`` is ``"superseded"`` or ``"retired"`` so the
        response has the same top-level boolean key the success path
        uses — agents can branch on `if not result[action_key]`.
        """
        # Always echo the ids so callers can key on result["old_id"] /
        # result["fact_id"] in both branches (was the documented but
        # missing field that caused agent KeyError).
        resp: dict = {
            action_key: False,
            "fact_id": body_id,
        }
        if new_id is not None:
            resp["old_id"] = body_id
            resp["new_id"] = new_id
            # drop the redundant fact_id alias when both ids exist
            resp.pop("fact_id", None)
        # Polymorphic kind detection — mirrors delete_body / explain_fact.
        if body_id in self._meta_facts:
            resp["error"] = "not_a_factpassport"
            resp["kind"] = "meta"
            resp["hint"] = (
                "Lifecycle ops (supersede / retire) are FactPassport-only "
                "by design — MetaFacts have no SPO slot. Use delete_body "
                "for destructive removal; record contradicting facts to "
                "let next-cycle collapse re-aggregate."
            )
            return resp
        if body_id in self._hole._singularity:
            resp["error"] = "not_a_factpassport"
            resp["kind"] = "singularity_fact"
            resp["hint"] = (
                "Body is already in the singularity (absorbed or "
                "previously superseded). Use delete_body if you need "
                "destructive removal."
            )
            return resp
        if body_id in self._hole._meta_singularity:
            resp["error"] = "not_a_factpassport"
            resp["kind"] = "singularity_meta"
            resp["hint"] = (
                "Body is a singularity MetaFact aggregate. Use "
                "delete_body for destructive removal; lifecycle ops "
                "do not apply to clusters."
            )
            return resp
        # Truly unknown.
        resp["error"] = "not_found"
        resp["reason"] = "id does not match any known body"
        return resp

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
        # Slot-replace is a SINGLE-transaction contract. Previously
        # ran as two independent transactions — add_fact committed,
        # then a second txn superseded occupants — so a failure of
        # the second txn left the new fact committed alongside the
        # old occupants, breaking slot uniqueness. Now: embed outside
        # the lock (slow HTTP call), then do add + supersede in one
        # atomic _txn(), wrapped in the standard rollback guard.
        key = self._normalize_spo(subject, predicate, obj)
        # Embed outside the lock — slow HTTP call, must not serialize.
        vec = embed(f"{subject} {predicate} {obj}")
        superseded: list[str] = []
        already_existed = False
        new_fact: Optional[FactPassport] = None
        try:
            with self._lock:
                with self._txn():
                    self._sync()
                    # Step 1: insert-or-touch the new fact. If the
                    # exact SPO already exists in the slot, touch it
                    # and reuse the existing fact_id; otherwise create
                    # a new FactPassport.
                    existing_id = self._spo_index.get(key)
                    if existing_id and existing_id in self._facts:
                        new_fact = self._touch_existing(
                            existing_id, session_id,
                        )
                        already_existed = True
                    else:
                        # Preflight dim BEFORE any state mutation.
                        if (vec and self._index.dim is not None
                                and len(vec) != self._index.dim):
                            raise DimensionMismatchError(
                                f"Embedding dimension mismatch: "
                                f"index has dim={self._index.dim}, "
                                f"incoming vector has dim={len(vec)}. "
                                f"Pin BIRCH_EMBED_MODEL or rebuild "
                                f"the store before set_fact."
                            )
                        sid = self._resolve_sid(session_id)
                        new_fact = FactPassport(
                            subject=subject,
                            predicate=predicate,
                            object=obj,
                            layer=1,
                            source_session=sid,
                        )
                        new_fact.vector = vec
                        self._facts[new_fact.fact_id] = new_fact
                        self._engine.register(new_fact)
                        self._index.add(new_fact.fact_id, vec)
                        self._spo_index[key] = new_fact.fact_id
                        self._auto_link_fact(new_fact.fact_id, vec)
                        if self._storage:
                            self._storage.save_fact(new_fact)
                        if sid is not None:
                            ctx = self._sessions.get(sid)
                            if ctx is not None:
                                self._attribute_to(
                                    ctx, new_fact.fact_id, 1.0,
                                )
                                self._persist_session_locked(ctx)
                        self._bump_mutation_locked()

                    # Step 2: supersede every OTHER live occupant of
                    # the (subject, predicate) slot — same transaction,
                    # so either both halves succeed or both roll back
                    # via the outer try/except + _reload below.
                    occupants = self._live_slot_occupants(
                        subject, predicate,
                    )
                    for old_id in occupants:
                        if old_id == new_fact.fact_id:
                            continue
                        result = self._supersede_fact_locked(
                            old_id, new_fact.fact_id,
                        )
                        if result.get("superseded"):
                            superseded.append(old_id)
        except Exception:
            # Either the insert or any supersede failed — SQLite
            # rolled back the disk, _reload() re-anchors in-memory
            # to disk truth so callers don't see the half-applied
            # slot replacement. Symmetric with collapse_singularity
            # / add_facts / query — see commit 0f23a62.
            with self._lock:
                self._reload()
            raise
        assert new_fact is not None

        # Response now carries the same per-fact metadata that
        # record_fact / record_facts items return — agents that
        # orchestrate writes through any of the three write tools
        # can key on a uniform field set (layer, gravity_score,
        # created). `created` is the explicit "this call inserted
        # a row" boolean; `already_existed` says only "the SPO was
        # already present" (could be touched, could be just
        # re-recorded by an earlier item in this batch).
        return {
            "set": True,
            "fact_id": new_fact.fact_id,
            "subject": subject,
            "predicate": predicate,
            "object": obj,
            "already_existed": already_existed,
            "created": not already_existed,
            "layer": new_fact.layer,
            "gravity_score": round(new_fact.gravity_score, 3),
            "superseded": superseded,
            "_hint": (
                "set_fact is the slot-replace primitive — every live "
                "fact sharing (subject, predicate) was superseded."
            ),
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
            try:
                with self._txn():
                    self._sync()
                    if fact_id not in self._facts:
                        return self._not_a_factpassport_failure(
                            "retired", fact_id,
                        )
                    fact = self._facts[fact_id]
                    fact.ttl = time.time()
                    if self._storage:
                        self._storage.save_fact(fact)
                    absorbed = self._absorb_dead()
                    self._bump_mutation_locked()
            except Exception:
                # Storage rolled back mid-write; in-memory caches
                # may be partially mutated (touches / unregisters /
                # pops / engine state). Re-anchor every cache to
                # disk truth before propagating — symmetric with
                # add_fact / add_facts / query / collapse_singularity.
                self._reload()
                raise
        return {
            "retired": True,
            "fact_id": fact_id,
            "absorbed": absorbed,
        }

    def _drop_from_spo_index(self, fact: FactPassport) -> None:
        key = self._normalize_spo(fact.subject, fact.predicate, fact.object)
        if self._spo_index.get(key) == fact.fact_id:
            del self._spo_index[key]
