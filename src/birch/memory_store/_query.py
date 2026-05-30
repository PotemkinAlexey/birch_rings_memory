"""QueryMixin — semantic retrieval, find_similar, echo check."""
from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, Any, Callable, Optional

from ..fact import FactPassport
from ..meta_fact import MetaFact
from ..thresholds import Thresholds
from ..vector_index import VectorIndex
from ._embed_proxy import embed
from ._models import QueryResult

if TYPE_CHECKING:  # pragma: no cover
    from ..black_hole import BlackHole
    from ..gravity import GravityEngine
    from ..resonance.echo import EchoStore
    from ..storage import StorageBackend
    from ._models import SessionContext

# Module-level alias — symmetric with the legacy single-file module.
# Read at import time so an operator setting BIRCH_* env vars before
# process start sees the pinned value here.
_ABSORPTION_THRESHOLD = Thresholds.ABSORPTION
_META_HAWKING_THRESHOLD = Thresholds.HAWKING_META


class QueryMixin:
    """Read-path methods. See ``MemoryStore`` for the assembled API."""

    _lock: "threading.RLock"
    _storage: "Optional[StorageBackend]"
    _facts: "dict[str, FactPassport]"
    _meta_facts: "dict[str, MetaFact]"
    _spo_index: "dict[tuple[str, str, str, str], str]"
    _index: "VectorIndex"
    _meta_index: "VectorIndex"
    _engine: "GravityEngine"
    _hole: "BlackHole"
    _echo: "EchoStore"
    _sessions: "dict[str, SessionContext]"
    _current_session_id: "Optional[str]"
    _mutation_version: int

    if TYPE_CHECKING:
        _sync: Callable[[], None]
        _txn: Callable[[], Any]
        _reload: Callable[[], None]
        _resolve_sid: Callable[[Optional[str]], Optional[str]]
        _normalize_spo: Callable[..., tuple[str, str, str, str]]
        # MemoryBricks Step 1: borrowed from FactsMixin for the
        # namespace_prefix filter on query / find_similar.
        _namespace_matches_prefix: Callable[[str, str], bool]
        _attribute_to: Callable[["SessionContext", str, float], None]
        _apply_recent_utility_locked: Callable[[dict[str, float], float], None]
        _apply_echo_gravity_locked: Callable[[Any], list[str]]
        _bump_mutation_locked: Callable[[], None]

    def find_similar(
        self,
        text: str,
        top_k: int = 5,
        min_similarity: float = 0.85,
        subject_prefix: Optional[str] = None,
        exclude_ids: Optional[set[str]] = None,
        *,
        namespace_prefix: Optional[str] = None,
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
        ``namespace_prefix`` (MemoryBricks Step 1) restricts results to
        facts in this namespace and its descendants (VB-style match).
        """
        # Library-API hardening: MCP validates this at the boundary,
        # but find_similar is also a public Python entry point. A
        # bare text.strip() on None / int / list raised AttributeError
        # at runtime — make the type contract explicit.
        if not isinstance(text, str):
            raise TypeError(
                f"text must be str, got {type(text).__name__}"
            )
        if not text.strip():
            return []
        vec = embed(text)
        return self.find_similar_by_vector(
            vec,
            top_k=top_k,
            min_similarity=min_similarity,
            subject_prefix=subject_prefix,
            exclude_ids=exclude_ids,
            namespace_prefix=namespace_prefix,
        )

    # Backward-compat alias — previous name was leading-underscore
    # "private". Server.py and external callers can keep importing
    # the old name while we migrate; will be removed after one
    # release cycle.
    def _find_similar_by_vector(self, *args, **kwargs) -> list[dict]:
        return self.find_similar_by_vector(*args, **kwargs)

    def find_similar_by_vector(
        self,
        vec: list[float],
        top_k: int = 5,
        min_similarity: float = 0.85,
        subject_prefix: Optional[str] = None,
        exclude_ids: Optional[set[str]] = None,
        *,
        namespace_prefix: Optional[str] = None,
    ) -> list[dict]:
        """Caller-provided embedding variant — used by record_fact's
        similar_existing hint to avoid embedding the same text twice.

        ``namespace_prefix`` (MemoryBricks Step 1) restricts results to
        facts in this namespace and its descendants.
        """
        prefix = subject_prefix.lower() if subject_prefix else None
        skip = exclude_ids or set()
        # Delegated import to avoid a cycle between _query and _facts.
        ns_match = self._namespace_matches_prefix  # type: ignore[attr-defined]
        # Keep the lock for the whole scan: reading self._facts after
        # releasing the lock previously allowed another thread to
        # delete / supersede / retire a fact between `_sync` and
        # `self._facts.get(fid)`, surfacing a stale FactPassport (or
        # missing one) in a read-only API. query() already holds the
        # lock through its results loop — find_similar_by_vector was
        # the odd sibling.
        hits: list[dict] = []
        with self._lock:
            self._sync()
            sims = self._index.all_similarities(vec)
            for fid, sim in sims.items():
                if fid in skip:
                    continue
                if sim < min_similarity:
                    continue
                fact = self._facts.get(fid)
                if (fact is None or fact.is_deprecated
                        or fact.is_expired):
                    continue
                if prefix and not fact.subject.lower().startswith(prefix):
                    continue
                if namespace_prefix is not None and not ns_match(
                    fact.namespace or "", namespace_prefix,
                ):
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
        *,
        namespace_prefix: Optional[str] = None,
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
        # non-positive top_k with a structured response, but
        # MemoryStore.query is a public core API used in tests and
        # embedded mode too — a negative top_k would slice
        # results[:top_k] from the right end (returning all-except-last),
        # not return "nothing". That is a Python-list semantics trap;
        # close it at the core. (Also skips a needless embed roundtrip.)
        if top_k <= 0:
            return []
        # Library-API hardening: MCP validates this at the boundary,
        # but query is also a public Python entry point. Same
        # contract as find_similar — explicit TypeError beats a
        # raw AttributeError deep inside embed().
        if not isinstance(text, str):
            raise TypeError(
                f"text must be str, got {type(text).__name__}"
            )
        # Embed outside the lock.
        vec = embed(text)
        prefix = subject_prefix.lower() if subject_prefix else None
        # MemoryBricks Step 1: namespace_prefix is a hierarchical
        # filter applied symmetrically across live facts, live metas,
        # and the Hawking emission predicates below. Borrows
        # ``_namespace_matches_prefix`` from FactsMixin (cycle-safe
        # because both mixins compose into the same MemoryStore).
        ns_match = self._namespace_matches_prefix  # type: ignore[attr-defined]

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
                if namespace_prefix is not None and not ns_match(
                    fact.namespace or "", namespace_prefix,
                ):
                    continue
                # Apply min_similarity to the RAW cosine, not the
                # 4-decimal display value. Asymmetric otherwise:
                # Hawking-candidate filter (line ~420) already uses
                # raw sim, so a live hit at sim=0.95004 (rounds to
                # 0.9500) would fail min_similarity=0.95005 while
                # a Hawking hit at the same raw score would pass.
                # Round only on output, never on decision.
                if sim < min_similarity:
                    continue
                results.append(QueryResult(
                    fact=fact,
                    similarity=sim,
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
                    # source_text starts with the prefix (case-insensitive
                    # — same `startswith` contract as the FactPassport
                    # subject filter, applied across the lineage).
                    if not any((st or "").lower().startswith(prefix)
                               for st in meta.source_texts):
                        continue
                if namespace_prefix is not None and not ns_match(
                    meta.namespace or "", namespace_prefix,
                ):
                    continue
                # Raw-sim filter symmetric with the live-fact branch.
                if sim < min_similarity:
                    continue
                results.append(QueryResult(
                    meta=meta,
                    similarity=sim,
                    source=layer_labels.get(meta.layer, "kinetic"),
                ))

            # Pre-Hawking sort / top_k slice — top selection is pure over
            # the live snapshot; mutation (touch / attribute / Hawking pop /
            # persist) happens together under the write transaction below.
            results.sort(key=lambda r: r.similarity, reverse=True)
            # NB: no post-loop min_similarity filter. Both append
            # sites (live facts, live metas) already filter on the
            # raw cosine — adding a second pass over r.similarity
            # would re-introduce the rounded-vs-raw asymmetry
            # versus the Hawking branch (which also filters on
            # raw at peek time). Decision on raw, round only on
            # serialisation.
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
            # Layer check helper that knows about Hawking emission:
            # singularity bodies (layer == -1) check against their
            # POST-EMIT layer (1 / kinetic) because that's the layer
            # they'll have if the predicate lets them through. Live
            # bodies check against their current layer.
            def _layer_ok(layer: int) -> bool:
                effective = 1 if layer == -1 else layer
                if not (min_layer <= effective <= max_layer):
                    return False
                if (allowed_layers is not None
                        and effective not in allowed_layers):
                    return False
                return True

            def _fact_predicate(f) -> bool:
                # Lifecycle: a fact that was superseded by set_fact /
                # supersede_fact, or expired via retire_fact, must NOT
                # come back through Hawking emission as if it were
                # current. The agent thinks it's reading live truth;
                # the body knows it has been retired.
                if f.is_deprecated or f.is_expired:
                    return False
                # Layer must match the caller's scope. Singularity
                # bodies pass when caller's allowed_layers includes
                # 1 (kinetic), since that's where Hawking emission
                # lands them.
                if not _layer_ok(f.layer):
                    return False
                if f.gravity_score < min_gravity:
                    return False
                if prefix and not f.subject.lower().startswith(prefix):
                    return False
                if namespace_prefix is not None and not ns_match(
                    f.namespace or "", namespace_prefix,
                ):
                    return False
                return True

            def _meta_predicate(m) -> bool:
                if getattr(m, "is_deprecated", False):
                    return False
                if getattr(m, "is_expired", False):
                    return False
                if not _layer_ok(m.layer):
                    return False
                if m.gravity_score < min_gravity:
                    return False
                if prefix:
                    if not any((st or "").lower().startswith(prefix)
                               for st in m.source_texts):
                        return False
                if namespace_prefix is not None and not ns_match(
                    m.namespace or "", namespace_prefix,
                ):
                    return False
                return True

            # query() does heavy mutation under the txn (Hawking
            # emission registers bodies back into _facts/_engine/
            # _index/_spo_index; touches mutate access_count and
            # last_accessed; session attribution writes to
            # ctx.facts). If any storage call below raises mid-
            # write, SQLite rolls back the disk but in-memory state
            # is already mutated and the next _sync sees the same
            # data_version — the divergence persists. Wrap in
            # try/except + _reload() — symmetric with add_facts
            # (1792d4f), collapse_singularity (b9412ab), and
            # add_fact (just shipped).
            try:
                with self._txn():
                    # Re-sync under the write lock; if another process committed
                    # we now hold the authoritative state.
                    self._sync()

                    # Revalidate the pre-sync top: another process may
                    # have deprecated / retired / deleted bodies, OR
                    # mutated state in a way that breaks our original
                    # filter predicates (layer migration, gravity drop
                    # below min_gravity, subject change). Re-run the
                    # SAME predicates that gated the initial scan so a
                    # body that no longer matches the caller's scope
                    # never gets returned (or touched / attributed).
                    # Backfill already re-applies filters; survivors
                    # used to be exempt and could return out-of-scope.
                    revalidated_top: list[QueryResult] = []
                    for r in top:
                        if r.fact is not None:
                            live = self._facts.get(r.fact.fact_id)
                            if (live is None or live.is_deprecated
                                    or live.is_expired):
                                continue
                            if not _fact_predicate(live):
                                # Filter no longer holds — caller's
                                # scope changed under our feet.
                                continue
                            r.fact = live
                            revalidated_top.append(r)
                        elif r.meta is not None:
                            live_meta = self._meta_facts.get(r.meta.meta_id)
                            if live_meta is None:
                                continue
                            if not _meta_predicate(live_meta):
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
                                similarity=sim,
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
                                similarity=sim,
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
                                similarity=sim,
                                source="hawking",
                            ))
                        for meta, sim in meta_candidates:
                            if sim < min_similarity:
                                continue
                            top.append(QueryResult(
                                meta=meta,
                                similarity=sim,
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
                                # MemoryBricks Step 1: Hawking-emitted
                                # facts re-enter the live SPO bucket
                                # under their own namespace, not the
                                # global root.
                                key = self._normalize_spo(
                                    fact.subject, fact.predicate, fact.object,
                                    fact.namespace,
                                )
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
                    # query() touches every returned body
                    # (access_count / last_accessed), which feeds gravity
                    # → galaxy → forecast. Without a mutation bump the
                    # forecast cache could return stale results on a
                    # back-to-back query+forecast pattern.
                    if touched_fact_ids or touched_meta_ids:
                        self._bump_mutation_locked()

            except Exception:
                self._reload()
                raise

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
            try:
                with self._txn():
                    self._sync()
                    result = self._echo.detect_echo(
                        vec, exclude_session_id=session_id)
                    # Gravity/EWMA propagation + persistence is shared with the
                    # deferred session_close path via this helper, so both
                    # routes mutate identically (incl. MetaFact penalties).
                    penalized_body_ids = self._apply_echo_gravity_locked(result)

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

            except Exception:
                # Storage rolled back mid-write; in-memory caches
                # may be partially mutated (touches / unregisters /
                # pops / engine state). Re-anchor every cache to
                # disk truth before propagating — symmetric with
                # add_fact / add_facts / query / collapse_singularity.
                self._reload()
                raise

    def peek_echo(self, first_message: str, session_id: str) -> dict:
        """
        Read-only echo detection for the streaming (open → close) path.

        Detects whether ``first_message`` echoes a past session's topic and,
        if so, records a *pending* echo marker on the named session's context
        — WITHOUT applying any penalty or touching gravity. The decision is
        deferred to ``session_close``, which:

          - applies the retroactive penalty only if THIS session also ends
            non-resonant (a genuine return-to-failure), or
          - cancels it if this session ends resonant (a productive revisit /
            continued use of what the past session built).

        This replaces the old apply-on-open behaviour, which guessed
        "returned ⇒ unresolved" and penalised immediately — firing on
        continued use as often as on real false closure. ``session_id`` is
        excluded from the match pool so a just-opened session can't echo
        itself.

        Returns a dict with ``pending`` (whether a marker was stored) rather
        than ``echo`` (nothing is applied here). The marker is in-memory only;
        a cross-process reload between open and close simply drops it.
        """
        # Embed outside the lock — slow HTTP call, must not serialise agents.
        vec = embed(first_message)
        with self._lock:
            try:
                with self._txn():
                    self._sync()
                    result = self._echo.peek_echo(
                        vec, exclude_session_id=session_id)
                    ctx = self._sessions.get(session_id)
                    pending = (
                        result.label == "echo"
                        and ctx is not None
                        and result.matched_session_id is not None
                    )
                    if pending and ctx is not None:
                        ctx.pending_echo = {
                            "matched_session_id": result.matched_session_id,
                            "similarity": result.similarity,
                        }
                    return {
                        # Nothing is applied at open — only a marker is set.
                        "echo": False,
                        "pending": pending,
                        "matched_session": result.matched_session_id,
                        "similarity": result.similarity,
                        # The would-be penalty, for transparency; not applied.
                        "would_be_penalty": result.penalty,
                        "penalized_fact_ids": [],
                    }
            except Exception:
                self._reload()
                raise