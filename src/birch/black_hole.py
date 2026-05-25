"""Black Hole — irreversible sink with Hawking emission for extreme retrieval."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .vector_index import DimensionMismatchError, VectorIndex

if TYPE_CHECKING:
    from .fact import FactPassport
    from .meta_fact import MetaFact

# Similarity threshold for Hawking emission — only the strongest queries
# pull facts back. Sourced from the env-overridable Thresholds module
# so an operator with a different embedding model's cosine
# distribution can tune it without forking.
from .thresholds import Thresholds  # noqa: E402

_HAWKING_THRESHOLD = Thresholds.HAWKING_FACT
# Gravity assigned to emitted facts — they return weakened
_HAWKING_GRAVITY = 0.30


@dataclass
class SingularityRecord:
    fact: "FactPassport"
    absorbed_at: float = field(default_factory=time.time)
    emission_count: int = 0         # how many times Hawking-emitted


@dataclass
class MetaSingularityRecord:
    meta: "MetaFact"
    absorbed_at: float = field(default_factory=time.time)
    emission_count: int = 0


class BlackHole:
    """
    Irreversible sink for facts that have lost all gravity.

    Two kinds of bodies live here:
      - FactPassports that fell past the gravity floor (single SPO triples)
      - MetaFacts created by the SingularityCompactor (a dense bundle of
        many absorbed facts collapsed into one centroid)

    Per-dimension vector indices
    ----------------------------
    Both kinds keep their flat `_singularity` / `_meta_singularity` dicts
    (body_id → record) for O(1) lookup by id, but the cosine search
    matrix is **partitioned by embedding dimension** in two dicts:
    `_indices: dict[int, VectorIndex]` for facts and
    `_meta_indices: dict[int, VectorIndex]` for metas. Each bucket is
    dim-pure by construction — a fact with dim=384 lands in
    `_indices[384]`, a fact with dim=768 lands in `_indices[768]`, and
    the dim mismatch error that earlier rounds had to defend against
    (with atomic three-phase absorb + rollback) can no longer arise
    in the first place: there is no shared matrix to mismatch with.

    Hawking emission scans only the bucket matching the query
    vector's dim — a query with dim=384 cannot resurrect a dim=768
    body and vice versa. This is the right semantics: vectors from
    different embedding spaces are not comparable, so a "similarity"
    cross-space would be undefined anyway.

    Buckets are lazy: `_indices[dim]` is created on first absorb with
    that dim and removed when its last body emits or is forgotten.
    """

    def __init__(self, hawking_threshold: float = _HAWKING_THRESHOLD) -> None:
        self._singularity: dict[str, SingularityRecord] = {}
        self._meta_singularity: dict[str, MetaSingularityRecord] = {}
        # dim → VectorIndex. Each bucket is dim-pure; a fact with a
        # never-before-seen dim creates a new bucket on absorb.
        self._indices: dict[int, VectorIndex] = {}
        self._meta_indices: dict[int, VectorIndex] = {}
        self._hawking_threshold = hawking_threshold
        self._total_emissions = 0   # cumulative, survives record removal

    # ── Internal index lookup ───────────────────────────────────────────────

    def _index_for(self, dim: int) -> VectorIndex:
        """Get or create the fact-bucket for ``dim``."""
        idx = self._indices.get(dim)
        if idx is None:
            idx = VectorIndex()
            self._indices[dim] = idx
        return idx

    def _meta_index_for(self, dim: int) -> VectorIndex:
        """Get or create the meta-bucket for ``dim``."""
        idx = self._meta_indices.get(dim)
        if idx is None:
            idx = VectorIndex()
            self._meta_indices[dim] = idx
        return idx

    def _prune_empty_fact_bucket(self, dim: int) -> None:
        """Drop a fact bucket whose VectorIndex went empty so the
        bucket dict doesn't accumulate stale zero-size entries."""
        idx = self._indices.get(dim)
        if idx is not None and len(idx) == 0:
            del self._indices[dim]

    def _prune_empty_meta_bucket(self, dim: int) -> None:
        """Meta counterpart of ``_prune_empty_fact_bucket``."""
        idx = self._meta_indices.get(dim)
        if idx is not None and len(idx) == 0:
            del self._meta_indices[dim]

    # ── Absorption ──────────────────────────────────────────────────────────

    def absorb(self, fact: "FactPassport") -> None:
        """Pull a FactPassport across the event horizon. Irreversible.

        With per-dim buckets the cross-dim mismatch hazard is removed
        at the source: a fact with vector dim D lands in
        ``_indices[D]``, which is dim-pure by construction. The
        previous round's atomic three-phase contract + rollback still
        applies as belt-and-suspenders against unexpected index
        failures (numpy / OOM), but DimensionMismatchError can no
        longer originate here.

        Rollback restores ``fact.layer`` to the value it had on
        entry — NOT a hardcoded constant. The earlier defensive code
        restored to ``2`` (core) on failure, which was wrong: a fact
        that entered at ``layer=0`` (surface) would silently land in
        core after a transient index error, perturbing the
        caller-visible state in a way the caller had no way to
        detect or undo. True atomic rollback returns the body to
        the exact state it was in before absorb was called.
        """
        old_layer = fact.layer
        fact.layer = -1     # sentinel: beyond core
        self._singularity[fact.fact_id] = SingularityRecord(fact=fact)
        if not fact.vector:
            # Vectorless bodies still live in the dict — they're
            # searchable by id but never returned from Hawking.
            return
        dim = len(fact.vector)
        idx = self._index_for(dim)
        try:
            idx.add(fact.fact_id, fact.vector)
        except Exception:
            # Defensive: per-dim buckets remove the dim-mismatch
            # cause, but any other failure (numpy alloc, OOM) still
            # rolls back the dict insert + layer mutation so the live
            # store stays consistent.
            self._singularity.pop(fact.fact_id, None)
            fact.layer = old_layer
            # Prune the bucket if we just created an empty one.
            self._prune_empty_fact_bucket(dim)
            raise

    def restore_fact(self, fact: "FactPassport") -> None:
        """Place a fact directly into the singularity without touching its
        metadata. Symmetric with ``restore_meta``: used by
        ``MemoryStore._load_from_storage`` to re-hydrate the black hole from
        SQLite rows whose ``layer == -1`` so absorbed facts survive a
        process restart and remain eligible for Hawking emission.

        Per-dim routing applies — bodies whose vectors survived the
        loader's _safe_vector gate land in their dim bucket; bodies
        with empty vectors live in the dict only.
        """
        self._singularity[fact.fact_id] = SingularityRecord(fact=fact)
        if fact.vector:
            self._index_for(len(fact.vector)).add(fact.fact_id, fact.vector)

    def absorb_meta(self, meta: "MetaFact") -> None:
        """Place a MetaFact in the singularity (typical: just after collapse).

        Per-dim bucketed symmetrically with ``absorb``. Rollback
        restores ``meta.layer`` to its on-entry value (NOT a
        hardcoded constant): a MetaFact promoted to ``layer=1``
        via prior Hawking emission shouldn't silently land in
        ``layer=0`` after a transient index error.
        """
        old_layer = meta.layer
        meta.layer = -1
        self._meta_singularity[meta.meta_id] = MetaSingularityRecord(meta=meta)
        if not meta.vector:
            return
        dim = len(meta.vector)
        idx = self._meta_index_for(dim)
        try:
            idx.add(meta.meta_id, meta.vector)
        except Exception:
            self._meta_singularity.pop(meta.meta_id, None)
            meta.layer = old_layer
            self._prune_empty_meta_bucket(dim)
            raise

    def restore_meta(self, meta: "MetaFact") -> None:
        """Rehydrate a MetaFact loaded from storage without touching its layer."""
        self._meta_singularity[meta.meta_id] = MetaSingularityRecord(meta=meta)
        if meta.vector:
            self._meta_index_for(len(meta.vector)).add(meta.meta_id, meta.vector)

    # ── Removal helpers (replace direct _index.remove() consumer code) ──────

    def forget_fact(self, fact_id: str) -> bool:
        """Remove a fact from the singularity AND its dim-index atomically.

        Returns True if removed, False if not present. Consumers
        (singularity compactor, MemoryStore.delete_body) should call
        this instead of doing ``_singularity.pop`` + per-dim index
        cleanup by hand — keeps the bucket lifecycle (lazy create,
        auto-prune when empty) entirely inside BlackHole.
        """
        rec = self._singularity.pop(fact_id, None)
        if rec is None:
            return False
        if rec.fact.vector:
            dim = len(rec.fact.vector)
            idx = self._indices.get(dim)
            if idx is not None:
                idx.remove(fact_id)
                self._prune_empty_fact_bucket(dim)
        return True

    def forget_meta(self, meta_id: str) -> bool:
        """MetaFact counterpart of ``forget_fact``."""
        rec = self._meta_singularity.pop(meta_id, None)
        if rec is None:
            return False
        if rec.meta.vector:
            dim = len(rec.meta.vector)
            idx = self._meta_indices.get(dim)
            if idx is not None:
                idx.remove(meta_id)
                self._prune_empty_meta_bucket(dim)
        return True

    # ── Hawking emission ────────────────────────────────────────────────────

    def peek_hawking_candidates(
        self,
        query_vector: list[float],
        predicate=None,
    ) -> list[tuple["FactPassport", float]]:
        """Return ``(fact, similarity)`` for every body that WOULD be
        Hawking-emitted by this query — without popping anything.

        Use ``peek_hawking_candidates`` to merge potential emissions into
        the live ranking and decide top_k FIRST, then call
        ``hawking_emit(..., only_ids=survivors)`` to commit only the
        bodies that actually made it into the returned results.
        Avoids the contract violation where a body was resurrected
        (state mutation, persistence) but the caller never received it
        because it was below top_k.

        Per-dim scan: only the bucket matching ``len(query_vector)``
        is searched — cross-dim cosine is undefined, so bodies in
        other dim buckets are correctly invisible to this query.
        """
        if not query_vector:
            return []
        dim = len(query_vector)
        idx = self._indices.get(dim)
        if idx is None:
            return []
        sims = idx.all_similarities(query_vector)
        out: list[tuple["FactPassport", float]] = []
        for fid, score in sims.items():
            if score < self._hawking_threshold:
                continue
            rec = self._singularity.get(fid)
            if rec is None:
                continue
            if predicate is not None and not predicate(rec.fact):
                continue
            out.append((rec.fact, float(score)))
        return out

    def peek_hawking_meta_candidates(
        self,
        query_vector: list[float],
        threshold: float | None = None,
        predicate=None,
    ) -> list[tuple["MetaFact", float]]:
        """MetaFact counterpart of ``peek_hawking_candidates``."""
        if not query_vector:
            return []
        dim = len(query_vector)
        idx = self._meta_indices.get(dim)
        if idx is None:
            return []
        thr = self._hawking_threshold if threshold is None else threshold
        sims = idx.all_similarities(query_vector)
        out: list[tuple["MetaFact", float]] = []
        for mid, score in sims.items():
            if score < thr:
                continue
            rec = self._meta_singularity.get(mid)
            if rec is None:
                continue
            if predicate is not None and not predicate(rec.meta):
                continue
            out.append((rec.meta, float(score)))
        return out

    def hawking_emit(
        self,
        query_vector: list[float],
        predicate=None,
        only_ids: set[str] | None = None,
    ) -> list["FactPassport"]:
        """
        Attempt Hawking emission of single FactPassports similar enough to
        the query. Emitted facts are restored to layer=1 (kinetic) with
        gravity reset to _HAWKING_GRAVITY and removed from the singularity.
        The caller re-registers them in the live store.

        ``predicate`` is an optional ``Callable[[FactPassport], bool]`` that
        must return True for a body to be emitted. Bodies that fail the
        predicate stay in the singularity — Hawking emission is a state
        mutation, and a scoped read (e.g. with ``subject_prefix``) must not
        resurrect bodies outside the requested scope as a side effect.

        MetaFact emission lives behind ``hawking_emit_metas`` so the typed
        return value stays narrow for the legacy call site.

        Per-dim scan: see ``peek_hawking_candidates``.
        """
        if not query_vector:
            return []
        dim = len(query_vector)
        idx = self._indices.get(dim)
        if idx is None:
            return []
        sims = idx.all_similarities(query_vector)
        # Filter BEFORE pop — so a rejected body stays in the singularity
        # entirely, with no mutation.
        emitted: list["FactPassport"] = []
        for fid, score in sims.items():
            if score < self._hawking_threshold:
                continue
            if only_ids is not None and fid not in only_ids:
                continue
            rec = self._singularity.get(fid)
            if rec is None:
                continue
            if predicate is not None and not predicate(rec.fact):
                continue
            # Now commit: pop from singularity, remove from bucket,
            # reset gravity, return. forget_fact handles bucket
            # lifecycle (lazy prune when empty).
            self._singularity.pop(fid)
            idx.remove(fid)
            rec.fact.gravity_score = _HAWKING_GRAVITY
            rec.fact.layer = 1
            self._total_emissions += 1
            emitted.append(rec.fact)
        self._prune_empty_fact_bucket(dim)
        return emitted

    def hawking_emit_metas(
        self,
        query_vector: list[float],
        threshold: float | None = None,
        predicate=None,
        only_ids: set[str] | None = None,
    ) -> list["MetaFact"]:
        """
        Attempt Hawking emission of MetaFacts.

        A MetaFact is a dense centroid — at full 0.95 the threshold would
        almost never fire because the centroid intentionally lives between
        its sources. The caller can pass a looser threshold (typical: 0.85);
        a MetaFact's gravity on emission is set by its own
        ``gravity_on_emission()`` (log-scaled in its weight).

        ``predicate`` is an optional ``Callable[[MetaFact], bool]`` that
        gates which bodies actually emerge — see ``hawking_emit`` for the
        scope-respecting motivation.
        """
        if not query_vector:
            return []
        dim = len(query_vector)
        idx = self._meta_indices.get(dim)
        if idx is None:
            return []
        thr = self._hawking_threshold if threshold is None else threshold
        sims = idx.all_similarities(query_vector)
        emitted: list["MetaFact"] = []
        for mid, score in sims.items():
            if score < thr:
                continue
            if only_ids is not None and mid not in only_ids:
                continue
            rec = self._meta_singularity.get(mid)
            if rec is None:
                continue
            if predicate is not None and not predicate(rec.meta):
                continue
            self._meta_singularity.pop(mid)
            idx.remove(mid)
            rec.meta.gravity_score = rec.meta.gravity_on_emission(_HAWKING_GRAVITY)
            rec.meta.layer = 1
            self._total_emissions += 1
            emitted.append(rec.meta)
        self._prune_empty_meta_bucket(dim)
        return emitted

    # ── Status ──────────────────────────────────────────────────────────────

    @property
    def mass(self) -> int:
        """Total bodies inside the singularity — facts plus MetaFacts."""
        return len(self._singularity) + len(self._meta_singularity)

    @property
    def fact_mass(self) -> int:
        return len(self._singularity)

    @property
    def meta_mass(self) -> int:
        return len(self._meta_singularity)

    @property
    def fact_dims(self) -> list[int]:
        """List of active fact dim buckets (sorted). Empty when the
        fact singularity holds nothing."""
        return sorted(self._indices.keys())

    @property
    def meta_dims(self) -> list[int]:
        """List of active meta dim buckets (sorted). Empty when the
        meta singularity holds nothing."""
        return sorted(self._meta_indices.keys())

    @property
    def total_emissions(self) -> int:
        """Cumulative Hawking emissions across the process lifetime."""
        return self._total_emissions

    def __contains__(self, body_id: str) -> bool:
        return body_id in self._singularity or body_id in self._meta_singularity


# DimensionMismatchError is re-exported for backward-compat with
# callers that imported it from this module before the per-dim
# refactor removed its direct usage in absorb / absorb_meta. Per-dim
# buckets eliminate the originating cause, but a downstream caller
# can still legitimately raise it through other vector_index paths.
__all__ = ["BlackHole", "DimensionMismatchError",
           "SingularityRecord", "MetaSingularityRecord"]
