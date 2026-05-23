"""Black Hole — irreversible sink with Hawking emission for extreme retrieval."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .vector_index import VectorIndex

if TYPE_CHECKING:
    from .fact import FactPassport
    from .meta_fact import MetaFact

# Similarity threshold for Hawking emission — only the strongest queries
# pull facts back. Sourced from the env-overridable Thresholds module
# (round 12) so an operator with a different embedding model's cosine
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

    They are kept in separate indices so Hawking emission can return each
    type with the right typed payload — a FactPassport plugs back into the
    live SPO store, while a MetaFact returns as a polymorphic context bundle.
    """

    def __init__(self, hawking_threshold: float = _HAWKING_THRESHOLD) -> None:
        self._singularity: dict[str, SingularityRecord] = {}
        self._meta_singularity: dict[str, MetaSingularityRecord] = {}
        # Separate indices: keeps hawking_emit() typed and avoids collisions
        # if a fact_id ever shared bytes with a meta_id.
        self._index = VectorIndex()              # FactPassport vectors
        self._meta_index = VectorIndex()         # MetaFact vectors
        self._hawking_threshold = hawking_threshold
        self._total_emissions = 0   # cumulative, survives record removal

    # ── Absorption ──────────────────────────────────────────────────────────

    def absorb(self, fact: "FactPassport") -> None:
        """Pull a FactPassport across the event horizon. Irreversible."""
        fact.layer = -1     # sentinel: beyond core
        self._singularity[fact.fact_id] = SingularityRecord(fact=fact)
        self._index.add(fact.fact_id, fact.vector)

    def restore_fact(self, fact: "FactPassport") -> None:
        """Place a fact directly into the singularity without touching its
        metadata. Symmetric with ``restore_meta``: used by
        ``MemoryStore._load_from_storage`` to re-hydrate the black hole from
        SQLite rows whose ``layer == -1`` so absorbed facts survive a
        process restart and remain eligible for Hawking emission.
        """
        self._singularity[fact.fact_id] = SingularityRecord(fact=fact)
        self._index.add(fact.fact_id, fact.vector)

    def absorb_meta(self, meta: "MetaFact") -> None:
        """Place a MetaFact in the singularity (typical: just after collapse)."""
        meta.layer = -1
        self._meta_singularity[meta.meta_id] = MetaSingularityRecord(meta=meta)
        self._meta_index.add(meta.meta_id, meta.vector)

    def restore_meta(self, meta: "MetaFact") -> None:
        """Rehydrate a MetaFact loaded from storage without touching its layer."""
        self._meta_singularity[meta.meta_id] = MetaSingularityRecord(meta=meta)
        self._meta_index.add(meta.meta_id, meta.vector)

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
        """
        sims = self._index.all_similarities(query_vector)
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
        thr = self._hawking_threshold if threshold is None else threshold
        sims = self._meta_index.all_similarities(query_vector)
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
        """
        sims = self._index.all_similarities(query_vector)
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
            # Now commit: pop from singularity, reset gravity, return.
            self._singularity.pop(fid)
            self._index.remove(fid)
            rec.fact.gravity_score = _HAWKING_GRAVITY
            rec.fact.layer = 1
            self._total_emissions += 1
            emitted.append(rec.fact)
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
        thr = self._hawking_threshold if threshold is None else threshold
        sims = self._meta_index.all_similarities(query_vector)
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
            self._meta_index.remove(mid)
            rec.meta.gravity_score = rec.meta.gravity_on_emission(_HAWKING_GRAVITY)
            rec.meta.layer = 1
            self._total_emissions += 1
            emitted.append(rec.meta)
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
    def total_emissions(self) -> int:
        return self._total_emissions

    def __contains__(self, body_id: str) -> bool:
        return body_id in self._singularity or body_id in self._meta_singularity
