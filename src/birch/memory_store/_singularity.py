"""SingularityMixin — black-hole absorption, collapse, forecast."""
from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Callable, Optional

from ..fact import FactPassport
from ..meta_fact import MetaFact
from ..singularity_compactor import (
    CollapseReport,
    collapse_singularity,
)
from ..thresholds import Thresholds

_logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover
    from ..black_hole import BlackHole
    from ..gravity import GravityEngine
    from ..storage import StorageBackend
    from ..vector_index import VectorIndex

# Module-level aliases — read at import time.
_ABSORPTION_THRESHOLD = Thresholds.ABSORPTION
_SALIENCE_NEIGHBOR = Thresholds.SALIENCE_NEIGHBOR
_SALIENCE_PROTECTION = Thresholds.SALIENCE_PROTECTION


class SingularityMixin:
    """Singularity / collapse / forecast methods."""

    _lock: "threading.RLock"
    _storage: "Optional[StorageBackend]"
    _facts: "dict[str, FactPassport]"
    _meta_facts: "dict[str, MetaFact]"
    _hole: "BlackHole"
    _index: "VectorIndex"
    _meta_index: "VectorIndex"
    _engine: "GravityEngine"
    _collapse_counter: int
    _collapse_async: bool
    _collapse_threshold: float
    _collapse_min_group_size: int
    _collapse_executor: Optional[ThreadPoolExecutor]
    _inflight_collapse: Optional[Future]
    _last_collapse_at: Optional[float]
    _last_collapse_attempt_at: Optional[float]
    _last_collapse_error: Optional[str]
    _total_collapses: int
    _total_collapse_attempts: int
    _mutation_version: int
    _data_version: int
    _salience_retained_ids: "set[str]"
    _forecast_cache: "Optional[tuple[tuple[int, int, int, int], dict]]"
    COLLAPSE_FACT_MASS_TRIGGER: int
    COLLAPSE_DELTA_TRIGGER: int

    if TYPE_CHECKING:
        _sync: Callable[[], None]
        _txn: Callable[[], Any]
        _reload: Callable[[], None]
        _drop_from_spo_index: Callable[[FactPassport], None]
        _bump_mutation_locked: Callable[[], None]
        _data_version_now: Callable[[], int]

    def _irreplaceability(self, fact: FactPassport) -> float:
        """Salience: 1 / (1 + same-namespace live neighbours at cosine ≥
        SALIENCE_NEIGHBOR). 1.0 = unique (no substitute in scope), → 0 as
        near-duplicates accumulate. Frequency-orthogonal — it reads only the
        semantic neighbourhood, so a rare fact can still be irreplaceable."""
        if not fact.vector or _SALIENCE_PROTECTION <= 0.0:
            return 0.0
        ns = fact.namespace or ""
        neighbours = 0
        for other_fid, sim in self._index.all_similarities(fact.vector).items():
            if other_fid == fact.fact_id or sim < _SALIENCE_NEIGHBOR:
                continue
            other = self._facts.get(other_fid)
            if other is None or other.is_deprecated:
                continue
            if (other.namespace or "") != ns:
                continue  # uniqueness is within-scope (MemoryBricks)
            neighbours += 1
        return 1.0 / (1.0 + neighbours)

    def _absorption_floor(self, fact: FactPassport) -> float:
        """Disuse-absorption floor, lowered for facts that are *both* unique and
        proven useful — "rare-but-critical".

            salience = irreplaceability · clamp(avg_resonance, 0, 1)
            floor    = ABSORPTION · (1 − SALIENCE_PROTECTION · salience)

        Uniqueness alone is NOT enough: in a real store almost every fact is
        unique, so protecting on uniqueness alone would halt absorption entirely
        and hoard junk. Coupling to proven value (avg_resonance) targets the
        fact that has *demonstrated* it matters and that nothing else covers —
        and both factors are frequency-orthogonal (avg_resonance is a mean,
        frozen on disuse), so a once-a-year fact still qualifies. An unproven
        unique fact decays normally."""
        if _SALIENCE_PROTECTION <= 0.0 or fact.resonance_count <= 0:
            return _ABSORPTION_THRESHOLD
        value = min(1.0, fact.avg_resonance)
        if value <= 0.0:
            return _ABSORPTION_THRESHOLD  # never proven useful → no protection
        salience = self._irreplaceability(fact) * value
        return _ABSORPTION_THRESHOLD * (1.0 - _SALIENCE_PROTECTION * salience)

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
            if fact.is_deprecated or fact.is_expired:
                falls_to_hole = True  # lifecycle exit — salience does not apply
            elif fact.gravity_score >= _ABSORPTION_THRESHOLD:
                falls_to_hole = False  # safe; skip the salience computation
            else:
                # Below the flat floor — keep it only if it's irreplaceable
                # enough that its salience-adjusted floor is below its gravity.
                falls_to_hole = fact.gravity_score < self._absorption_floor(fact)
                if not falls_to_hole:
                    self._salience_retained_ids.add(fid)
            if not falls_to_hole:
                continue
            # BlackHole.absorb is atomic and rolls back the fact's
            # layer / dict insert on failure. The per-dim singularity
            # refactor closed the mixed-dim cause; remaining failure
            # modes are unrelated (numpy alloc, OOM, corrupted vector
            # shape, monkeypatched index in tests). Catch here so one
            # bad body doesn't abort the whole sweep — body stays
            # live with its original layer restored, visible to the
            # next query.
            try:
                self._hole.absorb(fact)
            except Exception as exc:
                _logger.warning(
                    "_absorb_dead: absorb failed for fact_id=%r; "
                    "body left live for safety: %s",
                    fid, exc,
                )
                continue
            del self._facts[fid]
            self._index.remove(fid)
            self._drop_from_spo_index(fact)
            # Unregister from the gravity engine too. delete_fact /
            # delete_body do this; _absorb_dead used to forget,
            # leaving absorbed bodies tracked by GravityEngine.tick()
            # and apply_session_resonance() as if they were still
            # live. The `is_deprecated or is_expired` guard in tick()
            # catches the lifecycle-driven absorbs but NOT the
            # gravity-below-threshold path (line 76) — that body is
            # neither deprecated nor expired, just dim. After
            # restart _load_from_storage routes layer=-1 to the hole
            # and does NOT re-register in engine, so pre-restart and
            # post-restart behaviour diverged silently. Symmetric
            # unregister closes both gaps and re-aligns the runtime
            # with the disk truth.
            self._engine.unregister(fid)
            if self._storage:
                # Persist the layer=-1 transition so the body survives
                # restart inside the singularity (not as a live fact).
                self._storage.save_fact(fact)
            absorbed.append(fid)
        # Live MetaFacts use the same gravity floor — they came out of the
        # singularity once, they can fall back in.
        for mid, meta in list(self._meta_facts.items()):
            if meta.gravity_score < _ABSORPTION_THRESHOLD:
                # Symmetric with the fact-absorption catch above:
                # absorb_meta rolls back its own state on failure
                # (original layer restored, no half-state in
                # singularity), so one bad MetaFact shouldn't abort
                # the whole sweep and bubble up to session_close.
                # Body stays live with its original layer, visible
                # to the next query.
                try:
                    self._hole.absorb_meta(meta)
                except Exception as exc:
                    _logger.warning(
                        "_absorb_dead: absorb_meta failed for "
                        "meta_id=%r; body left live for safety: %s",
                        mid, exc,
                    )
                    continue
                del self._meta_facts[mid]
                self._meta_index.remove(mid)
                # Same engine-unregister symmetry for MetaFacts.
                self._engine.unregister(mid)
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
            # The compactor mutates self._hole (pops singularity
            # members, registers MetaFacts) BEFORE the storage writes
            # below. If a storage call raises mid-write, the SQLite
            # txn rolls back cleanly — but the in-memory _hole is
            # already mutated, and data_version doesn't bump on a
            # rolled-back txn, so the next _sync() won't trigger
            # _reload(). That would leave the in-memory view ahead of
            # the on-disk truth until something else triggers a
            # cross-process bump. Force a full _reload on any failure
            # inside the txn to re-anchor every cache to disk.
            collapse_succeeded = False
            try:
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
                    collapse_succeeded = True
            except Exception:
                # Storage rolled back; in-memory _hole is desynced.
                # Pull the authoritative state back from disk before
                # propagating so subsequent callers don't see a
                # phantom-collapsed view of the singularity.
                self._reload()
                raise
            # Bookkeeping runs only on a clean commit. The except
            # branch above re-raises, so we're guaranteed
            # collapse_succeeded here.
            assert collapse_succeeded
            now_ts = time.time()
            self._last_collapse_attempt_at = now_ts
            self._total_collapse_attempts += 1
            # Only count as a successful collapse if something actually
            # compressed; otherwise total_collapses would lie ("we
            # collapsed 47 times" when nothing was bundled).
            if report.groups > 0:
                self._last_collapse_at = now_ts
                self._total_collapses += 1
                # Successful collapse clears any prior captured
                # error — the worker is healthy again.
                self._last_collapse_error = None
            # Reset counter regardless so we don't re-trigger on the
            # same empty conditions in a tight loop.
            self._collapse_counter = 0
            # Collapse mutated _hole (singularity facts
            # removed) and _meta_facts (new MetaFacts registered)
            # if anything actually compressed. Forecast cache keys
            # on body counts and feature state — must invalidate.
            # Skip the bump on a no-op pass (report.groups == 0)
            # so we don't churn the cache for empty attempts.
            if report.groups > 0 or report.absorbed_facts > 0:
                self._bump_mutation_locked()
            return report

    def _maybe_trigger_collapse_locked(self, absorbed_count: int) -> None:
        """Caller must hold self._lock. Schedules collapse if thresholds met."""
        self._collapse_counter += absorbed_count
        if self._hole.fact_mass < self.COLLAPSE_FACT_MASS_TRIGGER:
            return
        if self._collapse_counter < self.COLLAPSE_DELTA_TRIGGER:
            return
        # Skip if a previous collapse is still running. If it finished
        # but raised, capture the error before scheduling another one
        # so a recurring background failure shows up in stats instead
        # of staying buried in the future.
        if self._inflight_collapse is not None:
            if not self._inflight_collapse.done():
                return
            try:
                self._inflight_collapse.result(timeout=0)
            except Exception as exc:
                self._last_collapse_error = repr(exc)
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
        """Run a galaxy forecast and write ``forecast_stability`` back to bodies.

        The galaxy module models every live body (FactPassport AND MetaFact
        — both carry ``forecast_stability``) as an N-body orbiting a central
        black hole. Running it forward gives a per-body prediction of how
        close that body will be to the event horizon after ``horizon_ticks``
        steps. Stability ∈ [0, 1]: 1.0 = predicted safely on surface,
        0.0 = predicted to fall, 0.5 = neutral prior (default for bodies
        the galaxy could not place).

        The value is stored on FactPassport.forecast_stability /
        MetaFact.forecast_stability and consumed by the adaptive gravity
        formula via ``w_stability`` — so this call materially feeds back
        into how the formula scores bodies on the next tick. The galaxy
        build + simulation is O(n²) per step in body count and pure numpy,
        fine for the few hundred to few thousand bodies a personal store
        holds. Returns a small summary with ``bodies_*`` keys and a
        per-type split (``facts_updated_count`` / ``metas_updated_count``);
        full per-body values are persisted, not returned. Legacy
        ``facts_forecasted`` / ``facts_updated`` keys are aliases for
        wire-format stability but actually count BODIES.
        """
        from ..galaxy.forecast import forecast_stability

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
                    self._mutation_version,
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
            try:
                with self._txn():
                    self._sync()
                        # Snapshot revalidation: the heavy N² simulation above
                    # ran OUTSIDE the lock so concurrent agents could keep
                    # querying. While we were away, another thread could
                    # have called set_fact / session_close / add_fact /
                    # delete_body and bumped _mutation_version. Writing
                    # stale scores into the surviving subset of bodies
                    # would silently feed the next tick's adaptive gravity
                    # with values computed from a phantom past. Recompute
                    # the same key under the lock and abort cleanly if
                    # the universe moved. Agent retries; the next call
                    # picks up the post-mutation state.
                    live_count = len(self._facts) + len(self._meta_facts)
                    writeback_key = (
                        self._data_version_now(),
                        self._mutation_version,
                        live_count,
                        horizon_ticks,
                    )
                    if writeback_key != cache_key:
                        return {
                            "ok": False,
                            "error": "forecast_snapshot_stale",
                            "horizon_ticks": horizon_ticks,
                            "snapshot_body_count": len(bodies_snapshot),
                            "writeback_body_count": live_count,
                            "hint": (
                                "Memory mutated between snapshot and "
                                "writeback. Retry forecast_memory; the "
                                "next call sees the post-mutation state."
                            ),
                        }
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
                    # Distribution snapshot + payload construction +
                    # cache-slot write all happen under the writeback
                    # lock — _forecast_cache is shared mutable state and
                    # two concurrent forecasts racing into the slot
                    # could tear the assignment. The lock that already
                    # guards _facts / _meta_facts / _mutation_version
                    # also owns the cache slot.
                    ranges = {
                        "safe": 0, "kinetic": 0,
                        "near_horizon": 0, "predicted_fall": 0,
                    }
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
                        # Kept for wire-format stability; aliases of the
                        # new keys.
                        "facts_forecasted": len(scores),
                        "facts_updated": updated,
                        # Clearer keys: forecast now covers both
                        # FactPassports and MetaFacts (both carry
                        # forecast_stability), so the operator can see
                        # how the update split across body types.
                        "bodies_forecasted": len(scores),
                        "bodies_updated": updated,
                        "facts_updated_count": facts_updated_n,
                        "metas_updated_count": metas_updated_n,
                        "distribution": ranges,
                        "_hint": (
                            "facts_forecasted / facts_updated are legacy "
                            "aliases — they actually count BODIES "
                            "(FactPassport + MetaFact). Prefer "
                            "bodies_forecasted / bodies_updated, or read "
                            "the per-type split via facts_updated_count "
                            "and metas_updated_count."
                        ),
                    }
                    # Cache the response keyed by the snapshot we
                    # forecasted against. Subsequent calls with no
                    # intervening writes hit the cache.
                    #
                    # INTENTIONAL non-bump: run_forecast writes
                    # forecast_stability into bodies but does NOT call
                    # _bump_mutation_locked(). Reason: this write is a
                    # *projection* of the (data_version, mutation_version)
                    # snapshot we just forecasted against — re-running on
                    # the same snapshot must hit the cache, so bumping
                    # would force a needless N² recompute on every
                    # consecutive call. The downstream consumers of
                    # forecast_stability (pre_resonance_features in
                    # gravity, SGD in session_close) read the new values
                    # on their next access; they don't subscribe to
                    # mutation_version for forecast specifically.
                    # If a future caller needs "tell me when forecast
                    # values changed" granularity, add a dedicated
                    # _forecast_version counter rather than coupling
                    # forecast writes to the general mutation bump.
                    self._forecast_cache = (
                        cache_key, dict(result_payload),
                    )
            except Exception:
                # Standard rollback-recovery pattern, symmetric with
                # add_fact / add_facts / query / collapse_singularity
                # / session_close. The storage txn rolls back the
                # disk truth, but in-memory fact.forecast_stability
                # / meta.forecast_stability writes above are not in
                # the txn — they mutate live objects. Without
                # _reload(), disk truth and RAM truth diverge:
                # next session_close trains adaptive_gravity with
                # phantom forecast scores, and a restart silently
                # snaps them back. forecast_stability feeds gravity
                # directly via pre_resonance_features, so the drift
                # propagates into layer migration decisions.
                self._reload()
                raise
        return result_payload
