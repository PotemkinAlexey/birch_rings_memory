"""StatsMixin — read-only ``stats`` view of the store."""
from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Callable, Optional

from ..fact import FactPassport
from ..meta_fact import MetaFact
from ..thresholds import Thresholds

if TYPE_CHECKING:  # pragma: no cover
    from ..black_hole import BlackHole
    from ..gravity import GravityEngine
    from ..resonance.echo import EchoStore
    from ._models import SessionContext


class StatsMixin:
    """``stats`` property — see ``MemoryStore`` for the assembled API."""

    _lock: "threading.RLock"
    _facts: "dict[str, FactPassport]"
    _meta_facts: "dict[str, MetaFact]"
    _hole: "BlackHole"
    _engine: "GravityEngine"
    _echo: "EchoStore"
    _sessions: "dict[str, SessionContext]"
    _salience_retained_ids: "set[str]"
    _pins_evicted: int
    _collapse_counter: int
    _total_collapses: int
    _total_collapse_attempts: int
    _last_collapse_at: Optional[float]
    _last_collapse_error: Optional[str]
    _last_collapse_attempt_at: Optional[float]

    if TYPE_CHECKING:
        _sync: Callable[[], None]

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
            # Pin telemetry DERIVED from persisted fact state (live + absorbed),
            # so the verdict survives restart. A pinned fact resists absorption,
            # so it's almost always live; scanning the singularity too keeps the
            # count honest for the rare pin that decayed and fell in.
            pinned_bodies = list(self._facts.values()) + [
                rec.fact for rec in self._hole._singularity.values()
            ]
            pins_created = sum(1 for f in pinned_bodies if f.was_pinned)
            pins_resonated = sum(
                1 for f in pinned_bodies if f.was_pinned and f.pin_resonated)
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
                "last_collapse_error": self._last_collapse_error,
                "last_collapse_attempt_at": self._last_collapse_attempt_at,
                "adaptive_weights": self._engine.weights.as_dict(),
                "echo_sessions": len(self._echo),
                "total_echoes_detected": self._echo.total_echoes_detected,
                "total_echoes_applied": self._echo.total_echoes_applied,
                "total_echoes_ignored": self._echo.total_echoes_ignored,
                # echoes session_close withheld (revisit ended resonant)
                "total_echoes_cancelled": self._echo.total_echoes_cancelled,
                # impulses the contrastive rule shrank for contradicting history
                "contrastive_attenuations": self._engine.contrastive_attenuations,
                # distinct facts salience (irreplaceability) kept from absorption
                "salience_retained": len(self._salience_retained_ids),
                # Declarative-pin telemetry — the verdict on the top-down channel.
                # pins_resonated / pins_created near zero over real traffic ⇒
                # people pin noise, the channel isn't earning its keep.
                "pins_active": sum(
                    1 for f in self._facts.values() if f.encode_salience > 0.0),
                "pins_created": pins_created,
                "pins_resonated": pins_resonated,
                "pins_evicted": self._pins_evicted,
                # Diagnostics: which thresholds the process actually
                # picked up. Operator can confirm BIRCH_* env vars
                # took effect without reading the process environment.
                "thresholds": Thresholds.as_dict(),
                # Thresholds are resolved at module import time. An
                # operator changing BIRCH_*_THRESHOLD env vars on a
                # running process will NOT see the new values here
                # until the process restarts. Flag is here so a
                # caller comparing stats["thresholds"] to current
                # env doesn't assume hot-reload.
                "thresholds_are_import_time": True,
            }
