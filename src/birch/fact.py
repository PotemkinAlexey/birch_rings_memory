"""FactPassport — atomic unit of knowledge in BirchKM."""
from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional


def _sanitize_float(
    value, default, *, lo: float | None = None, hi: float | None = None,
):
    """Constructor-side scalar sanitiser. Same contract as the SQLite
    backend's ``_finite_float``; lives here so direct construction
    (tests, in-memory migrations, library use) gets the same gate."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(out):
        return default
    if lo is not None:
        out = max(lo, out)
    if hi is not None:
        out = min(hi, out)
    return out


def _sanitize_nonneg_int(value, default: int) -> int:
    """Same as the loader's ``_nonnegative_int`` — defaults on
    non-coercible, clamps negatives to zero."""
    try:
        out = int(value)
    except (TypeError, ValueError):
        return default
    return max(0, out)


def _sanitize_namespace(value) -> str:
    """Coerce namespace to a stripped str.

    MemoryBricks Step 1: the namespace field is a path-style scope
    identifier (e.g. ``"WORK/DataArt/Databricks"``) borrowed from VB.
    Case-sensitive like VB paths, but trimmed of surrounding whitespace
    so ``" WORK "`` and ``"WORK"`` collapse to the same slot. None,
    non-string, or coercion failure all reduce to ``""`` (the
    global/unscoped root) so the field is never typed loosely.
    """
    if value is None:
        return ""
    try:
        out = str(value)
    except Exception:
        return ""
    return out.strip()


def _sanitize_layer(value, default: int = 1) -> int:
    """Layer must be one of (-1, 0, 1, 2). Unknown values revert to
    the default — same contract as the SQLite backend's ``_layer``."""
    try:
        out = int(value)
    except (TypeError, ValueError):
        return default
    if out not in (-1, 0, 1, 2):
        return default
    return out


@dataclass
class FactPassport:
    subject: str
    predicate: str
    object: str

    fact_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    vector: list[float] = field(default_factory=list)

    gravity_score: float = 0.5      # starts neutral, drifts with usage
    layer: int = 1                  # 0=surface, 1=kinetic, 2=core
    # MemoryBricks Step 1: scope identifier. Hierarchical path-style
    # string (e.g. "WORK/DataArt/Databricks") matching VB's namespace
    # convention; empty string means the global / unscoped root.
    # Reputation lives per-namespace per the "Reputation is scoped,
    # not global" invariant (see MemoryBricks
    # docs/STRUCTURED_LIVING_MEMORY.md). SPO dedup uses
    # (namespace, subject, predicate, object) — two facts with same
    # SPO under different namespaces are independent rows.
    namespace: str = ""

    created_at: float = field(default_factory=time.time)
    ttl: Optional[float] = None     # None = no expiry

    source_session: Optional[str] = None
    deprecated_by: Optional[str] = None   # fact_id that superseded this

    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    resonance_sum: float = 0.0      # cumulative (contrast-shrunk) R → gravity
    resonance_count: int = 0        # how many sessions contributed
    # Un-shrunk cumulative R — the fact's true track record, read by the
    # contrastive trust decision (resonance_sum is shrunk; this is not, so
    # trust never feeds on its own shrinking). See gravity.contrastive_impulse.
    raw_resonance_sum: float = 0.0
    # Declared (top-down) salience in [0, 1] — the one signal that cannot be
    # inferred bottom-up: "this is critical, don't forget", set at write time.
    # Floors absorption regardless of resonance_count, then decays use-it-or-
    # lose-it (eroded only when the fact surfaces into a non-positive session).
    # 0 = not pinned (the default; the system stays preponderantly inferential).
    encode_salience: float = 0.0
    # Durable pin telemetry (the channel's verdict metric, survives restart).
    # was_pinned: this fact was declared salient at least once (monotonic, never
    # cleared — distinct from encode_salience, which decays). pin_resonated: an
    # ever-pinned fact later rode a resonant session — declaration predicting
    # criticality. pins_created / pins_resonated in stats DERIVE from these, so
    # the verdict accrues across restarts instead of resetting per process.
    was_pinned: bool = False
    pin_resonated: bool = False

    # EWMA of recent contextual usefulness (closure-weighted resonance).
    # Default 0.5 = Bayesian neutral prior; untouched facts get a soft floor.
    recent_utility: float = 0.5

    # Galaxy-derived forecast: how far this fact will be from the horizon
    # after a short forward simulation. 1.0 = predicted safely on surface,
    # 0.0 = predicted to fall, 0.5 = unknown / no forecast run yet.
    # Updated by MemoryStore.run_forecast(); not touched per session.
    forecast_stability: float = 0.5

    def __post_init__(self) -> None:
        """Sanitise direct-construction values.

        Library users build FactPassport directly in tests, migrations,
        and ad-hoc scripts — bypassing the SQLite loader's
        ``_finite_float`` / ``_layer`` / ``_nonnegative_int`` gates.
        Without this hook, ``FactPassport(..., gravity_score=float
        ("nan"))`` would silently sit in memory until the next save
        crashes (write-side ``allow_nan=False``) or until the next
        ``compute_gravity`` returns NaN. Normalise on construction
        instead of failing late: poisoned scalars revert to defaults,
        legitimate-but-out-of-range scalars clamp. Strings / None /
        bool slot through ``float()``-coerce-or-default same as the
        loader.
        """
        self.gravity_score = _sanitize_float(
            self.gravity_score, 0.5, lo=0.0, hi=1.0,
        )
        self.layer = _sanitize_layer(self.layer, 1)
        self.created_at = _sanitize_float(
            self.created_at, time.time(),
        )
        if self.ttl is not None:
            self.ttl = _sanitize_float(self.ttl, None)
        self.access_count = _sanitize_nonneg_int(self.access_count, 0)
        self.last_accessed = _sanitize_float(
            self.last_accessed, time.time(),
        )
        self.resonance_sum = _sanitize_float(self.resonance_sum, 0.0)
        self.resonance_count = _sanitize_nonneg_int(
            self.resonance_count, 0,
        )
        self.raw_resonance_sum = _sanitize_float(self.raw_resonance_sum, 0.0)
        # Count invariant: every impulse ∈ [-1, 1], so |sum| ≤ count for both
        # accumulators. A corrupted external row (sum far outside that) would
        # otherwise skew avg_resonance → trust → gravity. Clamp to the
        # count-bound — the read-side completion of the storage symmetry. A
        # no-op for legitimate data, which always satisfies the bound.
        _bound = float(self.resonance_count)
        self.resonance_sum = max(-_bound, min(_bound, self.resonance_sum))
        self.raw_resonance_sum = max(-_bound, min(_bound, self.raw_resonance_sum))
        self.encode_salience = _sanitize_float(
            self.encode_salience, 0.0, lo=0.0, hi=1.0,
        )
        self.was_pinned = bool(self.was_pinned)
        self.pin_resonated = bool(self.pin_resonated)
        self.recent_utility = _sanitize_float(
            self.recent_utility, 0.5, lo=0.0, hi=1.0,
        )
        self.forecast_stability = _sanitize_float(
            self.forecast_stability, 0.5, lo=0.0, hi=1.0,
        )
        self.namespace = _sanitize_namespace(self.namespace)

    @property
    def is_deprecated(self) -> bool:
        return self.deprecated_by is not None

    @property
    def is_expired(self) -> bool:
        return self.ttl is not None and time.time() > self.ttl

    @property
    def avg_resonance(self) -> float:
        """Mean (contrast-shrunk) session resonance — the value gravity uses.
        Finite-safe: neutral 0.0 on a poisoned (NaN) sum so ranking stays
        deterministic."""
        if self.resonance_count <= 0:
            return 0.0
        avg = self.resonance_sum / self.resonance_count
        if not math.isfinite(avg):
            return 0.0
        return avg

    @property
    def raw_avg_resonance(self) -> float:
        """Mean un-shrunk resonance — true track record, read by the contrastive
        trust decision. Order-independent, never shaped by past shrinks."""
        if self.resonance_count <= 0:
            return 0.0
        avg = self.raw_resonance_sum / self.resonance_count
        if not math.isfinite(avg):
            return 0.0
        return avg

    def touch(self) -> None:
        self.access_count += 1
        self.last_accessed = time.time()

    @staticmethod
    def _clean_resonance(r: float) -> Optional[float]:
        try:
            value = float(r)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(value):
            return None
        return max(-1.0, min(1.0, value))

    def record_resonance(self, raw: float, gravity: float) -> None:
        """Record one session: ``raw`` (= effective_r·weight) into the track
        record, ``gravity`` (contrast-shrunk) into the gravity input. Both
        clamp to [-1, 1]; a bad value in either no-ops the whole record."""
        raw_v = self._clean_resonance(raw)
        grav_v = self._clean_resonance(gravity)
        if raw_v is None or grav_v is None:
            return
        self.raw_resonance_sum += raw_v
        self.resonance_sum += grav_v
        self.resonance_count += 1

    def apply_resonance(self, r: float) -> None:
        """Record a session at full strength to both accumulators (no shrink).
        Self-defending no-op on NaN/Inf. The engine uses record_resonance to
        apply a shrunk gravity impulse; this is the simple entry point."""
        self.record_resonance(r, r)

    def __repr__(self) -> str:
        return (
            f"Fact({self.fact_id!r}: {self.subject!r} {self.predicate!r} "
            f"{self.object!r} g={self.gravity_score:.2f} layer={self.layer})"
        )
