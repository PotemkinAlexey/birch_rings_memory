"""MetaFact — compressed representative of several absorbed facts.

A MetaFact is what remains after a cluster of facts has fallen into the
black hole: a single centroid vector, the list of original SPO texts it
swallowed, an integer weight (how many facts it represents) and, once
LLM summarisation lands, a natural-language summary.

It lives in the black hole at ``layer = -1`` and starts with a non-zero
``gravity_score`` so it has weight from the moment it is created — even
before Hawking emission can promote it back to the live layers.

A MetaFact implements the same gravity-relevant surface as
``FactPassport`` — ``access_count``, ``last_accessed``, ``resonance_sum``,
``resonance_count``, ``apply_resonance()``, ``touch()`` — so the
``GravityEngine`` and the ``BlackHole`` can treat it polymorphically
through duck typing.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MetaFact:
    meta_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    vector: list[float] = field(default_factory=list)
    weight: int = 1                                  # how many facts it absorbed
    source_texts: list[str] = field(default_factory=list)  # "subject predicate object"
    source_fact_ids: list[str] = field(default_factory=list)  # lineage to originals
    summary: str = ""                                # populated by later LLM pass
    gravity_score: float = 0.30                      # starts non-zero — meta is dense
    created_at: float = field(default_factory=time.time)
    layer: int = -1                                  # lives in the black hole until emitted

    # ── Feedback-loop participation (same surface as FactPassport) ──────────
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    resonance_sum: float = 0.0
    resonance_count: int = 0
    # Un-shrunk track record; see FactPassport.raw_resonance_sum.
    raw_resonance_sum: float = 0.0
    recent_utility: float = 0.5
    forecast_stability: float = 0.5
    # MemoryBricks Step 1: see FactPassport.namespace.
    namespace: str = ""

    def __post_init__(self) -> None:
        """Sanitise direct-construction values.

        Symmetric with ``FactPassport.__post_init__``. MetaFacts get
        built directly in compactor tests, in migrations, and by
        library users — bypassing the SQLite loader's gates. A
        poisoned scalar at construction time would sit in memory
        until the next save crashes (write-side ``allow_nan=False``)
        or until ``compute_gravity`` returns NaN. Normalise on
        construction instead of failing late.
        """
        import math as _math

        def _f(v, default, lo=None, hi=None):
            try:
                out = float(v)
            except (TypeError, ValueError):
                return default
            if not _math.isfinite(out):
                return default
            if lo is not None:
                out = max(lo, out)
            if hi is not None:
                out = min(hi, out)
            return out

        def _i(v, default):
            try:
                return max(0, int(v))
            except (TypeError, ValueError):
                return default

        def _layer(v, default=-1):
            try:
                out = int(v)
            except (TypeError, ValueError):
                return default
            return out if out in (-1, 0, 1, 2) else default

        self.weight = _i(self.weight, 1)
        self.gravity_score = _f(
            self.gravity_score, 0.30, lo=0.0, hi=1.0,
        )
        self.created_at = _f(self.created_at, time.time())
        self.layer = _layer(self.layer, -1)
        self.access_count = _i(self.access_count, 0)
        self.last_accessed = _f(self.last_accessed, time.time())
        self.resonance_sum = _f(self.resonance_sum, 0.0)
        self.resonance_count = _i(self.resonance_count, 0)
        self.raw_resonance_sum = _f(self.raw_resonance_sum, 0.0)
        # Count invariant: |sum| ≤ count for both accumulators (each impulse is
        # in [-1, 1]). Clamp a corrupted external row to the count-bound so it
        # can't skew avg_resonance. Symmetric with FactPassport.
        _bound = float(self.resonance_count)
        self.resonance_sum = max(-_bound, min(_bound, self.resonance_sum))
        self.raw_resonance_sum = max(-_bound, min(_bound, self.raw_resonance_sum))
        self.recent_utility = _f(
            self.recent_utility, 0.5, lo=0.0, hi=1.0,
        )
        self.forecast_stability = _f(
            self.forecast_stability, 0.5, lo=0.0, hi=1.0,
        )
        # MemoryBricks Step 1: namespace is a path-style scope
        # identifier shared with FactPassport / VB. Coerce loose
        # constructor values (None, non-str, surrounding whitespace)
        # to the canonical stripped string so SPO dedup and prefix
        # filters key on a single normalised form. Case-sensitive
        # like VB paths.
        if self.namespace is None:
            self.namespace = ""
        else:
            try:
                self.namespace = str(self.namespace).strip()
            except Exception:
                self.namespace = ""

    # ── Polymorphism shims for code that ducks on FactPassport ──────────────
    # FactPassport carries optional ttl/deprecated_by fields; MetaFacts cannot
    # currently be deprecated or expire, but exposing the same booleans keeps
    # GravityEngine and BlackHole._absorb_dead checks symmetric.
    @property
    def is_deprecated(self) -> bool:
        return False

    @property
    def is_expired(self) -> bool:
        return False

    @property
    def fact_id(self) -> str:
        """Alias so duck-typed callers can use .fact_id on either type."""
        return self.meta_id

    @property
    def avg_resonance(self) -> float:
        """Mean session resonance touched by this MetaFact.

        Finite-safe symmetric with ``FactPassport.avg_resonance``.
        A NaN ``resonance_sum`` injected via direct attribute
        mutation used to leak into ``compute_gravity`` and freeze
        the layer ranking. Neutral 0.0 on poison.
        """
        import math as _math

        if self.resonance_count <= 0:
            return 0.0
        avg = self.resonance_sum / self.resonance_count
        if not _math.isfinite(avg):
            return 0.0
        return avg

    @property
    def raw_avg_resonance(self) -> float:
        """Mean UN-shrunk session resonance — true track record, read by the
        contrastive trust decision. Symmetric with FactPassport."""
        import math as _math

        if self.resonance_count <= 0:
            return 0.0
        avg = self.raw_resonance_sum / self.resonance_count
        if not _math.isfinite(avg):
            return 0.0
        return avg

    def touch(self) -> None:
        self.access_count += 1
        self.last_accessed = time.time()

    @staticmethod
    def _clean_resonance(r: float):
        import math as _math

        try:
            value = float(r)
        except (TypeError, ValueError):
            return None
        if not _math.isfinite(value):
            return None
        return max(-1.0, min(1.0, value))

    def record_resonance(self, raw: float, gravity: float) -> None:
        """Record one session: raw track record + (shrunk) gravity input,
        separately. Symmetric with ``FactPassport.record_resonance``."""
        raw_v = self._clean_resonance(raw)
        grav_v = self._clean_resonance(gravity)
        if raw_v is None or grav_v is None:
            return
        self.raw_resonance_sum += raw_v
        self.resonance_sum += grav_v
        self.resonance_count += 1

    def apply_resonance(self, r: float) -> None:
        """Record a session at full strength to both accumulators (no shrink).
        Self-defending; symmetric with ``FactPassport.apply_resonance``."""
        self.record_resonance(r, r)

    # ── Hawking emission helper ─────────────────────────────────────────────
    def gravity_on_emission(self, base: float = 0.30) -> float:
        """Gravity to assign on Hawking emission.

        A MetaFact carries the weight of every fact it absorbed, but we
        do not want a MetaFact with weight=50 to leap straight into the
        surface layer: each of its constituents was dead. A logarithmic
        bonus on top of the base gives a defensible, bounded scaling.
        """
        import math
        weight = max(1, int(self.weight))
        bonus = 0.10 * math.log10(weight)       # log10(1)=0, log10(10)=0.10
        return float(max(0.0, min(0.70, base + bonus)))

    # ── Serialisation ───────────────────────────────────────────────────────
    #
    # to_dict() returns a row already shaped for SQLite: scalars stay scalar,
    # list-valued fields are JSON-encoded so they fit a TEXT column. from_dict
    # is symmetric and tolerant — it accepts both the JSON-encoded form and
    # raw Python lists, so callers can build a MetaFact from a hand-written
    # dict in tests without having to JSON-dump first.

    def to_dict(self) -> dict[str, Any]:
        # allow_nan=False on every json.dumps so a NaN sneaked into
        # the runtime vector / lineage raises ValueError here instead
        # of writing radioactive JSON to disk. The surrounding
        # storage txn rolls back, _reload restores the pre-write
        # snapshot, and the loader never has to "heroically" clean
        # data we never should have written.
        return {
            "meta_id": self.meta_id,
            "vector": json.dumps(self.vector, allow_nan=False),
            "weight": self.weight,
            "source_texts": json.dumps(
                self.source_texts, allow_nan=False,
            ),
            "source_fact_ids": json.dumps(
                self.source_fact_ids, allow_nan=False,
            ),
            "summary": self.summary,
            "gravity_score": self.gravity_score,
            "created_at": self.created_at,
            "layer": self.layer,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed,
            "resonance_sum": self.resonance_sum,
            "resonance_count": self.resonance_count,
            "raw_resonance_sum": self.raw_resonance_sum,
            "recent_utility": self.recent_utility,
            "forecast_stability": self.forecast_stability,
            "namespace": self.namespace,
        }

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "MetaFact":
        # Scalar sanitisation symmetric with FactPassport's SQLite
        # loader. ``_load_list`` already gates vector items against
        # NaN/Infinity; the scalar fields used to skip that gate, so a
        # gravity_score=NaN or created_at=Infinity stored row would
        # build a MetaFact whose downstream gravity math returns NaN
        # (adaptive_gravity SGD then freezes its weights silently).
        # Apply the same finite + clamp contract here so corrupt
        # storage rows never reach the live store.
        return cls(
            meta_id=row["meta_id"],
            vector=_load_list(row.get("vector"), float),
            weight=_finite_nonneg_int(row.get("weight"), 1),
            source_texts=_load_list(row.get("source_texts"), str),
            source_fact_ids=_load_list(row.get("source_fact_ids"), str),
            summary=row.get("summary", "") or "",
            gravity_score=_finite(
                row.get("gravity_score"), 0.30, lo=0.0, hi=1.0,
            ),
            created_at=_finite(
                row.get("created_at"), time.time(),
            ),
            layer=_meta_layer(row.get("layer"), -1),
            access_count=_finite_nonneg_int(row.get("access_count"), 0),
            last_accessed=_finite(
                row.get("last_accessed"), time.time(),
            ),
            resonance_sum=_finite(row.get("resonance_sum"), 0.0),
            resonance_count=_finite_nonneg_int(
                row.get("resonance_count"), 0,
            ),
            # Tolerant backfill: pre-#5 rows lack the column; their
            # resonance_sum was never shrunk, so it IS the raw history.
            raw_resonance_sum=_finite(
                row.get("raw_resonance_sum", row.get("resonance_sum")), 0.0,
            ),
            recent_utility=_finite(
                row.get("recent_utility"), 0.5, lo=0.0, hi=1.0,
            ),
            forecast_stability=_finite(
                row.get("forecast_stability"), 0.5, lo=0.0, hi=1.0,
            ),
            namespace=str(row.get("namespace") or ""),
        )


def _finite(
    value: Any, default: float, *, lo: float | None = None,
    hi: float | None = None,
) -> float:
    """Coerce to a finite float with optional clamp; defaults on failure.

    Same contract as the SQLite backend's ``_finite_float``; lives here
    too because ``MetaFact.from_dict`` is called both by the SQLite
    loader and by in-memory test fixtures / migrations. Either path
    must reject NaN / Infinity / non-numeric cells rather than build a
    body that poisons every downstream gravity calculation.
    """
    import math as _math

    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not _math.isfinite(out):
        return default
    if lo is not None:
        out = max(lo, out)
    if hi is not None:
        out = min(hi, out)
    return out


def _finite_nonneg_int(value: Any, default: int) -> int:
    """Coerce to a non-negative int; defaults on failure."""
    try:
        out = int(value)
    except (TypeError, ValueError):
        return default
    return max(0, out)


def _meta_layer(value: Any, default: int = -1) -> int:
    """Coerce layer to one of the known ids (-1, 0, 1, 2).

    A MetaFact at layer=99 would slip past every ``layer == -1``
    singularity scan and every ``layer >= 0`` live-layer predicate —
    silently invisible. Default to singularity (-1, the natural meta
    home) when the cell is corrupt.
    """
    try:
        out = int(value)
    except (TypeError, ValueError):
        return default
    if out not in (-1, 0, 1, 2):
        return default
    return out


def _load_list(value: Any, cast) -> list:
    """Accept a JSON-encoded string, a Python list, or None.

    Tolerant on item-level cast failures too — a list whose JSON
    parsed cleanly but contains values the caster can't accept
    (e.g. ``[1, "oops"]`` with ``float`` cast) returns ``[]``
    rather than raising. The SQLite loader catches the row-level
    case via its pre-validation, but direct callers (tests,
    in-memory migrations) get the same forgiving contract here
    — symmetric with FactPassport loading via ``_safe_vector``."""
    if value is None or value == "":
        return []
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            return []
    else:
        parsed = value
    if not isinstance(parsed, list):
        return []
    try:
        out = [cast(x) for x in parsed]
    except (TypeError, ValueError):
        return []
    # NaN / Infinity sneak through float() — they're valid floats per
    # Python but poison numpy cosine similarity (every comparison
    # becomes NaN, ranking undefined). FactPassport vectors go through
    # _safe_vector in the SQLite backend which checks math.isfinite;
    # MetaFacts used to skip that gate. Symmetric defence here.
    if cast is float:
        import math as _math
        if any(not _math.isfinite(x) for x in out):
            return []
    return out
