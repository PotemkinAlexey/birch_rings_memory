"""Three polish-grade defences after the radioactive-crystals round:

  1. ``compute_gravity`` final finite check. Every input is gated at
     storage / engine / public-method boundaries, but library mode
     can mutate ``fact.resonance_sum = float("nan")`` between load
     and next call here. ``avg_resonance`` then returns NaN, gravity
     arithmetic propagates it, and ``min/max(NaN, ...)`` is
     platform-dependent. Default to the neutral 0.0 (dead-weight
     body, absorption candidate) on poison.

  2. ``FactPassport.avg_resonance`` / ``MetaFact.avg_resonance``
     finite-safe. Symmetric defence: a NaN ``resonance_sum`` no
     longer leaks into ``compute_gravity`` even when the gravity
     final-check is bypassed.

  3. ``FactPassport.__post_init__`` / ``MetaFact.__post_init__``
     normalise direct-construction values. Library mode and tests
     build bodies without going through the SQLite loader; a
     poisoned scalar at construction time used to sit in memory
     until ``save`` crashed or ``compute_gravity`` returned NaN.
     Sanitise on construction instead of failing late.
"""
from __future__ import annotations

import math

from birch.fact import FactPassport
from birch.gravity import compute_gravity
from birch.meta_fact import MetaFact

# --- I1: compute_gravity final finite check ----------------------------


def test_compute_gravity_returns_zero_on_nan_path():
    """Exercise compute_gravity's final finite-check directly by
    bypassing avg_resonance's self-defence — subclass with the
    property forced to NaN. In real use ``avg_resonance`` fires
    first (belt-and-suspenders), but compute_gravity must still
    self-defend in case a future caller routes a non-FactPassport
    body through the gravity path.
    """
    class _PoisonedFact(FactPassport):
        @property
        def avg_resonance(self) -> float:   # type: ignore[override]
            return float("nan")

    f = _PoisonedFact(subject="a", predicate="b", object="c")
    f.resonance_count = 1   # force the resonance branch
    g = compute_gravity(f)
    assert math.isfinite(g)
    assert g == 0.0


def test_compute_gravity_returns_zero_on_inf_path():
    class _InfFact(FactPassport):
        @property
        def avg_resonance(self) -> float:   # type: ignore[override]
            return float("inf")

    f = _InfFact(subject="a", predicate="b", object="c")
    f.resonance_count = 1
    g = compute_gravity(f)
    assert math.isfinite(g)
    assert g == 0.0


def test_avg_resonance_is_first_line_for_real_factpassport():
    """Confirm the belt-and-suspenders: a NaN ``resonance_sum`` on a
    real FactPassport is caught by avg_resonance first, so
    compute_gravity returns the (finite) non-resonance contribution,
    NOT zero. The compute_gravity gate is the safety net for non-
    FactPassport bodies (e.g. duck-typed types from extensions)."""
    f = FactPassport(subject="a", predicate="b", object="c")
    f.resonance_sum = float("nan")
    f.resonance_count = 1
    g = compute_gravity(f)
    assert math.isfinite(g)
    # avg_resonance returned 0.0, so the resonance branch contributes
    # _W_RESONANCE * ((0 + 1) / 2) = 0.35 * 0.5 = 0.175 plus the
    # pre-resonance features. Just assert it's a sane finite number
    # in (0, 1), not the panicked 0.0 default.
    assert 0.0 < g <= 1.0


def test_compute_gravity_clean_path_still_works():
    f = FactPassport(subject="a", predicate="b", object="c")
    # Add some legitimate resonance and confirm we land in (0, 1).
    f.apply_resonance(0.5)
    g = compute_gravity(f)
    assert math.isfinite(g)
    assert 0.0 <= g <= 1.0


# --- I2: avg_resonance finite-safe -----------------------------------


def test_factpassport_avg_resonance_handles_nan_sum():
    f = FactPassport(subject="a", predicate="b", object="c")
    f.resonance_sum = float("nan")
    f.resonance_count = 5
    assert f.avg_resonance == 0.0


def test_factpassport_avg_resonance_handles_inf_sum():
    f = FactPassport(subject="a", predicate="b", object="c")
    f.resonance_sum = float("inf")
    f.resonance_count = 3
    assert f.avg_resonance == 0.0


def test_factpassport_avg_resonance_clean_path():
    f = FactPassport(subject="a", predicate="b", object="c")
    f.apply_resonance(0.6)
    f.apply_resonance(0.4)
    assert f.avg_resonance == 0.5
    assert f.resonance_count == 2


def test_metafact_avg_resonance_handles_nan_sum():
    m = MetaFact(meta_id="m", vector=[0.1])
    m.resonance_sum = float("nan")
    m.resonance_count = 5
    assert m.avg_resonance == 0.0


def test_metafact_avg_resonance_clean_path():
    m = MetaFact(meta_id="m", vector=[0.1])
    m.apply_resonance(0.3)
    m.apply_resonance(-0.3)
    assert m.avg_resonance == 0.0
    assert m.resonance_count == 2


def test_avg_resonance_zero_count_path():
    """Zero-resonance count → 0.0 (existing contract preserved)."""
    f = FactPassport(subject="a", predicate="b", object="c")
    assert f.avg_resonance == 0.0
    m = MetaFact(meta_id="m", vector=[0.1])
    assert m.avg_resonance == 0.0


# --- I3: __post_init__ sanitisation on direct construction -----------


def test_factpassport_post_init_drops_nan_gravity():
    f = FactPassport(
        subject="a", predicate="b", object="c",
        gravity_score=float("nan"),
    )
    assert math.isfinite(f.gravity_score)
    assert f.gravity_score == 0.5


def test_factpassport_post_init_drops_inf_utility():
    f = FactPassport(
        subject="a", predicate="b", object="c",
        recent_utility=float("inf"),
        forecast_stability=float("-inf"),
    )
    assert f.recent_utility == 0.5
    assert f.forecast_stability == 0.5


def test_factpassport_post_init_clamps_out_of_range():
    f = FactPassport(
        subject="a", predicate="b", object="c",
        gravity_score=2.5,        # above 1.0
        recent_utility=-0.5,      # below 0.0
    )
    assert f.gravity_score == 1.0
    assert f.recent_utility == 0.0


def test_factpassport_post_init_normalises_unknown_layer():
    f = FactPassport(
        subject="a", predicate="b", object="c",
        layer=99,
    )
    assert f.layer == 1   # kinetic default


def test_factpassport_post_init_clamps_negative_counters():
    f = FactPassport(
        subject="a", predicate="b", object="c",
        access_count=-50,
        resonance_count=-3,
    )
    assert f.access_count == 0
    assert f.resonance_count == 0


def test_factpassport_post_init_handles_string_typed_input():
    """A JSON-decoded dict that stringifies a scalar should still
    build (string-typed JSON from a poorly-typed client)."""
    f = FactPassport(
        subject="a", predicate="b", object="c",
        gravity_score="0.7",  # type: ignore[arg-type]
    )
    assert math.isclose(f.gravity_score, 0.7)


def test_factpassport_post_init_preserves_legitimate_ttl_none():
    f = FactPassport(
        subject="a", predicate="b", object="c",
        ttl=None,
    )
    assert f.ttl is None


def test_factpassport_post_init_sanitises_nan_ttl():
    f = FactPassport(
        subject="a", predicate="b", object="c",
        ttl=float("nan"),
    )
    assert f.ttl is None   # NaN ttl reverts to "no expiry"


def test_metafact_post_init_drops_nan_gravity():
    m = MetaFact(
        meta_id="m", vector=[0.1],
        gravity_score=float("nan"),
    )
    assert math.isfinite(m.gravity_score)
    assert m.gravity_score == 0.30


def test_metafact_post_init_normalises_unknown_layer():
    m = MetaFact(meta_id="m", vector=[0.1], layer=99)
    assert m.layer == -1   # singularity default


def test_metafact_post_init_clamps_negative_weight():
    m = MetaFact(meta_id="m", vector=[0.1], weight=-5)
    assert m.weight == 0


def test_metafact_post_init_drops_inf_utility():
    m = MetaFact(
        meta_id="m", vector=[0.1],
        recent_utility=float("inf"),
        forecast_stability=float("nan"),
    )
    assert m.recent_utility == 0.5
    assert m.forecast_stability == 0.5


def test_post_init_does_not_break_normal_construction():
    """The most important assertion: every existing test in the suite
    builds bodies through the constructor. __post_init__ must be a
    pure pass-through for clean values."""
    f = FactPassport(
        subject="api", predicate="uses", object="postgres",
        vector=[0.1, 0.2, 0.3],
        gravity_score=0.75,
        layer=0,
    )
    assert f.subject == "api"
    assert f.gravity_score == 0.75
    assert f.layer == 0
    assert f.vector == [0.1, 0.2, 0.3]
