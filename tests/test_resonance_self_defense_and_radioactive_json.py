"""Five contracts that turn "loader heroically cleans poison" into
"poison never reaches disk":

  1. FactPassport.apply_resonance + MetaFact.apply_resonance reject
     NaN / Infinity / non-numeric ``r``. Both are public object
     methods; library users can call them directly. ``max/min``
     in older code was not finite-aware, so a poisoned ``r`` used
     to land straight in ``resonance_sum`` and forever NaN every
     downstream ``avg_resonance`` / ``compute_gravity``.

  2. GravityEngine.apply_session_resonance sanitises both the
     session ``r`` and every per-fact weight before propagation.
     Bad ``r`` → whole-session no-op (so the agent sees a clean
     skip instead of every touched fact silently no-op'd at the
     body level); bad weight → pair-level skip (other facts
     still get propagation).

  3. _attribute_to (sessions) finite-guards the weight. ``max/min``
     is not NaN-aware: ``min(1.0, nan)`` is platform-dependent.
     Rejecting up front guarantees the persisted ``open_sessions
     .facts`` dict never carries a poisoned value, regardless of
     which caller wired the weight.

  4. ``json.dumps(..., allow_nan=False)`` on every storage write
     site (FactPassport vector, MetaFact vector + lineage, echo
     centroids + fact_weights, open_session messages + vectors +
     facts). Python's json silently writes the literal token
     ``NaN`` / ``Infinity`` by default (non-strict JSON); a poisoned
     in-memory body would otherwise persist radioactive cells that
     the loader has to "heroically" clean.

  5. Write-side scalar sanitisation in ``_fact_row`` / ``_meta_row``
     / ``save_echo_session`` / ``save_open_session``. Same helpers
     as the load path — the storage layer is now symmetric: it
     accepts radioactive runtime state and writes clean rows.
"""
from __future__ import annotations

import json
import math
import sqlite3

import pytest

from birch.fact import FactPassport
from birch.gravity import GravityEngine, _finite_clamped
from birch.memory_store import MemoryStore
from birch.meta_fact import MetaFact
from birch.storage.sqlite import SQLiteBackend

# --- I1: apply_resonance self-defence on FactPassport / MetaFact -------


def test_factpassport_apply_resonance_rejects_nan():
    f = FactPassport(subject="a", predicate="b", object="c")
    before_sum = f.resonance_sum
    before_count = f.resonance_count
    f.apply_resonance(float("nan"))
    assert f.resonance_sum == before_sum
    assert f.resonance_count == before_count


def test_factpassport_apply_resonance_rejects_inf():
    f = FactPassport(subject="a", predicate="b", object="c")
    f.apply_resonance(float("inf"))
    f.apply_resonance(float("-inf"))
    assert f.resonance_sum == 0.0
    assert f.resonance_count == 0


def test_factpassport_apply_resonance_rejects_non_numeric():
    f = FactPassport(subject="a", predicate="b", object="c")
    f.apply_resonance("totally-not-a-number")  # type: ignore[arg-type]
    f.apply_resonance(None)                    # type: ignore[arg-type]
    assert f.resonance_sum == 0.0
    assert f.resonance_count == 0


def test_factpassport_apply_resonance_clamps_to_unit_signed():
    f = FactPassport(subject="a", predicate="b", object="c")
    f.apply_resonance(5.0)
    assert f.resonance_sum == 1.0
    f.apply_resonance(-3.0)
    assert f.resonance_sum == 0.0   # 1.0 + (-1.0)
    assert f.resonance_count == 2


def test_factpassport_apply_resonance_accepts_normal_value():
    f = FactPassport(subject="a", predicate="b", object="c")
    f.apply_resonance(0.7)
    assert f.resonance_sum == pytest.approx(0.7)
    assert f.resonance_count == 1


def test_metafact_apply_resonance_rejects_nan():
    m = MetaFact(meta_id="m", vector=[0.1])
    m.apply_resonance(float("nan"))
    m.apply_resonance(float("inf"))
    assert m.resonance_sum == 0.0
    assert m.resonance_count == 0


def test_metafact_apply_resonance_clamps_and_accepts():
    m = MetaFact(meta_id="m", vector=[0.1])
    m.apply_resonance(10.0)
    assert m.resonance_sum == 1.0
    m.apply_resonance(-0.3)
    assert m.resonance_sum == pytest.approx(0.7)


# --- I2: GravityEngine.apply_session_resonance sanitisation -----------


def test_finite_clamped_helper():
    assert _finite_clamped(float("nan"), 0.0, 1.0) is None
    assert _finite_clamped(float("inf"), 0.0, 1.0) is None
    assert _finite_clamped("garbage", 0.0, 1.0) is None
    assert _finite_clamped(None, 0.0, 1.0) is None
    assert _finite_clamped(5.0, 0.0, 1.0) == 1.0
    assert _finite_clamped(-1.0, 0.0, 1.0) == 0.0
    assert _finite_clamped(0.3, 0.0, 1.0) == pytest.approx(0.3)


def test_engine_session_resonance_bad_r_is_whole_session_no_op():
    """Bad r should skip ALL facts, not propagate body-by-body
    no-ops. That way the agent sees a single clean "session
    skipped" outcome instead of silent per-fact misses."""
    eng = GravityEngine()
    f1 = FactPassport(subject="a", predicate="b", object="c", fact_id="f1")
    f2 = FactPassport(subject="x", predicate="y", object="z", fact_id="f2")
    eng.register(f1)
    eng.register(f2)
    eng.apply_session_resonance({"f1": 0.5, "f2": 0.8}, float("nan"))
    assert f1.resonance_sum == 0.0
    assert f1.resonance_count == 0
    assert f2.resonance_sum == 0.0
    assert f2.resonance_count == 0


def test_engine_session_resonance_bad_weight_is_pair_skip():
    """Bad weight on one fact must not abort propagation to the
    other facts in the same session."""
    eng = GravityEngine()
    f1 = FactPassport(subject="a", predicate="b", object="c", fact_id="f1")
    f2 = FactPassport(subject="x", predicate="y", object="z", fact_id="f2")
    eng.register(f1)
    eng.register(f2)
    eng.apply_session_resonance(
        {"f1": float("nan"), "f2": 0.5}, 0.8,
    )
    # f1 skipped
    assert f1.resonance_sum == 0.0
    assert f1.resonance_count == 0
    # f2 still propagated (0.5 * 0.8 = 0.4)
    assert f2.resonance_sum == pytest.approx(0.4)
    assert f2.resonance_count == 1


def test_engine_session_resonance_legacy_list_path_still_works():
    eng = GravityEngine()
    f1 = FactPassport(subject="a", predicate="b", object="c", fact_id="f1")
    eng.register(f1)
    eng.apply_session_resonance(["f1"], 0.6)
    assert f1.resonance_sum == pytest.approx(0.6)


# --- I3: _attribute_to finite guard -----------------------------------


def test_attribute_to_rejects_nan_weight(tmp_path):
    """A NaN weight piped into _attribute_to used to be ``clipped``
    by max/min — which is NOT NaN-aware. Now hard-rejected."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    mem.session_start("s1")
    ctx = mem._sessions["s1"]
    # Direct call to the static helper — bypass the public attribute
    # path to exercise the gate itself.
    mem._attribute_to(ctx, "fid", float("nan"))
    assert "fid" not in ctx.facts
    mem._attribute_to(ctx, "fid", float("inf"))
    assert "fid" not in ctx.facts
    mem._attribute_to(ctx, "fid", 0.7)
    assert ctx.facts["fid"] == pytest.approx(0.7)
    mem.close()


def test_attribute_to_clamps_out_of_range(tmp_path):
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    mem.session_start("s1")
    ctx = mem._sessions["s1"]
    mem._attribute_to(ctx, "fid", 5.0)
    assert ctx.facts["fid"] == 1.0
    mem.close()


# --- I4: allow_nan=False on every write path --------------------------


def test_save_fact_with_nan_vector_raises(tmp_path):
    """The whole point: a poisoned in-memory vector must not silently
    write radioactive JSON. json.dumps(allow_nan=False) raises so the
    surrounding txn rolls back."""
    be = SQLiteBackend(str(tmp_path / "m.db"))
    f = FactPassport(
        subject="a", predicate="b", object="c",
        vector=[1.0, float("nan"), 3.0],
    )
    with pytest.raises(ValueError):
        be.save_fact(f)
    be.close()


def test_save_meta_fact_with_inf_vector_raises(tmp_path):
    be = SQLiteBackend(str(tmp_path / "m.db"))
    m = MetaFact(meta_id="m", vector=[1.0, float("inf")])
    with pytest.raises(ValueError):
        be.save_meta_fact(m)
    be.close()


def test_save_echo_session_with_nan_centroid_raises(tmp_path):
    be = SQLiteBackend(str(tmp_path / "m.db"))
    with pytest.raises(ValueError):
        be.save_echo_session(
            session_id="s1",
            centroids=[[1.0, float("nan")]],
            r_score=0.5,
            recorded_at=100.0,
            fact_weights={"f1": 1.0},
        )
    be.close()


def test_save_open_session_with_nan_vector_raises(tmp_path):
    be = SQLiteBackend(str(tmp_path / "m.db"))
    with pytest.raises(ValueError):
        be.save_open_session(
            session_id="s1",
            messages=["hello"],
            vectors=[[1.0, float("nan")]],
            facts={"f1": 0.5},
            started_at=100.0,
        )
    be.close()


def test_save_open_session_with_nan_in_facts_dict_raises(tmp_path):
    be = SQLiteBackend(str(tmp_path / "m.db"))
    with pytest.raises(ValueError):
        be.save_open_session(
            session_id="s1",
            messages=["hello"],
            vectors=[[1.0, 2.0]],
            facts={"f1": float("nan")},
            started_at=100.0,
        )
    be.close()


# --- I5: write-side scalar sanitisation -------------------------------


def test_save_fact_with_nan_gravity_writes_default(tmp_path):
    """A FactPassport whose gravity got NaN'd by library API code
    should be sanitised at write — not propagate to disk and then
    require the loader to fix it on next boot."""
    be = SQLiteBackend(str(tmp_path / "m.db"))
    f = FactPassport(
        subject="a", predicate="b", object="c",
        fact_id="bad-fact",
        vector=[0.1, 0.2, 0.3],
        gravity_score=float("nan"),
    )
    be.save_fact(f)
    be._conn.commit()
    # Read raw row — should NOT contain NaN; should contain the field
    # default 0.5.
    row = be._conn.execute(
        "SELECT gravity_score FROM facts WHERE fact_id = 'bad-fact'",
    ).fetchone()
    assert row is not None
    g = row["gravity_score"]
    assert math.isfinite(g)
    assert g == 0.5
    be.close()


def test_save_fact_with_unknown_layer_writes_kinetic(tmp_path):
    be = SQLiteBackend(str(tmp_path / "m.db"))
    f = FactPassport(
        subject="a", predicate="b", object="c",
        fact_id="weird-layer-fact",
        vector=[0.1, 0.2, 0.3],
        layer=99,
    )
    be.save_fact(f)
    be._conn.commit()
    row = be._conn.execute(
        "SELECT layer FROM facts WHERE fact_id = 'weird-layer-fact'",
    ).fetchone()
    assert row["layer"] == 1
    be.close()


def test_save_meta_fact_with_nan_gravity_writes_default(tmp_path):
    be = SQLiteBackend(str(tmp_path / "m.db"))
    m = MetaFact(
        meta_id="bad-meta",
        vector=[0.1, 0.2, 0.3],
        gravity_score=float("nan"),
    )
    be.save_meta_fact(m)
    be._conn.commit()
    row = be._conn.execute(
        "SELECT gravity_score FROM meta_facts WHERE meta_id = 'bad-meta'",
    ).fetchone()
    g = row["gravity_score"]
    assert math.isfinite(g)
    assert g == 0.30
    be.close()


def test_save_echo_session_with_nan_score_writes_default(tmp_path):
    be = SQLiteBackend(str(tmp_path / "m.db"))
    be.save_echo_session(
        session_id="s-clean",
        centroids=[[0.1, 0.2]],
        r_score=float("nan"),
        recorded_at=float("inf"),
        fact_weights={"f1": 0.5},
        echo_penalty=float("nan"),
    )
    be._conn.commit()
    row = be._conn.execute(
        "SELECT r_score, recorded_at, echo_penalty FROM echo_sessions "
        "WHERE session_id = 's-clean'",
    ).fetchone()
    assert math.isfinite(row["r_score"])
    assert math.isfinite(row["recorded_at"])
    assert math.isfinite(row["echo_penalty"])
    be.close()


def test_roundtrip_through_storage_no_poison_on_disk(tmp_path):
    """End-to-end: poison a runtime body, save it, then read raw bytes
    from disk to confirm NaN never landed there."""
    be = SQLiteBackend(str(tmp_path / "m.db"))
    f = FactPassport(
        subject="x", predicate="y", object="z",
        fact_id="rt",
        vector=[0.5, 0.6],
        gravity_score=float("nan"),
        recent_utility=float("inf"),
        forecast_stability=float("-inf"),
    )
    be.save_fact(f)
    be._conn.commit()
    be.close()

    # Open the raw DB and assert no NaN tokens in the literal cell
    # values.
    conn = sqlite3.connect(str(tmp_path / "m.db"))
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT gravity_score, recent_utility, forecast_stability, "
        "vector FROM facts WHERE fact_id = 'rt'",
    ).fetchone()
    assert math.isfinite(row["gravity_score"])
    assert math.isfinite(row["recent_utility"])
    assert math.isfinite(row["forecast_stability"])
    # Vector cell stays clean too: a finite vector survives.
    parsed = json.loads(row["vector"])
    assert all(math.isfinite(x) for x in parsed)
    conn.close()
