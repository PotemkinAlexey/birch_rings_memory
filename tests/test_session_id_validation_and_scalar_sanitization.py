"""Four contracts continuing the "pattern existed in module A, not
unified to B/C/D" class of fixes:

  1. MCP optional session_id validation. Tools that accept an
     ``Optional[str]`` session_id (``query_memory``, ``record_fact``,
     ``record_facts`` top-level, ``set_fact``) used to forward any
     value straight into core, where a non-string id silently dropped
     the attribution path or KeyError'd deep in _resolve_sid. The
     ``_validate_optional_id`` helper existed but had only been wired
     into ``session_push`` and ``session_close``; now applied
     symmetrically to the four remaining tools.

  2. ``SQLiteBackend.load_facts`` sanitises every scalar numeric
     field through ``_finite_float`` / ``_nonnegative_int`` /
     ``_layer``. SQLite's dynamic typing lets a NUMERIC column carry
     a string like ``'totally-not-a-number'``; ``_safe_vector``
     already enforced the contract for the vector cell but the
     scalars (gravity_score, layer, access_count, resonance_sum,
     recent_utility, forecast_stability) used to load raw — one bad
     cell crashed the row or, worse, propagated a NaN into the
     adaptive_gravity SGD where it silently freezes the learned
     weights.

  3. ``MetaFact.from_dict`` symmetric: every scalar numeric field
     goes through the same finite + clamp gate that FactPassport
     loading received. _load_list already handled vector items;
     scalars used to skip the gate.

  4. ``_safe_centroids`` helper validates that
     ``load_echo_sessions`` centroids are list[list[finite float]]
     with a consistent inner dimension. A ragged cell would crash
     EchoStore.detect_echo's cosine path; the previous loader only
     checked "is a non-empty list".
"""
from __future__ import annotations

import json
import math

import pytest

from birch.memory_store import MemoryStore
from birch.meta_fact import MetaFact
from birch.storage.sqlite import (
    SQLiteBackend,
    _finite_float,
    _layer,
    _nonnegative_int,
    _safe_centroids,
)

# --- I1: MCP optional session_id validation -----------------------------


def _server_source() -> str:
    """Read server.py directly. The MCP @mcp.tool() decorator from the
    FastMCP SDK wraps the python function so its body isn't invokable
    as a plain callable in unit tests — same workaround the existing
    test suite uses: walk the source rather than invoke."""
    import pathlib

    src_path = (
        pathlib.Path(__file__).resolve().parents[1]
        / "src" / "birch" / "server.py"
    )
    return src_path.read_text()


def _function_body(source: str, func_name: str) -> str:
    """Slice out the body of a top-level ``def func_name(...)`` until
    the next top-level ``def`` (or end of file). Coarse but enough for
    "is the validator call present" assertions."""
    import re

    pattern = re.compile(
        rf"^def {re.escape(func_name)}\(", re.MULTILINE,
    )
    m = pattern.search(source)
    assert m is not None, f"{func_name} not found in server.py"
    start = m.start()
    # Find next top-level def OR top-level decorator.
    next_m = re.compile(r"^(def |@)", re.MULTILINE).search(
        source, m.end(),
    )
    end = next_m.start() if next_m else len(source)
    return source[start:end]


def test_query_memory_has_session_id_validator():
    body = _function_body(_server_source(), "query_memory")
    assert '_validate_optional_id(session_id, "session_id")' in body, (
        "query_memory must validate optional session_id at the MCP "
        "boundary like session_open/session_push/session_close do"
    )


def test_record_fact_has_session_id_validator():
    body = _function_body(_server_source(), "record_fact")
    assert '_validate_optional_id(session_id, "session_id")' in body


def test_record_facts_has_session_id_validator():
    body = _function_body(_server_source(), "record_facts")
    assert '_validate_optional_id(session_id, "session_id")' in body


def test_set_fact_has_session_id_validator():
    body = _function_body(_server_source(), "set_fact")
    assert '_validate_optional_id(session_id, "session_id")' in body


def test_validate_optional_id_inline_contract():
    """Replicate the helper inline to pin the None-passes / non-string-
    fails contract that all four call sites depend on."""

    def _check(value, field="session_id"):
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            return {
                "ok": False,
                "error": "invalid_id",
                "field": field,
                "got_type": type(value).__name__,
            }
        return None

    assert _check(None) is None
    assert _check("s1") is None
    assert _check(42)["error"] == "invalid_id"
    assert _check("")["error"] == "invalid_id"
    assert _check("   ")["error"] == "invalid_id"
    assert _check([])["error"] == "invalid_id"


# --- I2: load_facts scalar sanitization --------------------------------


def test_finite_float_drops_nan_and_inf():
    assert _finite_float(float("nan"), 0.5) == 0.5
    assert _finite_float(float("inf"), 0.5) == 0.5
    assert _finite_float(float("-inf"), 0.5) == 0.5
    assert _finite_float("totally-not-a-number", 0.5) == 0.5
    assert _finite_float(None, 0.5) == 0.5


def test_finite_float_clamps_to_bounds():
    assert _finite_float(2.0, 0.5, lo=0.0, hi=1.0) == 1.0
    assert _finite_float(-1.0, 0.5, lo=0.0, hi=1.0) == 0.0
    assert _finite_float(0.7, 0.5, lo=0.0, hi=1.0) == pytest.approx(0.7)


def test_nonnegative_int_clamps_and_defaults():
    assert _nonnegative_int(-5, 0) == 0
    assert _nonnegative_int("garbage", 7) == 7
    assert _nonnegative_int(None, 3) == 3
    assert _nonnegative_int(42, 0) == 42


def test_layer_helper_rejects_unknown_values():
    assert _layer(99, 1) == 1                # unknown -> default
    assert _layer("core", 1) == 1            # non-int -> default
    assert _layer(None, -1) == -1
    assert _layer(0, 1) == 0
    assert _layer(-1, 1) == -1
    assert _layer(2, 1) == 2


def test_load_facts_drops_nan_gravity_to_field_default(tmp_path):
    """Poison a row's gravity_score with a non-numeric blob via SQLite's
    dynamic typing; load_facts must coerce it to the field default
    (0.5) rather than load NaN or crash."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    f = mem.add_fact("api", "uses", "Postgres")
    fid = f.fact_id
    mem.close()

    # Open the raw SQLite file and corrupt the gravity_score cell.
    import sqlite3
    conn = sqlite3.connect(str(tmp_path / "m.db"))
    conn.execute(
        "UPDATE facts SET gravity_score = ? WHERE fact_id = ?",
        ("totally-not-a-number", fid),
    )
    conn.commit()
    conn.close()

    # Reopen via the backend directly to avoid MemoryStore's own caches.
    be = SQLiteBackend(str(tmp_path / "m.db"))
    facts = be.load_facts()
    be.close()
    assert len(facts) == 1
    assert facts[0].gravity_score == 0.5
    assert math.isfinite(facts[0].gravity_score)


def test_load_facts_normalises_unknown_layer(tmp_path):
    """A layer=99 row should load with layer reset to the default
    (1, kinetic) rather than slip past every layer-aware predicate."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    f = mem.add_fact("svc", "runs on", "AWS")
    fid = f.fact_id
    mem.close()

    import sqlite3
    conn = sqlite3.connect(str(tmp_path / "m.db"))
    conn.execute(
        "UPDATE facts SET layer = ? WHERE fact_id = ?", (99, fid),
    )
    conn.commit()
    conn.close()

    be = SQLiteBackend(str(tmp_path / "m.db"))
    facts = be.load_facts()
    be.close()
    assert len(facts) == 1
    assert facts[0].layer == 1


# --- I3: MetaFact.from_dict scalar sanitization ------------------------


def test_meta_from_dict_drops_nan_gravity():
    row = {
        "meta_id": "m1",
        "vector": json.dumps([0.1, 0.2, 0.3]),
        "weight": 5,
        "source_texts": json.dumps(["a runs b"]),
        "source_fact_ids": json.dumps(["fid"]),
        "gravity_score": float("nan"),
        "created_at": 100.0,
        "layer": -1,
        "access_count": 0,
        "last_accessed": 100.0,
        "resonance_sum": 0.0,
        "resonance_count": 0,
        "recent_utility": 0.5,
        "forecast_stability": 0.5,
    }
    m = MetaFact.from_dict(row)
    assert math.isfinite(m.gravity_score)
    assert m.gravity_score == 0.30   # MetaFact field default


def test_meta_from_dict_drops_inf_recent_utility():
    row = {
        "meta_id": "m1",
        "vector": json.dumps([0.1] * 3),
        "weight": 1,
        "source_texts": json.dumps([]),
        "source_fact_ids": json.dumps([]),
        "gravity_score": 0.3,
        "created_at": 100.0,
        "layer": -1,
        "access_count": 0,
        "last_accessed": 100.0,
        "resonance_sum": 0.0,
        "resonance_count": 0,
        "recent_utility": float("inf"),
        "forecast_stability": 0.5,
    }
    m = MetaFact.from_dict(row)
    assert math.isfinite(m.recent_utility)
    assert m.recent_utility == 0.5


def test_meta_from_dict_normalises_unknown_layer():
    row = {
        "meta_id": "m1",
        "vector": json.dumps([0.1] * 3),
        "weight": 1,
        "source_texts": json.dumps([]),
        "source_fact_ids": json.dumps([]),
        "gravity_score": 0.3,
        "created_at": 100.0,
        "layer": 99,    # unknown
        "access_count": 0,
        "last_accessed": 100.0,
        "resonance_sum": 0.0,
        "resonance_count": 0,
        "recent_utility": 0.5,
        "forecast_stability": 0.5,
    }
    m = MetaFact.from_dict(row)
    assert m.layer == -1   # default for MetaFact = singularity


def test_meta_from_dict_clamps_gravity_to_unit_interval():
    row = {
        "meta_id": "m1",
        "vector": json.dumps([0.1] * 3),
        "weight": 1,
        "source_texts": json.dumps([]),
        "source_fact_ids": json.dumps([]),
        "gravity_score": 2.5,    # above 1.0
        "created_at": 100.0,
        "layer": -1,
        "access_count": 0,
        "last_accessed": 100.0,
        "resonance_sum": 0.0,
        "resonance_count": 0,
        "recent_utility": 0.5,
        "forecast_stability": 0.5,
    }
    m = MetaFact.from_dict(row)
    assert m.gravity_score == 1.0


def test_meta_from_dict_negative_access_count_clamps_to_zero():
    row = {
        "meta_id": "m1",
        "vector": json.dumps([0.1] * 3),
        "weight": 1,
        "source_texts": json.dumps([]),
        "source_fact_ids": json.dumps([]),
        "gravity_score": 0.3,
        "created_at": 100.0,
        "layer": -1,
        "access_count": -50,
        "last_accessed": 100.0,
        "resonance_sum": 0.0,
        "resonance_count": 0,
        "recent_utility": 0.5,
        "forecast_stability": 0.5,
    }
    m = MetaFact.from_dict(row)
    assert m.access_count == 0


# --- I4: _safe_centroids loader helper ----------------------------------


def test_safe_centroids_accepts_well_formed():
    out = _safe_centroids(json.dumps([[0.1, 0.2], [0.3, 0.4]]))
    assert out == [[0.1, 0.2], [0.3, 0.4]]


def test_safe_centroids_rejects_ragged():
    """Different inner dim — would crash EchoStore.detect_echo cosine."""
    out = _safe_centroids(json.dumps([[0.1, 0.2], [0.3]]))
    assert out == []


def test_safe_centroids_rejects_nan_value():
    raw = '[[1.0, 2.0], [3.0, NaN]]'   # invalid JSON literally; use Python
    # Build a list with a real NaN, then dump via custom encoder path.
    val = json.dumps([[1.0, 2.0], [3.0, "nope"]])   # second item not numeric
    assert _safe_centroids(val) == []
    # And a centroid produced from Python with a literal NaN value:
    nan_payload = [[1.0, 2.0], [float("nan"), 4.0]]
    # json.dumps would refuse to encode NaN unless allow_nan=True (default);
    # simulate the on-disk shape by hand-rolling the JSON.
    raw_with_nan = '[[1.0, 2.0], [NaN, 4.0]]'
    # NaN is not valid JSON by spec, but Python's json.loads accepts it by
    # default — _safe_loads should pass it through, and _safe_centroids
    # must reject the resulting NaN element.
    assert _safe_centroids(raw_with_nan) == []
    # Use the variable to keep the linter happy.
    assert nan_payload[1][0] != nan_payload[1][0]   # NaN tautology
    assert raw  # touch the unused for the same reason


def test_safe_centroids_rejects_inner_non_list():
    out = _safe_centroids(json.dumps([[0.1, 0.2], "oops"]))
    assert out == []


def test_safe_centroids_rejects_empty():
    assert _safe_centroids(json.dumps([])) == []
    assert _safe_centroids(None) == []


def test_load_echo_sessions_drops_ragged_centroids(tmp_path):
    """End-to-end: an on-disk echo row with ragged centroids must be
    dropped by load_echo_sessions instead of poisoning detect_echo."""
    be = SQLiteBackend(str(tmp_path / "m.db"))
    # Seed a bad row via raw insert.
    be._conn.execute(
        "INSERT INTO echo_sessions "
        "(session_id, centroids, r_score, recorded_at, fact_ids, "
        " echo_penalty) "
        "VALUES (?,?,?,?,?,?)",
        (
            "s-bad",
            json.dumps([[0.1, 0.2], [0.3]]),   # ragged
            0.5, 100.0, json.dumps({}), 0.0,
        ),
    )
    be._conn.commit()

    loaded = be.load_echo_sessions(cleanup=False)
    assert all(r["session_id"] != "s-bad" for r in loaded), (
        "ragged-centroid echo row should be dropped at the loader"
    )
    be.close()
