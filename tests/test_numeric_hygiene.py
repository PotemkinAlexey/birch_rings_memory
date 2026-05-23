"""NaN / Infinity rejection across every vector boundary, plus
centroid / dispersion mixed-dim contract.

NaN and Infinity pass ``float()`` cleanly but poison every downstream
cosine: similarity becomes NaN, every comparison returns False
silently, sort order is undefined. Three boundaries needed the
defence:

  1. _validate_vector (HTTP embedding response)
  2. _safe_vector (SQLite on-disk vector)
  3. load_open_sessions vectors cell

Plus the in-memory utilities centroid / dispersion now raise loudly
on mixed-dimension input instead of silently truncating or
crashing deep in the stack — a direct caller (test, embedded mode,
in-memory migration) that bypasses the storage loaders still gets
a clear contract violation.

Plus belt-and-suspenders: Thresholds env reader explicitly rejects
NaN even though the [0, 1] range check would catch it (NaN
comparisons all return False) — clearer intent + faster reject.
"""
from __future__ import annotations

import importlib
import math
import os
import sqlite3
import sys

import pytest

from birch.resonance.centroid import centroid, dispersion
from birch.resonance.embeddings import EmbeddingError, _validate_vector
from birch.storage.sqlite import SQLiteBackend, _safe_vector

# --- I1: _validate_vector rejects NaN / inf ----------------------------


def test_validate_vector_rejects_nan():
    with pytest.raises(EmbeddingError, match="NaN or Infinity"):
        _validate_vector([1.0, float("nan"), 3.0], "test")


def test_validate_vector_rejects_pos_inf():
    with pytest.raises(EmbeddingError, match="NaN or Infinity"):
        _validate_vector([1.0, float("inf"), 3.0], "test")


def test_validate_vector_rejects_neg_inf():
    with pytest.raises(EmbeddingError, match="NaN or Infinity"):
        _validate_vector([1.0, float("-inf"), 3.0], "test")


def test_validate_vector_still_accepts_finite_floats():
    assert _validate_vector([0.0, -1.5, 3.14], "test") == [0.0, -1.5, 3.14]


# --- I2: _safe_vector returns [] on NaN / inf -------------------------


def test_safe_vector_returns_empty_on_nan():
    # JSON doesn't natively support NaN, so simulate via Python list.
    assert _safe_vector([1.0, float("nan"), 3.0]) == []


def test_safe_vector_returns_empty_on_inf():
    assert _safe_vector([1.0, float("inf"), 3.0]) == []


def test_safe_vector_accepts_finite_list():
    assert _safe_vector("[1.0, 2.0, 3.0]") == [1.0, 2.0, 3.0]


# --- I3: load_open_sessions rejects vectors with NaN / inf ------------


def test_load_open_sessions_drops_row_with_nan_vector(tmp_path):
    """A vectors cell containing NaN must drop the row at the loader
    instead of poisoning compute_resonance downstream. JSON doesn't
    natively encode NaN but JavaScript-style NaN/Infinity tokens
    slip in via permissive json.loads on some platforms — defence
    is at the float-coercion layer."""
    db = str(tmp_path / "m.db")
    backend = SQLiteBackend(db)
    backend.save_open_session(
        "ok", ["msg"], [[0.1, 0.2, 0.3]], {}, started_at=0.0,
    )
    backend.close()

    # Inject a row whose vectors cell decodes to a list-with-NaN via
    # the json5-ish encoding ``NaN`` literal. Python's json module
    # actually accepts NaN, Infinity, -Infinity by default.
    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT OR REPLACE INTO open_sessions "
        "(session_id, messages, vectors, facts, started_at) "
        "VALUES (?, ?, ?, ?, ?)",
        ("poison", "[\"x\"]", "[[0.1, NaN, 0.3]]", "{}", 0.0),
    )
    conn.commit()
    conn.close()

    backend2 = SQLiteBackend(db)
    sessions = backend2.load_open_sessions()
    backend2.close()
    ids = {s["session_id"] for s in sessions}
    assert "ok" in ids
    assert "poison" not in ids


def test_load_open_sessions_drops_row_with_inf_vector(tmp_path):
    db = str(tmp_path / "m.db")
    backend = SQLiteBackend(db)
    backend.close()

    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT OR REPLACE INTO open_sessions "
        "(session_id, messages, vectors, facts, started_at) "
        "VALUES (?, ?, ?, ?, ?)",
        ("infinite", "[\"x\"]", "[[0.1, Infinity, 0.3]]", "{}", 0.0),
    )
    conn.commit()
    conn.close()

    backend2 = SQLiteBackend(db)
    sessions = backend2.load_open_sessions()
    backend2.close()
    assert sessions == []


# --- I4: centroid / dispersion raise on mixed dims --------------------


def test_centroid_raises_on_mixed_dims():
    with pytest.raises(ValueError, match="mixed vector dimensions"):
        centroid([[1.0, 2.0, 3.0], [4.0, 5.0]])


def test_centroid_raises_on_empty_vector():
    with pytest.raises(ValueError, match="empty vector"):
        centroid([[]])


def test_centroid_works_on_clean_input():
    assert centroid([[1.0, 2.0], [3.0, 4.0]]) == [2.0, 3.0]


def test_dispersion_raises_on_mixed_dims():
    with pytest.raises(ValueError, match="mixed vector dimensions"):
        dispersion([[1.0, 2.0, 3.0], [4.0, 5.0]], [1.0, 1.0, 1.0])


def test_dispersion_raises_on_center_dim_mismatch():
    with pytest.raises(ValueError, match="center dim"):
        dispersion([[1.0, 2.0], [3.0, 4.0]], [1.0, 1.0, 1.0])


def test_dispersion_works_on_clean_input():
    # Two identical vectors → dispersion 0.
    d = dispersion([[1.0, 0.0], [1.0, 0.0]], [1.0, 0.0])
    assert d == 0.0


# --- I5: Thresholds reject NaN explicitly ------------------------------


def _reload_thresholds(env: dict[str, str]):
    saved = {k: os.environ.get(k) for k in env}
    for k, v in env.items():
        os.environ[k] = v
    try:
        sys.modules.pop("birch.thresholds", None)
        return importlib.import_module("birch.thresholds").Thresholds
    finally:
        for k, original in saved.items():
            if original is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = original
        sys.modules.pop("birch.thresholds", None)
        importlib.import_module("birch.thresholds")


def test_threshold_nan_falls_back_to_default():
    fresh = _reload_thresholds({"BIRCH_HAWKING_FACT_THRESHOLD": "nan"})
    assert fresh.HAWKING_FACT == 0.95
    assert math.isfinite(fresh.HAWKING_FACT)


def test_threshold_inf_falls_back_to_default():
    fresh = _reload_thresholds({"BIRCH_ECHO_THRESHOLD": "inf"})
    assert fresh.ECHO == 0.68
    assert math.isfinite(fresh.ECHO)


def test_threshold_neg_inf_falls_back_to_default():
    fresh = _reload_thresholds(
        {"BIRCH_ABSORPTION_THRESHOLD": "-inf"},
    )
    assert fresh.ABSORPTION == 0.10
    assert math.isfinite(fresh.ABSORPTION)
