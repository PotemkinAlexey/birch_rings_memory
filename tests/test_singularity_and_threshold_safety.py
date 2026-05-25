"""Singularity rehydration + threshold-range + open-session vector
shape regressions.

Five state-integrity contracts surfaced together:

  1. BlackHole.restore_fact / restore_meta tolerate mixed-dim
     singularity rows on startup — body lives without an index entry
     until reindex, doesn't crash boot.
  2. load_open_sessions validates vectors are list[list[float]] with
     consistent dim before passing to compute_resonance.
  3. Thresholds env values out of [0, 1] silently fall back to
     defaults so an operator typo can't poison the formula.
  4. MetaFact._load_list is tolerant on item-level cast errors —
     symmetric with FactPassport's _safe_vector contract.
  5. memory_stats carries thresholds_are_import_time flag so a
     caller doesn't assume hot-reload.
"""
from __future__ import annotations

import importlib
import logging
import os
import sqlite3
import sys

from birch.fact import FactPassport
from birch.memory_store import MemoryStore
from birch.meta_fact import MetaFact, _load_list
from birch.storage.sqlite import SQLiteBackend

# --- I1: BlackHole rehydration with mixed-dim singularity rows --------


def test_singularity_load_tolerates_mixed_dim_fact(tmp_path, caplog):
    """Absorbed fact with a different vector dim than the rest of the
    singularity must not crash boot. Body keeps its singularity record
    (visible via stats and dim-partitioned collapse) but is absent
    from the singularity vector index."""
    db = str(tmp_path / "m.db")
    mem = MemoryStore(db_path=db)
    # Seed one absorbed body at the active dim.
    seed_fact = FactPassport(
        subject="seed", predicate="is", object="absorbed",
    )
    seed_fact.vector = [0.1] * 8
    seed_fact.gravity_score = 0.05
    mem._facts[seed_fact.fact_id] = seed_fact
    mem._engine.register(seed_fact)
    mem._index.add(seed_fact.fact_id, seed_fact.vector)
    mem._storage.save_fact(seed_fact)
    mem._absorb_dead()

    # Inject a second absorbed body with a different dim straight to
    # storage.
    rogue = FactPassport(subject="rogue", predicate="is", object="absorbed")
    rogue.vector = [0.2] * 12  # different dim
    rogue.gravity_score = 0.05
    rogue.layer = -1
    mem._storage.save_fact(rogue)
    mem.close()

    with caplog.at_level(logging.WARNING):
        again = MemoryStore(db_path=db)
    # Both bodies show up in singularity records, regardless of dim.
    in_hole = {
        rec.fact.fact_id for rec in again._hole._singularity.values()
    }
    assert seed_fact.fact_id in in_hole
    assert rogue.fact_id in in_hole
    # Per-dim refactor: rogue's vector is RETAINED and lives in its
    # own dim bucket — no clear, no warning. Cross-dim singularity is
    # now the supported case, not a tolerated corruption.
    rogue_rehydrated = again._hole._singularity[rogue.fact_id].fact
    assert rogue_rehydrated.vector == [0.2] * 12
    # Two dim buckets coexist in the singularity.
    assert sorted(again._hole.fact_dims) == [8, 12]
    again.close()
    # No "singularity dim mismatch" warning — the per-dim refactor
    # eliminated the warning's cause.
    assert not any(
        "singularity dim mismatch" in rec.message
        for rec in caplog.records
    )


# --- I2: load_open_sessions vector numeric + same-dim validation -----


def test_load_open_sessions_drops_row_with_ragged_vectors(tmp_path):
    """A vectors cell that parses to list[list[?]] but rows have
    different lengths used to crash session_close downstream
    (centroid takes dim from vectors[0]). Drop the row at the loader."""
    db = str(tmp_path / "m.db")
    backend = SQLiteBackend(db)
    backend.save_open_session(
        "ok", ["msg"], [[0.1, 0.2, 0.3]], {}, started_at=0.0,
    )
    backend.close()

    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT OR REPLACE INTO open_sessions "
        "(session_id, messages, vectors, facts, started_at) "
        "VALUES (?, ?, ?, ?, ?)",
        # vectors is valid JSON list[list] but ragged.
        ("ragged", "[\"x\"]", "[[0.1, 0.2], [0.4]]", "{}", 0.0),
    )
    conn.commit()
    conn.close()

    backend2 = SQLiteBackend(db)
    sessions = backend2.load_open_sessions()
    backend2.close()
    ids = {s["session_id"] for s in sessions}
    assert "ok" in ids
    assert "ragged" not in ids


def test_load_open_sessions_drops_row_with_non_numeric_vector(tmp_path):
    db = str(tmp_path / "m.db")
    backend = SQLiteBackend(db)
    backend.close()

    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT OR REPLACE INTO open_sessions "
        "(session_id, messages, vectors, facts, started_at) "
        "VALUES (?, ?, ?, ?, ?)",
        ("strings", "[\"x\"]", "[[\"oops\", \"bad\"]]", "{}", 0.0),
    )
    conn.commit()
    conn.close()

    backend2 = SQLiteBackend(db)
    sessions = backend2.load_open_sessions()
    backend2.close()
    assert sessions == []


def test_load_open_sessions_drops_row_with_non_list_vector(tmp_path):
    db = str(tmp_path / "m.db")
    backend = SQLiteBackend(db)
    backend.close()

    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT OR REPLACE INTO open_sessions "
        "(session_id, messages, vectors, facts, started_at) "
        "VALUES (?, ?, ?, ?, ?)",
        ("scalar-vec", "[\"x\"]", "[42]", "{}", 0.0),
    )
    conn.commit()
    conn.close()

    backend2 = SQLiteBackend(db)
    sessions = backend2.load_open_sessions()
    backend2.close()
    assert sessions == []


# --- I3: Thresholds env clamps out-of-range to default ----------------


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


def test_threshold_negative_value_falls_back_to_default():
    fresh = _reload_thresholds(
        {"BIRCH_HAWKING_FACT_THRESHOLD": "-999"},
    )
    assert fresh.HAWKING_FACT == 0.95  # default, NOT -999


def test_threshold_above_one_falls_back_to_default():
    fresh = _reload_thresholds(
        {"BIRCH_COLLAPSE_THRESHOLD": "2.0"},
    )
    assert fresh.COLLAPSE == 0.92  # default, NOT 2.0


def test_threshold_in_range_takes_effect():
    fresh = _reload_thresholds(
        {"BIRCH_ECHO_THRESHOLD": "0.55"},
    )
    assert fresh.ECHO == 0.55


def test_threshold_exact_boundary_zero_accepted():
    fresh = _reload_thresholds(
        {"BIRCH_ABSORPTION_THRESHOLD": "0.0"},
    )
    assert fresh.ABSORPTION == 0.0


def test_threshold_exact_boundary_one_accepted():
    fresh = _reload_thresholds(
        {"BIRCH_HAWKING_FACT_THRESHOLD": "1.0"},
    )
    assert fresh.HAWKING_FACT == 1.0


# --- I4: MetaFact _load_list tolerant on item-level cast errors -------


def test_load_list_returns_empty_on_non_castable_items():
    """A JSON list with a non-castable element returns [] instead of
    raising. Symmetric with FactPassport's _safe_vector contract."""
    assert _load_list("[1, \"oops\", 3]", float) == []
    assert _load_list([1, "oops", 3], float) == []


def test_load_list_returns_clean_on_valid_items():
    assert _load_list("[1, 2, 3]", float) == [1.0, 2.0, 3.0]
    assert _load_list(["a", "b"], str) == ["a", "b"]


def test_metafact_from_dict_tolerant_on_bad_vector_items():
    """MetaFact.from_dict no longer raises on a list-with-bad-item;
    the field comes back empty so the body keeps its identity but
    can't be used in semantic search."""
    row = {
        "meta_id": "m1",
        "vector": "[1.0, \"oops\", 3.0]",
        "weight": 1,
        "source_texts": "[\"x\"]",
        "source_fact_ids": "[\"a\"]",
        "summary": "",
        "gravity_score": 0.5,
        "created_at": 0.0,
        "layer": 1,
        "access_count": 0,
        "last_accessed": 0.0,
        "resonance_sum": 0.0,
        "resonance_count": 0,
        "recent_utility": 0.5,
        "forecast_stability": 0.5,
    }
    meta = MetaFact.from_dict(row)
    assert meta.meta_id == "m1"
    assert meta.vector == []  # tolerated, not raised


# --- I5: stats carries thresholds_are_import_time flag ---------------


def test_stats_carries_thresholds_are_import_time_flag(tmp_path):
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    try:
        stats = mem.stats
        assert stats.get("thresholds_are_import_time") is True
    finally:
        mem.close()
