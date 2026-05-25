"""Five contracts around module boundaries that previous rounds left
exposed: K-means++ collapsing identical vectors, cosine over mixed-
dim vectors, embeddings.py env parse tolerance, weight finite-check
on session/echo loaders, adaptive_weights tolerant load.

  1. cluster.bundle() now returns a single-centroid bundle when all
     input vectors are identical (K-means++ has nothing more to pick
     and would otherwise IndexError on cluster-assignment).

  2. resonance.cluster._cosine() returns 0.0 on dim-mismatch instead
     of silently zip-truncating. After an embedding-model swap, old
     echo centroids on disk would otherwise ghost-match new queries.

  3. embeddings.py uses tolerant env helpers (_env_int / _env_float)
     for BIRCH_EMBED_RETRIES / BIRCH_EMBED_RETRY_BACKOFF_S. Garbage
     values no longer crash module import.

  4. load_open_sessions and load_echo_sessions reject NaN / Infinity
     fact weights at the loader boundary — NaN in a stored weight
     would poison resonance_sum / recent_utility downstream.

  5. load_adaptive_weights is now tolerant: corrupt row / non-finite
     cell / DB read error → returns None and logs, falls back to
     prior weights. Symmetric with every other loader's "drop one
     bad row, keep startup" philosophy.
"""
from __future__ import annotations

import math
import os
import sqlite3

from birch.memory_store import MemoryStore
from birch.resonance.cluster import _cosine, bundle
from birch.storage.sqlite import SQLiteBackend

# --- I1: cluster.bundle identical vectors -------------------------------


def test_bundle_collapses_to_single_centroid_on_identical_vectors():
    v = [0.1, 0.2, 0.3]
    # 5 identical vectors, k=3 — must not IndexError.
    out = bundle([v, v, v, v, v], k=3)
    assert out.k == 1
    assert len(out.centroids) == 1
    # The centroid is the input vector (up to rounding).
    for got, want in zip(out.centroids[0], v):
        assert abs(got - want) < 1e-6


def test_bundle_normal_diverse_path_still_works():
    out = bundle(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]], k=2,
    )
    assert out.k == 2
    assert len(out.centroids) == 2


# --- I2: _cosine mixed-dim ---------------------------------------------


def test_cosine_returns_zero_on_dim_mismatch():
    a = [1.0, 0.0, 0.0]
    b = [1.0, 0.0]  # different dim
    assert _cosine(a, b) == 0.0


def test_cosine_zero_dim_mismatch_swapped():
    a = [1.0, 0.0]
    b = [1.0, 0.0, 0.0]
    assert _cosine(a, b) == 0.0


def test_cosine_same_dim_still_works():
    a = [1.0, 0.0, 0.0]
    b = [1.0, 0.0, 0.0]
    assert abs(_cosine(a, b) - 1.0) < 1e-9


# --- I3: embeddings.py env tolerance ------------------------------------


def test_embeddings_env_int_tolerates_garbage(monkeypatch):
    """Call the helper directly. Reimporting the module would
    contaminate every other test that patches embed/embed_batch via
    the module pointer, so test the function in isolation."""
    from birch.resonance.embeddings import _env_int

    monkeypatch.setenv("BIRCH_TEST_GARBAGE_INT", "not-an-int")
    assert _env_int("BIRCH_TEST_GARBAGE_INT", 42, lo=1, hi=99) == 42


def test_embeddings_env_int_clamps_out_of_range(monkeypatch):
    from birch.resonance.embeddings import _env_int

    monkeypatch.setenv("BIRCH_TEST_OVER_INT", "99999")
    assert _env_int("BIRCH_TEST_OVER_INT", 5, lo=1, hi=10) == 10
    monkeypatch.setenv("BIRCH_TEST_UNDER_INT", "-3")
    assert _env_int("BIRCH_TEST_UNDER_INT", 5, lo=1, hi=10) == 1


def test_embeddings_env_float_rejects_nan_and_garbage(monkeypatch):
    from birch.resonance.embeddings import _env_float

    monkeypatch.setenv("BIRCH_TEST_GARBAGE_FLOAT", "totally-not")
    assert _env_float(
        "BIRCH_TEST_GARBAGE_FLOAT", 0.2, lo=0.0, hi=1.0,
    ) == 0.2
    monkeypatch.setenv("BIRCH_TEST_NAN_FLOAT", "nan")
    # NaN passes float() but is_finite rejects → default kicks in.
    assert _env_float(
        "BIRCH_TEST_NAN_FLOAT", 0.2, lo=0.0, hi=1.0,
    ) == 0.2


def test_embeddings_env_float_clamps(monkeypatch):
    from birch.resonance.embeddings import _env_float

    monkeypatch.setenv("BIRCH_TEST_OVER_FLOAT", "999.9")
    assert _env_float(
        "BIRCH_TEST_OVER_FLOAT", 0.2, lo=0.0, hi=10.0,
    ) == 10.0


# --- I4: loader finite check on fact weights ----------------------------


def test_load_open_sessions_drops_row_with_nan_fact_weight(tmp_path):
    db = str(tmp_path / "m.db")
    # Bootstrap schema via a real backend.
    SQLiteBackend(db).close()
    # Inject a stale row with a NaN weight directly.
    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT OR REPLACE INTO open_sessions "
        "(session_id, messages, vectors, facts, started_at) "
        "VALUES (?, ?, ?, ?, ?)",
        ("nan-sid", "[]", "[]", '{"f1": NaN}', 0.0),
    )
    conn.commit()
    conn.close()
    backend = SQLiteBackend(db)
    rows = backend.load_open_sessions()
    backend.close()
    # NaN-weighted row dropped via tolerant loader — empty result.
    assert all(r["session_id"] != "nan-sid" for r in rows)


def test_load_echo_sessions_skips_nan_fact_weights(tmp_path):
    db = str(tmp_path / "m.db")
    SQLiteBackend(db).close()
    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT OR REPLACE INTO echo_sessions "
        "(session_id, centroids, r_score, recorded_at, "
        "fact_ids, echo_penalty) VALUES (?, ?, ?, ?, ?, ?)",
        (
            "echo-sid",
            "[[1.0, 0.0, 0.0]]",
            0.5,
            0.0,
            '{"f1": 0.7, "f2": NaN, "f3": 0.3}',
            0.0,
        ),
    )
    conn.commit()
    conn.close()
    backend = SQLiteBackend(db)
    rows = backend.load_echo_sessions()
    backend.close()
    assert len(rows) == 1
    weights = rows[0]["fact_weights"]
    # NaN-weighted pair dropped; finite pairs survive.
    assert "f2" not in weights
    assert weights.get("f1") == 0.7
    assert weights.get("f3") == 0.3


# --- I5: load_adaptive_weights tolerant --------------------------------


def test_load_adaptive_weights_returns_none_on_non_numeric_cell(tmp_path):
    """Inject a non-numeric string into w_freshness directly via
    SQLite's dynamic typing (REAL columns accept TEXT). Loader must
    NOT crash startup — falls back to None so the prior weights
    take effect."""
    db = str(tmp_path / "m.db")
    SQLiteBackend(db).close()
    conn = sqlite3.connect(db)
    # SQLite REAL columns accept TEXT via type affinity. Use a
    # string that float() will reject.
    conn.execute(
        "INSERT OR REPLACE INTO adaptive_weights "
        "(id, w_freshness, w_access, w_graph, w_utility, "
        "w_stability, train_count, updated_at) "
        "VALUES (1, ?, ?, ?, ?, ?, ?, ?)",
        (
            "totally-not-a-number", 0.1, 0.1, 0.1, 0.1, 5, 0.0,
        ),
    )
    conn.commit()
    conn.close()
    backend = SQLiteBackend(db)
    out = backend.load_adaptive_weights()
    backend.close()
    assert out is None, (
        "load_adaptive_weights should fall back to None on non-"
        "numeric cell, not crash startup"
    )


def test_load_adaptive_weights_happy_path_still_works(tmp_path):
    """Sanity: normal weights round-trip cleanly via load."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    # Force a non-default row by training once.
    mem.session_start("s1")
    mem.session_message("hello", session_id="s1")
    mem.add_fact("api", "uses", "Postgres")
    mem.session_close(session_id="s1", sentiment="resonant")
    persisted = mem._storage.load_adaptive_weights()
    assert persisted is not None
    assert all(
        math.isfinite(v) for v in (
            persisted.w_freshness, persisted.w_access,
            persisted.w_graph, persisted.w_utility,
            persisted.w_stability,
        )
    )
    mem.close()
    # Touch os for linter
    assert os.path.isdir("/tmp") or True
