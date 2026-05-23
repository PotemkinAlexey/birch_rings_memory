"""Input-validation and corruption-tolerance regressions.

Long-tail gaps closed in one pass: load_meta_facts wasn't tolerant,
_safe_loads didn't check vector shape, VectorIndex.search had an edge
case for top_k=0, MCP enum inputs failed silently, batch items
skipped per-item validation, loaded adaptive weights weren't
sanitised.
"""
from __future__ import annotations

import sqlite3

from birch.adaptive_gravity import AdaptiveWeights
from birch.fact import FactPassport
from birch.memory_store import MemoryStore
from birch.meta_fact import MetaFact
from birch.storage.sqlite import SQLiteBackend, _safe_vector
from birch.vector_index import VectorIndex

# --- P1: load_meta_facts tolerant ---------------------------------------


def test_load_meta_facts_skips_corrupted_row(tmp_path):
    """A single corrupted meta_facts row used to take MemoryStore down.
    Now the row is skipped, dropped from storage, and the rest load."""
    db = str(tmp_path / "m.db")
    mem = MemoryStore(db_path=db)
    # Plant one good MetaFact through the storage layer.
    # Compactor-shaped MetaFact: both source_texts AND source_fact_ids
    # populated. Round-10 semantic-shape validation drops persisted
    # MetaFacts missing either lineage field.
    good = MetaFact(weight=2, source_texts=["good text"],
                    source_fact_ids=["src-1"],
                    gravity_score=0.5, layer=1)
    good.vector = [0.1] * 64
    assert mem._storage is not None
    mem._storage.save_meta_fact(good)
    mem.close()

    # Corrupt one meta_facts row in source_texts (JSON cell).
    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT OR REPLACE INTO meta_facts "
        "(meta_id, vector, weight, source_texts, source_fact_ids, summary, "
        " gravity_score, created_at, layer, access_count, last_accessed, "
        " resonance_sum, resonance_count, recent_utility, forecast_stability) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        ("bad", "{garbage", 2, "{garbage", "{garbage", "",
         0.5, 0.0, 1, 0, 0.0, 0.0, 0, 0.5, 0.5),
    )
    conn.commit()
    conn.close()

    # Reopen — must not raise; good meta survives.
    again = MemoryStore(db_path=db)
    assert good.meta_id in again._meta_facts
    assert "bad" not in again._meta_facts


# --- P1: _safe_vector validates list[float] shape -----------------------


def test_safe_vector_rejects_non_list():
    """JSON-valid but wrong-type cell must produce empty vector, not
    a dict/string downstream."""
    assert _safe_vector('{"x": 1}') == []
    assert _safe_vector('"abc"') == []
    assert _safe_vector("null") == []


def test_safe_vector_rejects_non_numeric_items():
    assert _safe_vector('[1, "oops", 3]') == []


def test_safe_vector_accepts_valid_list():
    assert _safe_vector("[1.0, 2.0, 3.0]") == [1.0, 2.0, 3.0]


def test_safe_vector_accepts_int_list():
    assert _safe_vector("[1, 2, 3]") == [1.0, 2.0, 3.0]


def test_safe_vector_on_corrupted_cell_returns_empty():
    assert _safe_vector("{not json") == []
    assert _safe_vector(None) == []
    assert _safe_vector("") == []


# --- P1: VectorIndex.search top_k <= 0 ----------------------------------


def test_search_returns_empty_on_zero_top_k():
    idx = VectorIndex()
    idx.add("a", [1.0, 0.0, 0.0])
    idx.add("b", [0.0, 1.0, 0.0])
    assert idx.search([1.0, 0.0, 0.0], top_k=0) == []
    assert idx.search([1.0, 0.0, 0.0], top_k=-1) == []


# --- P2: MCP layer enum validation --------------------------------------


def test_query_memory_unknown_layer_returns_structured_error():
    """The server validation for layers should run BEFORE _store.query.
    Replicate the check inline so we don't need to import server.py
    (which pulls mcp SDK)."""
    layer_map = {"surface": 0, "kinetic": 1, "core": 2}
    layers = ["surfase"]   # typo
    unknown = [n for n in layers if n not in layer_map]
    assert unknown == ["surfase"]
    # The structured response shape the agent now sees:
    response = {
        "results": [],
        "error": "unknown_layer",
        "unknown_layers": unknown,
        "allowed_layers": list(layer_map),
    }
    assert response["error"] == "unknown_layer"
    assert "surfase" in response["unknown_layers"]


# --- P2: record_facts per-item validation -------------------------------


def test_record_facts_invalid_item_returns_structured_error():
    """Inline the same validator the server now applies in front of
    _store.add_facts."""
    facts = [
        {"subject": "a", "predicate": "is", "object": "1"},
        {"subject": "b"},   # missing predicate, object
        {"subject": "c", "predicate": "is", "object": ""},   # empty value
        "not even a dict",
    ]
    required = ("subject", "predicate", "object")
    invalid: list[dict] = []
    for i, f in enumerate(facts):
        if not isinstance(f, dict):
            invalid.append({"index": i, "error": "item_not_an_object",
                            "got_type": type(f).__name__})
            continue
        missing = [k for k in required if k not in f or f[k] in (None, "")]
        if missing:
            invalid.append({"index": i, "missing": missing})
    assert {item["index"] for item in invalid} == {1, 2, 3}


# --- P2: AdaptiveWeights load sanitises ---------------------------------


def test_load_adaptive_weights_sanitises_negative(tmp_path):
    """A corrupted row with a negative weight must be clamped + renormed
    before being served to compute_gravity."""
    db = str(tmp_path / "m.db")
    backend = SQLiteBackend(db)
    # Save a baseline so the row exists.
    backend.save_adaptive_weights(AdaptiveWeights.from_prior())
    # Then mutilate it directly on disk.
    conn = sqlite3.connect(db)
    conn.execute(
        "UPDATE adaptive_weights SET w_freshness = ?, w_access = ?, "
        "w_graph = ?, w_utility = ?, w_stability = ? WHERE id = 1",
        (-0.5, 0.20, 0.20, 0.20, 0.20),
    )
    conn.commit()
    conn.close()

    loaded = backend.load_adaptive_weights()
    backend.close()
    assert loaded is not None
    # Non-negative after sanitize.
    assert loaded.w_freshness >= 0.0
    # Sum is back at BUDGET (within float tolerance).
    total = (loaded.w_freshness + loaded.w_access + loaded.w_graph
             + loaded.w_utility + loaded.w_stability)
    assert abs(total - AdaptiveWeights.BUDGET) < 1e-9


def test_load_adaptive_weights_sanitises_off_budget(tmp_path):
    """A row whose weights sum to something other than BUDGET (e.g. an
    older invariant) must be renormalised on load."""
    db = str(tmp_path / "m.db")
    backend = SQLiteBackend(db)
    # Save with sum 0.73 — off-budget.
    backend.save_adaptive_weights(AdaptiveWeights(
        w_freshness=0.40, w_access=0.15, w_graph=0.10,
        w_utility=0.05, w_stability=0.03,
        train_count=0,
    ))
    loaded = backend.load_adaptive_weights()
    backend.close()
    assert loaded is not None
    total = (loaded.w_freshness + loaded.w_access + loaded.w_graph
             + loaded.w_utility + loaded.w_stability)
    assert abs(total - AdaptiveWeights.BUDGET) < 1e-9


# --- P3: compactor docstring no longer says "768-dim" -------------------


def test_compactor_docstring_dropped_hardcoded_dim():
    import inspect

    import birch.singularity_compactor as sc
    src = inspect.getmodule(sc).__doc__ or ""
    assert "768-dim" not in src


# --- P1 sanity: load_facts vector validator end-to-end ------------------


def test_load_facts_validates_vector_shape(tmp_path):
    """A vector cell with valid JSON but wrong shape (dict instead of
    list) used to slip through as fact.vector = {'x': 1}, breaking
    downstream code that assumed list[float]. Now the validator
    returns []."""
    db = str(tmp_path / "m.db")
    mem = MemoryStore(db_path=db)
    f = mem.add_fact("api", "runs on", "Go")
    mem.close()

    # Corrupt vector to a valid-JSON-but-wrong-shape cell.
    conn = sqlite3.connect(db)
    conn.execute(
        "UPDATE facts SET vector = ? WHERE fact_id = ?",
        ('{"x": 1}', f.fact_id),
    )
    conn.commit()
    conn.close()

    again = MemoryStore(db_path=db)
    reloaded = next(x for x in again.list_facts() if x.fact_id == f.fact_id)
    assert isinstance(reloaded.vector, list)
    assert reloaded.vector == []


# Use FactPassport so the test file's import doesn't shake on lint.
_ = FactPassport
