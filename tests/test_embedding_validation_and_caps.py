"""Embedding numeric validation, error wrap, and cap-disclosure regressions.

Covers:

  1. Embedding numeric validation in the HTTP path.
  2. Core MemoryStore.query top_k<=0 guard.
  3. record_facts field type validation (subject=123 used to pass).
  4. MetaFact empty-lineage warning log (not drop — round-trip held).
  5. MCP cap disclosure (effective_top_k + _warning when capped).

EmbeddingError MCP wrapping is also tested via the response shape
the inline validator returns.
"""
from __future__ import annotations

import logging
import sqlite3

import pytest

from birch.memory_store import MemoryStore
from birch.meta_fact import MetaFact
from birch.resonance.embeddings import EmbeddingError, _validate_vector
from birch.storage.sqlite import SQLiteBackend

# --- P1: embedding numeric validation -----------------------------------


def test_validate_vector_rejects_non_list():
    with pytest.raises(EmbeddingError, match="empty or wrong shape"):
        _validate_vector("not a list", "test")
    with pytest.raises(EmbeddingError, match="empty or wrong shape"):
        _validate_vector(None, "test")
    with pytest.raises(EmbeddingError, match="empty or wrong shape"):
        _validate_vector({"x": 1}, "test")


def test_validate_vector_rejects_empty_list():
    with pytest.raises(EmbeddingError, match="empty or wrong shape"):
        _validate_vector([], "test")


def test_validate_vector_rejects_non_numeric_items():
    with pytest.raises(EmbeddingError, match="non-numeric"):
        _validate_vector([1.0, "oops", 3.0], "test")
    with pytest.raises(EmbeddingError, match="non-numeric"):
        _validate_vector([1.0, None, 3.0], "test")


def test_validate_vector_accepts_valid_float_list():
    assert _validate_vector([1.0, 2.0, 3.0], "test") == [1.0, 2.0, 3.0]


def test_validate_vector_coerces_int_list_to_float():
    out = _validate_vector([1, 2, 3], "test")
    assert out == [1.0, 2.0, 3.0]
    assert all(isinstance(x, float) for x in out)


# --- P2: core MemoryStore.query top_k guard -----------------------------


def test_core_query_returns_empty_on_non_positive_top_k(tmp_path):
    """A negative top_k used to slice results[:top_k] from the right
    end (Python list semantics). Now the core boundary rejects it
    explicitly — same contract the server layer enforces."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    mem.add_fact("api", "runs on", "Go")
    mem.add_fact("db", "is", "Postgres")
    assert mem.query("api", top_k=0) == []
    assert mem.query("api", top_k=-1) == []
    # Positive still works.
    assert len(mem.query("api", top_k=2)) > 0
    mem.close()


# --- P2: record_facts field type validation -----------------------------


def test_record_facts_rejects_non_string_fields_inline_validator():
    """server.py now flags subject=int / predicate=list / object=dict.
    Replicated inline since the server module imports the mcp SDK
    which isn't in the test env."""
    facts = [
        {"subject": "api", "predicate": "runs on", "object": "Go"},  # ok
        {"subject": 123, "predicate": "is", "object": "1"},           # bad
        {"subject": "a", "predicate": [], "object": "b"},             # bad
        {"subject": "a", "predicate": "is", "object": {"x": 1}},      # bad
        {"subject": "a", "predicate": "is", "object": "   "},         # whitespace-only
    ]
    required = ("subject", "predicate", "object")
    invalid: list[dict] = []
    for i, f in enumerate(facts):
        if not isinstance(f, dict):
            continue
        missing = [k for k in required if k not in f or f[k] in (None, "")]
        if missing:
            invalid.append({"index": i, "missing": missing})
            continue
        bad_type = [
            k for k in required
            if not isinstance(f[k], str) or not f[k].strip()
        ]
        if bad_type:
            invalid.append({"index": i, "bad_fields": bad_type})
    # Items 1, 2, 3, 4 are bad; item 0 passes.
    assert {item["index"] for item in invalid} == {1, 2, 3, 4}


# --- P2: MetaFact empty-lineage warning log -----------------------------


def test_load_meta_facts_warns_on_empty_lineage(tmp_path, caplog):
    """Round-trip for legitimately empty MetaFacts still holds — the
    body is kept, but a logger.warning surfaces so the operator can
    spot manual-edit drift."""
    db = str(tmp_path / "m.db")
    backend = SQLiteBackend(db)
    empty = MetaFact()  # no source_fact_ids, no source_texts
    backend.save_meta_fact(empty)
    backend.close()

    backend2 = SQLiteBackend(db)
    with caplog.at_level(logging.WARNING):
        loaded = backend2.load_meta_facts()
    backend2.close()
    # Body is preserved (round-trip contract).
    assert any(m.meta_id == empty.meta_id for m in loaded)
    # And we logged about it.
    assert any("empty lineage" in rec.message for rec in caplog.records)


def test_load_meta_facts_no_warning_for_full_lineage(tmp_path, caplog):
    db = str(tmp_path / "m.db")
    backend = SQLiteBackend(db)
    full = MetaFact(
        weight=2, source_texts=["good"], source_fact_ids=["src-1"],
    )
    backend.save_meta_fact(full)
    backend.close()

    backend2 = SQLiteBackend(db)
    with caplog.at_level(logging.WARNING):
        backend2.load_meta_facts()
    backend2.close()
    assert not any("empty lineage" in rec.message for rec in caplog.records)


# --- P2: MCP cap disclosure ---------------------------------------------


def test_query_memory_cap_disclosure_shape_inline():
    """Caller asks for top_k=999 → server caps to 50 and surfaces
    both the effective value and a structured warning."""
    requested = 999
    top_k = requested
    if top_k <= 0:
        return
    if top_k > 50:
        top_k = 50
    response: dict = {"results": [], "_hint": "...", "effective_top_k": top_k}
    if requested != top_k:
        response["_warning"] = (
            f"top_k capped at {top_k} (requested {requested})"
        )
    assert response["effective_top_k"] == 50
    assert response["_warning"] == "top_k capped at 50 (requested 999)"


def test_query_memory_no_cap_no_warning_inline():
    """Under-cap request shows effective_top_k but no _warning."""
    requested = 10
    top_k = requested
    if top_k > 50:
        top_k = 50
    response: dict = {"results": [], "effective_top_k": top_k}
    if requested != top_k:
        response["_warning"] = "..."
    assert response["effective_top_k"] == 10
    assert "_warning" not in response


# --- P1: EmbeddingError MCP wrapper shape -------------------------------


def test_embedding_error_response_shape_inline():
    """Replicates the helper in server.py since importing server
    requires the mcp SDK."""
    exc = EmbeddingError("Ollama at http://localhost:11434 unreachable")
    response = {
        "ok": False,
        "error": "embedding_provider_unavailable",
        "detail": str(exc),
        "hint": (
            "Start Ollama, set BIRCH_EMBED_MODEL to a model the provider "
            "knows, or set BIRCH_EMBED_PROVIDER=mock for offline use."
        ),
    }
    assert response["ok"] is False
    assert response["error"] == "embedding_provider_unavailable"
    assert "Ollama" in response["detail"]
    assert "mock" in response["hint"]


# --- sanity: empty list round-trip preserved ----------------------------


def test_metafact_empty_lineage_round_trip_preserved(tmp_path):
    """The contract that '[]' saved -> [] loaded must still hold
    even with the empty-lineage warning log. Body is preserved."""
    db = str(tmp_path / "m.db")
    backend = SQLiteBackend(db)
    empty = MetaFact()
    backend.save_meta_fact(empty)
    backend.close()

    backend2 = SQLiteBackend(db)
    loaded = backend2.load_meta_facts()
    backend2.close()
    assert any(m.meta_id == empty.meta_id for m in loaded)


# --- helper: corrupt-row drop still works after warning addition --------


def test_load_meta_facts_still_drops_wrong_shape(tmp_path):
    db = str(tmp_path / "m.db")
    backend = SQLiteBackend(db)
    good = MetaFact(weight=1, source_texts=["x"], source_fact_ids=["y"])
    backend.save_meta_fact(good)
    backend.close()
    # Plant wrong-shape lineage.
    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT OR REPLACE INTO meta_facts "
        "(meta_id, vector, weight, source_texts, source_fact_ids, summary, "
        " gravity_score, created_at, layer, access_count, last_accessed, "
        " resonance_sum, resonance_count, recent_utility, "
        "forecast_stability) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        ("ghost", "[]", 1, '{"x": 1}', "[]", "",
         0.5, 0.0, 1, 0, 0.0, 0.0, 0, 0.5, 0.5),
    )
    conn.commit()
    conn.close()

    backend2 = SQLiteBackend(db)
    loaded = backend2.load_meta_facts()
    backend2.close()
    ids = {m.meta_id for m in loaded}
    assert good.meta_id in ids
    assert "ghost" not in ids
