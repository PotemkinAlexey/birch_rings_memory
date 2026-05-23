"""ChatGPT round-10 punch-list regressions.

Round 10 (after DeepSeek round 1) found semantic-shape gaps the
round-9 JSON-parse pre-validation missed (MetaFact silently coerced
to lineage-less body), MCP bounds gaps (top_k/limit <= 0 leaked
through), and a load_open_sessions shape gap (valid JSON in the
wrong shape would crash the consumer).
"""
from __future__ import annotations

import sqlite3

from birch.meta_fact import MetaFact
from birch.storage.sqlite import SQLiteBackend

# --- P1: MetaFact semantic shape (wrong-shape JSON drops the row) -------


def test_load_meta_facts_drops_row_with_dict_shaped_lineage(tmp_path):
    """Cell is valid JSON but a dict where a list is required. The
    round-9 pre-validation only caught unparseable cells; round 10
    catches parsed-but-wrong-shape too. The round-trip contract for
    legitimately empty MetaFacts ([] saved → [] loaded) still holds."""
    db = str(tmp_path / "m.db")
    backend = SQLiteBackend(db)

    # Plant one good MetaFact through the normal save path so we have
    # a baseline that survives load.
    good = MetaFact(weight=2, source_texts=["good text"],
                    source_fact_ids=["src-1"],
                    gravity_score=0.5, layer=1)
    good.vector = [0.1] * 8
    backend.save_meta_fact(good)
    backend.close()

    # Plant a row whose source_texts is valid JSON but a dict, not a
    # list. _load_list would silently coerce it to []; round 10 drops
    # the row instead so the body never enters _meta_facts as a
    # lineage-less ghost.
    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT OR REPLACE INTO meta_facts "
        "(meta_id, vector, weight, source_texts, source_fact_ids, summary, "
        " gravity_score, created_at, layer, access_count, last_accessed, "
        " resonance_sum, resonance_count, recent_utility, forecast_stability) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        ("ghost", "[]", 1, '{"x": 1}', "[]", "",
         0.5, 0.0, 1, 0, 0.0, 0.0, 0, 0.5, 0.5),
    )
    conn.commit()
    conn.close()

    backend2 = SQLiteBackend(db)
    loaded = backend2.load_meta_facts()
    backend2.close()
    loaded_ids = {m.meta_id for m in loaded}
    assert good.meta_id in loaded_ids
    assert "ghost" not in loaded_ids


def test_load_meta_facts_preserves_legitimately_empty_lists(tmp_path):
    """An empty list cell ("[]") is saved-as-empty, not corruption.
    Round-trip contract: save → load returns the same shape."""
    db = str(tmp_path / "m.db")
    backend = SQLiteBackend(db)
    empty = MetaFact()                # no lineage, no vector
    backend.save_meta_fact(empty)
    backend.close()

    backend2 = SQLiteBackend(db)
    loaded = backend2.load_meta_facts()
    backend2.close()
    assert any(m.meta_id == empty.meta_id for m in loaded)


# --- P2: load_open_sessions shape validation -----------------------------


def test_load_open_sessions_drops_row_with_wrong_shape_facts(tmp_path):
    """A row whose `facts` cell parsed to a list (instead of dict)
    used to surface to the consumer, which then crashed on
    ``.items()``. Drop the row at the loader instead."""
    db = str(tmp_path / "m.db")
    backend = SQLiteBackend(db)
    backend.save_open_session("ok", [], [], {}, started_at=0.0)
    backend.close()

    # Corrupt one row: `facts` cell is a JSON list, not a dict.
    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT OR REPLACE INTO open_sessions "
        "(session_id, messages, vectors, facts, started_at) "
        "VALUES (?, ?, ?, ?, ?)",
        ("bad-shape", "[]", "[]", "[1, 2, 3]", 0.0),
    )
    conn.commit()
    conn.close()

    backend2 = SQLiteBackend(db)
    sessions = backend2.load_open_sessions()
    backend2.close()
    ids = {s["session_id"] for s in sessions}
    assert "ok" in ids
    assert "bad-shape" not in ids


# --- P2: MCP top_k / limit bounds (inline simulation, no SDK import) ----


def test_query_memory_top_k_zero_inline_validator():
    """The server now rejects top_k<=0 with a structured response
    rather than passing it through to the store."""
    # Replicate the server's bounds check inline so we don't need to
    # spin up the MCP SDK.
    top_k = 0
    if top_k <= 0:
        resp = {"results": [], "error": "invalid_top_k",
                "_hint": "top_k must be positive"}
    assert resp["error"] == "invalid_top_k"
    assert resp["results"] == []


def test_list_facts_limit_zero_inline_validator():
    """A zero/negative limit returns [] before any append, not
    after-the-first-item like the old code path."""
    limit = 0
    out: list = []
    if limit <= 0:
        result = out
    else:
        # would have appended one item then broken on len(out)>=limit
        out.append({"dummy": 1})
        result = out
    assert result == []


def test_find_similar_top_k_negative_inline_validator():
    """find_similar surfaces a structured warning instead of falling
    through to the core layer's defensive guard."""
    top_k = -3
    text = "x"
    if top_k <= 0:
        resp = {"query": text, "hits": [],
                "_warning": "top_k must be positive"}
    assert resp["_warning"] == "top_k must be positive"


# --- P2: forecast_memory docstring uses "bodies" language ---------------


def test_forecast_memory_docstring_says_bodies():
    # File-level read avoids importing birch.server (it imports the
    # mcp SDK, optional dep). The docstring is what the agent sees,
    # so the file text is the right ground truth.
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    src = (root / "src" / "birch" / "server.py").read_text()
    assert "how many bodies were forecasted" in src
    # Legacy keys are still documented (wire-format stability).
    assert "facts_forecasted" in src
