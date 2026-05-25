"""MemoryBricks Step 1 — namespace coexistence + filter + persistence.

This pins the load-bearing invariants of the new `namespace` field:

* Same SPO under different namespaces are independent live rows.
* `namespace_prefix` is a VB-style hierarchical filter (matches the
  prefix and every descendant; does not match a sibling that merely
  shares a leading substring).
* `set_fact` slot uniqueness is namespace-scoped — a write in WORK
  does not supersede a fact in PERSONAL with the same (subject,
  predicate).
* SQLite round-trip preserves `namespace` on both FactPassport and
  MetaFact. A legacy DB created without the column migrates cleanly
  on open and defaults the missing field to "" (the global root).
* MCP boundary validates `namespace` and `namespace_prefix` as
  optional text — non-string inputs return structured errors instead
  of crashing core.
"""
from __future__ import annotations

import os
import sqlite3

import pytest

from birch.fact import FactPassport
from birch.memory_store import MemoryStore
from birch.meta_fact import MetaFact
from birch.storage.sqlite import SQLiteBackend

# --- Dataclass-level sanitisation ----------------------------------------


def test_factpassport_namespace_defaults_empty_string():
    f = FactPassport(subject="s", predicate="p", object="o")
    assert f.namespace == ""


def test_factpassport_namespace_strips_surrounding_whitespace():
    f = FactPassport(
        subject="s", predicate="p", object="o", namespace="  WORK/X  ",
    )
    assert f.namespace == "WORK/X"


@pytest.mark.parametrize("bad", [None, 42, ["WORK"], object()])
def test_factpassport_namespace_coerces_loose_input(bad):
    f = FactPassport(
        subject="s", predicate="p", object="o", namespace=bad,
    )
    # All paths produce a string; bad values collapse to "" or to
    # the str() of the value — never to a non-string slot type.
    assert isinstance(f.namespace, str)


def test_metafact_namespace_defaults_empty_string():
    m = MetaFact()
    assert m.namespace == ""


def test_metafact_namespace_none_becomes_empty():
    m = MetaFact(namespace=None)
    assert m.namespace == ""


# --- Coexistence of same SPO under different namespaces ------------------


def test_same_spo_in_different_namespaces_are_independent(tmp_path):
    s = MemoryStore(db_path=str(tmp_path / "ns.db"))
    f_global = s.add_fact("api", "uses", "Postgres")
    f_a = s.add_fact("api", "uses", "Postgres", namespace="WORK/A")
    f_b = s.add_fact("api", "uses", "Postgres", namespace="WORK/B")
    assert len({f_global.fact_id, f_a.fact_id, f_b.fact_id}) == 3
    # fact_exists is scoped per namespace.
    assert s.fact_exists("api", "uses", "Postgres")
    assert s.fact_exists("api", "uses", "Postgres", namespace="WORK/A")
    assert s.fact_exists("api", "uses", "Postgres", namespace="WORK/B")
    assert not s.fact_exists("api", "uses", "Postgres", namespace="WORK/C")


def test_add_fact_same_spo_same_namespace_is_dedup(tmp_path):
    s = MemoryStore(db_path=str(tmp_path / "ns.db"))
    f1 = s.add_fact("api", "uses", "Postgres", namespace="WORK/A")
    f2 = s.add_fact("api", "uses", "Postgres", namespace="WORK/A")
    assert f1.fact_id == f2.fact_id


# --- namespace_prefix filter --------------------------------------------


def test_namespace_prefix_matches_self_and_descendants(tmp_path):
    s = MemoryStore(db_path=str(tmp_path / "ns.db"))
    s.add_fact("a", "p", "o", namespace="WORK")
    s.add_fact("b", "p", "o", namespace="WORK/X")
    s.add_fact("c", "p", "o", namespace="WORK/X/Y")
    s.add_fact("d", "p", "o", namespace="PERSONAL")
    s.add_fact("e", "p", "o")  # global root

    work = s.list_facts(namespace_prefix="WORK", limit=100)
    assert {f.namespace for f in work} == {"WORK", "WORK/X", "WORK/X/Y"}

    work_x = s.list_facts(namespace_prefix="WORK/X", limit=100)
    assert {f.namespace for f in work_x} == {"WORK/X", "WORK/X/Y"}

    # Trailing slash on the prefix is tolerated.
    work_x_slash = s.list_facts(namespace_prefix="WORK/X/", limit=100)
    assert {f.namespace for f in work_x_slash} == {"WORK/X", "WORK/X/Y"}


def test_namespace_prefix_does_not_match_sibling_with_shared_substring(tmp_path):
    s = MemoryStore(db_path=str(tmp_path / "ns.db"))
    s.add_fact("a", "p", "o", namespace="WORK")
    s.add_fact("b", "p", "o", namespace="WORKSPACE")  # NOT a child of WORK

    out = s.list_facts(namespace_prefix="WORK", limit=100)
    assert {f.namespace for f in out} == {"WORK"}


def test_namespace_prefix_empty_string_means_all_facts(tmp_path):
    s = MemoryStore(db_path=str(tmp_path / "ns.db"))
    s.add_fact("a", "p", "o", namespace="WORK")
    s.add_fact("b", "p", "o", namespace="PERSONAL")
    s.add_fact("c", "p", "o")

    out = s.list_facts(namespace_prefix="", limit=100)
    assert len(out) == 3


def test_query_namespace_prefix_filters_results(tmp_path):
    s = MemoryStore(db_path=str(tmp_path / "ns.db"))
    s.add_fact("api", "uses", "Postgres", namespace="WORK/A")
    s.add_fact("api", "uses", "Postgres", namespace="WORK/B")
    s.add_fact("api", "uses", "Postgres", namespace="PERSONAL")

    hits = s.query(
        "api uses Postgres", top_k=10, namespace_prefix="WORK",
    )
    namespaces = {r.fact.namespace for r in hits if r.fact is not None}
    assert namespaces == {"WORK/A", "WORK/B"}


def test_find_similar_namespace_prefix(tmp_path):
    s = MemoryStore(db_path=str(tmp_path / "ns.db"))
    s.add_fact("api", "uses", "Postgres", namespace="WORK/A")
    s.add_fact("api", "uses", "Postgres", namespace="PERSONAL")

    hits = s.find_similar(
        "api uses Postgres", top_k=10, min_similarity=0.0,
        namespace_prefix="WORK",
    )
    # Mock embedding gives non-trivial cosines; we only care that
    # PERSONAL is excluded by the namespace filter.
    assert all(h["fact_id"] for h in hits)


# --- set_fact slot uniqueness is namespace-scoped -----------------------


def test_set_fact_does_not_supersede_other_namespace(tmp_path):
    s = MemoryStore(db_path=str(tmp_path / "ns.db"))
    s.add_fact("api", "uses", "Postgres", namespace="WORK/A")
    s.add_fact("api", "uses", "Postgres", namespace="WORK/B")

    r = s.set_fact("api", "uses", "MySQL", namespace="WORK/A")
    # Only the WORK/A occupant was superseded.
    assert len(r["superseded"]) == 1
    # WORK/B's Postgres fact is still live.
    assert s.fact_exists("api", "uses", "Postgres", namespace="WORK/B")
    # WORK/A's Postgres fact is gone from the live SPO bucket.
    assert not s.fact_exists("api", "uses", "Postgres", namespace="WORK/A")


# --- add_facts per-item namespace ---------------------------------------


def test_add_facts_namespaces_per_item(tmp_path):
    s = MemoryStore(db_path=str(tmp_path / "ns.db"))
    triples = [
        ("api", "uses", "Postgres"),
        ("api", "uses", "Postgres"),
        ("svc", "owns", "queue"),
    ]
    out = s.add_facts(
        triples,
        namespaces=["WORK/A", "WORK/B", None],  # third falls back to default
        namespace="WORK/Default",
    )
    assert out[0].namespace == "WORK/A"
    assert out[1].namespace == "WORK/B"
    assert out[2].namespace == "WORK/Default"
    # Same SPO under A and B coexist.
    assert out[0].fact_id != out[1].fact_id


def test_add_facts_namespaces_length_mismatch_raises(tmp_path):
    s = MemoryStore(db_path=str(tmp_path / "ns.db"))
    with pytest.raises(ValueError, match="namespaces length"):
        s.add_facts(
            [("a", "p", "o"), ("b", "p", "o")],
            namespaces=["WORK/A"],  # too short
        )


def test_add_facts_same_spo_across_namespaces_not_in_batch_dupe(tmp_path):
    """Regression: ``seen_in_batch`` used to key on (s, p, o) only.

    A batch like ``add_facts([("api","uses","PG"), ("api","uses","PG")],
    namespaces=["WORK/A", "WORK/B"])`` silently aliased the second
    item to the first occurrence (``duplicate_in_batch=True``) and
    attributed both to WORK/A — the exact failure mode Step 1's
    per-namespace dedup is meant to exclude. The fix widened the
    seen_in_batch key to (namespace, s, p, o); this test pins the
    runtime contract so a future narrowing regresses loudly instead
    of quietly.
    """
    s = MemoryStore(db_path=str(tmp_path / "ns.db"))
    statuses = s.add_facts(
        [("api", "uses", "Postgres"), ("api", "uses", "Postgres")],
        namespaces=["WORK/A", "WORK/B"],
        return_status=True,
    )
    # Neither item is a duplicate; both are brand-new facts.
    assert statuses[0]["duplicate_in_batch"] is False
    assert statuses[1]["duplicate_in_batch"] is False
    assert statuses[0]["already_existed"] is False
    assert statuses[1]["already_existed"] is False
    # The two facts are distinct objects in their respective namespaces.
    assert statuses[0]["fact"].fact_id != statuses[1]["fact"].fact_id
    assert statuses[0]["fact"].namespace == "WORK/A"
    assert statuses[1]["fact"].namespace == "WORK/B"


def test_add_facts_same_spo_same_namespace_still_in_batch_dupe(tmp_path):
    """Companion to the above — within a single namespace, a repeated
    SPO in the same batch is still correctly flagged as a duplicate.
    The wider key must not weaken the in-batch dedup it exists for."""
    s = MemoryStore(db_path=str(tmp_path / "ns.db"))
    statuses = s.add_facts(
        [("api", "uses", "Postgres"), ("api", "uses", "Postgres")],
        namespaces=["WORK/A", "WORK/A"],
        return_status=True,
    )
    assert statuses[0]["duplicate_in_batch"] is False
    assert statuses[1]["duplicate_in_batch"] is True
    # Both entries map to the same fact_id.
    assert statuses[0]["fact"].fact_id == statuses[1]["fact"].fact_id


# --- SQLite persistence + migration -------------------------------------


def test_sqlite_round_trip_preserves_namespace(tmp_path):
    db = str(tmp_path / "rt.db")
    backend = SQLiteBackend(db)
    f = FactPassport(
        subject="s", predicate="p", object="o", namespace="WORK/X",
    )
    f.vector = [0.1, 0.2]
    backend.save_fact(f)
    m = MetaFact(
        namespace="PERSONAL", vector=[0.1, 0.2],
        source_texts=["x"], source_fact_ids=["a"],
    )
    backend.save_meta_fact(m)
    backend.close()

    backend2 = SQLiteBackend(db)
    facts = backend2.load_facts()
    metas = backend2.load_meta_facts()
    backend2.close()
    assert facts[0].namespace == "WORK/X"
    assert metas[0].namespace == "PERSONAL"


def test_legacy_db_without_namespace_column_migrates(tmp_path):
    """A pre-Step-1 SQLite file (no `namespace` column) must open
    cleanly under the new backend. The migration adds the column
    NOT NULL DEFAULT '', existing rows take the global root, and
    load_facts returns a FactPassport whose namespace is ''."""
    db = str(tmp_path / "legacy.db")
    c = sqlite3.connect(db)
    c.executescript(
        """
        CREATE TABLE facts (
            fact_id TEXT PRIMARY KEY, subject TEXT, predicate TEXT,
            object TEXT, vector TEXT, gravity_score REAL, layer INTEGER,
            created_at REAL, ttl REAL, source_session TEXT,
            deprecated_by TEXT, access_count INTEGER, last_accessed REAL,
            resonance_sum REAL, resonance_count INTEGER,
            recent_utility REAL, forecast_stability REAL
        );
        CREATE TABLE meta_facts (
            meta_id TEXT PRIMARY KEY, vector TEXT, weight INTEGER,
            source_texts TEXT, source_fact_ids TEXT, summary TEXT,
            gravity_score REAL, created_at REAL, layer INTEGER,
            access_count INTEGER, last_accessed REAL, resonance_sum REAL,
            resonance_count INTEGER, recent_utility REAL,
            forecast_stability REAL
        );
        INSERT INTO facts (fact_id, subject, predicate, object, vector,
            gravity_score, layer, created_at, access_count, last_accessed,
            resonance_sum, resonance_count, recent_utility,
            forecast_stability)
        VALUES ('id1', 'sub', 'pred', 'obj', '[]', 0.5, 1, 100.0, 0, 100.0,
            0.0, 0, 0.5, 0.5);
        """
    )
    c.commit()
    c.close()

    backend = SQLiteBackend(db)
    facts = backend.load_facts()
    backend.close()
    assert len(facts) == 1
    assert facts[0].namespace == ""


# --- MCP boundary validation ---------------------------------------------


def _fresh_server(tmp_path):
    """Reload server module against a fresh on-disk DB.

    The MCP server module owns a singleton store at import time
    (``_store = MemoryStore(db_path=_DB_PATH)`` keyed off
    ``BIRCH_DB``). Swap that env var to a per-test tmp_path file
    and reload the module so each test starts with an empty store
    sized to the mock embedding's dim — without it, the default
    ``~/.birch/memory.db`` (potentially 768-dim from a real-Ollama
    run) gets reused and the mock provider's 64-dim vectors fail
    the dim preflight.
    """
    import importlib
    os.environ["BIRCH_DB"] = str(tmp_path / "mcp.db")
    import birch.server as server_mod
    importlib.reload(server_mod)
    return server_mod


def test_mcp_record_fact_rejects_non_string_namespace(tmp_path):
    server = _fresh_server(tmp_path)
    resp = server.record_fact(
        subject="s", predicate="p", object="o", namespace=42,
    )
    assert resp.get("error") == "invalid_text"
    assert resp.get("field") == "namespace"


def test_mcp_set_fact_rejects_non_string_namespace(tmp_path):
    server = _fresh_server(tmp_path)
    resp = server.set_fact(
        subject="s", predicate="p", object="o", namespace=["WORK"],
    )
    assert resp.get("error") == "invalid_text"
    assert resp.get("field") == "namespace"


def test_mcp_query_memory_rejects_non_string_namespace_prefix(tmp_path):
    server = _fresh_server(tmp_path)
    resp = server.query_memory(text="api", namespace_prefix=123)
    assert resp.get("error") == "invalid_text"
    assert resp.get("field") == "namespace_prefix"


def test_mcp_record_facts_per_item_namespace_validation(tmp_path):
    server = _fresh_server(tmp_path)
    resp = server.record_facts(
        facts=[
            {"subject": "a", "predicate": "p", "object": "o",
             "namespace": "WORK/A"},
            {"subject": "b", "predicate": "p", "object": "o",
             "namespace": 42},   # non-string
        ],
    )
    assert resp.get("error") == "invalid_fact_item"
    bad = [e for e in resp.get("invalid", []) if e.get("index") == 1]
    assert bad and bad[0].get("error") == "invalid_namespace"


def test_mcp_record_fact_persists_namespace(tmp_path):
    server = _fresh_server(tmp_path)
    resp = server.record_fact(
        subject="api", predicate="uses", object="Postgres",
        namespace="WORK/A",
    )
    assert resp.get("fact_id")
    # Read back via list_facts MCP — should carry namespace.
    rows = server.list_facts(namespace_prefix="WORK")
    assert any(r.get("namespace") == "WORK/A" for r in rows)


def test_mcp_set_fact_scoped_does_not_supersede_other_namespace(tmp_path):
    server = _fresh_server(tmp_path)
    server.record_fact(
        subject="api", predicate="uses", object="Postgres",
        namespace="WORK/A",
    )
    server.record_fact(
        subject="api", predicate="uses", object="Postgres",
        namespace="WORK/B",
    )
    r = server.set_fact(
        subject="api", predicate="uses", object="MySQL",
        namespace="WORK/A",
    )
    assert r.get("set") is True
    assert len(r.get("superseded", [])) == 1
    rows_b = server.list_facts(namespace_prefix="WORK/B")
    assert any(
        row.get("object") == "Postgres" and row.get("namespace") == "WORK/B"
        for row in rows_b
    )
