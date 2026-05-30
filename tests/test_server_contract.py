"""MCP error-contract tests bound to the REAL server functions.

Unlike the historical inline-mirror approach (test_mcp_contract.py), this calls
``birch.server``'s actual tool functions and pins the envelopes they really
return — so the contract cannot silently drift from the code (the prior mirror
had already drifted: it asserted ``invalid_top_k`` while the server returns
``invalid_int`` via the shared ``_validate_int``).

``@mcp.tool()`` returns the plain function, so each tool is directly callable.
The module-level BIRCH_DB override points the server's singleton store at a
throwaway temp DB so importing it never touches a real ~/.birch brain; the
embedding provider is the deterministic mock (pytest default).
"""
from __future__ import annotations

import os
import tempfile

# Set BEFORE importing birch.server — it builds its MemoryStore singleton at
# import time from BIRCH_DB. Force a throwaway path so the real brain is safe.
os.environ["BIRCH_DB"] = os.path.join(
    tempfile.mkdtemp(prefix="birch-contract-"), "contract.db")
os.environ.setdefault("BIRCH_EMBED_PROVIDER", "mock")

import birch.server as server  # noqa: E402


# --- query_memory -------------------------------------------------------

def test_query_memory_top_k_zero_is_invalid_int():
    r = server.query_memory("anything", top_k=0)
    assert r["error"] == "invalid_int"
    assert r["results"] == []


def test_query_memory_negative_top_k_is_invalid_int():
    assert server.query_memory("anything", top_k=-3)["error"] == "invalid_int"


def test_query_memory_unknown_layer_is_structured():
    r = server.query_memory("anything", layers=["surfase"])
    assert r["error"] == "unknown_layer"
    assert r["unknown_layers"] == ["surfase"]
    assert set(r["allowed_layers"]) == {"surface", "kinetic", "core"}


# --- find_similar -------------------------------------------------------

def test_find_similar_top_k_zero_is_invalid_int():
    assert server.find_similar("x", top_k=0)["error"] == "invalid_int"


# --- list_facts ---------------------------------------------------------

def test_list_facts_limit_zero_is_invalid_int():
    out = server.list_facts(limit=0)
    # list_facts returns a list; the error envelope is a single-item list.
    assert isinstance(out, list) and out and out[0]["error"] == "invalid_int"


def test_list_facts_unknown_layer_is_structured():
    out = server.list_facts(layer="surfase")
    assert out[0]["error"] == "unknown_layer"
    assert set(out[0]["allowed"]) == {"surface", "kinetic", "core"}


# --- record_fact / set_fact: SPO field types ---------------------------

def test_record_fact_non_string_subject_is_invalid_fact_fields():
    r = server.record_fact(123, "is", "x")
    assert r["error"] == "invalid_fact_fields"
    assert "subject" in r["bad_fields"]


def test_set_fact_whitespace_predicate_is_invalid_fact_fields():
    r = server.set_fact("a", "   ", "b")
    assert r["error"] == "invalid_fact_fields"
    assert "predicate" in r["bad_fields"]


# --- record_facts batch -------------------------------------------------

def test_record_facts_missing_field_is_invalid_fact_item():
    r = server.record_facts([
        {"subject": "a", "predicate": "is", "object": "1"},
        {"subject": "b"},
    ])
    assert r["error"] == "invalid_fact_item"
    assert r["invalid"][0]["index"] == 1


def test_record_facts_non_dict_item_is_structured():
    r = server.record_facts(["not a dict"])
    assert r["invalid"][0]["error"] == "item_not_an_object"


# --- session_push -------------------------------------------------------

def test_session_push_unknown_session_is_structured():
    r = server.session_push("hello", session_id="ghost-session-xyz")
    assert r["error"] == "unknown_session"
    assert r["ok"] is False


# --- session_close ------------------------------------------------------

def test_session_close_invalid_sentiment_is_structured():
    r = server.session_close(session_id="whatever", sentiment="bogus")
    assert r["error"] == "invalid_sentiment"
    assert "resonant" in r["allowed"]


def test_session_close_invalid_r_override_is_structured():
    r = server.session_close(session_id="whatever", r_override=float("nan"))
    assert r["error"] == "invalid_r_override"


# --- record_session messages -------------------------------------------

def test_record_session_string_messages_is_invalid_messages():
    r = server.record_session("hello")
    assert r["error"] == "invalid_messages"


def test_record_session_non_string_item_is_invalid_message_item():
    r = server.record_session(["ok", 123, "fine"])
    assert r["error"] == "invalid_message_item"
    assert 1 in r["indices"]


def test_record_session_empty_list_is_empty_messages():
    assert server.record_session([])["error"] == "empty_messages"


# --- P3.8: record_session full close contract --------------------------

def test_record_session_surfaces_full_close_contract():
    """record_session must expose the same close fields as session_close —
    no one-shot contract drift."""
    r = server.record_session(["how do I tune the cache", "perfect, that worked"])
    for key in ("scoring_source", "confidence", "effective_r", "echo_outcome",
                "label", "r_score"):
        assert key in r, f"record_session response missing {key!r}"


# --- P3.9: conflicts are namespace-scoped ------------------------------

def test_conflicts_do_not_cross_namespaces():
    """Same (subject, predicate), different object, but in DIFFERENT namespaces
    are independent live rows — not a conflict."""
    server.record_fact("api", "runs on", "Go", namespace="WORK")
    server.record_fact("api", "runs on", "Rust", namespace="HOME")
    r = server.query_memory("api runs on", top_k=5)
    for c in r.get("conflicts", []):
        # No conflict group may mix namespaces; each is scoped.
        cands_ns = {c.get("namespace")}
        assert len(cands_ns) == 1
    # The cross-namespace pair specifically must not form a conflict.
    cross = [
        c for c in r.get("conflicts", [])
        if c.get("subject") == "api" and c.get("predicate") == "runs on"
        and len(c.get("candidates", [])) > 1
    ]
    # Either no such conflict, or if present its candidates share one namespace.
    for c in cross:
        objs = {x["object"] for x in c["candidates"]}
        assert not ({"Go", "Rust"} <= objs), (
            "Go (WORK) and Rust (HOME) must not be grouped as one conflict"
        )
