"""MCP serialization — polymorphic query_memory payloads."""
from __future__ import annotations

from birch.fact import FactPassport
from birch.memory_store import QueryResult
from birch.meta_fact import MetaFact


def test_query_result_fact_payload():
    f = FactPassport(
        fact_id="f-1",
        subject="mailer",
        predicate="runs on",
        object="Go",
        layer=1,
        gravity_score=0.55,
    )
    r = QueryResult(similarity=0.91, source="kinetic", fact=f)
    d = r.to_mcp_dict()
    assert d["kind"] == "fact"
    assert d["body_id"] == "f-1"
    assert d["fact_id"] == "f-1"
    assert d["subject"] == "mailer"
    assert d["predicate"] == "runs on"
    assert d["object"] == "Go"
    assert d["layer"] == 1
    assert d["gravity_score"] == 0.55
    assert d["source"] == "kinetic"
    assert d["similarity"] == 0.91
    assert "meta_id" not in d
    assert "weight" not in d


def test_query_result_meta_payload():
    m = MetaFact(
        meta_id="m-1",
        weight=4,
        source_texts=["a uses b", "c uses d"],
        source_fact_ids=["f-a", "f-b"],
        summary="cluster hint",
        layer=-1,
        gravity_score=0.42,
    )
    r = QueryResult(similarity=0.88, source="hawking_meta", meta=m)
    d = r.to_mcp_dict()
    assert d["kind"] == "meta"
    assert d["body_id"] == "m-1"
    assert d["meta_id"] == "m-1"
    assert d["weight"] == 4
    assert d["source_texts"] == ["a uses b", "c uses d"]
    assert d["source_fact_ids"] == ["f-a", "f-b"]
    assert d["summary"] == "cluster hint"
    assert d["source"] == "hawking_meta"
    assert "subject" not in d
    assert "fact_id" not in d
