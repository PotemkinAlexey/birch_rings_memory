"""MemoryStore.query surfaces MetaFacts via Hawking emission and live store."""
from __future__ import annotations

from birch.memory_store import MemoryStore
from birch.meta_fact import MetaFact


def test_query_result_kind_property_classifies_polymorphic_hits():
    from birch.memory_store import QueryResult
    from birch.fact import FactPassport

    f = FactPassport("x", "is", "y", fact_id="f1")
    r_fact = QueryResult(similarity=0.9, source="kinetic", fact=f)
    assert r_fact.kind == "fact"
    assert r_fact.body_id == "f1"

    r_meta = QueryResult(similarity=0.9, source="hawking_meta", meta=MetaFact(meta_id="m1"))
    assert r_meta.kind == "meta"
    assert r_meta.body_id == "m1"


def test_hawking_emit_meta_through_query_returns_kind_meta():
    """A MetaFact in the singularity matched by query emerges as kind=meta."""
    mem = MemoryStore()
    # Use add_fact to get a real embedding vector; then point the MetaFact
    # at the same vector so the query text below matches it tightly.
    f = mem.add_fact("mailer service", "runs on", "Go")
    target_vec = list(f.vector)
    # Move the fact to the black hole so the live scan won't outshine the
    # MetaFact hit (and so the query can only land on the meta).
    f.gravity_score = 0.05
    mem._absorb_dead()
    assert mem.stats["black_hole_fact_mass"] == 1

    meta = MetaFact(
        meta_id="m-mailer",
        vector=target_vec,
        weight=10,                       # log10(10)=1 → bonus 0.10 → gravity 0.40
        source_texts=["mailer service runs on Go", "mailer relies on Go"],
        source_fact_ids=["x", "y"],
    )
    mem._hole.absorb_meta(meta)

    # Use the same source text so the embedding hits exactly the meta vector.
    results = mem.query("mailer service runs on Go", top_k=5)
    meta_hits = [r for r in results if r.kind == "meta"]
    assert len(meta_hits) == 1
    hit = meta_hits[0]
    assert hit.source == "hawking_meta"
    assert hit.meta.meta_id == "m-mailer"
    assert abs(hit.meta.gravity_score - 0.40) < 1e-6
    assert hit.meta.layer == 1, "emitted MetaFact must land in kinetic"
    assert "m-mailer" not in mem._hole
    assert "m-mailer" in mem._meta_facts


def test_live_metafact_is_returned_on_subsequent_queries_without_hawking():
    """Once a MetaFact is alive it answers queries from the live layers."""
    mem = MemoryStore()
    f = mem.add_fact("mailer service", "runs on", "Go")
    target_vec = list(f.vector)

    meta = MetaFact(meta_id="m-live", vector=target_vec, weight=10)
    mem._meta_facts[meta.meta_id] = meta
    mem._meta_index.add(meta.meta_id, meta.vector)
    mem._engine.register(meta)
    meta.layer = 1
    meta.gravity_score = 0.50

    results = mem.query("mailer service Go", top_k=5)
    meta_hits = [r for r in results if r.kind == "meta"]
    assert len(meta_hits) == 1
    assert meta_hits[0].source == "kinetic"


def test_metafact_attribution_propagates_session_resonance():
    """A queried MetaFact participates in the feedback loop just like a fact."""
    mem = MemoryStore()
    f = mem.add_fact("mailer service", "runs on", "Go")
    target_vec = list(f.vector)

    meta = MetaFact(meta_id="m-feedback", vector=target_vec, weight=10)
    mem._meta_facts[meta.meta_id] = meta
    mem._meta_index.add(meta.meta_id, meta.vector)
    mem._engine.register(meta)
    meta.layer = 1
    meta.gravity_score = 0.50
    assert meta.resonance_count == 0

    mem.session_start("s-meta")
    mem.session_message("how do I configure the mailer service")
    mem.query("mailer service Go", top_k=3, session_id="s-meta")
    mem.session_message("works, thanks!")
    summary = mem.session_close(session_id="s-meta")

    assert summary["label"] == "resonant"
    assert meta.resonance_count == 1
    assert meta.resonance_sum > 0


def test_metafact_falls_back_into_singularity_when_gravity_drops():
    """A live MetaFact whose gravity falls below 0.10 is re-absorbed."""
    mem = MemoryStore()
    f = mem.add_fact("mailer service", "runs on", "Go")
    target_vec = list(f.vector)

    meta = MetaFact(meta_id="m-falling", vector=target_vec, weight=10)
    mem._meta_facts[meta.meta_id] = meta
    mem._meta_index.add(meta.meta_id, meta.vector)
    mem._engine.register(meta)
    meta.layer = 1
    meta.gravity_score = 0.05            # below the floor

    absorbed = mem._absorb_dead()
    assert "m-falling" in absorbed
    assert "m-falling" not in mem._meta_facts
    assert "m-falling" in mem._hole
    assert mem._hole.meta_mass == 1


def test_stats_breaks_down_facts_and_metas(tmp_path):
    mem = MemoryStore(db_path=str(tmp_path / "stats.db"))
    mem.add_fact("alpha", "is", "first")
    mem.add_fact("beta",  "is", "second")
    # Force one MetaFact into the live store.
    meta = MetaFact(meta_id="m-stat", vector=[1.0, 0.0], weight=4)
    meta.layer = 1
    mem._meta_facts[meta.meta_id] = meta
    mem._meta_index.add(meta.meta_id, meta.vector)
    mem._engine.register(meta)

    s = mem.stats
    assert s["total_live"] == 2
    assert s["total_live_metas"] == 1
    assert "black_hole_fact_mass" in s
    assert "black_hole_meta_mass" in s


def test_emitted_meta_persists_across_restart(tmp_path):
    """An emitted MetaFact must come back from storage on next open as live."""
    db = tmp_path / "meta-emit.db"
    mem = MemoryStore(db_path=str(db))
    f = mem.add_fact("mailer service", "runs on", "Go")
    target_vec = list(f.vector)
    f.gravity_score = 0.05
    mem._absorb_dead()

    meta = MetaFact(
        meta_id="m-persist",
        vector=target_vec,
        weight=10,
        source_texts=["mailer Go"],
    )
    mem._hole.absorb_meta(meta)
    mem._storage.save_meta_fact(meta)

    results = mem.query("mailer service Go", top_k=3)
    meta_hits = [r for r in results if r.kind == "meta"]
    assert len(meta_hits) == 1
    mem._storage.close()

    reopened = MemoryStore(db_path=str(db))
    assert "m-persist" in reopened._meta_facts, \
        "emitted MetaFact must be persisted with layer >= 0"
    assert reopened._meta_facts["m-persist"].layer == 1
