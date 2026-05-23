"""Agent-facing ergonomics: set_fact, find_similar, explain_fact, filters.

These cover the write-time hygiene and read-time sanity additions:
- set_fact upserts a (subject, predicate) slot — old bodies go to singularity
- find_similar surfaces paraphrase candidates without writing
- explain_fact decomposes gravity for debugging
- list_facts filters (subject_prefix, min_gravity, layer, exclude_deprecated)
"""
from birch.memory_store import MemoryStore


def test_set_fact_creates_first_value(tmp_path):
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    out = mem.set_fact("project-x", "HEAD on master", "abc123")
    assert out["set"] is True
    assert out["superseded"] == []
    assert out["fact_id"]
    live = mem.list_facts(subject="project-x")
    assert len(live) == 1
    assert live[0].object == "abc123"


def test_set_fact_supersedes_existing_slot_occupants(tmp_path):
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    first = mem.add_fact("project-x", "HEAD on master", "abc123")
    second = mem.add_fact("project-x", "HEAD on master", "def456")
    # Both live (record_fact is append on different objects).
    assert {first.fact_id, second.fact_id} <= {f.fact_id for f in mem.list_facts()}

    out = mem.set_fact("project-x", "HEAD on master", "ghi789")
    assert out["set"] is True
    assert set(out["superseded"]) == {first.fact_id, second.fact_id}

    # The live SPO slot now belongs to the new fact only.
    live = [
        f for f in mem.list_facts(subject="project-x")
        if not f.is_deprecated and not f.is_expired
    ]
    assert [(f.object, f.fact_id) for f in live] == [("ghi789", out["fact_id"])]

    # Old bodies survived in the singularity with deprecated_by intact.
    for old_id in [first.fact_id, second.fact_id]:
        assert old_id in mem._hole._singularity
        assert mem._hole._singularity[old_id].fact.deprecated_by == out["fact_id"]


def test_set_fact_idempotent_when_value_unchanged(tmp_path):
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    first = mem.set_fact("project-x", "HEAD on master", "abc123")
    again = mem.set_fact("project-x", "HEAD on master", "abc123")
    assert again["fact_id"] == first["fact_id"]
    assert again["already_existed"] is True
    assert again["superseded"] == []


def test_find_similar_returns_paraphrase_hits(tmp_path):
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    a = mem.add_fact("project-x", "test count", "263 passed")
    mem.add_fact("project-x", "test count", "263 passed")  # exact dedup → same id
    # Different SPO, similar vector.
    b = mem.add_fact("project-x", "test count", "263 tests passed")

    hits = mem.find_similar(
        "project-x test count 263", top_k=5, min_similarity=0.3
    )
    ids = {h["fact_id"] for h in hits}
    assert a.fact_id in ids
    assert b.fact_id in ids


def test_find_similar_excludes_deprecated(tmp_path):
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    old = mem.add_fact("project-x", "HEAD on master", "abc123")
    new = mem.add_fact("project-x", "HEAD on master", "def456")
    mem.supersede_fact(old.fact_id, new.fact_id)
    hits = mem.find_similar(
        "project-x HEAD on master", top_k=5, min_similarity=0.3
    )
    ids = {h["fact_id"] for h in hits}
    assert old.fact_id not in ids
    assert new.fact_id in ids


def test_find_similar_respects_subject_prefix(tmp_path):
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    mem.add_fact("project-x", "uses", "Postgres")
    mem.add_fact("project-y", "uses", "Postgres")
    hits = mem.find_similar(
        "uses Postgres", top_k=5, min_similarity=0.3, subject_prefix="project-x"
    )
    assert all(h["subject"] == "project-x" for h in hits)


def test_explain_fact_decomposes_gravity(tmp_path):
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    fact = mem.add_fact("api", "runs on", "Go")
    report = mem.explain_fact(fact.fact_id)
    assert report["found"] is True
    assert report["fact_id"] == fact.fact_id
    # Six contributions, one per feature plus resonance.
    assert set(report["contributions"]) == {
        "freshness", "access", "graph",
        "recent_utility", "forecast_stability", "resonance",
    }
    # Contributions sum to the live (recomputed) gravity, not necessarily
    # the stored one (which is the value at the last tick).
    summed = sum(report["contributions"].values())
    assert abs(summed - report["live_gravity_score"]) < 0.01
    # Sanity: every contribution is non-negative.
    assert all(v >= 0.0 for v in report["contributions"].values())


def test_explain_fact_missing_id(tmp_path):
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    out = mem.explain_fact("not-a-real-id")
    assert out["found"] is False
