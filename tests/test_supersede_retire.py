"""supersede_fact / retire_fact — soft removal that preserves the singularity.

The agent-facing path for "stale / wrong / outdated" data must NOT be the
destructive ``delete_fact`` primitive. Both supersede and retire move the
body to the black hole on the same call and leave the row in storage so
the singularity collapse can still compress it and Hawking emission can
still rescue it.
"""
from birch.memory_store import MemoryStore


def test_supersede_moves_body_to_singularity(tmp_path):
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    old = mem.add_fact("api", "runs on", "Go")
    new = mem.add_fact("api", "runs on", "Rust")

    result = mem.supersede_fact(old.fact_id, new.fact_id)
    assert result["superseded"] is True
    assert old.fact_id in result["absorbed"]

    # Body left the live layers …
    assert old.fact_id not in mem._facts
    # … and ended up in the singularity …
    assert old.fact_id in mem._hole._singularity
    # … with the lineage pointer intact.
    assert (mem._hole._singularity[old.fact_id].fact.deprecated_by
            == new.fact_id)


def test_supersede_keeps_row_in_storage_for_lineage(tmp_path):
    """Hard delete drops the row; supersede must NOT."""
    db = str(tmp_path / "m.db")
    mem = MemoryStore(db_path=db)
    old = mem.add_fact("api", "runs on", "Go")
    new = mem.add_fact("api", "runs on", "Rust")
    mem.supersede_fact(old.fact_id, new.fact_id)
    mem.close()

    # Reopen — the deprecated fact must still be on disk with the pointer.
    reopened = MemoryStore(db_path=db)
    assert reopened._storage is not None
    loaded = {f.fact_id: f for f in reopened._storage.load_facts()}
    assert old.fact_id in loaded
    assert loaded[old.fact_id].deprecated_by == new.fact_id


def test_supersede_frees_the_spo_slot(tmp_path):
    """After supersede, the SPO slot belongs to the new fact, not the old."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    old = mem.add_fact("api", "runs on", "Go")
    mem.supersede_fact(old.fact_id, "some-new-uuid")
    # A subsequent record of the same SPO returns a NEW fact, not the
    # deprecated one — the slot was freed.
    fresh = mem.add_fact("api", "runs on", "Go")
    assert fresh.fact_id != old.fact_id


def test_retire_moves_body_to_singularity(tmp_path):
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    f = mem.add_fact("legacy feature", "lived in", "v1")

    result = mem.retire_fact(f.fact_id)
    assert result["retired"] is True
    assert f.fact_id in result["absorbed"]

    assert f.fact_id not in mem._facts
    assert f.fact_id in mem._hole._singularity


def test_supersede_unknown_id_is_a_noop(tmp_path):
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    out = mem.supersede_fact("missing", "also-missing")
    assert out["superseded"] is False


def test_retire_unknown_id_is_a_noop(tmp_path):
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    out = mem.retire_fact("missing")
    assert out["retired"] is False


def test_delete_fact_remains_destructive(tmp_path):
    """The hard primitive must still exist for secrets / GDPR."""
    db = str(tmp_path / "m.db")
    mem = MemoryStore(db_path=db)
    f = mem.add_fact("secret", "is", "redacted")
    assert mem.delete_fact(f.fact_id) is True

    # Gone from live, gone from the singularity, gone from storage.
    assert f.fact_id not in mem._facts
    assert f.fact_id not in mem._hole._singularity
    mem.close()

    again = MemoryStore(db_path=db)
    assert again._storage is not None
    loaded_ids = {x.fact_id for x in again._storage.load_facts()}
    assert f.fact_id not in loaded_ids
