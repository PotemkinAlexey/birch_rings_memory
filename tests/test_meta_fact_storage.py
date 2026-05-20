"""SQLite persistence for MetaFact."""
from birch.meta_fact import MetaFact
from birch.storage.sqlite import SQLiteBackend


def test_save_and_load_meta_fact_round_trips(tmp_path):
    db = tmp_path / "meta.db"
    backend = SQLiteBackend(str(db))

    m = MetaFact(
        meta_id="m-rt",
        vector=[0.1, 0.2, 0.3],
        weight=5,
        source_texts=["a b c", "d e f"],
        source_fact_ids=["id-a", "id-b", "id-c"],
        summary="combined",
        gravity_score=0.41,
        layer=-1,
        access_count=2,
        resonance_sum=0.6,
        resonance_count=3,
    )
    backend.save_meta_fact(m)
    backend.close()

    # Re-open the DB and load.
    reopened = SQLiteBackend(str(db))
    loaded = reopened.load_meta_facts()
    assert len(loaded) == 1
    restored = loaded[0]
    assert restored.meta_id == m.meta_id
    assert restored.vector == m.vector
    assert restored.weight == m.weight
    assert restored.source_texts == m.source_texts
    assert restored.source_fact_ids == m.source_fact_ids
    assert restored.summary == m.summary
    assert restored.gravity_score == m.gravity_score
    assert restored.layer == m.layer
    assert restored.access_count == m.access_count
    assert restored.resonance_sum == m.resonance_sum
    assert restored.resonance_count == m.resonance_count


def test_save_meta_facts_batch_uses_one_transaction(tmp_path):
    db = tmp_path / "meta_batch.db"
    backend = SQLiteBackend(str(db))

    metas = [
        MetaFact(meta_id=f"m-{i}", vector=[float(i), 0.0], weight=i + 1)
        for i in range(5)
    ]
    backend.save_meta_facts(metas)
    backend.close()

    reopened = SQLiteBackend(str(db))
    loaded = {m.meta_id: m for m in reopened.load_meta_facts()}
    assert set(loaded.keys()) == {f"m-{i}" for i in range(5)}
    assert loaded["m-3"].weight == 4
    assert loaded["m-3"].vector == [3.0, 0.0]


def test_delete_meta_fact(tmp_path):
    db = tmp_path / "meta_del.db"
    backend = SQLiteBackend(str(db))
    m = MetaFact(meta_id="to-drop", vector=[1.0, 0.0])
    backend.save_meta_fact(m)
    assert len(backend.load_meta_facts()) == 1

    backend.delete_meta_fact("to-drop")
    assert backend.load_meta_facts() == []


def test_load_returns_empty_when_no_meta_facts(tmp_path):
    backend = SQLiteBackend(str(tmp_path / "empty.db"))
    assert backend.load_meta_facts() == []


def test_meta_facts_table_added_on_existing_db_via_executescript(tmp_path):
    """CREATE TABLE IF NOT EXISTS lets older DBs upgrade transparently."""
    import sqlite3

    db = tmp_path / "legacy.db"
    # Simulate a pre-MetaFact DB: open with sqlite3 directly and create only
    # the older tables.
    conn = sqlite3.connect(str(db))
    conn.executescript(
        "CREATE TABLE facts (fact_id TEXT PRIMARY KEY);"
        "CREATE TABLE edges (from_id TEXT, to_id TEXT);"
        "CREATE TABLE echo_sessions (session_id TEXT PRIMARY KEY);"
        "CREATE TABLE open_sessions (session_id TEXT PRIMARY KEY);"
    )
    conn.commit()
    conn.close()

    # SQLiteBackend should add meta_facts on open without error.
    backend = SQLiteBackend(str(db))
    backend.save_meta_fact(MetaFact(meta_id="post-migration", vector=[1.0]))
    assert any(m.meta_id == "post-migration" for m in backend.load_meta_facts())
