"""Multi-process cache coherence.

Each MCP client spawns its own birch.server process, so several MemoryStore
instances share one SQLite file. These tests pin the contract that makes that
safe: a store reloads when another writes (data_version), writers serialize
(BEGIN IMMEDIATE), and a lone store never reloads — it stays hot.
"""
from birch.memory_store import MemoryStore
from birch.storage.sqlite import SQLiteBackend


def test_wal_mode_enabled(tmp_path):
    """WAL lets readers and the single writer run concurrently."""
    be = SQLiteBackend(str(tmp_path / "m.db"))
    mode = be._conn.execute("PRAGMA journal_mode").fetchone()[0]
    be.close()
    assert mode.lower() == "wal"


def test_store_reloads_when_another_writes(tmp_path):
    """A store with a stale cache sees a peer's write on its next operation."""
    db = str(tmp_path / "m.db")
    a = MemoryStore(db_path=db)
    b = MemoryStore(db_path=db)  # loaded empty, before a writes

    a.add_fact("alpha", "is", "first")

    # b's cache predates the write; list_facts() must _sync() and pick it up.
    assert any(f.subject == "alpha" for f in b.list_facts())


def test_concurrent_adds_both_survive(tmp_path):
    """Two stores writing in turn must not clobber each other."""
    db = str(tmp_path / "m.db")
    a = MemoryStore(db_path=db)
    b = MemoryStore(db_path=db)

    a.add_fact("alpha", "is", "first")
    b.add_fact("beta", "is", "second")  # b was stale; reloads inside the txn

    expected = {"alpha", "beta"}
    assert {f.subject for f in MemoryStore(db_path=db).list_facts()} == expected
    assert {f.subject for f in a.list_facts()} == expected
    assert {f.subject for f in b.list_facts()} == expected


def test_session_close_preserves_foreign_facts(tmp_path):
    """session_close does a full-dump save_facts — it must not lose facts a
    peer created after this store's cache was loaded."""
    db = str(tmp_path / "m.db")
    a = MemoryStore(db_path=db)
    a.add_fact("alpha", "is", "first")
    a.session_start("sa")
    a.session_message("hello there alpha")
    a.session_message("bye")

    b = MemoryStore(db_path=db)
    b.add_fact("beta", "is", "second")  # written after a's snapshot

    a.session_close("sa")  # full-dump write; must reload and keep beta

    assert {f.subject for f in MemoryStore(db_path=db).list_facts()} == {"alpha", "beta"}


def test_stale_process_does_not_clobber_resonance(tmp_path):
    """The original bug: a stale process' tick rewinds another's work.

    B applies resonance to a fact, then a stale A closes an unrelated session.
    A's full-dump write must build on B's state, not overwrite it with the
    pre-resonance value its cache still holds.
    """
    db = str(tmp_path / "m.db")
    a = MemoryStore(db_path=db)
    a.add_fact("birch", "is", "a tree")

    b = MemoryStore(db_path=db)
    b.session_start("sb")
    b.session_message("tell me about birch")
    b.query("birch is a tree", session_id="sb")  # attribute the fact to sb
    b.session_message("perfect, exactly what i needed")
    b.session_close("sb")  # resonant session → resonance applied to the fact

    rc_after_b = MemoryStore(db_path=db).list_facts()[0].resonance_count
    assert rc_after_b == 1

    # A's cache predates B's whole session.
    a.session_start("sa")
    a.session_message("something totally unrelated")
    a.session_message("ok")
    a.session_close("sa")

    rc_final = MemoryStore(db_path=db).list_facts()[0].resonance_count
    assert rc_final == 1, "stale A must not reset B's resonance_count"


def test_foreign_write_triggers_exactly_one_reload(tmp_path):
    """A peer's write reloads the cache once; further reads do not re-reload."""
    db = str(tmp_path / "m.db")
    a = MemoryStore(db_path=db)
    b = MemoryStore(db_path=db)

    reloads = []
    original = b._reload
    b._reload = lambda: (reloads.append(1), original())[1]

    a.add_fact("alpha", "is", "first")
    b.list_facts()  # detects a's commit → one reload
    b.list_facts()  # nothing changed → no reload
    assert len(reloads) == 1


def test_solo_store_never_reloads(tmp_path):
    """With a single process, data_version never moves — the store stays hot
    and never pays a reload, no matter how many writes it makes itself."""
    db = str(tmp_path / "m.db")
    a = MemoryStore(db_path=db)

    reloads = []
    original = a._reload
    a._reload = lambda: (reloads.append(1), original())[1]

    a.add_fact("alpha", "is", "first")
    a.add_fact("beta", "is", "second")
    a.add_facts([("gamma", "is", "third"), ("delta", "is", "fourth")])
    a.query("alpha")
    a.list_facts()
    a.session_start("s")
    a.session_message("a message")
    a.session_close("s")

    assert reloads == []
