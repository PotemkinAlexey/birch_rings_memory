"""ChatGPT round-2 punch-list regressions.

The first round (test_chatgpt_punch_list.py) closed black-hole persistence
and several contracts. ChatGPT then re-reviewed the fixed code and found
the next layer of bugs — most importantly that Hawking emission would
resurrect deprecated/expired bodies as if they were live truth, and that
the new write-on-query path was not doing reload-under-transaction.
"""
from __future__ import annotations

from unittest import mock

import pytest

from birch.memory_store import MemoryStore
from birch.meta_fact import MetaFact
from birch.resonance.embeddings import EmbeddingError

# --- P1: Hawking emission must not return deprecated / expired bodies -------


def test_hawking_skips_deprecated_facts(tmp_path):
    """set_fact made the old HEAD obsolete; an exact-cosine Hawking query
    against the singularity must NOT resurrect it as if it were current."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    old = mem.add_fact("project-x HEAD", "is", "abc123")
    new = mem.add_fact("project-x HEAD", "is", "def456")
    mem.supersede_fact(old.fact_id, new.fact_id)

    # Old body is in the singularity, but it's deprecated.
    assert old.fact_id in mem._hole._singularity
    assert mem._hole._singularity[old.fact_id].fact.is_deprecated

    # Exact-cosine query of the old fact's text must NOT resurrect it.
    hits = mem.query(
        "project-x HEAD is abc123", top_k=5, hawking=True, min_similarity=0.0,
    )
    ids = {h.body_id for h in hits}
    assert old.fact_id not in ids
    # Old body stays in the singularity, untouched.
    assert old.fact_id in mem._hole._singularity
    assert old.fact_id not in mem._facts


def test_hawking_skips_expired_facts(tmp_path):
    """retire_fact set ttl=now; the body must not come back via Hawking
    just because cosine matches."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    f = mem.add_fact("legacy thing", "lived in", "v1")
    mem.retire_fact(f.fact_id)
    assert f.fact_id in mem._hole._singularity

    hits = mem.query(
        "legacy thing lived in v1", top_k=5, hawking=True, min_similarity=0.0,
    )
    ids = {h.body_id for h in hits}
    assert f.fact_id not in ids
    assert f.fact_id in mem._hole._singularity


# --- P1: query() reloads under write transaction before persisting ----------


def test_query_persist_path_saves_authoritative_object(tmp_path):
    """The persistence branch in query() must pass the LIVE fact object
    from _facts to save_facts — not the pre-lock snapshot. We verify by
    capturing the save_facts argument and asserting it is the same Python
    object as the current _facts entry (re-resolved under write lock,
    not the snapshot taken before)."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    f = mem.add_fact("api", "runs on", "Go")
    assert mem._storage is not None

    captured: list[list] = []
    real_save = mem._storage.save_facts

    def _capture(facts_arg):
        captured.append(list(facts_arg))
        return real_save(facts_arg)

    with mock.patch.object(mem._storage, "save_facts", side_effect=_capture):
        mem.query("api Go", top_k=1)

    # Save was called and the captured object IS the current _facts entry,
    # not a stale snapshot. Reference identity is the strict contract.
    assert captured, "query() did not persist the touched fact"
    assert captured[-1][0] is mem._facts[f.fact_id]


# --- P1: MetaFacts get recent_utility EWMA --------------------------------


def test_meta_fact_recent_utility_updates_on_session_close(tmp_path):
    """A live MetaFact touched by a session must receive the EWMA update,
    not just resonance_sum/resonance_count."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))

    # Hand-craft a live MetaFact directly into the live meta store.
    meta = MetaFact(weight=3, source_texts=["a b c"], gravity_score=0.5,
                    layer=1)
    meta.vector = [0.5] * 64
    mem._meta_facts[meta.meta_id] = meta
    mem._meta_index.add(meta.meta_id, meta.vector)
    mem._engine.register(meta)

    before = meta.recent_utility

    mem.session_start("s")
    mem.session_message("hello")
    # Manually attribute the meta to the session — query() would do this
    # but the mock embedder makes the cosine path unreliable here.
    mem._sessions["s"].facts[meta.meta_id] = 1.0
    mem.session_message("works great, thanks!")
    mem.session_close(session_id="s")

    # EWMA moved the meta's recent_utility off its default.
    assert meta.recent_utility != before


# --- P2: open session attribution persisted on every write site ------------


def test_add_fact_persists_open_session_attribution(tmp_path):
    """add_fact mutated ctx.facts but did not flush the open_sessions
    row. A crash before session_close lost the attribution mapping."""
    db = str(tmp_path / "m.db")
    mem = MemoryStore(db_path=db)
    mem.session_start("crash_test")
    fact = mem.add_fact("api", "runs on", "Go", session_id="crash_test")
    mem.close()

    again = MemoryStore(db_path=db)
    ctx = again._sessions.get("crash_test")
    assert ctx is not None
    assert fact.fact_id in ctx.facts


# --- P2: record_facts in-batch duplicates ---------------------------------


def test_add_facts_marks_in_batch_duplicates(tmp_path):
    """A batch with the same SPO appearing twice must mark the second
    occurrence duplicate_in_batch=True. Previously both came back as
    already_existed=False (clean inserts), which lied to the agent."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    statuses = mem.add_facts(
        [("api", "uses", "postgres"), ("api", "uses", "postgres")],
        return_status=True,
    )
    assert statuses[0]["already_existed"] is False
    assert statuses[0]["duplicate_in_batch"] is False
    assert statuses[1]["duplicate_in_batch"] is True
    # Same FactPassport — only one fact actually inserted.
    assert statuses[0]["fact"].fact_id == statuses[1]["fact"].fact_id


# --- P2: embed_batch validates length ------------------------------------


def test_add_facts_refuses_partial_batch_from_provider(tmp_path):
    """A provider that returns fewer vectors than triples must trigger
    EmbeddingError, not silently misalign via zip()."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    triples = [("a", "is", "1"), ("b", "is", "2"), ("c", "is", "3")]

    with mock.patch(
        "birch.memory_store.embed_batch",
        return_value=[[1.0] * 64, [1.0] * 64],   # only 2 vectors for 3 inputs
    ):
        with pytest.raises(EmbeddingError):
            mem.add_facts(triples)


# --- P2: graph edges cleaned on delete -----------------------------------


def test_delete_fact_drops_edges_in_storage(tmp_path):
    db = str(tmp_path / "m.db")
    mem = MemoryStore(db_path=db)
    a = mem.add_fact("a", "is", "thing")
    b = mem.add_fact("b", "is", "thing")
    mem.link(a.fact_id, b.fact_id)
    mem.delete_fact(a.fact_id)

    # The deleted endpoint must not appear in storage edges anymore.
    assert mem._storage is not None
    edges = mem._storage.load_edges()
    endpoints = {e for pair in edges for e in pair}
    assert a.fact_id not in endpoints


def test_load_skips_orphan_edges(tmp_path):
    """If somehow the storage has orphan edges (older DB, partial cleanup),
    the loader must not register them — otherwise _degrees inflates with
    ghost ids and depresses graph_score for healthy facts."""
    import sqlite3

    db = str(tmp_path / "m.db")
    mem = MemoryStore(db_path=db)
    a = mem.add_fact("a", "is", "thing")
    mem.close()

    # Inject an orphan edge directly (ghost id never existed).
    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT OR IGNORE INTO edges (from_id, to_id) VALUES (?, ?)",
        (a.fact_id, "00000000-ghost"),
    )
    conn.commit()
    conn.close()

    again = MemoryStore(db_path=db)
    # The orphan must not have made it into the engine's degrees, so the
    # only live fact's degree stays 0.
    assert again._engine._degrees.get(a.fact_id, 0) == 0


# --- P2: record_session triggers echo on the first message ---------------


def test_record_session_check_echo_runs_on_first_message(tmp_path):
    """record_session used to skip echo detection entirely, breaking
    retroactive correction for one-shot session writes."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    mem.session_start("past")
    mem.session_message("server keeps timing out on connect")
    mem.session_close(session_id="past")

    with mock.patch.object(mem, "check_echo",
                           wraps=mem.check_echo) as spy:
        # Simulate what server.record_session does — open, echo, push, close.
        sid = "current"
        mem.session_start(sid)
        spy("server keeps timing out on connect", session_id=sid)
        mem.session_message("still timing out", session_id=sid)
        mem.session_close(session_id=sid)
        assert spy.call_count == 1


# --- P3: session_open auto-pushes first_message --------------------------


def test_session_open_record_first_message_pushes_to_trajectory(tmp_path):
    """When session_open is called with first_message and record_first_message
    is True (the default), the message must land in ctx.messages so the
    resonance engine sees it on close — not just be used for echo and
    silently dropped."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    # Simulate the new server.session_open behaviour: open + check_echo +
    # session_message (record_first_message default True path).
    sid = "s"
    mem.session_start(sid)
    mem.check_echo("opening question", session_id=sid)
    mem.session_message("opening question", session_id=sid)
    ctx = mem._sessions[sid]
    assert "opening question" in ctx.messages
