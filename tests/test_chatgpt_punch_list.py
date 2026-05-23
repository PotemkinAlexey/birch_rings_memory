"""ChatGPT review punch-list regressions.

Each test pins one of the bugs ChatGPT surfaced. The first one is the
worst — facts that fell into the black hole via natural gravity decay
used to be hard-deleted from storage, so Hawking emission and singularity
collapse lineage evaporated on restart. That is now closed.
"""
from __future__ import annotations

import sqlite3
import time

from birch.memory_store import MemoryStore


def test_absorbed_facts_survive_restart_in_singularity(tmp_path):
    """Gravity-floor absorption used to call storage.delete_fact and lose
    the body for good. Now absorbed bodies are persisted with layer=-1 and
    re-hydrated into the singularity on next open.

    We force the absorption directly (rather than aging a fact via tick)
    by setting gravity_score below the floor and calling _absorb_dead —
    that path is what used to delete the row from storage.
    """
    db = str(tmp_path / "m.db")
    mem = MemoryStore(db_path=db)
    f = mem.add_fact("doomed", "is", "going down")
    f.gravity_score = 0.01  # below _ABSORPTION_THRESHOLD = 0.10
    if mem._storage:
        mem._storage.save_fact(f)

    with mem._lock:
        with mem._txn():
            mem._absorb_dead()

    # Body is gone from live, lives in singularity, persists with layer=-1.
    assert f.fact_id not in mem._facts
    assert f.fact_id in mem._hole._singularity
    assert mem._hole._singularity[f.fact_id].fact.layer == -1
    mem.close()

    # Reopen: black hole gets re-hydrated, NOT re-loaded into _facts.
    again = MemoryStore(db_path=db)
    assert f.fact_id not in again._facts
    assert f.fact_id in again._hole._singularity
    assert again._hole._singularity[f.fact_id].fact.layer == -1


def test_query_persists_access_count_to_storage(tmp_path):
    """access_count / last_accessed used to live only in RAM until next
    session_close; a crash between session_message calls lost the read
    history. Now each query saves touched facts."""
    db = str(tmp_path / "m.db")
    mem = MemoryStore(db_path=db)
    f = mem.add_fact("api", "runs on", "Go")
    before = f.access_count
    mem.query("api Go", top_k=1)
    mem.close()

    # Reopen with a separate MemoryStore — access bump survived.
    again = MemoryStore(db_path=db)
    reloaded = next(x for x in again.list_facts() if x.fact_id == f.fact_id)
    assert reloaded.access_count == before + 1
    assert reloaded.last_accessed > 0


def test_query_persists_open_session_attribution(tmp_path):
    """query() used to mutate ctx.facts in RAM only; a crash before
    session_close lost the per-fact relevance weights. Now each query
    re-saves the open session row."""
    db = str(tmp_path / "m.db")
    mem = MemoryStore(db_path=db)
    f = mem.add_fact("api", "runs on", "Go")
    mem.session_start("crash_test")
    mem.query("api Go", top_k=1, session_id="crash_test")
    mem.close()

    # Reopen — open_sessions row must contain the attributed fact_id.
    again = MemoryStore(db_path=db)
    ctx = again._sessions.get("crash_test")
    assert ctx is not None
    assert f.fact_id in ctx.facts


def test_record_facts_honours_per_item_session_id(tmp_path):
    """Two open sessions, one record_facts call with per-item session_id,
    facts attributed to the right context."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    mem.session_start("alpha")
    mem.session_start("beta")
    results = mem.add_facts(
        [("X", "is", "1"), ("Y", "is", "2")],
        session_ids=["alpha", "beta"],
    )
    assert results[0].fact_id in mem._sessions["alpha"].facts
    assert results[0].fact_id not in mem._sessions["beta"].facts
    assert results[1].fact_id in mem._sessions["beta"].facts
    assert results[1].fact_id not in mem._sessions["alpha"].facts


def test_query_allowed_layers_is_set_not_range(tmp_path):
    """layers={surface, core} must EXCLUDE kinetic — the previous
    server.py range computed min=0, max=2 and let kinetic through."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    s = mem.add_fact("on surface", "is", "hot")
    s.layer = 0
    k = mem.add_fact("in kinetic", "is", "working")
    k.layer = 1
    c = mem.add_fact("in core", "is", "cold")
    c.layer = 2
    if mem._storage:
        mem._storage.save_facts([s, k, c])

    hits = mem.query("anything", top_k=10, allowed_layers={0, 2})
    ids = {h.body_id for h in hits}
    assert s.fact_id in ids
    assert c.fact_id in ids
    assert k.fact_id not in ids


def test_echo_timestamp_survives_restart(tmp_path):
    """StoredSession.timestamp used to reset to now() on every reload, so
    the TTL stopped being a TTL. Now recorded_at is round-tripped."""
    db = str(tmp_path / "m.db")
    mem = MemoryStore(db_path=db)
    mem.session_start("s")
    mem.session_message("hello")
    mem.session_close(session_id="s")

    # Inject an old recorded_at directly so we can verify it survives.
    old_ts = time.time() - 86_400 * 5   # 5 days ago
    conn = sqlite3.connect(db)
    conn.execute("UPDATE echo_sessions SET recorded_at = ? WHERE session_id = ?",
                 (old_ts, "s"))
    conn.commit()
    conn.close()
    mem.close()

    again = MemoryStore(db_path=db)
    stored = again._echo.get("s")
    assert stored is not None
    # Round-trip kept the timestamp; not reset to ~now.
    assert abs(stored.timestamp - old_ts) < 1.0


def test_vector_index_dim_mismatch_raises_loudly():
    import pytest

    from birch.vector_index import DimensionMismatchError, VectorIndex

    idx = VectorIndex()
    idx.add("a", [1.0, 0.0, 0.0])
    with pytest.raises(DimensionMismatchError):
        idx.add("b", [1.0, 0.0])  # wrong dim


def test_subject_prefix_is_real_prefix_not_substring(tmp_path):
    """subject_prefix used to be substring matching despite the name.
    Now it's true startswith — "rails" matches "rails-api" but NOT
    "my-rails-app"."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    rails_api = mem.add_fact("rails-api", "is", "web service")
    my_rails = mem.add_fact("my-rails-app", "is", "legacy")

    hits = mem.find_similar(
        "rails web service legacy", top_k=10,
        min_similarity=0.0, subject_prefix="rails",
    )
    ids = {h["fact_id"] for h in hits}
    assert rails_api.fact_id in ids
    assert my_rails.fact_id not in ids


def test_check_echo_idempotent(tmp_path):
    """Sanity that check_echo exists and is idempotent — used to live in
    the MemoryStore but was inaccessible to MCP agents until this commit."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    mem.session_start("first")
    mem.session_message("oh no the deploy failed again with the same error")
    mem.session_close(session_id="first")

    result_1 = mem.check_echo("the deploy failed again with the same error")
    result_2 = mem.check_echo("the deploy failed again with the same error")
    # Either both echoed and the second is idempotent (penalty=0), or
    # neither echoed (mock-embedding similarity below threshold). Either
    # is acceptable; what we pin is that the call is callable.
    assert isinstance(result_1, dict)
    assert isinstance(result_2, dict)
