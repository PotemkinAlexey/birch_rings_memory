"""EchoStore.expire — TTL sweep that bounds the store without dropping signal."""
from __future__ import annotations

import time

from birch.resonance.echo import (
    EchoStore,
    StoredSession,
    TTL_DEFAULT,
    TTL_PENALIZED,
    TTL_RESOLVED,
)
from birch.resonance.cluster import ClusterBundle


def _store_with(sessions: list[StoredSession]) -> EchoStore:
    store = EchoStore()
    for s in sessions:
        store._sessions[s.session_id] = s
    return store


def _stale(session_id: str, age: float, r: float = 0.0, penalty: float = 0.0) -> StoredSession:
    return StoredSession(
        session_id=session_id,
        bundle=ClusterBundle(centroids=[[1.0, 0.0]], k=1, inertia=0.0),
        r_score=r,
        fact_weights={},
        timestamp=time.time() - age,
        echo_penalty=penalty,
    )


def test_resolved_sessions_drop_after_short_ttl():
    """A session that closed resonant a week ago has no echo value."""
    store = _store_with([
        _stale("recent",      age=3 * 24 * 3600, r=0.8),     # 3 days, keep
        _stale("week-stale",  age=TTL_RESOLVED + 1, r=0.8),  # > 7 days, drop
    ])
    dropped = store.expire()
    assert dropped == ["week-stale"]
    assert "recent" in store._sessions
    assert "week-stale" not in store._sessions


def test_penalized_sessions_kept_longer_than_resolved():
    """An already-echoed session still survives to its own TTL tier."""
    store = _store_with([
        _stale("penalized-1week", age=8 * 24 * 3600, r=0.5, penalty=-0.6),  # < 14d
        _stale("penalized-3week", age=TTL_PENALIZED + 1, r=0.5, penalty=-0.6),
    ])
    dropped = store.expire()
    assert dropped == ["penalized-3week"]
    assert "penalized-1week" in store._sessions


def test_neutral_sessions_capped_at_default_ttl():
    """Everything else lasts 30 days, then it's just memory pressure."""
    store = _store_with([
        _stale("midlife",  age=20 * 24 * 3600, r=0.0),       # 20d, keep
        _stale("ancient",  age=TTL_DEFAULT + 1, r=0.0),      # > 30d, drop
    ])
    dropped = store.expire()
    assert dropped == ["ancient"]


def test_expire_returns_empty_when_nothing_stale():
    store = _store_with([_stale("fresh", age=10.0, r=0.5)])
    assert store.expire() == []
    assert len(store) == 1


def test_expire_with_custom_thresholds_is_honored():
    store = _store_with([_stale("aging", age=120.0, r=0.0)])
    assert store.expire(ttl_default=60.0) == ["aging"]


def test_expire_runs_from_session_close_and_drops_in_storage(tmp_path):
    """The lifecycle wiring: session_close calls expire() and persists."""
    from birch.memory_store import MemoryStore

    db = tmp_path / "echo-ttl.db"
    mem = MemoryStore(db_path=str(db))

    # Plant a stale session directly in both stores.
    stale = _stale("stale-from-storage", age=TTL_DEFAULT + 100)
    mem._echo._sessions[stale.session_id] = stale
    mem._storage.save_echo_session(
        stale.session_id,
        stale.bundle.centroids,
        stale.r_score,
        stale.timestamp,
    )
    assert any(
        r["session_id"] == "stale-from-storage"
        for r in mem._storage.load_echo_sessions()
    )

    # Close a fresh session — this triggers expire() in the lifecycle.
    mem.session_start("fresh")
    mem.session_message("a normal session message", session_id="fresh")
    mem.session_close(session_id="fresh")

    rows = mem._storage.load_echo_sessions()
    sids = {r["session_id"] for r in rows}
    assert "stale-from-storage" not in sids, \
        "session_close() should expire and delete stale echo session from disk"
    assert "stale-from-storage" not in mem._echo._sessions
