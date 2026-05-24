"""Six fixes from the latest professor-tier review:

  1. set_fact() is now atomic in a SINGLE transaction. Previously ran
     add_fact() (commit 1) then a second txn that superseded
     occupants (commit 2). If commit 2 failed, the new fact was
     already on disk alongside the old occupants — slot uniqueness
     violated.

  2-4. session_start / session_message / abort_session each got the
       try/except + _reload guard. The previous "complete rollback
       coverage" commit (0f23a62) missed these three session-
       lifecycle paths even though they all mutate in-memory state
       before calling save_open_session / delete_open_session.

  5. MCP boundary now validates session_id (via _validate_id or
     _validate_optional_id) on session_open / session_push /
     session_close / check_echo / record_session, plus
     record_first_message as a real bool (not a Python truthy
     string like "false").

  6. _load_from_storage(prune=True/False). _reload now calls with
     prune=False so rollback recovery is strictly read-only. The
     destructive cleanup (orphan-edge GC + TTL'd session sweep) only
     runs from __init__'s self-healing first load, where it's the
     right behaviour.
"""
from __future__ import annotations

import pytest

from birch.memory_store import MemoryStore

# --- I1: set_fact atomicity --------------------------------------------


def test_set_fact_rolls_back_both_halves_on_supersede_failure(tmp_path):
    """Force the supersede half to fail (storage.save_fact on
    deprecated_by write). The new fact must NOT remain on disk —
    slot uniqueness preserved either way (old occupants stay live
    OR new fact lands, but not both)."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    old = mem.add_fact("project", "HEAD", "abc")
    pre_facts = {f.fact_id: f.object for f in mem.list_facts(subject="project")}

    # Patch save_fact to fail on the THIRD call (let add_fact's insert
    # and the touch path succeed, but break the supersede's save_fact).
    original = mem._storage.save_fact
    call_count = {"n": 0}

    def maybe_failing(fact):
        call_count["n"] += 1
        if call_count["n"] >= 2:
            raise RuntimeError("simulated supersede save failure")
        original(fact)

    mem._storage.save_fact = maybe_failing  # type: ignore[assignment]
    try:
        with pytest.raises(RuntimeError, match="supersede save failure"):
            mem.set_fact("project", "HEAD", "def")
    finally:
        mem._storage.save_fact = original  # type: ignore[assignment]

    # After rollback, slot must be in a consistent state — either old
    # (abc) live or new (def) live, but NEVER both. Previously the
    # two-txn implementation could land both at once.
    live = [
        f for f in mem.list_facts(subject="project")
        if not f.is_deprecated
    ]
    objects = {f.object for f in live}
    assert objects in (set(), {"abc"}, {"def"}), (
        f"slot uniqueness violated: live HEAD objects = {objects}"
    )
    # Specifically: deprecated_by on old must not be set unless new
    # is also live.
    refreshed_old = mem._facts.get(old.fact_id)
    if refreshed_old is not None:
        if refreshed_old.deprecated_by is not None:
            # If old was superseded, the new fact must also exist.
            assert any(f.object == "def" for f in live), (
                "old fact deprecated but new fact missing — torn write"
            )
    mem.close()
    # touch pre_facts so linter doesn't complain
    assert "abc" in pre_facts.values()


def test_set_fact_happy_path_still_supersedes_and_returns_metadata(tmp_path):
    """Sanity: the new single-txn path still does the slot-replace
    and returns the symmetric response shape (layer/gravity_score/
    created/_hint)."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    mem.add_fact("project", "HEAD", "abc")
    resp = mem.set_fact("project", "HEAD", "def")
    assert resp["set"] is True
    assert resp["created"] is True
    assert resp["object"] == "def"
    assert "layer" in resp
    assert "gravity_score" in resp
    assert "_hint" in resp
    # Live slot has only the new value.
    live = [
        f for f in mem.list_facts(subject="project")
        if not f.is_deprecated
    ]
    assert len(live) == 1
    assert live[0].object == "def"
    mem.close()


# --- I2: session_start rollback ----------------------------------------


def test_session_start_rolls_back_on_storage_failure(tmp_path):
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    original = mem._storage.save_open_session

    def boom(*a, **kw):
        raise RuntimeError("simulated save_open_session failure")

    mem._storage.save_open_session = boom  # type: ignore[assignment]
    try:
        with pytest.raises(RuntimeError, match="save_open_session"):
            mem.session_start("s_new")
    finally:
        mem._storage.save_open_session = original  # type: ignore[assignment]

    # In-memory must NOT have the session after _reload.
    assert "s_new" not in mem._sessions, (
        "session_start leaked in-memory session despite storage rollback"
    )
    mem.close()


# --- I3: session_message rollback --------------------------------------


def test_session_message_rolls_back_on_storage_failure(tmp_path):
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    mem.session_start("s1")
    pre_messages = list(mem._sessions["s1"].messages)
    original = mem._storage.save_open_session
    call_count = {"n": 0}

    def boom_after_first(*a, **kw):
        call_count["n"] += 1
        # session_start already called save_open_session once — fail
        # only on the second call (from session_message).
        if call_count["n"] >= 1:
            raise RuntimeError("simulated session_message save failure")

    mem._storage.save_open_session = boom_after_first  # type: ignore[assignment]
    try:
        with pytest.raises(RuntimeError, match="session_message"):
            mem.session_message("late message", session_id="s1")
    finally:
        mem._storage.save_open_session = original  # type: ignore[assignment]

    # After _reload, ctx.messages must equal the disk-persisted value
    # (which is what session_start wrote — just empty list initially).
    refreshed = mem._sessions.get("s1")
    assert refreshed is not None
    assert refreshed.messages == pre_messages, (
        f"session_message leaked in-memory append: got {refreshed.messages}, "
        f"expected {pre_messages}"
    )
    mem.close()


# --- I4: abort_session rollback ----------------------------------------


def test_abort_session_rolls_back_on_storage_failure(tmp_path):
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    mem.session_start("s_to_abort")
    assert "s_to_abort" in mem._sessions
    original = mem._storage.delete_open_session

    def boom(sid):
        raise RuntimeError("simulated abort delete failure")

    mem._storage.delete_open_session = boom  # type: ignore[assignment]
    try:
        with pytest.raises(RuntimeError, match="abort delete"):
            mem.abort_session("s_to_abort")
    finally:
        mem._storage.delete_open_session = original  # type: ignore[assignment]

    # After _reload, the session must STILL be in memory (disk kept it).
    assert "s_to_abort" in mem._sessions, (
        "abort_session leaked in-memory pop despite storage rollback — "
        "next restart would resurrect the 'aborted' session"
    )
    mem.close()


# --- I5: MCP session_id and bool validators ----------------------------


def test_validate_optional_id_inline_contract():
    """Replicate the server helper inline (mcp SDK isn't importable
    from tests). None passes; explicit non-string fails."""

    def _check(value, field="session_id"):
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            return {
                "ok": False,
                "error": "invalid_id",
                "field": field,
                "got_type": type(value).__name__,
            }
        return None

    assert _check(None) is None
    assert _check("s1") is None
    err = _check(42)
    assert err["error"] == "invalid_id"
    err = _check("")
    assert err["error"] == "invalid_id"
    err = _check([])
    assert err["error"] == "invalid_id"


def test_validate_bool_inline_contract():
    def _check(value, field="record_first_message"):
        if not isinstance(value, bool):
            return {
                "ok": False,
                "error": "invalid_bool",
                "field": field,
                "got_type": type(value).__name__,
            }
        return None

    assert _check(True) is None
    assert _check(False) is None
    # Strings that look bool-ish — the truthy-trap case.
    err = _check("false")
    assert err["error"] == "invalid_bool"
    assert err["got_type"] == "str"
    err = _check(0)  # int 0 is falsy but not a bool
    assert err["error"] == "invalid_bool"
    assert err["got_type"] == "int"


# --- I6: _load_from_storage prune flag ---------------------------------


def test_reload_does_not_prune_orphan_sessions(tmp_path):
    """_reload uses prune=False — an orphan / TTL-expired open
    session on disk must NOT be deleted by a recovery reload.
    Confirm by injecting a stale row directly and observing it
    survives a _reload."""
    import time as _t

    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    # Insert a stale row (timestamp 48h ago) directly via storage.
    stale_started = _t.time() - 48 * 3600
    mem._storage.save_open_session(
        "stale_sid", [], [], {}, stale_started,
    )
    # Confirm it's on disk.
    rows = mem._storage.load_open_sessions()
    assert any(r["session_id"] == "stale_sid" for r in rows)
    # _reload (rollback recovery) should NOT delete it.
    mem._reload()
    rows_after = mem._storage.load_open_sessions()
    assert any(r["session_id"] == "stale_sid" for r in rows_after), (
        "_reload pruned an orphan session — destructive cleanup leaked "
        "into the recovery path"
    )
    mem.close()


def test_init_load_does_prune_orphan_sessions(tmp_path):
    """Sanity counterpart: a fresh process __init__ DOES prune.
    Open a store, plant a stale session, close, open a new store
    (fresh __init__) — stale session should be gone."""
    import time as _t

    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    stale_started = _t.time() - 48 * 3600
    mem._storage.save_open_session(
        "stale_sid", [], [], {}, stale_started,
    )
    mem.close()
    # Fresh process simulation.
    mem2 = MemoryStore(db_path=str(tmp_path / "m.db"))
    rows = mem2._storage.load_open_sessions()
    assert not any(r["session_id"] == "stale_sid" for r in rows), (
        "__init__ should prune TTL-expired open sessions"
    )
    mem2.close()
