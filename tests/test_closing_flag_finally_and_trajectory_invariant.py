"""Four Vader torpedoes shipped together:

  1. ``_closing_sessions`` flag is now wrapped in try/finally that
     guarantees discard on EVERY exit path. Pre-fix, r_override /
     sentiment validation raised ValueError BETWEEN the flag-set
     and the writeback try/except, so the flag would stick and
     permanently brick the sid for any future session_message.
     The fix moves validation BEFORE the snapshot so bad input
     never sets the flag in the first place, AND wraps the whole
     post-snapshot block in try/finally as belt-and-suspenders.

  2. MCP session_push now catches the RuntimeError("session_closing")
     that the per-sid gate raises. Pre-fix the structured rejection
     existed at the core level but leaked as a raw exception to
     the MCP boundary.

  3. SQLiteBackend.load_open_sessions drops rows where
     ``len(messages) != len(vectors)``. The trajectory invariant
     was not enforced — a corrupted row with messages=["hi"] and
     vectors=[] used to load and then crash session_close with
     IndexError on vectors_snapshot[0]. Plus core session_close
     re-checks the invariant BEFORE marking the flag so even if
     a row sneaks past the loader, the flag never sticks.

  4. The stale comment in session_close about "late messages
     remain available as the agent's open ctx but are effectively
     dropped" no longer matches reality — late messages are
     REJECTED by the _closing_sessions gate now. Comment
     refreshed to reflect current behaviour.

DEFERRED (same as previous round): record_session non-atomic echo.
The architectural decision still requires user input on which of
three resolutions to ship (reorder; partial flag; opt-out kwarg);
not picking one unilaterally.
"""
from __future__ import annotations

import pytest

from birch import server as srv
from birch.memory_store import MemoryStore
from birch.storage.sqlite import SQLiteBackend

# --- I1: closing flag finally ---------------------------------------


def test_bad_sentiment_does_not_brick_closing_flag(tmp_path):
    """A ValueError from sentiment validation must NOT leave the
    sid in _closing_sessions. Pre-fix, the flag stuck and every
    future session_message raised 'session_closing'."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    mem.session_start("s1")
    mem.session_message("hello", session_id="s1")

    with pytest.raises(ValueError, match="sentiment"):
        mem.session_close(session_id="s1", sentiment="bogus_label")

    # The bad input was caught BEFORE the flag was set — sid is
    # still usable for a follow-up message or a clean close.
    assert "s1" not in mem._closing_sessions
    # And a follow-up push succeeds.
    mem.session_message("retry", session_id="s1")
    mem.close()


def test_bad_r_override_does_not_brick_closing_flag(tmp_path):
    """Same contract for r_override: NaN / non-numeric raises but
    leaves the sid clean."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    mem.session_start("s1")
    mem.session_message("hello", session_id="s1")

    with pytest.raises(ValueError, match="r_override"):
        mem.session_close(session_id="s1", r_override=float("nan"))
    assert "s1" not in mem._closing_sessions

    with pytest.raises(ValueError, match="r_override"):
        mem.session_close(session_id="s1", r_override="garbage")
    assert "s1" not in mem._closing_sessions
    mem.close()


def test_flag_clears_after_any_writeback_exception(tmp_path):
    """If the storage write inside session_close fails, the finally
    block must still discard the flag — symmetric with the existing
    except path but now via finally so it covers every exception
    class, not just the ones the inner except catches."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    mem.session_start("s1")
    mem.session_message("hello", session_id="s1")
    original = mem._storage.save_facts

    def boom(*a, **kw):
        raise RuntimeError("simulated storage failure")

    mem._storage.save_facts = boom
    try:
        with pytest.raises(RuntimeError, match="simulated"):
            mem.session_close(session_id="s1", sentiment="resonant")
    finally:
        mem._storage.save_facts = original
    # Flag cleared by the new finally block, sid retryable.
    assert "s1" not in mem._closing_sessions
    mem.close()


def test_flag_clears_on_successful_close(tmp_path):
    """Sanity: happy path also clears the flag via finally."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    mem.session_start("s1")
    mem.session_message("hello", session_id="s1")
    mem.session_close(session_id="s1", sentiment="resonant")
    assert "s1" not in mem._closing_sessions
    mem.close()


# --- I2: MCP session_push catches RuntimeError ----------------------


def test_mcp_session_push_returns_session_closing_structured():
    """Source-level wiring check: server.session_push must catch the
    RuntimeError("session_closing") and return a structured response,
    not let the raw exception escape."""
    import pathlib
    import re
    src = pathlib.Path(srv.__file__).read_text()
    m = re.search(r"^def session_push\(", src, re.MULTILINE)
    assert m is not None
    next_m = re.compile(r"^(def |@)", re.MULTILINE).search(
        src, m.end(),
    )
    body = src[m.start():next_m.start() if next_m else len(src)]
    # New except branch:
    assert "except RuntimeError" in body
    assert "session_closing" in body
    assert "ok" in body and "False" in body


# --- I3: trajectory invariant in loader + core ----------------------


def test_load_open_sessions_drops_mismatched_messages_vectors(tmp_path):
    """A row with messages=["hi"] and vectors=[] used to pass the
    loader (each individually well-formed) and crash session_close
    on vectors_snapshot[0]. Drop the row at the loader."""
    import json
    import sqlite3

    db = str(tmp_path / "m.db")
    backend = SQLiteBackend(db)
    backend.close()

    # Hand-craft a corrupted row directly.
    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT INTO open_sessions VALUES (?,?,?,?,?)",
        (
            "corrupt-sid",
            json.dumps(["hello", "world"]),    # 2 messages
            json.dumps([[0.1, 0.2, 0.3]]),     # 1 vector
            json.dumps({}),
            100.0,
        ),
    )
    conn.commit()
    conn.close()

    # Reopen — the loader must drop the bad row.
    backend = SQLiteBackend(db)
    sessions = backend.load_open_sessions(cleanup=False)
    assert all(s["session_id"] != "corrupt-sid" for s in sessions), (
        "session with mismatched messages/vectors lengths should be "
        "dropped at the loader, not loaded as a half-state ctx"
    )
    backend.close()


def test_core_session_close_rejects_mismatched_trajectory(tmp_path):
    """Even if a row sneaks past the loader (e.g. legacy DB before
    the invariant), session_close's pre-snapshot check refuses to
    set the closing flag on a mismatched ctx — the flag never
    sticks."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    mem.session_start("s1")
    # Hand-corrupt the in-memory ctx to simulate a slipped-through
    # row (loader-bypass scenario).
    ctx = mem._sessions["s1"]
    ctx.messages = ["hi", "there"]
    ctx.vectors = []   # mismatch

    with pytest.raises(ValueError, match="trajectory corrupted"):
        mem.session_close(session_id="s1", sentiment="resonant")
    # And the flag never got set — sid still recoverable via
    # abort_session.
    assert "s1" not in mem._closing_sessions
    mem.close()


# --- I4: stale comment refreshed (source-level audit) ---------------


def test_session_close_comment_does_not_lie_about_late_messages():
    """The pre-fix comment said late messages 'remain available as
    the agent's open ctx but are effectively dropped'. After the
    _closing_sessions gate, late messages are REJECTED outright.
    Audit ensures the misleading wording is gone."""
    import pathlib
    src = pathlib.Path(
        "src/birch/memory_store/_sessions.py"
    ).read_text() if pathlib.Path(
        "src/birch/memory_store/_sessions.py"
    ).exists() else None
    if src is None:
        # Fall back to module file resolution.
        import birch.memory_store._sessions as sm
        src = pathlib.Path(sm.__file__).read_text()
    assert "effectively dropped from THIS round" not in src
    assert "push them BEFORE session_close" not in src
    # And the new wording is present.
    assert "REJECTED entirely by the" in src or \
           "_closing_sessions gate in session_message" in src
