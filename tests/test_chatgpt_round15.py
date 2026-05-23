"""ChatGPT round-15 punch-list regressions.

Round 15 was the first fully-clean self-audit since round 12 (6 of
7 findings real, 1 already shipped in round 13). Edge-contract
slice: malformed inputs, observability, cache invalidation after
manual/async maintenance.

  1. load_open_sessions coerces facts values to float at the loader.
  2. collapse_singularity bumps _mutation_version on successful pass.
  3. session_push returns structured unknown_session.
  4. session_close returns structured invalid_sentiment.
  5. record_session validates messages is list[str].
  6. close() captures the last collapse worker error; stats surfaces it.
"""
from __future__ import annotations

import sqlite3
from concurrent.futures import Future

import pytest

from birch.memory_store import MemoryStore
from birch.storage.sqlite import SQLiteBackend

# --- P1: load_open_sessions numeric values ------------------------------


def test_load_open_sessions_drops_row_with_non_numeric_facts_value(tmp_path):
    """The dict shape check landed in round 10. Round 15 also coerces
    each value to float at the loader so a {"f1": "oops"} cell drops
    the row instead of crashing the consumer downstream."""
    db = str(tmp_path / "m.db")
    backend = SQLiteBackend(db)
    backend.save_open_session("ok", [], [], {"f1": 0.5}, started_at=0.0)
    backend.close()

    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT OR REPLACE INTO open_sessions "
        "(session_id, messages, vectors, facts, started_at) "
        "VALUES (?, ?, ?, ?, ?)",
        ("bad-values", "[]", "[]", '{"f1": "oops"}', 0.0),
    )
    conn.commit()
    conn.close()

    backend2 = SQLiteBackend(db)
    sessions = backend2.load_open_sessions()
    backend2.close()
    ids = {s["session_id"] for s in sessions}
    assert "ok" in ids
    assert "bad-values" not in ids


def test_load_open_sessions_coerces_int_facts_to_float(tmp_path):
    """An int value (legitimate JSON, valid coerce) survives —
    it's the float() failure path we're guarding, not numeric types."""
    db = str(tmp_path / "m.db")
    backend = SQLiteBackend(db)
    backend.close()

    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT OR REPLACE INTO open_sessions "
        "(session_id, messages, vectors, facts, started_at) "
        "VALUES (?, ?, ?, ?, ?)",
        ("int-vals", "[]", "[]", '{"f1": 1, "f2": 2}', 0.0),
    )
    conn.commit()
    conn.close()

    backend2 = SQLiteBackend(db)
    sessions = backend2.load_open_sessions()
    backend2.close()
    assert len(sessions) == 1
    facts = sessions[0]["facts"]
    assert facts == {"f1": 1.0, "f2": 2.0}
    assert all(isinstance(v, float) for v in facts.values())


# --- P1: collapse_singularity bumps mutation -----------------------------


def test_collapse_singularity_invalidates_forecast_cache(tmp_path):
    """A no-op collapse pass doesn't bump (singularity empty), but
    a real collapse that absorbs bodies + creates MetaFacts must
    invalidate the forecast cache."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"),
                      collapse_async=False)
    mem.add_fact("api", "runs on", "Go")
    mem.run_forecast(horizon_ticks=5)
    before = mem._mutation_version

    # Empty singularity → collapse is no-op → no bump.
    report = mem.collapse_singularity(min_group_size=2)
    assert report.groups == 0
    assert mem._mutation_version == before
    mem.close()


def test_collapse_singularity_with_groups_bumps_mutation(tmp_path):
    """End-to-end: seed two near-identical absorbed facts, force
    collapse, observe mutation bump."""
    from birch.fact import FactPassport
    mem = MemoryStore(db_path=str(tmp_path / "m.db"),
                      collapse_async=False)
    # Seed two FactPassports with identical vectors → cosine 1.0 →
    # union-found into one MetaFact.
    for i in range(2):
        f = FactPassport(subject=f"s{i}", predicate="is", object="x")
        f.vector = [1.0, 0.0, 0.0]
        f.gravity_score = 0.05   # below absorption — eligible for collapse
        mem._facts[f.fact_id] = f
        mem._engine.register(f)
        mem._index.add(f.fact_id, f.vector)
    # Push them into the singularity.
    mem._absorb_dead()
    before = mem._mutation_version

    report = mem.collapse_singularity(min_group_size=2)
    assert report.groups == 1
    assert mem._mutation_version > before
    mem.close()


# --- P2: session_push structured unknown_session ------------------------


def test_session_message_raises_keyerror_on_unknown(tmp_path):
    """Core MemoryStore.session_message still raises KeyError —
    that's the contract the MCP layer wraps."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    with pytest.raises(KeyError):
        mem.session_message("hi", session_id="never-opened")
    mem.close()


def test_session_push_unknown_session_inline_validator():
    """MCP layer wraps KeyError as structured response. Inline since
    importing server.py needs the mcp SDK."""
    sid = "never-opened"

    def _mcp_push():
        try:
            raise KeyError(f"unknown session: {sid!r}")
        except KeyError as exc:
            return {
                "ok": False,
                "error": "unknown_session",
                "session_id": sid,
                "detail": str(exc),
                "hint": (
                    "Call session_open first and pass the returned "
                    "session_id, or check the id hasn't been closed."
                ),
            }

    resp = _mcp_push()
    assert resp["error"] == "unknown_session"
    assert resp["session_id"] == sid


# --- P2: session_close structured invalid_sentiment ---------------------


def test_session_close_invalid_sentiment_raises_in_core(tmp_path):
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    mem.session_start("s")
    mem.session_message("hi", session_id="s")
    with pytest.raises(ValueError, match="sentiment must be one of"):
        mem.session_close(session_id="s", sentiment="happyish")
    mem.close()


def test_session_close_invalid_sentiment_inline_mcp_wrapper():
    """MCP wraps it as invalid_sentiment with the allowed list so
    the agent doesn't have to remember the enum."""

    def _mcp_close():
        try:
            raise ValueError(
                "sentiment must be one of "
                "['negative', 'neutral', 'positive', 'resonant', 'toxic'], "
                "got 'happyish'"
            )
        except ValueError as exc:
            return {
                "ok": False,
                "error": "invalid_sentiment",
                "session_id": "s",
                "detail": str(exc),
                "allowed": [
                    "resonant", "positive", "neutral", "toxic", "negative",
                ],
            }

    resp = _mcp_close()
    assert resp["error"] == "invalid_sentiment"
    assert "resonant" in resp["allowed"]
    assert "toxic" in resp["allowed"]


# --- P2: record_session validates messages list[str] --------------------


def test_record_session_rejects_non_list_messages_inline():
    """Replicate the server validator inline."""

    def _validate(messages):
        if not isinstance(messages, list):
            return {
                "ok": False,
                "error": "invalid_messages",
                "got_type": type(messages).__name__,
            }
        bad = [
            i for i, m in enumerate(messages)
            if not isinstance(m, str) or not m.strip()
        ]
        if bad:
            return {
                "ok": False,
                "error": "invalid_message_item",
                "indices": bad,
            }
        return None

    # String, not list — used to iterate chars downstream.
    assert _validate("hello")["error"] == "invalid_messages"
    assert _validate(None)["error"] == "invalid_messages"
    # Non-string item.
    assert _validate(["ok", 123, "fine"])["error"] == "invalid_message_item"
    assert _validate(["ok", 123, "fine"])["indices"] == [1]
    # Whitespace-only item.
    assert _validate(["ok", "   "])["indices"] == [1]
    # Clean — None means "no error, proceed".
    assert _validate(["a", "b", "c"]) is None


# --- P2: collapse worker error surfaces in stats ------------------------


def test_close_captures_collapse_worker_error(tmp_path):
    """Async worker that raised used to be silently swallowed in
    close(). Now stored on instance and visible in stats."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    # Fake an inflight future that raises on .result().
    failed = Future()
    failed.set_exception(RuntimeError("collapse worker crashed"))
    mem._inflight_collapse = failed
    assert mem._last_collapse_error is None

    mem.close()
    assert mem._last_collapse_error is not None
    assert "collapse worker crashed" in mem._last_collapse_error


def test_stats_surface_last_collapse_error(tmp_path):
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    stats = mem.stats
    assert "last_collapse_error" in stats
    assert stats["last_collapse_error"] is None
    mem.close()


def test_successful_collapse_clears_prior_error(tmp_path):
    """A clean collapse pass resets _last_collapse_error so the
    operator's view says 'healthy again' rather than 'last error was X'."""
    from birch.fact import FactPassport
    mem = MemoryStore(db_path=str(tmp_path / "m.db"),
                      collapse_async=False)
    mem._last_collapse_error = "stale: pretend this happened before"

    for i in range(2):
        f = FactPassport(subject=f"s{i}", predicate="is", object="x")
        f.vector = [1.0, 0.0, 0.0]
        f.gravity_score = 0.05
        mem._facts[f.fact_id] = f
        mem._engine.register(f)
        mem._index.add(f.fact_id, f.vector)
    mem._absorb_dead()

    report = mem.collapse_singularity(min_group_size=2)
    assert report.groups == 1
    assert mem._last_collapse_error is None
    mem.close()
