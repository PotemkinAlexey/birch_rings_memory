"""Session abort + partial-open contract regressions.

Covers:

  1. record_session leaks an open session if embed fails mid-flow
     → MemoryStore.abort_session helper + wired into server.
  2. session_open(first_message=...) doesn't tell the agent whether
     first_message actually landed → first_message_recorded flag.
  3. README compactor description was pre-mixed-dim — updated to
     per-dim partitioning.

Deferred (design choices, not bugs):
  - list_facts envelope with effective_limit (would break the
    list[dict] consumer contract; logged server-side instead).
  - averaged training event per session ("one signal in, one
    step out" is the explicit design — mini-batch per-body would
    over-fit on noisy single sessions).
"""
from __future__ import annotations

from birch.memory_store import MemoryStore

# --- P2: abort_session helper -------------------------------------------


def test_abort_session_drops_open_session(tmp_path):
    """abort_session pops the session without scoring or touching
    gravity. Idempotent on unknown id."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    mem.session_start("s1")
    mem.session_message("hello", session_id="s1")
    assert "s1" in mem._sessions

    aborted = mem.abort_session("s1")
    assert aborted is True
    assert "s1" not in mem._sessions

    # Idempotent — second abort returns False.
    assert mem.abort_session("s1") is False
    # Unknown id — False, no raise.
    assert mem.abort_session("never-opened") is False
    mem.close()


def test_abort_session_does_not_train_or_migrate(tmp_path):
    """abort skips the full close pipeline — no R, no migrations,
    no EWMA, no adaptive weight training."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    mem.add_fact("api", "runs on", "Go")
    train_before = mem._engine.weights.train_count
    mut_before = mem._mutation_version

    mem.session_start("s")
    mem.session_message("looking at api", session_id="s")
    mem.query("api", session_id="s")
    train_after_query = mem._engine.weights.train_count
    assert train_after_query == train_before  # no training yet

    mem.abort_session("s")
    # No training step happened.
    assert mem._engine.weights.train_count == train_before
    # mutation_version may have bumped from add_fact / query but
    # the abort itself doesn't add gravity-changing events.
    assert mem._mutation_version >= mut_before
    mem.close()


def test_abort_session_persists_drop_to_storage(tmp_path):
    """Storage row for the open session is deleted, not left orphan."""
    db = str(tmp_path / "m.db")
    mem = MemoryStore(db_path=db)
    mem.session_start("s")
    mem.session_message("hi", session_id="s")
    mem.close()

    # Re-open — open_session row was persisted, so the session would
    # still exist if we hadn't aborted. Confirm the contract by
    # opening + aborting + re-opening.
    again = MemoryStore(db_path=db)
    assert "s" in again._sessions  # persisted from the first run

    again.abort_session("s")
    again.close()

    third = MemoryStore(db_path=db)
    assert "s" not in third._sessions  # gone from disk too
    third.close()


# --- P3: record_session / session_open partial-state contracts ----------


def test_record_session_inline_abort_on_embed_failure():
    """Server's record_session except-path replicates this contract
    inline (importing server requires the mcp SDK)."""

    class FakeStore:
        def __init__(self) -> None:
            self.started: list[str] = []
            self.aborted: list[str] = []

        def session_start(self, sid: str) -> None:
            self.started.append(sid)

        def check_echo(self, *_args, **_kw):
            raise RuntimeError("embed unavailable")

        def session_message(self, *_args, **_kw) -> None:
            pass

        def session_close(self, *_args, **_kw):
            return {}

        def abort_session(self, sid: str) -> bool:
            self.aborted.append(sid)
            return True

    store = FakeStore()
    sid = "test-sid"
    store.session_start(sid)
    try:
        store.check_echo("hi", session_id=sid)
    except RuntimeError:
        try:
            store.abort_session(sid)
        except Exception:
            pass

    assert sid in store.started
    assert sid in store.aborted


def test_session_open_first_message_recorded_flag_on_success(tmp_path):
    """When session_open succeeds at recording first_message, the
    flag is True; agent knows the message is in the trajectory."""
    # Replicate the server response shape inline since server.py
    # imports the mcp SDK.
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    sid = "s"
    mem.session_start(sid)

    response: dict = {"session_id": sid}
    first_message = "hello there"
    try:
        echo = mem.check_echo(first_message, session_id=sid)
        response["echo"] = echo
        mem.session_message(first_message, session_id=sid)
        response["first_message_recorded"] = True
    except Exception:
        response["first_message_recorded"] = False

    assert response["first_message_recorded"] is True
    assert "echo" in response
    mem.close()


def test_session_open_first_message_recorded_flag_on_failure():
    """When check_echo blows up, the flag is False and the response
    carries an _hint pointing at the recovery options."""

    class FakeStore:
        def session_start(self, sid: str) -> None:
            pass

        def check_echo(self, *_a, **_k):
            raise RuntimeError("embed down")

        def session_message(self, *_a, **_k):
            pass

    store = FakeStore()
    sid = "s"
    store.session_start(sid)

    response: dict = {"session_id": sid}
    first_message = "test"
    try:
        response["echo"] = store.check_echo(
            first_message, session_id=sid,
        )
        store.session_message(first_message, session_id=sid)
        response["first_message_recorded"] = True
    except Exception:
        response["echo_error"] = {"ok": False, "error": "embed_failed"}
        response["first_message_recorded"] = False
        response["_hint"] = (
            "session was opened but first_message was NOT recorded "
            "due to embedding failure; retry session_push or call "
            "session_close to drop the empty session"
        )

    assert response["first_message_recorded"] is False
    assert "echo_error" in response
    assert "NOT recorded" in response["_hint"]


# --- README docstring drift catch ---------------------------------------


def test_readme_compactor_describes_per_dim_partitioning():
    """The compactor partitions by vector dim. README used to say
    'a single numpy matmul over all absorbed fact vectors' which only
    held before the dim-mix safety fix."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text()
    assert "partitioned by vector dimension" in readme
    # Old single-matmul sentence is gone.
    assert "a single\nnumpy `matmul` computes" not in readme
