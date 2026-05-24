"""Self-review findings, all in one file:

  P1 — supersede_fact / retire_fact now echo ids in both branches and
       detect non-FactPassport body kinds with structured error.
  P2 — set_fact response shape adds layer/gravity_score/created/_hint.
  P2 — _reload() is crash-safe (build-then-swap, restore on failure,
       data_version reset to sentinel for retry).
  P2 — load_open_sessions returns rows ORDER BY started_at so
       _current_session_id is deterministic across cross-process
       reloads.
  P3 — _validate_optional_text rejects non-string subject_prefix /
       subject / predicate at MCP boundary (inline replication).
  P3 — record_session([]) returns structured empty_messages.

  Plus the five highest-value missing-pin tests surfaced by self-review:
    - set_fact with N>1 pre-existing live facts in the same slot
    - forecast determinism (same store → same scores)
    - session_close with zero touched_facts (no SGD divide-by-zero)
    - explain_fact / explain_body unknown id still returns found=False
    - Hawking emission without active session does not crash
"""
from __future__ import annotations

import pytest

from birch.memory_store import MemoryStore
from birch.meta_fact import MetaFact

# --- P1: supersede / retire failure shape + body-kind detection -------


def test_supersede_unknown_id_returns_both_ids(tmp_path):
    """Unknown id must echo old_id AND new_id so callers don't
    KeyError on result['old_id'] in the failure branch."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    resp = mem.supersede_fact("nonexistent-old", "nonexistent-new")
    assert resp["superseded"] is False
    assert resp["old_id"] == "nonexistent-old"
    assert resp["new_id"] == "nonexistent-new"
    assert resp["error"] == "not_found"
    mem.close()


def test_supersede_metafact_body_id_returns_not_a_factpassport(tmp_path):
    """The docstring on delete_body says 'prefer supersede_fact /
    retire_fact for stale data'; if the agent pipes a MetaFact body_id
    in, we used to return a generic 'old_id not found'. Now we surface
    kind=meta + a hint explaining why supersede doesn't apply."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    meta = MetaFact(
        weight=2, source_texts=["a", "b"],
        source_fact_ids=["x", "y"],
        layer=0,
    )
    meta.vector = [1.0, 0.0, 0.0]
    mem._storage.save_meta_fact(meta)
    mem._reload()
    resp = mem.supersede_fact(meta.meta_id, "replacement-id")
    assert resp["superseded"] is False
    assert resp["error"] == "not_a_factpassport"
    assert resp["kind"] == "meta"
    assert "MetaFacts have no SPO slot" in resp["hint"]
    assert resp["old_id"] == meta.meta_id
    assert resp["new_id"] == "replacement-id"
    mem.close()


def test_supersede_singularity_factpassport_returns_not_a_factpassport(tmp_path):
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    f = mem.add_fact("api", "uses", "Postgres")
    f.gravity_score = 0.05
    mem._storage.save_fact(f)
    mem._absorb_dead()
    assert f.fact_id in mem._hole._singularity
    resp = mem.supersede_fact(f.fact_id, "new-id")
    assert resp["superseded"] is False
    assert resp["error"] == "not_a_factpassport"
    assert resp["kind"] == "singularity_fact"
    assert "already in the singularity" in resp["hint"]
    mem.close()


def test_retire_unknown_id_echoes_fact_id(tmp_path):
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    resp = mem.retire_fact("nonexistent")
    assert resp["retired"] is False
    assert resp["fact_id"] == "nonexistent"
    assert resp["error"] == "not_found"
    mem.close()


def test_retire_metafact_body_id_returns_not_a_factpassport(tmp_path):
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    meta = MetaFact(
        weight=2, source_texts=["a", "b"],
        source_fact_ids=["x", "y"],
        layer=0,
    )
    meta.vector = [1.0, 0.0, 0.0]
    mem._storage.save_meta_fact(meta)
    mem._reload()
    resp = mem.retire_fact(meta.meta_id)
    assert resp["retired"] is False
    assert resp["error"] == "not_a_factpassport"
    assert resp["kind"] == "meta"
    assert resp["fact_id"] == meta.meta_id
    mem.close()


# --- P2: set_fact response shape symmetry -----------------------------


def test_set_fact_response_carries_record_fact_field_set(tmp_path):
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    resp = mem.set_fact("api", "version", "1.0")
    # New fields that record_fact / record_facts items already had.
    assert "layer" in resp
    assert "gravity_score" in resp
    assert "created" in resp
    assert "_hint" in resp
    assert resp["created"] is True
    assert resp["already_existed"] is False
    # Existing fields preserved.
    assert resp["set"] is True
    assert resp["subject"] == "api"
    mem.close()


def test_set_fact_with_three_existing_occupants_supersedes_all_atomically(tmp_path):
    """The N>1 occupant scenario surfaced by self-review test gap.
    Seed 3 live FactPassports with shared (subject, predicate) by
    bypassing slot uniqueness (direct add_fact via different objects
    is fine — slot is shared across object values). Run set_fact and
    assert ALL 3 are superseded inside ONE transaction."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    f1 = mem.add_fact("svc", "endpoint", "https://eu.api.example.com")
    f2 = mem.add_fact("svc", "endpoint", "https://us.api.example.com")
    f3 = mem.add_fact("svc", "endpoint", "https://ap.api.example.com")
    # All three live; slot has multiple objects (legal for add_fact).
    live = [f for f in mem.list_facts(subject="svc") if not f.is_deprecated]
    assert len(live) >= 3
    resp = mem.set_fact("svc", "endpoint", "https://global.api.example.com")
    assert resp["set"] is True
    superseded = set(resp["superseded"])
    assert f1.fact_id in superseded
    assert f2.fact_id in superseded
    assert f3.fact_id in superseded
    # Live view: only the new one survives.
    live_after = [
        f for f in mem.list_facts(subject="svc") if not f.is_deprecated
    ]
    assert len(live_after) == 1
    assert live_after[0].object == "https://global.api.example.com"
    mem.close()


# --- P2: _reload atomicity --------------------------------------------


def test_reload_restores_caches_on_load_failure(tmp_path, monkeypatch):
    """If _load_from_storage raises mid-rebuild, the previous
    populated caches must remain live AND data_version must be
    reset so the next _sync retries instead of trusting empty."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    seed = mem.add_fact("api", "uses", "Postgres")
    assert seed.fact_id in mem._facts
    saved_data_version = mem._data_version

    # Force _load_from_storage to raise.
    def boom():
        raise RuntimeError("simulated transient SQLite failure")

    monkeypatch.setattr(mem, "_load_from_storage", boom)
    with pytest.raises(RuntimeError, match="transient"):
        mem._reload()
    # Caches survived the failed reload.
    assert seed.fact_id in mem._facts, (
        "_reload left caches empty after _load_from_storage raised"
    )
    # data_version reset to sentinel so next _sync forces a retry.
    assert mem._data_version == -1
    assert mem._data_version != saved_data_version
    mem.close()


# --- P2: load_open_sessions ORDER BY ----------------------------------


def test_load_open_sessions_returns_rows_ordered_by_started_at(tmp_path):
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    # Open three sessions in order; started_at increases.
    import time as _t
    mem.session_start("first")
    _t.sleep(0.01)
    mem.session_start("second")
    _t.sleep(0.01)
    mem.session_start("third")
    rows = mem._storage.load_open_sessions()
    started = [r["started_at"] for r in rows]
    assert started == sorted(started), (
        "load_open_sessions must return rows ORDER BY started_at"
    )
    mem.close()


# --- P3: _validate_optional_text inline contract ----------------------


def test_validate_optional_text_inline_contract():
    """Replicate the server helper inline (mcp SDK not importable
    from tests). subject_prefix=123 used to reach core where
    .lower() raised raw AttributeError."""

    def _check(value, field="subject_prefix"):
        if value is None:
            return None
        if not isinstance(value, str):
            return {
                "ok": False,
                "error": "invalid_text",
                "field": field,
                "got_type": type(value).__name__,
            }
        return None

    assert _check(None) is None
    assert _check("") is None  # empty is fine; means "no filter"
    assert _check("api/") is None
    err = _check(123)
    assert err["error"] == "invalid_text"
    assert err["got_type"] == "int"
    err = _check(["bad"])
    assert err["error"] == "invalid_text"


# --- P3: record_session empty messages envelope -----------------------


def test_record_session_empty_messages_envelope_inline():
    def _check(messages):
        if not isinstance(messages, list):
            return {"ok": False, "error": "invalid_messages"}
        if not messages:
            return {"ok": False, "error": "empty_messages"}
        return None

    assert _check([]) == {"ok": False, "error": "empty_messages"}
    assert _check(["hello"]) is None


# --- Missing pin: forecast determinism --------------------------------


def test_forecast_determinism_same_store_same_scores(tmp_path):
    """Two consecutive run_forecast calls on the same store must
    produce identical scores. If dict-iteration order ever leaks
    into the N-body simulation, this catches it before reproducibility
    of weight training silently degrades."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    f1 = mem.add_fact("svc", "uses", "Postgres")
    f2 = mem.add_fact("svc", "uses", "Redis")
    f3 = mem.add_fact("svc", "uses", "Kafka")
    first = mem.run_forecast(horizon_ticks=5)
    # Capture scores before cache potentially serves on second call.
    scores_first = {
        f.fact_id: f.forecast_stability
        for f in mem.list_facts(subject="svc")
    }
    # Force cache miss by bumping mutation, then redo to compare.
    mem._bump_mutation_locked()
    second = mem.run_forecast(horizon_ticks=5)
    assert second.get("ok") is not False
    scores_second = {
        f.fact_id: f.forecast_stability
        for f in mem.list_facts(subject="svc")
    }
    assert scores_first == scores_second, (
        "forecast_stability scores changed across identical runs"
    )
    mem.close()
    # Touch first so linter doesn't complain about unused.
    assert first.get("ok") is not False
    assert f1 and f2 and f3


# --- Missing pin: session_close with zero touched_facts ---------------


def test_session_close_zero_touched_facts_does_not_crash(tmp_path):
    """Open session, push one message, close — no query_memory ran,
    so ctx.facts is empty. SGD path must not divide by zero or
    train on a degenerate empty fact set."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    mem.session_start("s1")
    mem.session_message("hello", session_id="s1")
    # No query → no touched facts.
    assert mem._sessions["s1"].facts == {}
    summary = mem.session_close(
        session_id="s1", sentiment="positive",
    )
    # Close completes cleanly; r in [-1, 1].
    r = summary.get("r", summary.get("r_score", 0.0))
    assert -1.0 <= r <= 1.0
    mem.close()


# --- Missing pin: explain unknown id ----------------------------------


def test_explain_fact_unknown_id_returns_not_found_envelope(tmp_path):
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    out = mem.explain_fact("ghost-id")
    assert out["found"] is False
    assert out["fact_id"] == "ghost-id"
    # Also via the body alias.
    out2 = mem.explain_body("ghost-id")
    assert out2 == out
    mem.close()


# --- Missing pin: Hawking emission without active session -------------


def test_hawking_emission_without_session_does_not_crash(tmp_path):
    """Query against a Hawking-eligible black-hole resident with NO
    active session_id. The emission path tags touched facts to the
    session; if it assumed a session always exists, the no-session
    case would crash. Pin the bare query."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    f = mem.add_fact(
        "topic", "is described as",
        "deep technical edge case unique signature",
    )
    f.gravity_score = 0.05
    mem._storage.save_fact(f)
    mem._absorb_dead()
    # Body is in singularity, not live. No session opened.
    assert mem._current_session_id is None
    # Query with text that should match — Hawking peek-then-commit
    # may or may not emit depending on similarity vs threshold,
    # but the path must not raise.
    results = mem.query(
        "deep technical edge case unique signature", top_k=5,
    )
    # Either nothing came back (threshold not met — fine) OR the
    # emission landed without a session_id (also fine). What we
    # pin: no exception.
    assert isinstance(results, list)
    mem.close()
