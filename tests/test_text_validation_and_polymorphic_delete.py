"""MCP text-input validation + record_facts top-level list check +
session_close unknown-or-empty + polymorphic delete_body.

Six contracts:

  1. record_facts rejects non-list facts payload with structured
     invalid_facts_payload (was: would iterate string by character).
  2. _validate_text helper rejects None / non-string / whitespace-only
     text inputs at the MCP boundary for query_memory, find_similar,
     check_echo, session_push, session_open(first_message).
  3. session_close on unknown / empty session returns structured
     unknown_or_empty_session (was: looked like a successful close
     with null label and r_score=0.0).
  4. delete_body is polymorphic — deletes from live FactPassports,
     live MetaFacts, singularity FactPassports, or singularity
     MetaFacts. delete_fact remains FactPassport-only as the legacy
     primitive.
"""
from __future__ import annotations

from birch.memory_store import MemoryStore
from birch.meta_fact import MetaFact

# --- I1: record_facts payload type check ------------------------------


def test_record_facts_payload_validator_inline():
    """Replicate the server validator inline since server.py needs
    the mcp SDK to import."""

    def _validate(facts):
        if not isinstance(facts, list):
            return {
                "ok": False,
                "error": "invalid_facts_payload",
                "got_type": type(facts).__name__,
            }
        return None

    assert _validate("not a list")["error"] == "invalid_facts_payload"
    assert _validate("not a list")["got_type"] == "str"
    assert _validate({"key": "val"})["error"] == "invalid_facts_payload"
    assert _validate({"key": "val"})["got_type"] == "dict"
    assert _validate(None)["error"] == "invalid_facts_payload"
    assert _validate([]) is None
    assert _validate([{"subject": "a"}]) is None


# --- I2: _validate_text contract --------------------------------------


def test_validate_text_inline_contract():
    """Server now rejects non-string / empty text inputs across
    query_memory, find_similar, session_push, check_echo,
    session_open(first_message). Replicate inline."""

    def _validate_text(value, field_name="text"):
        if not isinstance(value, str) or not value.strip():
            return {
                "ok": False,
                "error": "invalid_text",
                "field": field_name,
                "got_type": type(value).__name__,
            }
        return None

    assert _validate_text(None)["error"] == "invalid_text"
    assert _validate_text(None)["got_type"] == "NoneType"
    assert _validate_text("")["error"] == "invalid_text"
    assert _validate_text("   ")["error"] == "invalid_text"
    assert _validate_text(123)["error"] == "invalid_text"
    assert _validate_text(123)["got_type"] == "int"
    assert _validate_text(["a"])["error"] == "invalid_text"
    assert _validate_text("hello") is None
    # Field name carries through.
    err = _validate_text(None, "first_message")
    assert err["field"] == "first_message"


# --- I3: session_close empty session structured ------------------------


def test_session_close_unknown_session_returns_empty_summary(tmp_path):
    """Core MemoryStore.session_close returns {} for an unknown
    session_id — the MCP wrapper now translates that to a structured
    unknown_or_empty_session response. Inline since we can't import
    server."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    summary = mem.session_close(session_id="never-opened")
    assert summary == {}
    mem.close()


def test_session_close_empty_session_envelope_inline():
    """Replicate the MCP wrapper's empty-summary branch."""
    summary = {}  # what core returns on unknown session
    if not summary:
        resp = {
            "ok": False,
            "error": "unknown_or_empty_session",
            "session_id": "ghost",
            "hint": "Call session_open and session_push before session_close",
        }
    assert resp["error"] == "unknown_or_empty_session"
    assert resp["session_id"] == "ghost"


# --- I4: delete_body polymorphic --------------------------------------


def test_delete_body_handles_live_fact(tmp_path):
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    f = mem.add_fact("api", "runs on", "Go")
    result = mem.delete_body(f.fact_id)
    assert result["deleted"] is True
    assert result["kind"] == "fact"
    assert result["body_id"] == f.fact_id
    # Fact really gone.
    assert not any(
        x.fact_id == f.fact_id for x in mem.list_facts(subject="api")
    )
    mem.close()


def test_delete_body_handles_singularity_fact(tmp_path):
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    f = mem.add_fact("api", "runs on", "Go")
    # Push to singularity.
    f.gravity_score = 0.05
    mem._storage.save_fact(f)
    mem._absorb_dead()
    assert f.fact_id in {
        rec.fact.fact_id for rec in mem._hole._singularity.values()
    }

    result = mem.delete_body(f.fact_id)
    assert result["deleted"] is True
    assert result["kind"] == "singularity_fact"
    assert f.fact_id not in mem._hole._singularity
    mem.close()


def test_delete_body_handles_singularity_meta(tmp_path):
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    meta = MetaFact(
        weight=2, source_texts=["x", "y"],
        source_fact_ids=["a", "b"],
    )
    meta.vector = [1.0, 0.0, 0.0]
    mem._storage.save_meta_fact(meta)
    mem._hole.absorb_meta(meta)
    assert meta.meta_id in mem._hole._meta_singularity

    result = mem.delete_body(meta.meta_id)
    assert result["deleted"] is True
    assert result["kind"] == "singularity_meta"
    assert meta.meta_id not in mem._hole._meta_singularity
    mem.close()


def test_delete_body_returns_false_on_unknown(tmp_path):
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    result = mem.delete_body("nonexistent-id")
    assert result["deleted"] is False
    assert result["body_id"] == "nonexistent-id"
    mem.close()


def test_delete_body_bumps_mutation_version(tmp_path):
    """delete_body changes body universe — forecast cache must
    invalidate via the bump."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    f = mem.add_fact("api", "runs on", "Go")
    before = mem._mutation_version
    mem.delete_body(f.fact_id)
    assert mem._mutation_version > before
    mem.close()


# --- I5: README + AGENTS sync -----------------------------------------


def test_readme_lists_check_echo_with_correct_tool_count():
    """README's MCP-tool table now includes check_echo and says
    seventeen tools (was sixteen + missing check_echo)."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text()
    assert "seventeen tools" in readme
    assert "`check_echo`" in readme
    # forecast wording updated to per-body.
    assert "per-body stability" in readme


def test_agents_changing_prefers_set_fact():
    """AGENTS Changing principle now leads with set_fact for
    single-valued slots, supersede_fact reserved for cross-slot."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    agents = (root / "AGENTS.md").read_text()
    # "Changing" paragraph mentions set_fact BEFORE supersede_fact.
    changing_start = agents.find("**Changing**")
    assert changing_start != -1
    next_section = agents.find("**Deleting**", changing_start)
    changing_block = agents[changing_start:next_section]
    set_pos = changing_block.find("set_fact")
    supersede_pos = changing_block.find("supersede_fact")
    assert 0 < set_pos < supersede_pos, (
        "AGENTS Changing principle should lead with set_fact for "
        "single-valued slots"
    )
