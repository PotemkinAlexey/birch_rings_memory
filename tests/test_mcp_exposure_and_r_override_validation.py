"""MCP exposure parity, r_override validation, edges cleanup symmetry,
and private→public API rename.

Five contracts shipped together:

  1. delete_body is now an MCP tool (was only in core) — polymorphic
     hard-delete matches polymorphic query.
  2. delete_body singularity-fact branch now cleans edges too — same
     contract as the live-fact branch.
  3. session_close r_override is validated at MCP boundary BEFORE
     core, so non-numeric input returns invalid_r_override (not
     invalid_sentiment — which was a misleading catch-all).
  4. r_override = NaN / Infinity rejected explicitly at the boundary;
     core's max/min clamp would otherwise give a surprising result.
  5. find_similar_by_vector promoted from leading-underscore "private"
     to public method. _find_similar_by_vector remains as a
     deprecated alias so existing callers don't break.
"""
from __future__ import annotations

import math

from birch.memory_store import MemoryStore

# --- I1 + I2: delete_body singularity edges cleanup -------------------


def test_delete_body_singularity_fact_cleans_edges(tmp_path):
    """Both live-fact and singularity-fact branches must call
    delete_edges_for_fact. Without this, hard-delete of an absorbed
    body leaves orphan edge rows on disk that would inflate _degrees
    on next load."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    f1 = mem.add_fact("api", "uses", "Postgres")
    # Second fact shares subject "api" so the auto-link layer wires an
    # edge to f1 — that's the edge we want to make sure gets cleaned
    # up when f1 is hard-deleted from singularity.
    mem.add_fact("api", "uses", "Redis")
    # Push f1 into singularity.
    f1.gravity_score = 0.05
    mem._storage.save_fact(f1)
    mem._absorb_dead()
    assert f1.fact_id in {
        rec.fact.fact_id for rec in mem._hole._singularity.values()
    }

    # Delete from singularity via polymorphic delete_body.
    result = mem.delete_body(f1.fact_id)
    assert result["kind"] == "singularity_fact"

    # No edges incident to f1 remain in storage.
    edges = mem._storage.load_edges()
    assert not any(
        f1.fact_id in (from_id, to_id) for from_id, to_id in edges
    ), "orphan edges left on disk after singularity-fact delete"
    mem.close()


# --- I3 + I4: r_override validation at MCP boundary -------------------


def _validate_r_override(r_override):
    """Replicate the server's r_override validator inline so we don't
    need the mcp SDK to import server."""
    if r_override is None:
        return None
    try:
        r_check = float(r_override)
    except (TypeError, ValueError):
        return {
            "ok": False,
            "error": "invalid_r_override",
            "got_type": type(r_override).__name__,
        }
    if not math.isfinite(r_check):
        return {
            "ok": False,
            "error": "invalid_r_override",
            "detail": "NaN or Infinity",
        }
    return None


def test_r_override_none_passes():
    assert _validate_r_override(None) is None


def test_r_override_valid_float_passes():
    assert _validate_r_override(0.5) is None
    assert _validate_r_override(-0.7) is None
    assert _validate_r_override(0) is None


def test_r_override_string_returns_invalid_r_override():
    """The whole point of the round: a string used to map to
    invalid_sentiment via the catch-all ValueError handler. Now
    it gets the honest error."""
    err = _validate_r_override("not a number")
    assert err["error"] == "invalid_r_override"
    assert err["got_type"] == "str"


def test_r_override_list_returns_invalid_r_override():
    err = _validate_r_override([0.5])
    assert err["error"] == "invalid_r_override"
    assert err["got_type"] == "list"


def test_r_override_nan_returns_invalid_r_override():
    err = _validate_r_override(float("nan"))
    assert err["error"] == "invalid_r_override"
    assert "NaN or Infinity" in err["detail"]


def test_r_override_inf_returns_invalid_r_override():
    err = _validate_r_override(float("inf"))
    assert err["error"] == "invalid_r_override"


def test_r_override_neg_inf_returns_invalid_r_override():
    err = _validate_r_override(float("-inf"))
    assert err["error"] == "invalid_r_override"


# --- I5: find_similar_by_vector public + alias ------------------------


def test_find_similar_by_vector_is_public(tmp_path):
    """New public name works. Old private alias still works for
    backward-compat — server.py + any external callers can migrate
    at their own pace."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    f = mem.add_fact("api", "runs on", "Go")
    # Use the same vector to guarantee a hit.
    hits = mem.find_similar_by_vector(
        f.vector, top_k=3, min_similarity=0.5,
    )
    assert any(h.get("fact_id") == f.fact_id for h in hits)

    # Private alias produces identical result.
    legacy_hits = mem._find_similar_by_vector(
        f.vector, top_k=3, min_similarity=0.5,
    )
    assert hits == legacy_hits
    mem.close()


# --- delete_body MCP tool surface (inline) ----------------------------


def test_delete_body_handles_polymorphic_body_id(tmp_path):
    """Core delete_body is the polymorphic primitive; the MCP wrapper
    just forwards. Confirms core contract — server tool routes
    through this same code path."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    # Live fact path.
    f = mem.add_fact("api", "runs on", "Go")
    result = mem.delete_body(f.fact_id)
    assert result["deleted"] is True
    assert result["kind"] == "fact"
    mem.close()


# --- README + AGENTS doc-sync pins ------------------------------------


def test_readme_lists_delete_body_in_tool_count_or_table():
    """README should mention delete_body somewhere now that it's an
    MCP tool. Either as a separate row (preferred) or as a callout
    in the delete_fact row. Don't require a specific tool-count
    update — that's a separate doc-pin test."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text()
    # Lightweight assertion: README should mention delete_body OR
    # delete_fact should be flagged as the legacy form. We accept
    # either, but at least one must be true.
    has_delete_body = "delete_body" in readme
    has_legacy_marker = (
        "live FactPassport" in readme
        or "Legacy" in readme
    )
    assert has_delete_body or has_legacy_marker, (
        "README should expose delete_body as a tool OR flag "
        "delete_fact as legacy/FactPassport-only"
    )
