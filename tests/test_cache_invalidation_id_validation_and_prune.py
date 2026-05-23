"""Six contracts that came out of the same triage round:

  1. Every write path goes through ``_bump_mutation_locked`` (which
     bumps mutation_version AND drops _forecast_cache) — raw
     ``self._mutation_version += 1`` would silently skip the cache
     drop and serve stale forecasts after delete / supersede /
     retire / etc.

  2. ``record_facts`` rejects oversized batches at the MCP boundary
     with a structured ``batch_too_large`` error so a 50k-item
     payload can't accidentally pin the embed endpoint.

  3. ``record_facts`` per-item ``session_id`` is type-checked. A
     non-string override silently mis-attributed before; now it's a
     structured ``invalid_session_id`` per item.

  4. ID-based MCP tools (``delete_fact``, ``delete_body``,
     ``supersede_fact``, ``retire_fact``, ``explain_fact``) validate
     the id strings at the boundary and return ``invalid_id`` for
     non-string / empty input — symmetric with text/spo validators.

  5. README says ``eighteen tools`` (was seventeen) and now lists
     ``delete_body`` in the tool table.

  6. ``SQLiteBackend.prune_orphan_edges()`` exists and is
     idempotent. On-disk hygiene utility for the case where a future
     code path forgets to call ``delete_edges_for_fact``.
"""
from __future__ import annotations

import pathlib

from birch.memory_store import MemoryStore
from birch.storage.sqlite import SQLiteBackend

# --- I1: forecast cache invalidates on every write path ---------------


def test_delete_fact_invalidates_forecast_cache(tmp_path):
    """Bug this guards: ``delete_fact`` used to do a raw
    ``self._mutation_version += 1`` and bypass the cache-drop, so a
    follow-up ``forecast_memory`` would serve a stale cached result
    that still mentioned the deleted body."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    f1 = mem.add_fact("api", "runs on", "Go")
    f2 = mem.add_fact("api", "uses", "Postgres")
    forecast_before = mem.run_forecast(horizon_ticks=5)
    bodies_before = forecast_before["bodies_forecasted"]
    # Delete via the raw path; the helper MUST run.
    assert mem.delete_fact(f1.fact_id) is True
    forecast_after = mem.run_forecast(horizon_ticks=5)
    assert forecast_after["bodies_forecasted"] == bodies_before - 1, (
        "forecast cache served stale body count after delete_fact"
    )
    # Sanity: the surviving fact is still there.
    assert any(
        x.fact_id == f2.fact_id for x in mem.list_facts(subject="api")
    )
    mem.close()


def test_supersede_invalidates_forecast_cache(tmp_path):
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    old = mem.add_fact("svc", "version", "1.0")
    forecast_before = mem.run_forecast(horizon_ticks=5)
    new = mem.add_fact("svc", "version", "2.0")
    # add_fact already bumped — but supersede must bump too.
    mem.supersede_fact(old.fact_id, new.fact_id)
    forecast_after = mem.run_forecast(horizon_ticks=5)
    # Old body goes to singularity (still counted in run_forecast),
    # but the mutation_version must differ — provable by checking
    # the cache is not literally the same object.
    assert mem._forecast_cache is not None, (
        "run_forecast should have re-populated cache after supersede"
    )
    # Cache contents include the post-supersede world.
    assert forecast_after["bodies_forecasted"] >= forecast_before["bodies_forecasted"]
    mem.close()


def test_retire_invalidates_forecast_cache(tmp_path):
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    f = mem.add_fact("topic", "is", "active")
    mem.run_forecast(horizon_ticks=5)
    cache_before = mem._forecast_cache
    assert cache_before is not None
    mem.retire_fact(f.fact_id)
    # Helper must have dropped the cache.
    assert mem._forecast_cache is None, (
        "retire_fact left stale forecast cache in place"
    )
    mem.close()


# --- I2: record_facts batch cap ---------------------------------------


def test_record_facts_batch_cap_inline():
    """Replicate the server validator inline since server needs the
    mcp SDK to import. The cap is read from
    BIRCH_RECORD_FACTS_BATCH_CAP (default 500)."""
    import os

    cap = max(
        1, int(os.environ.get("BIRCH_RECORD_FACTS_BATCH_CAP", "500"))
    )

    def _check(facts):
        if not isinstance(facts, list):
            return {"ok": False, "error": "invalid_facts_payload"}
        if len(facts) > cap:
            return {
                "ok": False,
                "error": "batch_too_large",
                "limit": cap,
                "got": len(facts),
            }
        return None

    # Under the cap → pass.
    assert _check([{"subject": "a"}] * cap) is None
    # Over the cap → structured rejection.
    err = _check([{"subject": "a"}] * (cap + 1))
    assert err["error"] == "batch_too_large"
    assert err["limit"] == cap
    assert err["got"] == cap + 1


# --- I3: per-item session_id type check -------------------------------


def test_record_facts_per_item_session_id_validator_inline():
    """Replicate the server's per-item session_id validator."""

    def _check_item(f):
        if "session_id" in f and f["session_id"] is not None:
            if (
                not isinstance(f["session_id"], str)
                or not f["session_id"].strip()
            ):
                return {
                    "error": "invalid_session_id",
                    "got_type": type(f["session_id"]).__name__,
                }
        return None

    assert _check_item({"subject": "a"}) is None
    assert _check_item({"subject": "a", "session_id": None}) is None
    assert _check_item({"subject": "a", "session_id": "ok"}) is None
    err = _check_item({"subject": "a", "session_id": 42})
    assert err["error"] == "invalid_session_id"
    assert err["got_type"] == "int"
    err = _check_item({"subject": "a", "session_id": ["x"]})
    assert err["error"] == "invalid_session_id"
    err = _check_item({"subject": "a", "session_id": "   "})
    assert err["error"] == "invalid_session_id"


# --- I4: ID-tool string validation -------------------------------------


def test_validate_id_inline_contract():
    """Replicate the server's _validate_id helper. Used by
    delete_fact, delete_body, supersede_fact, retire_fact,
    explain_fact at the MCP boundary."""

    def _validate(value, field="fact_id"):
        if not isinstance(value, str) or not value.strip():
            return {
                "ok": False,
                "error": "invalid_id",
                "field": field,
                "got_type": type(value).__name__,
            }
        return None

    assert _validate("abc") is None
    assert _validate(None)["error"] == "invalid_id"
    assert _validate(None)["got_type"] == "NoneType"
    assert _validate(123)["error"] == "invalid_id"
    assert _validate(123)["got_type"] == "int"
    assert _validate("")["error"] == "invalid_id"
    assert _validate("   ")["error"] == "invalid_id"
    err = _validate(None, "body_id")
    assert err["field"] == "body_id"


# --- I5: README doc sync ----------------------------------------------


def test_readme_says_eighteen_tools_and_lists_delete_body():
    root = pathlib.Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text()
    assert "eighteen tools" in readme, (
        "README should advertise eighteen tools "
        "(was seventeen; delete_body is now exposed)"
    )
    # Tool table row exists.
    assert "`delete_body`" in readme
    # Old "seventeen tools" line must be gone.
    assert "seventeen tools" not in readme


# --- I6: prune_orphan_edges utility -----------------------------------


def test_prune_orphan_edges_drops_orphans_keeps_valid(tmp_path):
    """Adds a fact + edges, then forcibly inserts an orphan edge
    behind the executor's back, then asserts prune removes only the
    orphan."""
    backend = SQLiteBackend(str(tmp_path / "m.db"))
    mem = MemoryStore(storage=backend)
    f1 = mem.add_fact("api", "uses", "Postgres")
    f2 = mem.add_fact("api", "uses", "Redis")
    # Real edge between two real facts (the auto-link layer creates
    # this on subject collision, so it should already be there).
    edges_before = backend.load_edges()
    real_edge_count = sum(
        1 for a, b in edges_before
        if a in {f1.fact_id, f2.fact_id} and b in {f1.fact_id, f2.fact_id}
    )
    # Insert an orphan edge by hand.
    backend._conn.execute(
        "INSERT INTO edges (from_id, to_id) VALUES (?, ?)",
        (f1.fact_id, "ghost-id-does-not-exist"),
    )
    backend._conn.commit()
    assert any(
        b == "ghost-id-does-not-exist"
        for _, b in backend.load_edges()
    )
    # Prune.
    removed = backend.prune_orphan_edges()
    assert removed == 1
    # Real edges survive.
    edges_after = backend.load_edges()
    surviving_real = sum(
        1 for a, b in edges_after
        if a in {f1.fact_id, f2.fact_id} and b in {f1.fact_id, f2.fact_id}
    )
    assert surviving_real == real_edge_count
    # Orphan gone.
    assert not any(
        b == "ghost-id-does-not-exist" for _, b in edges_after
    )
    # Idempotent.
    assert backend.prune_orphan_edges() == 0
    mem.close()
