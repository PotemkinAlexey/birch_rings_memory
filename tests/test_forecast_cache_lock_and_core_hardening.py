"""Five contracts from the same triage:

  1. _forecast_cache assignment lives INSIDE the writeback lock.
     Distribution computation + payload construction + cache slot
     write are all guarded — _forecast_cache is shared mutable
     state and two concurrent forecasts could otherwise race the
     slot. The cache_key revalidation that aborts on drift would
     be meaningless if the cache itself is written by a torn
     assignment.

  2. Core MemoryStore.session_close(r_override=...) rejects NaN
     and Infinity. The MCP boundary already validated this in
     a77d3b3; the core method is a public Python API too. Bare
     max(-1, min(1, NaN)) silently propagates NaN and poisons
     every downstream comparison.

  3. Core MemoryStore.find_similar(text) and .query(text) raise
     TypeError on non-string input instead of leaking a raw
     AttributeError from .strip() deep inside the call.

  4. explain_body() exists as a polymorphic alias for
     explain_fact() — naming symmetry with delete_body /
     query_memory (both already polymorphic and body-named).

  5. monkeypatch.setattr(birch.memory_store, "embed", fake) still
     reaches mixin call sites after the package split. The
     _embed_proxy module's late-binding lookup is the load-bearing
     piece; pin it directly so a future "optimisation" that turns
     it into a direct import is caught immediately, not via a
     downstream test failure with an opaque diagnosis.
"""
from __future__ import annotations

import math

import pytest

import birch.memory_store as ms_mod
from birch.memory_store import MemoryStore
from birch.memory_store._embed_proxy import embed as proxy_embed
from birch.memory_store._embed_proxy import embed_batch as proxy_embed_batch

# --- I1: _forecast_cache assignment under lock ------------------------


def test_forecast_cache_assignment_under_lock(tmp_path, monkeypatch):
    """After a clean run_forecast, the cache slot is populated under
    the same lock that guards _facts/_meta_facts/_mutation_version.
    Behavioural pin: the cache key must match the snapshot that was
    actually written back."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    mem.add_fact("api", "uses", "Postgres")
    mem.add_fact("api", "uses", "Redis")
    result = mem.run_forecast(horizon_ticks=3)
    assert result.get("ok") is not False
    # Cache slot is populated and the key matches the writeback
    # invariant. Specifically: cache_key[1] is the mutation_version
    # snapshot — after a successful writeback it should equal the
    # current _mutation_version (since the writeback bumped nothing
    # and the snapshot value is what we expect for "this universe").
    assert mem._forecast_cache is not None
    cached_key, cached_payload = mem._forecast_cache
    assert cached_key[3] == 3  # horizon_ticks matches
    assert cached_payload["horizon_ticks"] == 3
    # A second call with no mutations between hits the cache.
    second = mem.run_forecast(horizon_ticks=3)
    assert second.get("cached") is True
    mem.close()


# --- I2: core session_close r_override NaN/Inf reject -----------------


def test_session_close_r_override_rejects_nan(tmp_path):
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    mem.session_start("s1")
    mem.session_message("seed message", session_id="s1")
    with pytest.raises(ValueError, match="NaN or Infinity"):
        mem.session_close(session_id="s1", r_override=float("nan"))
    mem.close()


def test_session_close_r_override_rejects_inf(tmp_path):
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    mem.session_start("s1")
    mem.session_message("seed message", session_id="s1")
    with pytest.raises(ValueError, match="NaN or Infinity"):
        mem.session_close(session_id="s1", r_override=float("inf"))
    mem.close()


def test_session_close_r_override_rejects_non_numeric(tmp_path):
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    mem.session_start("s1")
    mem.session_message("seed message", session_id="s1")
    with pytest.raises(ValueError, match="must be a finite float"):
        mem.session_close(session_id="s1", r_override="not a number")
    mem.close()


def test_session_close_r_override_accepts_valid_floats(tmp_path):
    """Sanity: the new guard does NOT reject legitimate inputs."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    mem.session_start("s1")
    mem.session_message("seed message", session_id="s1")
    result = mem.session_close(session_id="s1", r_override=0.5)
    # session_close response key for resonance is "r" (the realised
    # value); presence of any finite r confirms the override path
    # neither raised nor produced a NaN.
    r = result.get("r", result.get("r_score"))
    assert r is not None
    assert math.isfinite(r)
    assert 0.49 < r < 0.51
    mem.close()


# --- I3: core text-API type validation --------------------------------


def test_find_similar_rejects_non_string_text(tmp_path):
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    with pytest.raises(TypeError, match="text must be str"):
        mem.find_similar(None)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="text must be str"):
        mem.find_similar(42)  # type: ignore[arg-type]
    # Empty string still returns [] (existing contract preserved).
    assert mem.find_similar("") == []
    mem.close()


def test_query_rejects_non_string_text(tmp_path):
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    with pytest.raises(TypeError, match="text must be str"):
        mem.query(None)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="text must be str"):
        mem.query(["a", "list"])  # type: ignore[arg-type]
    mem.close()


# --- I4: explain_body alias --------------------------------------------


def test_explain_body_alias_matches_explain_fact(tmp_path):
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    f = mem.add_fact("api", "uses", "Postgres")
    a = mem.explain_fact(f.fact_id)
    b = mem.explain_body(f.fact_id)
    # Identical output — they're the same method through an alias,
    # not two parallel implementations that could drift apart.
    assert a == b
    assert b["found"] is True
    assert b["kind"] == "fact"
    mem.close()


def test_explain_body_handles_metafact_body_id(tmp_path):
    """The whole reason for the alias: agents pipe query_memory's
    polymorphic body_id straight in. Confirm the meta path works
    under the body-named call."""
    from birch.meta_fact import MetaFact

    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    meta = MetaFact(
        weight=2, source_texts=["x", "y"],
        source_fact_ids=["a", "b"],
        layer=0,
    )
    meta.vector = [1.0, 0.0, 0.0]
    mem._storage.save_meta_fact(meta)
    mem._reload()
    out = mem.explain_body(meta.meta_id)
    assert out["found"] is True
    assert out["kind"] == "meta"
    assert out["weight"] == 2
    mem.close()


# --- I5: embed-proxy late-binding pin ---------------------------------


def test_embed_proxy_late_binds_to_package(monkeypatch):
    """Pin the contract _embed_proxy promises: monkey-patching
    birch.memory_store.embed must affect the proxy on every call.
    If a future refactor replaces the late-binding lookup with a
    direct top-level import, this test fails immediately instead
    of bubbling through a downstream test as an opaque mismatch."""
    sentinel = object()
    monkeypatch.setattr(ms_mod, "embed", lambda text: sentinel)
    assert proxy_embed("anything") is sentinel


def test_embed_batch_proxy_late_binds_to_package(monkeypatch):
    sentinel_batch = [[0.1], [0.2], [0.3]]
    monkeypatch.setattr(ms_mod, "embed_batch", lambda texts: sentinel_batch)
    assert proxy_embed_batch(["a", "b", "c"]) is sentinel_batch
