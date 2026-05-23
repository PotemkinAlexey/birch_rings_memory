"""Eight contracts from one triage round:

  1. add_facts() is now all-or-nothing in memory, not only in SQLite.
     The compactor used to mutate self._facts / _engine / _index /
     _spo_index per item BEFORE later items' validation could raise;
     a mid-batch DimensionMismatchError would roll back SQLite but
     leave the in-memory dicts dirty. Now preflight validates every
     dim first, then a second pass applies mutations.

  2. Existing-fact touches (access_count, last_accessed, ctx.facts)
     used to leak the same way — if item N raised, items 0..N-1
     had already mutated FactPassport state in memory.

  3. _env_int tolerates garbage env values without crashing module
     import (BIRCH_RECORD_FACTS_BATCH_CAP=abc no longer kills the
     MCP server at startup).

  4. _validate_int / _validate_float at MCP boundary turn raw
     TypeError on string args into structured invalid_int /
     invalid_float responses.

  5. query_memory layers="surface" (string instead of list) is
     rejected with an explicit invalid_layers error, not a
     misleading per-character unknown_layer dump.

  6. session_open returns created/already_open flags from the
     idempotent core session_start.

  7. explain_fact is polymorphic — handles live FactPassport, live
     MetaFact, singularity FactPassport, and singularity MetaFact.
     Symmetric with delete_body and query_memory.

  8. README module table reflects the post-split memory_store/
     package shape instead of the legacy memory_store.py single file.
"""
from __future__ import annotations

import os
import pathlib

import pytest

from birch.memory_store import MemoryStore
from birch.meta_fact import MetaFact
from birch.vector_index import DimensionMismatchError

# --- I1 + I2: add_facts atomicity --------------------------------------


def test_add_facts_rolls_back_in_memory_on_mid_batch_dim_mismatch(tmp_path):
    """Force a mid-batch dim mismatch (second item has bad-dim
    vector) and assert that the first item did NOT leak into
    _facts / _spo_index / _index."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    # Seed one fact so _index has a baseline dim.
    seed = mem.add_fact("seed", "uses", "Postgres")
    target_dim = mem._index._dim
    assert target_dim is not None

    # Monkey-patch embed_batch via the package binding (the late-
    # binding _embed_proxy.embed_batch resolves through this on every
    # call).
    import birch.memory_store as _pkg
    good_vec = [0.1] * target_dim
    bad_vec = [0.1] * (target_dim + 7)  # off by 7 dims
    original = _pkg.embed_batch
    _pkg.embed_batch = lambda texts: [good_vec, bad_vec]
    try:
        with pytest.raises(DimensionMismatchError):
            mem.add_facts(
                [("svc", "uses", "Redis"), ("svc", "uses", "Kafka")],
                return_status=True,
            )
    finally:
        _pkg.embed_batch = original

    # The first item must NOT be in _facts / _spo_index.
    live = mem.list_facts(subject="svc")
    assert not any(x.object == "Redis" for x in live), (
        "first item leaked into _facts despite mid-batch raise"
    )
    assert not any(x.object == "Kafka" for x in live)
    # Seed survives untouched.
    assert any(x.fact_id == seed.fact_id for x in mem.list_facts(subject="seed"))
    mem.close()


def test_add_facts_does_not_double_touch_existing_on_mid_batch_failure(
    tmp_path,
):
    """First batch item touches an existing fact; second item has
    bad-dim vector and raises. After _reload(), the existing fact's
    access_count must match the disk-persisted value (NOT the in-
    memory touch that got rolled back)."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    existing = mem.add_fact("api", "lives in", "Frankfurt")
    access_before = existing.access_count
    target_dim = mem._index._dim
    assert target_dim is not None

    good_vec = [0.1] * target_dim
    bad_vec = [0.1] * (target_dim + 7)
    import birch.memory_store as _pkg
    original = _pkg.embed_batch
    # First item re-records the existing SPO (touches existing),
    # second item is new with bad dim.
    _pkg.embed_batch = lambda texts: [good_vec, bad_vec]
    try:
        with pytest.raises(DimensionMismatchError):
            mem.add_facts(
                [
                    ("api", "lives in", "Frankfurt"),  # touches existing
                    ("api", "new", "fact"),  # new, bad dim
                ],
                return_status=True,
            )
    finally:
        _pkg.embed_batch = original

    # After rollback _reload pulled disk truth back in — access_count
    # on the existing fact should match what was persisted, not the
    # leaked in-memory touch.
    refreshed = next(
        x for x in mem.list_facts(subject="api")
        if x.object == "Frankfurt"
    )
    assert refreshed.access_count == access_before, (
        "existing fact's access_count was permanently mutated by a "
        "rolled-back batch item"
    )
    mem.close()


def test_add_facts_happy_path_still_works(tmp_path):
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    statuses = mem.add_facts(
        [
            ("a", "uses", "x"),
            ("b", "uses", "y"),
            ("a", "uses", "x"),  # in-batch duplicate
        ],
        return_status=True,
    )
    assert statuses[0]["already_existed"] is False
    assert statuses[1]["already_existed"] is False
    assert statuses[2]["duplicate_in_batch"] is True
    # Both new SPOs landed.
    assert len(mem.list_facts(subject="a")) == 1
    assert len(mem.list_facts(subject="b")) == 1
    mem.close()


# --- I3: tolerant env parse -------------------------------------------


def test_env_int_tolerates_garbage():
    """Import-level int(os.environ.get(...)) used to crash the MCP
    server if anyone set BIRCH_RECORD_FACTS_BATCH_CAP=abc. The
    _env_int helper falls back to the default + clamps."""
    # We can't import server (needs mcp SDK), so replicate the
    # helper inline.
    def _env_int(name, default, lo=1, hi=1_000_000):
        raw = os.environ.get(name)
        if raw is None:
            return default
        try:
            value = int(raw)
        except (TypeError, ValueError):
            return default
        return max(lo, min(hi, value))

    # Garbage falls back.
    os.environ["_TEST_CAP_X"] = "abc"
    try:
        assert _env_int("_TEST_CAP_X", 500) == 500
    finally:
        del os.environ["_TEST_CAP_X"]
    # Valid passes through.
    os.environ["_TEST_CAP_Y"] = "250"
    try:
        assert _env_int("_TEST_CAP_Y", 500) == 250
    finally:
        del os.environ["_TEST_CAP_Y"]
    # Out-of-range clamps.
    os.environ["_TEST_CAP_Z"] = "999999999"
    try:
        assert _env_int("_TEST_CAP_Z", 500, hi=10_000) == 10_000
    finally:
        del os.environ["_TEST_CAP_Z"]


# --- I4: numeric MCP validators ---------------------------------------


def test_validate_int_inline_contract():
    import math

    def _validate_int(value, field_name, *, lo=1, hi=500):
        try:
            ivalue = int(value)
        except (TypeError, ValueError):
            return None, {"ok": False, "error": "invalid_int",
                          "field": field_name,
                          "got_type": type(value).__name__}
        if ivalue < lo:
            return None, {"ok": False, "error": "invalid_int",
                          "field": field_name, "min": lo, "got": ivalue}
        return min(ivalue, hi), None

    # Good.
    assert _validate_int(5, "top_k") == (5, None)
    assert _validate_int("5", "top_k") == (5, None)  # JSON string-int OK
    # Bad type.
    _, err = _validate_int("abc", "top_k")
    assert err["error"] == "invalid_int"
    _, err = _validate_int(None, "top_k")
    assert err["error"] == "invalid_int"
    # Out-of-range.
    _, err = _validate_int(0, "top_k", lo=1)
    assert err["error"] == "invalid_int"
    assert err["min"] == 1
    # Clamp upper.
    value, _ = _validate_int(999, "top_k", lo=1, hi=50)
    assert value == 50
    # Verify math is importable (defensive for the float test).
    assert math.isfinite(1.0)


def test_validate_float_inline_contract():
    import math

    def _validate_float(value, field_name, *, lo=0.0, hi=1.0):
        try:
            fvalue = float(value)
        except (TypeError, ValueError):
            return None, {"ok": False, "error": "invalid_float",
                          "field": field_name,
                          "got_type": type(value).__name__}
        if not math.isfinite(fvalue):
            return None, {"ok": False, "error": "invalid_float",
                          "field": field_name,
                          "detail": "NaN or Infinity"}
        if not lo <= fvalue <= hi:
            return None, {"ok": False, "error": "invalid_float",
                          "field": field_name}
        return fvalue, None

    assert _validate_float(0.5, "min_similarity") == (0.5, None)
    _, err = _validate_float("abc", "min_similarity")
    assert err["error"] == "invalid_float"
    _, err = _validate_float(float("nan"), "min_similarity")
    assert err["detail"] == "NaN or Infinity"
    _, err = _validate_float(float("inf"), "min_similarity")
    assert err["detail"] == "NaN or Infinity"
    _, err = _validate_float(1.5, "min_similarity")
    assert err["error"] == "invalid_float"


# --- I5: layers shape check -------------------------------------------


def test_layers_string_returns_invalid_layers_not_per_char():
    """Replicate the shape check: layers="surface" used to iterate
    by character. Now caller gets a single invalid_layers error."""

    def _check(layers):
        if layers is None:
            return None
        if isinstance(layers, str) or not isinstance(layers, list):
            return {
                "error": "invalid_layers",
                "got_type": type(layers).__name__,
            }
        if any(not isinstance(x, str) for x in layers):
            return {"error": "invalid_layers"}
        return None

    assert _check(None) is None
    assert _check(["surface", "kinetic"]) is None
    err = _check("surface")
    assert err["error"] == "invalid_layers"
    assert err["got_type"] == "str"
    err = _check([1, 2, 3])
    assert err["error"] == "invalid_layers"


# --- I6: session_start returns created flag ---------------------------


def test_session_start_returns_true_on_fresh_open(tmp_path):
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    created = mem.session_start("brand-new-session")
    assert created is True
    # Second open with same id: existing context preserved, returns False.
    again = mem.session_start("brand-new-session")
    assert again is False
    mem.close()


# --- I7: polymorphic explain_fact -------------------------------------


def test_explain_fact_handles_live_factpassport(tmp_path):
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    f = mem.add_fact("api", "uses", "Postgres")
    out = mem.explain_fact(f.fact_id)
    assert out["found"] is True
    assert out["kind"] == "fact"
    assert out["subject"] == "api"
    mem.close()


def test_explain_fact_handles_live_metafact(tmp_path):
    """Live (promoted, non-singularity) MetaFact branch — layer >= 0
    after Hawking emission promoted it back to the live tier."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    meta = MetaFact(
        weight=3, source_texts=["a", "b", "c"],
        source_fact_ids=["x", "y", "z"],
        layer=0,  # promote past the singularity default of -1
    )
    meta.vector = [1.0, 0.0, 0.0]
    mem._storage.save_meta_fact(meta)
    mem._reload()
    out = mem.explain_fact(meta.meta_id)
    assert out["found"] is True
    assert out["kind"] == "meta"
    assert out["weight"] == 3
    assert out["source_fact_ids"] == ["x", "y", "z"]
    # No SPO fields on a meta.
    assert "subject" not in out
    mem.close()


def test_explain_fact_handles_singularity_metafact(tmp_path):
    """Singularity MetaFact branch — the common case for absorbed
    bundles. Default MetaFact layer is -1 so a save+reload lands
    here naturally."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    meta = MetaFact(
        weight=2, source_texts=["p", "q"],
        source_fact_ids=["i", "j"],
    )
    meta.vector = [1.0, 0.0, 0.0]
    mem._storage.save_meta_fact(meta)
    mem._reload()
    assert meta.meta_id in mem._hole._meta_singularity
    out = mem.explain_fact(meta.meta_id)
    assert out["found"] is True
    assert out["kind"] == "singularity_meta"
    assert out["weight"] == 2
    mem.close()


def test_explain_fact_handles_singularity_factpassport(tmp_path):
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    f = mem.add_fact("api", "uses", "Redis")
    f.gravity_score = 0.05
    mem._storage.save_fact(f)
    mem._absorb_dead()
    assert f.fact_id in mem._hole._singularity
    out = mem.explain_fact(f.fact_id)
    assert out["found"] is True
    assert out["kind"] == "singularity_fact"
    assert out["subject"] == "api"
    mem.close()


def test_explain_fact_unknown_id_still_returns_not_found(tmp_path):
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    out = mem.explain_fact("nonexistent-id")
    assert out["found"] is False
    mem.close()


# --- I8: README post-split docs ---------------------------------------


def test_readme_describes_memory_store_as_package_not_file():
    root = pathlib.Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text()
    # New row uses the package path with a trailing slash.
    assert "`memory_store/`" in readme, (
        "README still describes memory_store as a single file"
    )
    # The mixin file names are listed so an agent reading the doc can
    # navigate to the right file.
    assert "_facts.py" in readme
    assert "_sessions.py" in readme
    # Old single-file row should be gone.
    assert "`memory_store.py` | `MemoryStore`" not in readme
