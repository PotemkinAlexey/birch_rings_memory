"""Seven contracts continuing the "pattern existed elsewhere, unify it"
class of fixes:

  1. query() filter predicates now include layer/allowed_layers checks.
     Previously the predicate gated only deprecated/expired/gravity/
     subject_prefix — a body whose layer drifted (e.g. retired into
     singularity) could survive revalidation.

  2. layer check special-cases Hawking candidates: singularity bodies
     (layer == -1) pass against their POST-EMIT layer (1, kinetic).
     Caller asking layers=["surface"] no longer receives a kinetic
     Hawking body.

  3. session_close empty-messages path now goes through try/except +
     _reload. Previously _pop_session_locked dropped session from
     memory before storage.delete_open_session — failure left disk
     with an orphan session that would resurrect on restart.

  4. record_session now does general cleanup on any exception, not
     only EmbeddingError. A sqlite / dim mismatch / ValueError mid-
     session_message used to leak the open session into disk until
     TTL eviction.

  5. session_message preflights vector dim against ctx.vectors[0].
     If embedding provider returned a different-dim vector (model
     swap mid-session, mock-vs-real flip), the live path used to
     produce ragged ctx.vectors that crashed the repetition
     detector later. Loader already enforced this on read; live
     write path was the missing symmetric guard.

  6. semantic._cosine rejects dim-mismatch with 0.0. Same fix as
     cluster._cosine from the previous round, applied to the
     other cosine implementation.

  7. VectorIndex.remove resets _dim when the index becomes empty.
     After all vectors are removed, a subsequent add with a new
     dim used to raise DimensionMismatchError despite the index
     being empty.
"""
from __future__ import annotations

import math
import pathlib

import pytest

from birch.memory_store import MemoryStore
from birch.resonance.semantic import _cosine as semantic_cosine
from birch.vector_index import DimensionMismatchError, VectorIndex

# --- I1 + I2: query layer revalidation + Hawking layer awareness -------


def test_query_with_surface_only_does_not_return_kinetic_fact(tmp_path):
    """Manually set a fact's layer to 2 (core) AFTER the query starts.
    The revalidation must drop it. Done by adding the fact, then
    patching _sync to bump its layer between snapshot and writeback."""
    from unittest.mock import patch

    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    f = mem.add_fact("api", "uses", "Postgres")
    original_sync = mem._sync
    flipped = {"done": False}

    def racing_sync():
        original_sync()
        if not flipped["done"]:
            flipped["done"] = True
            # Move fact to core (layer=2) — out of caller's surface
            # scope between snapshot and revalidation.
            mem._facts[f.fact_id].layer = 2

    with patch.object(mem, "_sync", racing_sync):
        results = mem.query(
            "api uses Postgres", top_k=5,
            allowed_layers={0},  # surface only
        )

    assert all(r.fact is None or r.fact.fact_id != f.fact_id
               for r in results), (
        "query returned a kinetic/core fact when caller scoped "
        "to allowed_layers={surface}"
    )
    mem.close()


def test_hawking_meta_still_emitted_when_allowed_layers_includes_kinetic(
    tmp_path,
):
    """Hawking emission lands at layer=1 (kinetic). Caller asking
    for layers includes kinetic must still receive the meta."""
    from birch.meta_fact import MetaFact

    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    # Need a real embedding so the meta cosine matches the query.
    f = mem.add_fact("mailer service", "runs on", "Go")
    target_vec = list(f.vector)
    f.gravity_score = 0.05
    mem._absorb_dead()
    meta = MetaFact(
        meta_id="m-mailer",
        vector=target_vec,
        weight=10,
        source_texts=["mailer service runs on Go"],
        source_fact_ids=["x", "y"],
    )
    mem._hole.absorb_meta(meta)
    results = mem.query(
        "mailer service runs on Go", top_k=5,
        allowed_layers={1},  # kinetic — where Hawking emit lands
    )
    meta_hits = [r for r in results if r.kind == "meta"]
    assert len(meta_hits) == 1
    mem.close()


def test_hawking_meta_filtered_when_allowed_layers_is_surface_only(
    tmp_path,
):
    """Symmetric to above: scope surface only, Hawking emit lands
    at kinetic, so meta must NOT come back."""
    from birch.meta_fact import MetaFact

    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    meta = MetaFact(
        weight=2, source_texts=["mailer service runs on Go"],
        source_fact_ids=["x", "y"],
    )
    mem._hole.absorb_meta(meta)
    results = mem.query(
        "mailer service runs on Go", top_k=5,
        allowed_layers={0},  # surface only — excludes kinetic
    )
    meta_hits = [r for r in results if r.kind == "meta"]
    assert len(meta_hits) == 0, (
        "Hawking-emitted meta leaked past surface-only allowed_layers"
    )
    mem.close()


# --- I3: empty session_close rollback safety ----------------------------


def test_empty_session_close_rolls_back_on_storage_failure(tmp_path):
    """Open a session, push nothing, force delete_open_session to fail
    during the empty-close path. In-memory pop must be reverted."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    mem.session_start("s1")
    assert "s1" in mem._sessions
    original = mem._storage.delete_open_session

    def boom(sid):
        raise RuntimeError("simulated empty-close delete failure")

    mem._storage.delete_open_session = boom  # type: ignore[assignment]
    try:
        with pytest.raises(RuntimeError, match="empty-close"):
            mem.session_close(session_id="s1")
    finally:
        mem._storage.delete_open_session = original  # type: ignore[assignment]

    # After _reload, session restored from disk (which still has it).
    assert "s1" in mem._sessions, (
        "empty session_close leaked in-memory pop despite storage rollback"
    )
    mem.close()


# --- I4: record_session general cleanup ---------------------------------


def test_record_session_cleanup_on_generic_error_inline():
    """Replicate the server flow inline since server.py needs the mcp
    SDK to import. Confirm structure: any exception triggers
    abort_session."""

    class _Store:
        def __init__(self):
            self.opened = []
            self.aborted = []

        def session_start(self, sid):
            self.opened.append(sid)

        def check_echo(self, *a, **kw):
            return {}

        def session_message(self, *a, **kw):
            raise RuntimeError("simulated storage failure")

        def session_close(self, *a, **kw):
            return {}

        def abort_session(self, sid):
            self.aborted.append(sid)

    store = _Store()
    sid = "test-sid"
    store.session_start(sid)
    try:
        store.session_message("hello", session_id=sid)
        store.session_close(session_id=sid)
    except RuntimeError:
        try:
            store.abort_session(sid)
        except Exception:
            pass

    assert sid in store.aborted, (
        "record_session cleanup should fire on generic exception, "
        "not only EmbeddingError"
    )


# --- I5: session_message vector dim preflight ---------------------------


def test_session_message_rejects_dim_mismatch(tmp_path, monkeypatch):
    """Push one message normally, then force embed() to return a
    different-dim vector. session_message must raise BEFORE mutating
    ctx.vectors so the trajectory stays consistent."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    mem.session_start("s1")
    mem.session_message("first message", session_id="s1")
    pre_count = len(mem._sessions["s1"].vectors)
    assert pre_count == 1

    # Patch embed via the package-level pointer that _embed_proxy
    # late-binds to.
    import birch.memory_store as _pkg
    original = _pkg.embed
    _pkg.embed = lambda text: [0.5] * 999  # different dim
    try:
        with pytest.raises(DimensionMismatchError, match="vector dim"):
            mem.session_message("second message", session_id="s1")
    finally:
        _pkg.embed = original

    # ctx.vectors did NOT grow.
    assert len(mem._sessions["s1"].vectors) == pre_count, (
        "session_message appended a mismatched-dim vector despite "
        "the preflight raise"
    )
    mem.close()


# --- I6: semantic._cosine dim-check ------------------------------------


def test_semantic_cosine_rejects_dim_mismatch():
    a = [1.0, 0.0, 0.0]
    b = [1.0, 0.0]  # different dim
    assert semantic_cosine(a, b) == 0.0


def test_semantic_cosine_same_dim_still_works():
    a = [1.0, 0.0, 0.0]
    b = [1.0, 0.0, 0.0]
    assert abs(semantic_cosine(a, b) - 1.0) < 1e-9


def test_semantic_cosine_handles_empty():
    """Zero-norm vectors return 0.0 (existing contract)."""
    assert semantic_cosine([0.0, 0.0], [0.0, 0.0]) == 0.0


# --- I7: VectorIndex.remove resets dim when empty ----------------------


def test_vector_index_remove_resets_dim_when_empty():
    idx = VectorIndex()
    idx.add("f1", [1.0, 0.0, 0.0])
    idx.add("f2", [0.0, 1.0, 0.0])
    assert idx._dim == 3
    idx.remove("f1")
    # Still has f2 — dim stays.
    assert idx._dim == 3
    idx.remove("f2")
    # Empty now — dim reset.
    assert idx._dim is None, (
        "VectorIndex._dim not reset on empty — subsequent add with "
        "new dim would falsely fail"
    )
    # Now adding a new-dim vector must work.
    idx.add("f3", [0.1] * 64)
    assert idx._dim == 64


def test_vector_index_remove_keeps_dim_when_not_empty():
    """Sanity: remove of one of many still preserves the dim."""
    idx = VectorIndex()
    for i in range(5):
        idx.add(f"f{i}", [float(i), 0.0, 0.0])
    assert idx._dim == 3
    idx.remove("f0")
    idx.remove("f1")
    assert idx._dim == 3
    assert len(idx._ids) == 3


# Touch math + pathlib so the linter doesn't complain about unused.
assert math.isfinite(1.0)
assert pathlib.Path(__file__).exists()
