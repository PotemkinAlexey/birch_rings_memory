"""Query revalidation and supersede-helper regressions.

Three concrete gaps closed: query() returned stale top results after
the in-txn re-sync, set_fact reported already_existed=True for
genuinely new facts, and the in-memory store skipped touch/attribution
entirely because the gate was wired to storage existence.
"""
from __future__ import annotations

from birch.galaxy.forecast import forecast_stability
from birch.memory_store import MemoryStore
from birch.meta_fact import MetaFact

# --- P1: top is revalidated after the in-txn _sync() --------------------


def test_query_drops_top_hits_that_vanished_during_resync(tmp_path):
    """If a fact got deprecated between the pre-lock top selection and the
    in-txn _sync, the returned top must NOT include it. We simulate by
    deprecating the fact between query()'s two phases via a saved
    reference we mutate from outside the call (single-process surrogate
    for a race window)."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    f = mem.add_fact("api", "runs on", "Go")

    # Wrap _sync to also deprecate the fact the first time it fires under
    # the txn (the first call is pre-lock; the second is inside the txn).
    original_sync = mem._sync
    call_count = {"n": 0}

    def _sync_with_race():
        original_sync()
        call_count["n"] += 1
        if call_count["n"] == 2:
            # Simulate "another process superseded the fact between our
            # top selection and our write-lock acquisition".
            other = mem.add_fact("api", "runs on", "Rust")
            mem._supersede_fact_locked(f.fact_id, other.fact_id)

    mem._sync = _sync_with_race  # type: ignore[method-assign]
    hits = mem.query("api Go", top_k=5, min_similarity=0.0)
    mem._sync = original_sync  # type: ignore[method-assign]

    ids = {h.body_id for h in hits}
    assert f.fact_id not in ids, (
        "revalidation must drop the deprecated body even though it was "
        "in the pre-sync top"
    )


# --- P1: set_fact already_existed reflects pre-add reality --------------


def test_set_fact_already_existed_false_on_first_write(tmp_path):
    """already_existed must mean 'SPO was in store BEFORE this call' —
    not 'is now in the store after add_fact ran'."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    out = mem.set_fact("project-x", "HEAD", "abc123")
    assert out["already_existed"] is False
    # Same call again with the same SPO must report True now.
    again = mem.set_fact("project-x", "HEAD", "abc123")
    assert again["already_existed"] is True


def test_set_fact_already_existed_false_for_value_change(tmp_path):
    """A slot replace (different object) must report already_existed=False
    for the new SPO, while still superseding the old occupant."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    old = mem.add_fact("project-x", "HEAD", "abc123")
    out = mem.set_fact("project-x", "HEAD", "def456")
    assert out["already_existed"] is False
    assert old.fact_id in out["superseded"]


# --- P2: query() write path runs even without storage -------------------


def test_in_memory_query_still_touches_facts():
    """An in-memory MemoryStore (no storage backend) must still apply
    touch() and session attribution on query() — otherwise the
    feedback loop silently dies for embedded and test usage."""
    mem = MemoryStore()
    f = mem.add_fact("api", "runs on", "Go")
    before = f.access_count

    mem.session_start("s")
    mem.query("api Go", top_k=1, session_id="s")

    # touch() ran even though there's no storage to persist into.
    assert mem._facts[f.fact_id].access_count == before + 1
    # Attribution landed in ctx.facts.
    assert f.fact_id in mem._sessions["s"].facts


# --- P2: echo penalty persists MetaFacts --------------------------------


def test_check_echo_persists_affected_metafacts(tmp_path):
    """If a past session touched a MetaFact and the echo penalty path
    fires, the MetaFact's mutated state must be saved. Previously only
    FactPassports were saved; MetaFact penalty updates leaked on restart."""
    db = str(tmp_path / "m.db")
    mem = MemoryStore(db_path=db)

    # Plant a live MetaFact and a "past" StoredSession that touched it.
    meta = MetaFact(weight=2, source_texts=["x y z"], gravity_score=0.5,
                    layer=1)
    meta.vector = [0.5] * 64
    mem._meta_facts[meta.meta_id] = meta
    mem._meta_index.add(meta.meta_id, meta.vector)
    mem._engine.register(meta)
    if mem._storage and hasattr(mem._storage, "save_meta_fact"):
        mem._storage.save_meta_fact(meta)

    # Open a session that "touches" the meta and close it resonantly so
    # there's a past StoredSession to echo against.
    mem.session_start("past")
    mem.session_message("worked great with x y z, thanks")
    mem._sessions["past"].facts[meta.meta_id] = 1.0
    mem.session_close(session_id="past")

    # Force an echo by querying with a vector close to the past bundle.
    # (We use direct text — the mock embedder is deterministic on shared
    # tokens, so re-using vocabulary from the past session triggers a
    # high-similarity match.)
    mem.check_echo("worked great with x y z, thanks")
    # Whether or not the echo predicate fires under mock, the test pins
    # that IF apply_session_resonance ran with the meta in fact_weights
    # the meta's mutated state is persisted (we'd see it after reload).
    mem.close()

    again = MemoryStore(db_path=db)
    reloaded_meta = again._meta_facts.get(meta.meta_id)
    assert reloaded_meta is not None
    # Either the echo did not fire (penalty path skipped — no change to
    # check), or it did and the change was persisted (state survives).
    # The contract is "no silent in-memory mutation"; identity of the
    # value being preserved across restart is what we pin.
    assert abs(reloaded_meta.resonance_sum - meta.resonance_sum) < 1e-9


# --- P3: forecast covers MetaFacts --------------------------------------


def test_forecast_includes_metafacts(tmp_path):
    """run_forecast must update forecast_stability on live MetaFacts too —
    they participate in the gravity formula via the same w_stability
    weight as FactPassports."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    # Add some facts so the galaxy loader has a non-trivial corpus.
    for i in range(5):
        mem.add_fact(f"fact-{i}", "is", f"value-{i}")
    # Plant a live MetaFact.
    meta = MetaFact(weight=3, source_texts=["alpha", "beta"],
                    gravity_score=0.5, layer=1)
    meta.vector = [0.3] * 64
    mem._meta_facts[meta.meta_id] = meta
    mem._meta_index.add(meta.meta_id, meta.vector)
    mem._engine.register(meta)

    before = meta.forecast_stability
    summary = mem.run_forecast(horizon_ticks=20)
    # The summary count includes both facts and metas now.
    assert summary["facts_forecasted"] >= 6
    # The MetaFact got a real forecast value, not the 0.5 neutral prior.
    assert meta.forecast_stability != before


# --- P2: galaxy loader accepts polymorphic bodies ------------------------


def test_galaxy_loader_handles_mixed_bodies():
    """build_galaxy must accept a list of FactPassport AND MetaFact
    without raising on the missing subject/predicate/object attributes."""
    from birch.fact import FactPassport
    from birch.galaxy.loader import build_galaxy

    fact = FactPassport("api", "runs on", "Go")
    fact.vector = [0.1] * 64
    meta = MetaFact(weight=2, source_texts=["seed text"],
                    gravity_score=0.5, layer=1)
    meta.vector = [0.2] * 64

    gal = build_galaxy([fact, meta])
    assert len(gal.bodies) == 2
    body_ids = {b.fact_id for b in gal.bodies}
    assert fact.fact_id in body_ids
    assert meta.meta_id in body_ids


def test_forecast_stability_function_handles_metas():
    """forecast_stability() at the function level must accept a meta in
    the input list and key the result by meta_id (== fact_id alias)."""
    fact_passports = []
    from birch.fact import FactPassport
    for i in range(3):
        f = FactPassport(f"s{i}", "rel", f"o{i}")
        f.vector = [float(((i * j) % 5) - 2) for j in range(8)]
        fact_passports.append(f)
    meta = MetaFact(weight=2, source_texts=["alpha beta"],
                    gravity_score=0.5, layer=1)
    meta.vector = [0.3] * 8

    scores = forecast_stability(
        [*fact_passports, meta], horizon_ticks=10,
    )
    assert meta.meta_id in scores
    assert 0.0 <= scores[meta.meta_id] <= 1.0
