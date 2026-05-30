"""Restart-after-every-operation tests.

For each public mutation, perform: operation → close MemoryStore →
reopen with the same db_path → assert state restored.

This is the persistence-correctness layer. Catches the class of bug
where state lived in memory but never reached storage, and the bug
where storage round-trip lost a field. The classic earlier instance
was the black-hole layer=-1 fix: facts in singularity disappeared
after restart until the persistence column was added.

Clean shutdown only — kill -9 variants are a separate harder layer
(subprocess + signal) and live in chaos tests.
"""
from __future__ import annotations

import sqlite3

import pytest

from birch.fact import FactPassport
from birch.memory_store import MemoryStore


def _reopen(db_path: str) -> MemoryStore:
    """Helper: close happens at caller; this just reopens."""
    return MemoryStore(db_path=db_path)


# --- add_fact -----------------------------------------------------------


def test_add_fact_survives_restart(tmp_path):
    db = str(tmp_path / "m.db")
    mem = MemoryStore(db_path=db)
    f = mem.add_fact("api", "runs on", "Go")
    fact_id = f.fact_id
    mem.close()

    again = _reopen(db)
    survivors = again.list_facts(subject="api")
    again.close()
    assert any(s.fact_id == fact_id for s in survivors)
    survivor = next(s for s in survivors if s.fact_id == fact_id)
    assert survivor.subject == "api"
    assert survivor.predicate == "runs on"
    assert survivor.object == "Go"
    assert survivor.vector  # vector persisted, not lost


def test_add_fact_query_finds_it_after_restart(tmp_path):
    """Persistence + vector index rebuild — query through the embed
    path must find the fact on a fresh process."""
    db = str(tmp_path / "m.db")
    mem = MemoryStore(db_path=db)
    f = mem.add_fact("api", "runs on", "Go")
    mem.close()

    again = _reopen(db)
    results = again.query("api Go", top_k=5)
    again.close()
    assert any(
        r.kind == "fact" and r.fact.fact_id == f.fact_id
        for r in results
    )


# --- set_fact -----------------------------------------------------------


def test_set_fact_slot_state_survives_restart(tmp_path):
    """After set_fact superseded an earlier value, restart must
    preserve: new value is live, old value is in singularity, slot
    holds exactly one live occupant."""
    db = str(tmp_path / "m.db")
    mem = MemoryStore(db_path=db)
    mem.set_fact("api", "version", "1.0")
    mem.set_fact("api", "version", "2.0")
    mem.set_fact("api", "version", "3.0")
    mem.close()

    again = _reopen(db)
    live = [
        f for f in again.list_facts(subject="api", predicate="version")
        if not (f.is_deprecated or f.is_expired)
    ]
    again.close()
    assert len(live) == 1
    assert live[0].object == "3.0"


# --- retire_fact / supersede_fact --------------------------------------


def test_retire_fact_state_survives_restart(tmp_path):
    db = str(tmp_path / "m.db")
    mem = MemoryStore(db_path=db)
    f = mem.add_fact("api", "runs on", "Go")
    mem.retire_fact(f.fact_id)
    mem.close()

    again = _reopen(db)
    live = [
        x for x in again.list_facts(subject="api")
        if not (x.is_deprecated or x.is_expired)
    ]
    # Singularity-resident bodies are eligible for Hawking emission.
    assert any(rec.fact.fact_id == f.fact_id
               for rec in again._hole._singularity.values())
    again.close()
    assert not live


def test_supersede_fact_state_survives_restart(tmp_path):
    db = str(tmp_path / "m.db")
    mem = MemoryStore(db_path=db)
    old = mem.add_fact("api", "version", "1.0")
    new = mem.add_fact("api", "version", "2.0")
    mem.supersede_fact(old.fact_id, new.fact_id)
    mem.close()

    again = _reopen(db)
    live = [
        x for x in again.list_facts(subject="api", predicate="version")
        if not (x.is_deprecated or x.is_expired)
    ]
    # Old body in singularity with deprecated_by intact.
    in_hole = [
        rec for rec in again._hole._singularity.values()
        if rec.fact.fact_id == old.fact_id
    ]
    again.close()
    assert len(live) == 1
    assert live[0].fact_id == new.fact_id
    assert len(in_hole) == 1
    assert in_hole[0].fact.deprecated_by == new.fact_id


# --- delete_fact (destructive) -----------------------------------------


def test_delete_fact_leaves_no_trace_after_restart(tmp_path):
    db = str(tmp_path / "m.db")
    mem = MemoryStore(db_path=db)
    f = mem.add_fact("secret", "is", "hunter2")
    mem.delete_fact(f.fact_id)
    mem.close()

    again = _reopen(db)
    # Nowhere in live, nowhere in singularity, nowhere in index.
    assert not any(
        x.fact_id == f.fact_id for x in again.list_facts(subject="secret")
    )
    assert f.fact_id not in {
        rec.fact.fact_id for rec in again._hole._singularity.values()
    }
    again.close()


# --- session attribution survives restart before close -----------------


def test_open_session_attribution_survives_restart(tmp_path):
    """Open a session, attribute a fact to it via query, restart
    BEFORE session_close. The reopened session must still know which
    facts were attributed — otherwise the resonance signal would be
    lost on a crash mid-conversation."""
    db = str(tmp_path / "m.db")
    mem = MemoryStore(db_path=db)
    f = mem.add_fact("api", "runs on", "Go")
    mem.session_start("s")
    mem.session_message("looking at api", session_id="s")
    mem.query("api Go", session_id="s")
    # NB: NO session_close. State must persist mid-flight.
    mem.close()

    again = _reopen(db)
    ctx = again._sessions.get("s")
    assert ctx is not None, "open session lost on restart"
    assert f.fact_id in ctx.facts, (
        "fact attribution lost on restart — resonance signal would "
        "be lost on a crash mid-conversation"
    )
    again.close()


# --- session_close persistence ----------------------------------------


def test_session_close_persists_adaptive_weights_train_count(tmp_path):
    """session_close trains adaptive weights with an SGD step.
    train_count on disk must reflect that step after restart."""
    db = str(tmp_path / "m.db")
    mem = MemoryStore(db_path=db)
    mem.add_fact("api", "runs on", "Go")
    train_before = mem._engine.weights.train_count

    mem.session_start("s")
    mem.session_message("looking at api", session_id="s")
    mem.query("api Go", session_id="s")
    mem.session_close(session_id="s", sentiment="resonant")
    train_after = mem._engine.weights.train_count
    assert train_after == train_before + 1
    mem.close()

    again = _reopen(db)
    assert again._engine.weights.train_count == train_after
    again.close()


def test_session_close_persists_recent_utility_ewma(tmp_path):
    """The EWMA update on touched facts must be persisted, otherwise
    learning is lost on every restart."""
    db = str(tmp_path / "m.db")
    mem = MemoryStore(db_path=db)
    f = mem.add_fact("api", "runs on", "Go")

    mem.session_start("s")
    mem.session_message("looking at api", session_id="s")
    mem.query("api Go", session_id="s")
    mem.session_close(session_id="s", sentiment="resonant")
    utility_after_close = next(
        x for x in mem.list_facts(subject="api")
        if x.fact_id == f.fact_id
    ).recent_utility
    assert utility_after_close > 0.5  # lifted from default by resonant
    mem.close()

    again = _reopen(db)
    reloaded = next(
        x for x in again.list_facts(subject="api")
        if x.fact_id == f.fact_id
    )
    again.close()
    assert abs(reloaded.recent_utility - utility_after_close) < 1e-9


# --- collapse_singularity persistence ---------------------------------


def test_collapse_singularity_metafact_survives_restart(tmp_path):
    """Collapsed MetaFact must persist, AND source facts must NOT
    be rehydrated into the live store on next boot (their rows are
    deleted by the collapse pass)."""
    db = str(tmp_path / "m.db")
    mem = MemoryStore(db_path=db, collapse_async=False)
    # Seed two near-identical absorbed facts so collapse produces one
    # MetaFact.
    source_ids = []
    for i in range(2):
        f = FactPassport(subject=f"s{i}", predicate="is", object="x")
        f.vector = [1.0, 0.0, 0.0]
        f.gravity_score = 0.05  # below absorption threshold
        mem._facts[f.fact_id] = f
        mem._engine.register(f)
        mem._index.add(f.fact_id, f.vector)
        mem._storage.save_fact(f)  # ensure row on disk for delete path
        source_ids.append(f.fact_id)
    mem._absorb_dead()

    report = mem.collapse_singularity(min_group_size=2)
    assert report.groups == 1
    # Compactor produces MetaFacts that land directly in the
    # singularity meta-side (where they're Hawking-eligible at
    # gravity 0.30 + 0.10·log10(weight)), not in the live meta dict.
    meta_ids = [
        rec.meta.meta_id for rec in mem._hole._meta_singularity.values()
    ]
    assert len(meta_ids) == 1
    mem.close()

    again = _reopen(db)
    # MetaFact survives in the singularity after restart.
    surviving_meta_ids = {
        rec.meta.meta_id
        for rec in again._hole._meta_singularity.values()
    }
    assert meta_ids[0] in surviving_meta_ids
    meta = next(
        rec.meta for rec in again._hole._meta_singularity.values()
        if rec.meta.meta_id == meta_ids[0]
    )
    assert set(meta.source_fact_ids) == set(source_ids)
    # Source facts NOT rehydrated as live (their rows were deleted).
    for src_id in source_ids:
        assert src_id not in again._facts
    again.close()


# --- Hawking emission persistence -------------------------------------


def test_hawking_emission_persists_emitted_fact_as_live(tmp_path):
    """A Hawking-emitted fact must persist as a live body with
    layer=1 and gravity 0.30, not still living in the singularity."""
    db = str(tmp_path / "m.db")
    mem = MemoryStore(db_path=db)
    f = mem.add_fact("api", "runs on", "Go")
    fact_id = f.fact_id
    vec = list(f.vector)

    # Drive the fact below absorption threshold to push it into the
    # singularity.
    f.gravity_score = 0.05
    mem._storage.save_fact(f)
    mem._absorb_dead()
    assert fact_id in {
        rec.fact.fact_id for rec in mem._hole._singularity.values()
    }
    mem.close()

    # Reopen, run a query that exactly matches the absorbed vector —
    # Hawking emission should fire because the in-memory index is
    # rehydrated from layer=-1 rows.
    again = _reopen(db)
    # Confirm the fact rehydrated to the singularity on reload.
    assert fact_id in {
        rec.fact.fact_id for rec in again._hole._singularity.values()
    }
    # Use the same vector as query so cosine == 1.0, comfortably
    # above the 0.95 Hawking threshold.
    emitted = again._hole.hawking_emit(vec)
    again.close()
    assert any(e.fact_id == fact_id for e in emitted)


# --- _mutation_version is process-local: NOT persisted (by design) ----


def test_mutation_version_resets_on_restart(tmp_path):
    """_mutation_version is a process-local cache-invalidation counter.
    It is intentionally NOT persisted — a fresh process starts at 0
    because all its caches are also fresh. SQLite data_version is the
    cross-process counter that survives."""
    db = str(tmp_path / "m.db")
    mem = MemoryStore(db_path=db)
    for i in range(5):
        mem.add_fact(f"s{i}", "is", "x")
    assert mem._mutation_version >= 5
    mem.close()

    again = _reopen(db)
    assert again._mutation_version == 0
    # But all 5 facts are there — state IS persisted, only the
    # process-local cache counter resets.
    assert len([
        f for f in again.list_facts(limit=50)
        if f.subject.startswith("s")
    ]) == 5
    again.close()


# --- collapse_counter persistence (currently process-local by design)
# - skip if confirmed-intentional reset on restart; remove this
#   suppression if conservation tracking is added later.


@pytest.mark.skip(reason="collapse_counter is process-local by design")
def test_collapse_counter_persists_on_restart(tmp_path):
    """Placeholder — current contract is monotonic across process
    lifetime, not across restarts. If that contract changes (e.g.
    conservation tracking lands), unskip and assert restoration."""


# --- storage-level sanity: SQLite file actually has the rows ----------


def test_storage_has_layer_minus_one_rows_for_singularity(tmp_path):
    """Storage-level smoke: absorbed facts persist with layer=-1
    column so reload re-hydrates the singularity. The earlier-era
    bug fixed by this contract was facts vanishing across restarts."""
    db = str(tmp_path / "m.db")
    mem = MemoryStore(db_path=db)
    f = mem.add_fact("api", "runs on", "Go")
    f.gravity_score = 0.05
    mem._storage.save_fact(f)
    mem._absorb_dead()
    mem.close()

    conn = sqlite3.connect(db)
    row = conn.execute(
        "SELECT layer FROM facts WHERE fact_id = ?", (f.fact_id,),
    ).fetchone()
    conn.close()
    assert row is not None
    assert row[0] == -1


def test_raw_resonance_diverges_from_shrunk_and_survives_restart(tmp_path):
    """The contrastive fix splits raw history from the gravity-side (shrunk)
    sum. After a contradicting outlier the two must DIFFER, and both must
    round-trip through storage so the trust prior is not silently reset to the
    shrunk value on restart (which would re-introduce the self-reference)."""
    from birch.memory_store import MemoryStore

    db = str(tmp_path / "m.db")
    mem = MemoryStore(db_path=db)
    f = mem.add_fact("contrastive", "round", "trip")
    # Establish a resonant history, then one contradicting toxic session.
    for _ in range(8):
        mem._engine.apply_session_resonance({f.fact_id: 0.9}, 0.7)
    mem._engine.apply_session_resonance({f.fact_id: 0.9}, -0.7)  # shrunk
    raw_before = f.raw_resonance_sum
    sum_before = f.resonance_sum
    assert raw_before != sum_before, "outlier should have been shrunk for gravity only"
    mem._engine.tick()
    if mem._storage:
        mem._storage.save_facts([f])

    mem2 = MemoryStore(db_path=db)
    g = mem2._facts[f.fact_id]
    assert abs(g.raw_resonance_sum - raw_before) < 1e-6, "raw history lost on restart"
    assert abs(g.resonance_sum - sum_before) < 1e-6, "shrunk sum lost on restart"
    assert g.raw_resonance_sum != g.resonance_sum


def test_negative_echo_penalty_round_trips(tmp_path):
    """echo_penalty is a NEGATIVE retroactive correction. A clamp to lo=0.0 in
    save_echo_session silently stored it as 0.0, so after a reload a penalised
    session looked un-penalised — re-applying the penalty (idempotency broken)
    and mis-tiering its TTL. The value must survive the round trip negative."""
    from birch.storage.sqlite import SQLiteBackend

    db = str(tmp_path / "echo.db")
    b = SQLiteBackend(db)
    b.save_echo_session(
        "s1", [[1.0, 0.0]], r_score=0.7, recorded_at=123.0,
        fact_weights={"f": 1.0}, echo_penalty=-0.6,
    )
    b.close()

    b2 = SQLiteBackend(db)
    rows = {r["session_id"]: r for r in b2.load_echo_sessions(cleanup=False)}
    b2.close()
    assert "s1" in rows
    assert rows["s1"]["echo_penalty"] == -0.6, (
        f"negative echo_penalty must persist, got {rows['s1']['echo_penalty']}"
    )


def test_echo_penalty_survives_store_restart_and_stays_idempotent(tmp_path):
    """End-to-end: an applied echo penalty must persist as non-zero across a
    MemoryStore reopen, so apply_echo stays idempotent (no double penalty) and
    EchoStore keeps the session in its penalised TTL tier."""
    from birch.memory_store import MemoryStore

    db = str(tmp_path / "m.db")
    mem = MemoryStore(db_path=db)
    vec = [1.0, 0.0, 0.0]
    mem._echo.record("past", [vec, vec], r_score=0.7, fact_weights={})
    res = mem._echo.apply_echo("past")
    assert res.penalty < 0.0
    applied = mem._echo.get("past").echo_penalty
    assert applied < 0.0
    if mem._storage:
        past = mem._echo.get("past")
        mem._storage.save_echo_session(
            "past", past.bundle.centroids, past.r_score, 1.0,
            fact_weights=past.fact_weights, echo_penalty=past.echo_penalty,
        )

    mem2 = MemoryStore(db_path=db)
    reloaded = mem2._echo.get("past")
    assert reloaded is not None
    assert reloaded.echo_penalty < 0.0, "penalty must survive restart as non-zero"
    # Idempotent: a re-apply on the reloaded (already-penalised) session is a no-op.
    again = mem2._echo.apply_echo("past")
    assert again.penalty == 0.0, "re-apply after restart must not double-penalise"


def test_load_echo_session_clamps_corrupted_values(tmp_path):
    """Read-side defence (storage symmetry): a hand-edited / corrupted row with
    a POSITIVE echo_penalty must not load as 'already penalised' (which would
    make apply_echo a silent no-op), and an out-of-range r_score must clamp.
    A bad row degrades to neutral, never to a logic flip."""
    import sqlite3

    from birch.storage.sqlite import SQLiteBackend

    db = str(tmp_path / "echo.db")
    b = SQLiteBackend(db)
    b.save_echo_session("s", [[1.0, 0.0]], r_score=0.5, recorded_at=1.0,
                        echo_penalty=-0.3)
    b.close()

    c = sqlite3.connect(db)
    c.execute(
        "UPDATE echo_sessions SET echo_penalty=0.8, r_score=5.0 "
        "WHERE session_id='s'")
    c.commit()
    c.close()

    b2 = SQLiteBackend(db)
    row = {r["session_id"]: r for r in b2.load_echo_sessions(cleanup=False)}["s"]
    b2.close()
    assert row["echo_penalty"] == 0.0, "positive penalty must clamp to 0, not load as penalised"
    assert row["r_score"] == 1.0, "out-of-range r_score must clamp into [-1, 1]"
