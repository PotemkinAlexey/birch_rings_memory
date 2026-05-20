"""MemoryStore — full lifecycle integration test."""
from birch.memory_store import MemoryStore


def _build_store():
    mem = MemoryStore()
    f_go     = mem.add_fact("mailer service", "runs on",    "Go")
    f_python = mem.add_fact("legacy script",  "written in", "Python")
    f_db     = mem.add_fact("database",       "uses",       "PostgreSQL")
    mem.link(f_go.fact_id, f_db.fact_id)
    mem.deprecate(f_python.fact_id, f_go.fact_id)
    return mem, f_go, f_python, f_db


def test_deprecated_fact_absorbed_on_resonant_session():
    mem, f_go, f_python, f_db = _build_store()
    mem.session_start("session_A")
    mem.session_message("how to configure the mailer service on Go")
    mem.session_message("how to connect it to PostgreSQL")
    mem.session_message("everything works, thanks!")
    mem._session_fact_ids = [f_go.fact_id, f_db.fact_id]
    summary = mem.session_close()
    assert summary["label"] == "resonant"
    assert f_python.fact_id in summary["absorbed"], "deprecated fact should be absorbed"
    assert mem.stats["black_hole_mass"] >= 1


def test_query_returns_relevant_facts():
    mem, f_go, _, f_db = _build_store()
    results = mem.query("how does the mailer service work", top_k=3)
    assert len(results) > 0
    assert results[0].fact.fact_id == f_go.fact_id, "mailer service fact should rank first"
    assert results[0].similarity > 0.7


def test_echo_detected_after_toxic_session():
    mem, f_go, f_python, f_db = _build_store()
    mem.session_start("session_B")
    mem.session_message("why is the old python script not working")
    mem.session_message("still not working")
    mem.session_message("I don't understand why it's not working")
    mem.session_close()
    echo = mem.check_echo("old python script not working again")
    assert echo["echo"] is True
    assert echo["similarity"] > 0.60


def test_hawking_emission():
    mem, _, _, _ = _build_store()
    f_dead = mem.add_fact("expired token", "expired at", "2024-01-01")
    f_dead.gravity_score = 0.05
    mem._engine.register(f_dead)
    mem._absorb_dead()
    assert mem.stats["black_hole_mass"] >= 1

    results = mem.query("expired token expired at 2024-01-01", top_k=1, hawking=True)
    assert results, "Hawking emission should return at least one fact"
    assert results[0].source == "hawking"
    assert results[0].similarity > 0.95


def test_hawking_removes_fact_from_singularity_and_persists():
    """Fix #3: emitted facts must exit the singularity and reappear as live."""
    mem = MemoryStore()
    f_dead = mem.add_fact("expired token", "expired at", "2024-01-01")
    f_dead.gravity_score = 0.05
    mem._absorb_dead()
    assert mem.stats["black_hole_mass"] == 1
    assert f_dead.fact_id not in mem._facts

    first = mem.query("expired token expired at 2024-01-01", top_k=1, hawking=True)
    assert first[0].source == "hawking"
    assert mem.stats["black_hole_mass"] == 0, "singularity should release the emitted fact"
    assert f_dead.fact_id in mem._facts, "emitted fact must be re-registered in live store"

    second = mem.query("expired token expired at 2024-01-01", top_k=1, hawking=True)
    assert second, "live fact should still be queryable on the second call"
    assert second[0].source != "hawking", "live fact should not be double-emitted"
    assert mem._hole.total_emissions == 1


def test_query_attributes_facts_to_active_session():
    """Fix #1: facts returned by query() must inherit the session's resonance."""
    mem = MemoryStore()
    f = mem.add_fact("mailer service", "runs on", "Go")
    assert f.resonance_count == 0
    initial_access = f.access_count

    mem.session_start("s_attr")
    mem.session_message("how does the mailer service work")
    results = mem.query("mailer service Go", top_k=3)
    assert results, "query should return at least one fact"
    assert f.fact_id in mem._session_fact_ids, "queried fact must be attributed to the session"
    assert f.access_count > initial_access, "query() must touch the returned fact"

    mem.session_message("works, thanks!")
    summary = mem.session_close()
    assert summary["label"] == "resonant"
    assert f.resonance_count == 1, "session R must propagate to the queried fact"
    assert f.resonance_sum > 0


def test_query_without_active_session_does_not_tag():
    """Touching is fine outside a session, but attribution must stay scoped."""
    mem = MemoryStore()
    f = mem.add_fact("mailer service", "runs on", "Go")
    initial_access = f.access_count

    results = mem.query("mailer service Go", top_k=1)
    assert results
    assert mem._session_fact_ids == [], "no session active → no attribution"
    assert f.access_count > initial_access, "touch should still happen on direct query"


def test_query_dedupes_attribution_within_session():
    """Repeated queries to the same fact must not double-count in resonance."""
    mem = MemoryStore()
    f = mem.add_fact("mailer service", "runs on", "Go")

    mem.session_start("s_dedup")
    mem.query("mailer service", top_k=1)
    mem.query("mailer service Go", top_k=1)
    mem.query("Go mailer", top_k=1)

    assert mem._session_fact_ids.count(f.fact_id) == 1, "fact must appear at most once per session"


def test_sqlite_save_facts_batches_in_one_transaction(tmp_path):
    """save_facts must round-trip via SQLite in a single transaction."""
    from birch.storage.sqlite import SQLiteBackend

    db_path = tmp_path / "batch.db"
    mem = MemoryStore(db_path=str(db_path))
    f1 = mem.add_fact("alpha", "is", "first")
    f2 = mem.add_fact("beta",  "is", "second")
    f3 = mem.add_fact("gamma", "is", "third")

    f1.gravity_score = 0.81
    f2.gravity_score = 0.42
    f3.gravity_score = 0.13
    mem._storage.save_facts([f1, f2, f3])

    reopened = SQLiteBackend(str(db_path))
    by_id = {f.fact_id: f for f in reopened.load_facts()}
    assert by_id[f1.fact_id].gravity_score == 0.81
    assert by_id[f2.fact_id].gravity_score == 0.42
    assert by_id[f3.fact_id].gravity_score == 0.13


def test_query_attribution_uses_similarity_as_weight():
    """C5: facts returned with high similarity weigh more than near-noise hits."""
    mem = MemoryStore()
    f_relevant = mem.add_fact("mailer service", "runs on", "Go")
    f_unrelated = mem.add_fact("siamese cat", "fur color", "cream")

    mem.session_start("s_weight")
    mem.query("mailer service Go programming language", top_k=2)

    assert f_relevant.fact_id in mem._session_facts
    assert f_unrelated.fact_id in mem._session_facts
    w_rel = mem._session_facts[f_relevant.fact_id]
    w_unr = mem._session_facts[f_unrelated.fact_id]
    assert w_rel > w_unr, f"relevant fact must outweigh noise: {w_rel} vs {w_unr}"
    assert 0.0 <= w_unr <= w_rel <= 1.0


def test_add_fact_pins_weight_to_one():
    """Explicit add_fact during a session must pin weight=1.0."""
    mem = MemoryStore()
    mem.session_start("s_pin")
    f = mem.add_fact("mailer service", "runs on", "Go")
    assert mem._session_facts[f.fact_id] == 1.0


def test_weighted_resonance_propagates_proportionally():
    """A high-weight fact must receive more resonance than a low-weight one."""
    mem = MemoryStore()
    f_relevant = mem.add_fact("mailer service", "runs on", "Go")
    f_unrelated = mem.add_fact("siamese cat", "fur color", "cream")

    mem.session_start("s_prop")
    mem.query("mailer service Go programming language", top_k=2)
    mem.session_message("how do I configure the mailer service")
    mem.session_message("works, thanks!")
    summary = mem.session_close()
    assert summary["label"] == "resonant"

    # Both received some resonance, but the relevant fact should have received
    # strictly more (positive R × bigger weight).
    assert f_relevant.resonance_sum > f_unrelated.resonance_sum
    assert f_relevant.resonance_count == 1
    assert f_unrelated.resonance_count == 1


def test_add_fact_dedupes_identical_triples():
    """Re-adding the same SPO triple should not create a duplicate fact."""
    mem = MemoryStore()
    a = mem.add_fact("user", "prefers", "dark mode")
    initial_access = a.access_count
    b = mem.add_fact("user", "prefers", "dark mode")
    assert a.fact_id == b.fact_id, "identical SPO must collapse to one fact"
    assert mem.stats["total_live"] == 1
    assert b.access_count > initial_access, "dedupe path must still touch the fact"


def test_add_fact_dedupe_is_case_and_whitespace_insensitive():
    mem = MemoryStore()
    a = mem.add_fact("User",  "Prefers", "dark mode")
    b = mem.add_fact("user ", "prefers", "DARK   MODE")
    assert a.fact_id == b.fact_id


def test_add_fact_dedupe_attributes_to_active_session():
    mem = MemoryStore()
    a = mem.add_fact("user", "prefers", "dark mode")

    mem.session_start("s_dedup_attr")
    b = mem.add_fact("user", "prefers", "dark mode")
    assert a.fact_id == b.fact_id
    assert mem._session_fact_ids == [a.fact_id], (
        "dedupe path must still attribute the fact to the active session"
    )


def test_deprecated_fact_frees_spo_slot():
    """A deprecated fact must not block a new fact with the same SPO."""
    mem = MemoryStore()
    old = mem.add_fact("mailer", "runs on", "Python")
    new_replacement = mem.add_fact("mailer", "runs on", "Go")
    mem.deprecate(old.fact_id, new_replacement.fact_id)

    revived = mem.add_fact("mailer", "runs on", "Python")
    assert revived.fact_id != old.fact_id, "deprecated fact must release its SPO slot"


def test_echo_drags_down_gravity_of_misleading_facts():
    """Fix #2: when echo is detected, facts of the past session lose resonance."""
    mem = MemoryStore()
    f_misleading = mem.add_fact("legacy script", "written in", "Python")
    assert f_misleading.resonance_count == 0

    # Pretend a past session leaned on this fact and looked resonant at close.
    mem.session_start("session_past")
    mem.session_message("how does the legacy python script work")
    mem.query("legacy script python", top_k=3)
    assert f_misleading.fact_id in mem._session_fact_ids
    mem.session_message("ok, got it!")
    mem.session_close()

    rcount_before = f_misleading.resonance_count
    rsum_before = f_misleading.resonance_sum

    # User comes back stuck — echo is triggered.
    echo = mem.check_echo("legacy python script still not working")
    assert echo["echo"] is True, "high-similarity follow-up must trigger echo"
    assert f_misleading.fact_id in echo["penalized_fact_ids"]
    assert f_misleading.resonance_count == rcount_before + 1
    assert f_misleading.resonance_sum < rsum_before, "echo penalty must pull resonance down"

    # Idempotency: a second check on the same topic must not stack penalties.
    rsum_after_first = f_misleading.resonance_sum
    echo2 = mem.check_echo("legacy python script still not working again")
    assert echo2["echo"] is True
    assert echo2["penalty"] == 0.0, "second echo on same session must not re-apply penalty"
    assert f_misleading.resonance_sum == rsum_after_first


if __name__ == "__main__":
    import traceback
    tests = [
        test_deprecated_fact_absorbed_on_resonant_session,
        test_query_returns_relevant_facts,
        test_echo_detected_after_toxic_session,
        test_hawking_emission,
        test_hawking_removes_fact_from_singularity_and_persists,
        test_query_attributes_facts_to_active_session,
        test_query_without_active_session_does_not_tag,
        test_query_dedupes_attribution_within_session,
        test_echo_drags_down_gravity_of_misleading_facts,
    ]
    for t in tests:
        try:
            t()
            print(f"✓  {t.__name__}")
        except Exception as e:
            print(f"✗  {t.__name__}: {e}")
