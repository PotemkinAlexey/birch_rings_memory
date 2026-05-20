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


if __name__ == "__main__":
    import traceback
    tests = [
        test_deprecated_fact_absorbed_on_resonant_session,
        test_query_returns_relevant_facts,
        test_echo_detected_after_toxic_session,
        test_hawking_emission,
    ]
    for t in tests:
        try:
            t()
            print(f"✓  {t.__name__}")
        except Exception as e:
            print(f"✗  {t.__name__}: {e}")
