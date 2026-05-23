"""recent_utility — EWMA of closure-weighted resonance per fact.

The signal is orthogonal to access_count: it does not care how often the
fact was touched, only how the sessions that touched it ended up.
"""
from birch.fact import FactPassport
from birch.gravity import pre_resonance_features
from birch.memory_store import MemoryStore
from birch.storage.sqlite import SQLiteBackend


def test_default_is_neutral_prior():
    f = FactPassport("api", "runs on", "Go")
    assert f.recent_utility == 0.5


def test_pre_resonance_features_returns_five_values():
    f = FactPassport("api", "runs on", "Go")
    feats = pre_resonance_features(f, graph_degree=0, max_degree=1)
    assert len(feats) == 5
    freshness, access, graph, utility, stability = feats
    assert 0.0 <= utility <= 1.0
    assert 0.0 <= stability <= 1.0
    # An untouched fact carries the 0.5 prior on both EWMA / forecast.
    assert abs(utility - 0.5) < 1e-9
    assert abs(stability - 0.5) < 1e-9


def test_sqlite_roundtrips_recent_utility(tmp_path):
    backend = SQLiteBackend(str(tmp_path / "m.db"))
    f = FactPassport("api", "runs on", "Go")
    f.recent_utility = 0.87
    backend.save_fact(f)
    loaded = {x.fact_id: x for x in backend.load_facts()}[f.fact_id]
    backend.close()
    assert abs(loaded.recent_utility - 0.87) < 1e-9


def test_session_close_raises_utility_on_resonant_session(tmp_path):
    """A positive closure must lift recent_utility above the 0.5 prior."""
    db = str(tmp_path / "m.db")
    mem = MemoryStore(db_path=db)
    fact = mem.add_fact("api", "runs on", "Go")
    assert abs(fact.recent_utility - 0.5) < 1e-9

    mem.session_start("s")
    mem.session_message("how do I run the API on Go")
    mem.query("api Go", session_id="s")
    mem.session_message("perfect, exactly what i needed, thanks")
    summary = mem.session_close(session_id="s")
    assert summary.get("label") == "resonant"

    # Cache may have been swapped during session_close; reload via stats path.
    refreshed = mem.list_facts(subject="api")[0]
    assert refreshed.recent_utility > 0.5

    # And it must survive a process restart.
    again = MemoryStore(db_path=db).list_facts(subject="api")[0]
    assert again.recent_utility > 0.5


def test_session_close_lowers_utility_on_toxic_session(tmp_path):
    """A negative closure must drag recent_utility below the 0.5 prior."""
    db = str(tmp_path / "m.db")
    mem = MemoryStore(db_path=db)
    mem.add_fact("api", "runs on", "Go")

    mem.session_start("s")
    mem.session_message("trying to make the API run")
    mem.query("api Go", session_id="s")
    mem.session_message("опять не работает, ничего не помогло")
    summary = mem.session_close(session_id="s")
    assert summary.get("label") == "toxic"

    refreshed = mem.list_facts(subject="api")[0]
    assert refreshed.recent_utility < 0.5
