"""AdaptiveWeights — learned pre-resonance weights instead of magic numbers."""
from birch.adaptive_gravity import AdaptiveWeights
from birch.fact import FactPassport
from birch.gravity import compute_gravity
from birch.memory_store import MemoryStore
from birch.storage.sqlite import SQLiteBackend


def test_from_prior_matches_the_legacy_formula():
    w = AdaptiveWeights.from_prior()
    assert (
        w.w_freshness, w.w_access, w.w_graph, w.w_utility
    ) == AdaptiveWeights.PRIOR
    assert w.train_count == 0


def test_consistent_signal_nudges_the_relevant_weight_up():
    """A stream of 'freshness predicts realised value' steps grows w_freshness."""
    w = AdaptiveWeights.from_prior()
    start = w.w_freshness
    for _ in range(40):
        w.update(
            freshness=1.0, access=0.0, graph=0.0, utility=0.0, target=1.0,
        )
    assert w.w_freshness > start
    assert w.train_count == 40


def test_weights_stay_in_budget_and_non_negative():
    w = AdaptiveWeights.from_prior()
    for _ in range(200):
        w.update(
            freshness=1.0, access=0.0, graph=0.0, utility=0.0, target=1.0,
        )
    total = w.w_freshness + w.w_access + w.w_graph + w.w_utility
    assert abs(total - AdaptiveWeights.BUDGET) < 1e-9
    assert min(
        w.w_freshness, w.w_access, w.w_graph, w.w_utility
    ) >= 0.0


def test_regularisation_pulls_back_toward_the_prior():
    """Stop signalling and reg, every step, drifts each weight toward the prior."""
    w = AdaptiveWeights.from_prior()
    for _ in range(80):
        w.update(
            freshness=1.0, access=0.0, graph=0.0, utility=0.0, target=1.0,
        )
    pumped = w.w_freshness
    # Feed neutral signal (target equals the model's own prediction → err = 0)
    # so only the regularisation term acts.
    for _ in range(80):
        pred = w.predict(0.5, 0.5, 0.5, 0.5)
        w.update(0.5, 0.5, 0.5, 0.5, target=pred)
    assert w.w_freshness < pumped


def test_utility_weight_learns_when_utility_predicts_value():
    """If recent_utility lines up with the target, w_utility climbs."""
    w = AdaptiveWeights.from_prior()
    start = w.w_utility
    for _ in range(60):
        w.update(
            freshness=0.0, access=0.0, graph=0.0, utility=1.0, target=1.0,
        )
    assert w.w_utility > start


def test_compute_gravity_with_prior_matches_the_default():
    """At zero training data, compute_gravity behaves exactly as before."""
    fact = FactPassport("api", "runs on", "Go")
    fact.created_at -= 86400
    fact.access_count = 5
    g_with_prior = compute_gravity(fact, weights=AdaptiveWeights.from_prior())
    g_default = compute_gravity(fact, weights=None)
    assert g_with_prior == g_default


def test_sqlite_roundtrip_of_adaptive_weights(tmp_path):
    backend = SQLiteBackend(str(tmp_path / "m.db"))
    assert backend.load_adaptive_weights() is None
    weights = AdaptiveWeights(
        w_freshness=0.40, w_access=0.15, w_graph=0.10, w_utility=0.05,
        train_count=7,
    )
    backend.save_adaptive_weights(weights)
    loaded = backend.load_adaptive_weights()
    backend.close()
    assert loaded is not None
    assert abs(loaded.w_freshness - 0.40) < 1e-9
    assert abs(loaded.w_access - 0.15) < 1e-9
    assert abs(loaded.w_graph - 0.10) < 1e-9
    assert abs(loaded.w_utility - 0.05) < 1e-9
    assert loaded.train_count == 7


def test_memory_store_learns_one_step_per_resonant_session(tmp_path):
    """A closed resonant session trains the adaptive weights and persists them."""
    db = str(tmp_path / "m.db")
    mem = MemoryStore(db_path=db)
    mem.add_fact("api", "runs on", "Go")
    mem.add_fact("db", "is", "Postgres")

    mem.session_start("s")
    mem.session_message("how do I connect the API to Postgres")
    mem.query("api Go", session_id="s")
    mem.session_message("perfect, exactly what i needed, thanks")
    summary = mem.session_close(session_id="s")
    assert summary.get("label") == "resonant"

    stats = mem.stats
    weights = stats["adaptive_weights"]
    assert weights["train_count"] >= 1
    total = (
        weights["w_freshness"]
        + weights["w_access"]
        + weights["w_graph"]
        + weights["w_utility"]
    )
    assert abs(total - AdaptiveWeights.BUDGET) < 1e-3

    # A fresh process must read the same learned weights from disk.
    again = MemoryStore(db_path=db).stats["adaptive_weights"]
    assert again["train_count"] == weights["train_count"]
    assert abs(again["w_freshness"] - weights["w_freshness"]) < 1e-9
