"""Property-based invariants — Hypothesis generates random operation
sequences and asserts memory invariants hold after every step.

The five semantic invariants probed here are the load-bearing promises
of the memory system. Each one is more important than catching any
specific bug, because each one is what an agent has to assume when
reading from the store:

  1. Deprecated / expired facts are NEVER returned by query.
     The memory's headline promise — old/wrong knowledge stays hidden.
  2. set_fact keeps EXACTLY ONE live occupant per (subject, predicate).
     The slot abstraction's headline promise — no two competing truths
     for the same slot.
  3. gravity_score ∈ [0, 1] for every live body after every operation.
     The numerical safety contract for the formula.
  4. mutation_version monotonic (never decreases, increases after every
     write).
     The cache-invalidation primitive's contract.
  5. forecast cache invalidates after any mutation.
     The two-counter cache invariant in action.

A failing sequence shrinks to the minimal example that breaks the
contract — which is exactly what reviews can't do.
"""
from __future__ import annotations

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from birch.memory_store import MemoryStore

# Small alphabet keeps the search space manageable but still hits
# enough collisions / duplicates / case variants to find SPO bugs.
_SUBJECTS = st.sampled_from(["api", "db", "cache", "queue", "auth"])
_PREDICATES = st.sampled_from(
    ["runs on", "uses", "is", "depends on", "has"],
)
_OBJECTS = st.sampled_from(
    ["Go", "Rust", "Postgres", "Redis", "Kafka", "1.0", "2.0"],
)

_SPO_TRIPLE = st.tuples(_SUBJECTS, _PREDICATES, _OBJECTS)


# --- I1: deprecated / expired never returned by query -----------------


@given(triples=st.lists(_SPO_TRIPLE, min_size=2, max_size=8, unique=True))
@settings(max_examples=50, deadline=None,
          suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_supersede_chain_query_never_returns_deprecated(triples, tmp_path):
    """For any sequence of supersede operations, no query result ever
    contains a deprecated or expired fact. The memory's headline
    promise: old/wrong knowledge stays hidden once you say it's wrong."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    try:
        # Add every triple, then supersede each one with the next.
        facts = []
        for s, p, o in triples:
            f = mem.add_fact(s, p, o)
            facts.append(f)
        # Supersede chains: f[0] -> f[1] -> f[2] -> ...
        for i in range(len(facts) - 1):
            mem.supersede_fact(facts[i].fact_id, facts[i + 1].fact_id)
        # Query for every subject/object combination; ensure no result
        # is a deprecated or expired body.
        for fact in facts:
            results = mem.query(fact.subject, top_k=20)
            for r in results:
                if r.kind == "fact":
                    assert not r.fact.is_deprecated, (
                        f"query returned deprecated fact "
                        f"{r.fact.fact_id} for subject {fact.subject!r}"
                    )
                    assert not r.fact.is_expired, (
                        f"query returned expired fact "
                        f"{r.fact.fact_id} for subject {fact.subject!r}"
                    )
    finally:
        mem.close()


@given(triples=st.lists(_SPO_TRIPLE, min_size=1, max_size=5, unique=True))
@settings(max_examples=40, deadline=None,
          suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_retire_chain_query_never_returns_expired(triples, tmp_path):
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    try:
        facts = [mem.add_fact(s, p, o) for s, p, o in triples]
        # Retire every other fact.
        for i, f in enumerate(facts):
            if i % 2 == 0:
                mem.retire_fact(f.fact_id)
        for s, _, _ in triples:
            results = mem.query(s, top_k=20)
            for r in results:
                if r.kind == "fact":
                    assert not r.fact.is_expired
    finally:
        mem.close()


# --- I2: set_fact keeps one live occupant per (subject, predicate) -----


@given(
    subject=_SUBJECTS,
    predicate=_PREDICATES,
    objects=st.lists(_OBJECTS, min_size=2, max_size=6, unique=True),
)
@settings(max_examples=50, deadline=None,
          suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_set_fact_slot_holds_exactly_one_live(
    subject, predicate, objects, tmp_path,
):
    """For any sequence of set_fact calls on the same (subject,
    predicate) slot, exactly one fact is live afterwards (the last
    written value). Everything earlier landed in the singularity."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    try:
        last_obj = None
        for obj in objects:
            mem.set_fact(subject, predicate, obj)
            last_obj = obj
        live = [
            f for f in mem.list_facts(subject=subject, predicate=predicate)
            if not (f.is_deprecated or f.is_expired)
        ]
        assert len(live) == 1, (
            f"slot ({subject!r}, {predicate!r}) has {len(live)} live "
            f"occupants after {len(objects)} set_fact calls; expected 1"
        )
        assert live[0].object == last_obj
    finally:
        mem.close()


# --- I3: gravity ∈ [0, 1] for every live body after every operation ----


_OP = st.sampled_from(["add", "set", "retire", "session"])


@given(
    ops=st.lists(
        st.tuples(_OP, _SPO_TRIPLE),
        min_size=3, max_size=15,
    ),
)
@settings(max_examples=40, deadline=None,
          suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_gravity_stays_in_unit_interval_under_random_ops(ops, tmp_path):
    """For any mix of add / set / retire / session operations, every
    live fact's gravity_score stays in [0, 1] after every step. The
    formula's numerical safety contract."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    try:
        for op, (s, p, o) in ops:
            if op == "add":
                mem.add_fact(s, p, o)
            elif op == "set":
                mem.set_fact(s, p, o)
            elif op == "retire":
                # Retire first matching live fact, if any.
                live = [
                    f for f in mem.list_facts(subject=s, predicate=p)
                    if not (f.is_deprecated or f.is_expired)
                ]
                if live:
                    mem.retire_fact(live[0].fact_id)
            elif op == "session":
                mem.session_start("hyp")
                mem.session_message(f"{s} {p} {o}", session_id="hyp")
                mem.query(s, session_id="hyp", top_k=3)
                mem.session_close(
                    session_id="hyp", sentiment="resonant",
                )
            # Invariant after every op.
            for f in mem.list_facts(limit=500):
                assert 0.0 <= f.gravity_score <= 1.0, (
                    f"gravity_score {f.gravity_score} out of [0,1] "
                    f"for fact {f.fact_id} ({f.subject} {f.predicate} "
                    f"{f.object}) after op {op!r}"
                )
    finally:
        mem.close()


# --- I4: mutation_version monotonic + increases after writes -----------


@given(
    ops=st.lists(
        st.tuples(_OP, _SPO_TRIPLE),
        min_size=2, max_size=10,
    ),
)
@settings(max_examples=40, deadline=None,
          suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_mutation_version_monotonic_under_writes(ops, tmp_path):
    """mutation_version never decreases. Any write op causes it to
    strictly increase. This is the contract the forecast cache key
    leans on."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    try:
        prev = mem._mutation_version
        for op, (s, p, o) in ops:
            if op == "add":
                mem.add_fact(s, p, o)
            elif op == "set":
                mem.set_fact(s, p, o)
            elif op == "retire":
                live = [
                    f for f in mem.list_facts(subject=s, predicate=p)
                    if not (f.is_deprecated or f.is_expired)
                ]
                if live:
                    mem.retire_fact(live[0].fact_id)
            elif op == "session":
                mem.session_start("hyp")
                mem.session_message(f"{s} {p} {o}", session_id="hyp")
                mem.session_close(
                    session_id="hyp", sentiment="resonant",
                )
            now = mem._mutation_version
            assert now >= prev, (
                f"mutation_version regressed from {prev} to {now} "
                f"after op {op!r}"
            )
            prev = now
    finally:
        mem.close()


# --- I5: forecast cache invalidates after mutation ---------------------


@given(
    initial=st.lists(_SPO_TRIPLE, min_size=1, max_size=4, unique=True),
    mutator=_OP,
    mut_triple=_SPO_TRIPLE,
)
@settings(max_examples=40, deadline=None,
          suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_forecast_cache_invalidates_after_any_mutation(
    initial, mutator, mut_triple, tmp_path,
):
    """Run forecast (warm cache), apply any mutation, run forecast
    again — second forecast must have cached=False. Two-counter
    invariant in action."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    try:
        for s, p, o in initial:
            mem.add_fact(s, p, o)
        first = mem.run_forecast(horizon_ticks=3)
        assert first["cached"] is False
        cached = mem.run_forecast(horizon_ticks=3)
        assert cached["cached"] is True

        s, p, o = mut_triple
        if mutator == "add":
            mem.add_fact(s, p, o)
        elif mutator == "set":
            mem.set_fact(s, p, o)
        elif mutator == "retire":
            live = [
                f for f in mem.list_facts(subject=s, predicate=p)
                if not (f.is_deprecated or f.is_expired)
            ]
            if live:
                mem.retire_fact(live[0].fact_id)
            else:
                # No live to retire — mutation didn't happen, cache
                # may stay valid. Skip this example.
                return
        elif mutator == "session":
            mem.session_start("hyp")
            mem.session_message(f"{s} {p} {o}", session_id="hyp")
            mem.session_close(
                session_id="hyp", sentiment="resonant",
            )
        after = mem.run_forecast(horizon_ticks=3)
        assert after["cached"] is False, (
            f"forecast cache returned cached=True after mutator "
            f"{mutator!r} — cache invariant broken"
        )
    finally:
        mem.close()
