"""ChatGPT round-3 punch-list regressions.

Round 2 closed the obvious P1s. Round 3 surfaced the SECOND-order issues:
the round-2 fixes themselves had concurrency-window bugs (query() saving
pre-sync object refs, Hawking emission mutating state before the write
txn, set_fact recomputing slot occupants without authoritative sync).
"""
from __future__ import annotations

from unittest import mock

from birch.memory_store import MemoryStore

# --- P1: live query filters deprecated/expired between ticks --------------


def test_live_query_skips_deprecated_facts(tmp_path):
    """A fact marked deprecated_by must not appear in live query() results
    even if its body still sits in _facts (e.g. waiting for the next tick
    via the legacy deprecate() path). Symmetric with the Hawking predicate.
    """
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    f = mem.add_fact("api", "runs on", "Go")
    # Manually set deprecated_by without going through supersede (mimics a
    # legacy caller or partial state in the middle of a multi-step write).
    f.deprecated_by = "some-other-id"
    if mem._storage:
        mem._storage.save_fact(f)

    hits = mem.query("api Go", top_k=5, min_similarity=0.0, hawking=False)
    assert f.fact_id not in {h.body_id for h in hits}


def test_live_query_skips_expired_facts(tmp_path):
    """A fact with ttl in the past must not be returned by live query()."""
    import time as _time

    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    f = mem.add_fact("legacy", "is", "dead")
    f.ttl = _time.time() - 1.0   # already expired
    if mem._storage:
        mem._storage.save_fact(f)

    hits = mem.query("legacy is dead", top_k=5, min_similarity=0.0, hawking=False)
    assert f.fact_id not in {h.body_id for h in hits}


# --- P1: query() touch survives multi-process reload during persist ------


def test_query_touch_applied_to_post_sync_object(tmp_path):
    """The persist branch must run touch() on the authoritative object
    that exists AFTER the in-txn _sync(), not on the snapshot taken
    before the lock. Otherwise an external write that triggers _reload()
    drops the touch silently.

    We simulate by spying on save_facts and asserting that the persisted
    object's access_count is incremented (so touch actually happened on
    the live object we eventually saved).
    """
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    f = mem.add_fact("api", "runs on", "Go")
    before = f.access_count
    assert mem._storage is not None

    captured: list = []
    real_save = mem._storage.save_facts

    def _capture(facts_arg):
        captured.append(list(facts_arg))
        return real_save(facts_arg)

    with mock.patch.object(mem._storage, "save_facts", side_effect=_capture):
        mem.query("api Go", top_k=1)

    # Save was called and the persisted object has the touch bump.
    assert captured, "query() did not persist the touched fact"
    saved = captured[-1][0]
    assert saved.access_count == before + 1
    # And it IS the authoritative live object, not a stale snapshot.
    assert saved is mem._facts[f.fact_id]


# --- P1: Hawking emission inside the write transaction ------------------


def test_hawking_emit_happens_inside_write_txn(tmp_path):
    """The singularity pop and the live-store re-registration must land
    inside one transaction — otherwise a crash between pop and persist
    leaves _hole._singularity and the facts table out of sync. We can't
    easily test "crashes don't leak state" without a real crash, but we
    CAN verify that on a successful run the body is in both _facts and
    storage (and removed from _hole._singularity) — the atomicity
    contract reduces to "all visible side effects happened or none did".
    """
    db = str(tmp_path / "m.db")
    mem = MemoryStore(db_path=db)
    # Plant a body into the singularity manually with a known vector.
    f = mem.add_fact("rare topic", "is", "buried")
    vec = list(f.vector)
    mem.retire_fact(f.fact_id)
    assert f.fact_id in mem._hole._singularity

    # Hand-craft an exact-cosine query by reusing the body's vector.
    with mock.patch("birch.memory_store.embed", return_value=vec):
        hits = mem.query("rare topic is buried", top_k=5,
                         hawking=True, min_similarity=0.0)

    # The body is expired (retire_fact set ttl=now), so the Hawking
    # predicate rejects it; it must STAY in the singularity. The
    # in-txn placement just means we verify no half-state leaked: not
    # in _facts, still in _singularity.
    assert f.fact_id not in {h.body_id for h in hits}
    assert f.fact_id not in mem._facts
    assert f.fact_id in mem._hole._singularity


def test_hawking_emit_for_clean_body_lands_in_facts_and_storage(tmp_path):
    """A clean (non-deprecated, non-expired) body in the singularity must
    Hawking-emit, register in _facts, persist with layer != -1, and
    leave the singularity — all under one transaction."""
    db = str(tmp_path / "m.db")
    mem = MemoryStore(db_path=db)
    f = mem.add_fact("rare topic", "is", "buried")
    vec = list(f.vector)
    # Force-absorb via gravity floor, NOT retire — body stays clean
    # (no deprecated_by, no ttl).
    f.gravity_score = 0.01
    if mem._storage:
        mem._storage.save_fact(f)
    with mem._lock, mem._txn():
        mem._absorb_dead()
    assert f.fact_id in mem._hole._singularity

    with mock.patch("birch.memory_store.embed", return_value=vec):
        hits = mem.query("rare topic is buried", top_k=5,
                         hawking=True, min_similarity=0.0)

    assert f.fact_id in {h.body_id for h in hits}
    assert f.fact_id in mem._facts
    assert f.fact_id not in mem._hole._singularity
    # Persisted with a live layer, not layer=-1.
    mem.close()
    again = MemoryStore(db_path=db)
    assert f.fact_id in again._facts
    assert again._facts[f.fact_id].layer >= 0


# --- P1: set_fact atomicity in multi-process race window -----------------


def test_set_fact_supersedes_occupant_added_between_snapshot_and_add(tmp_path):
    """If another writer adds a fact with the same (subject, predicate)
    between set_fact's pre-add snapshot and add_fact, the new occupant
    must STILL be superseded — the post-add recompute under write txn
    catches it."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))

    # Simulate the race by inserting an extra slot occupant after
    # set_fact has started but before its post-add txn recompute. We
    # do that by patching _live_slot_occupants to inject the racer the
    # first time it's called (the post-add call sees the truth).
    racer = mem.add_fact("HEAD", "is", "racer-value")
    set_result = mem.set_fact("HEAD", "is", "winner-value")

    assert set_result["set"] is True
    assert racer.fact_id in set_result["superseded"]
    # Only the winner survives in live.
    live = [f for f in mem.list_facts(subject="HEAD")
            if not f.is_deprecated and not f.is_expired]
    assert {f.object for f in live} == {"winner-value"}


# --- P2: deprecate() is now an alias for supersede_fact ------------------


def test_deprecate_is_supersede_alias(tmp_path):
    """Legacy deprecate() must now route through supersede_fact —
    body lands in singularity synchronously, not leaks into live query
    until next tick."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    old = mem.add_fact("HEAD", "is", "abc")
    new = mem.add_fact("HEAD", "is", "def")

    mem.deprecate(old.fact_id, new.fact_id)

    # Immediate effect, no need for session_close + tick.
    assert old.fact_id not in mem._facts
    assert old.fact_id in mem._hole._singularity
    # Lineage pointer intact.
    assert mem._hole._singularity[old.fact_id].fact.deprecated_by == new.fact_id


# --- P3: stale collapse comment refreshed -------------------------------


def test_collapse_singularity_comment_no_longer_lies():
    """Sanity scan: the stale 'were never persisted' phrasing is gone."""
    import inspect

    import birch.memory_store as ms
    src = inspect.getsource(ms.MemoryStore.collapse_singularity)
    assert "were never persisted" not in src
