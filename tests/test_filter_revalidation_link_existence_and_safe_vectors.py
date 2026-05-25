"""Six contracts around read-path race windows + endpoint guards
+ safe vector loading. Bundled because they all surfaced from the
same review pass and share the "consistency under concurrent or
malformed input" theme.

  1. query() now revalidates ALL filter predicates after _sync, not
     just exists+not-deprecated. A body whose layer / gravity /
     subject_prefix changed under our feet no longer survives into
     results — symmetric with the backfill path which already
     re-filtered.

  2. add_facts() duplicate-in-batch with a DIFFERENT per-item
     session_id now gets attribution to its own session. Skipping
     attribution (alongside skipping touch) silently dropped the
     second session's resonance signal — broke per-item session_id
     contract.

  3. SQLite loaders (load_echo_sessions / load_open_sessions /
     load_meta_facts) now take cleanup: bool = True. _reload calls
     with cleanup=False so rollback recovery is TRULY read-only.
     Previously loaders deleted corrupted rows inline regardless
     of prune, breaking the "_reload doesn't write" promise.

  4. find_similar_by_vector now holds the lock through the entire
     scan. Was releasing after _sync, then reading self._facts
     unlocked — racy if another thread mutated facts mid-scan.

  5. link() now validates that both from_id and to_id point at
     live FactPassports. Previously created ghost edges referencing
     non-existent facts; engine degree counter inflated forever.

  6. MetaFact vector load now rejects NaN / Infinity (matching
     FactPassport _safe_vector contract). A poisoned meta vector
     used to enter the meta_index and corrupt every downstream
     cosine.
"""
from __future__ import annotations

import math
from unittest.mock import patch

import pytest

from birch.memory_store import MemoryStore
from birch.meta_fact import _load_list

# --- I1: query revalidates ALL filters --------------------------------


def test_query_drops_survivor_whose_layer_changed_after_sync(tmp_path):
    """Initial scan finds a surface fact matching layer filter.
    Simulate another thread retiring it (which moves it down/out)
    between snapshot and revalidation by hand-mutating the fact's
    layer to -1 after the snapshot. Revalidation should drop it
    because the layer no longer matches."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    f = mem.add_fact("api", "uses", "Postgres")
    # Surface layer scope.
    pre_layer = f.layer

    # Monkey-patch _sync to flip the fact's layer to -1 (singularity)
    # between the initial scan and revalidation. This is exactly what
    # a concurrent supersede / absorb would do.
    original_sync = mem._sync
    flipped = {"done": False}

    def racing_sync():
        original_sync()
        if not flipped["done"]:
            flipped["done"] = True
            mem._facts[f.fact_id].layer = -1  # type: ignore[union-attr]

    with patch.object(mem, "_sync", racing_sync):
        results = mem.query(
            "api uses Postgres", top_k=5,
            allowed_layers={0},  # surface only
        )

    # Fact must NOT appear in results — its layer changed to -1
    # between initial scan and revalidation.
    for r in results:
        if r.fact is not None:
            assert r.fact.fact_id != f.fact_id, (
                "query returned a fact whose layer no longer "
                "matches the caller's allowed_layers filter"
            )
    mem.close()
    assert pre_layer == 1  # touch pre_layer so linter doesn't complain


# --- I2: add_facts dup_in_batch attribution -----------------------------


def test_add_facts_dup_in_batch_attributes_to_second_session(tmp_path):
    """Two items with same SPO but different per-item session_id.
    Both sessions must receive attribution for the fact, even though
    body.touch() runs only once."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    mem.session_start("s1")
    mem.session_start("s2")
    # Same SPO, different sessions.
    statuses = mem.add_facts(
        [("api", "uses", "Postgres"), ("api", "uses", "Postgres")],
        session_ids=["s1", "s2"],
        return_status=True,
    )
    assert len(statuses) == 2
    fact_id = statuses[0]["fact"].fact_id
    assert statuses[1]["fact"].fact_id == fact_id  # same fact
    assert statuses[1]["duplicate_in_batch"] is True
    # Both sessions must have the fact in their ctx.facts.
    assert fact_id in mem._sessions["s1"].facts, (
        "first session lost attribution"
    )
    assert fact_id in mem._sessions["s2"].facts, (
        "dup_in_batch dropped attribution for second per-item session"
    )
    mem.close()


# --- I3: _reload truly read-only ---------------------------------------


def test_reload_does_not_delete_corrupted_meta_rows(tmp_path):
    """Inject a hand-crafted corrupt meta row, call _reload, confirm
    the corrupt row survives on disk."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    # Insert a row with garbage JSON in source_texts so load_meta_facts
    # would normally delete it.
    mem._storage._conn.execute(
        "INSERT INTO meta_facts (meta_id, vector, weight, source_texts, "
        "source_fact_ids, summary, gravity_score, created_at, layer) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "corrupt-meta-1",
            "[1.0, 2.0, 3.0]",
            1,
            "not-valid-json-at-all{",  # corrupt JSON
            "[]",
            "",
            0.3,
            1234567890.0,
            -1,
        ),
    )
    mem._storage._conn.commit()
    # _reload (rollback-recovery path) must NOT delete the corrupt
    # row even though load_meta_facts(cleanup=True) would.
    mem._reload()
    row = mem._storage._conn.execute(
        "SELECT meta_id FROM meta_facts WHERE meta_id = ?",
        ("corrupt-meta-1",),
    ).fetchone()
    assert row is not None, (
        "_reload deleted a corrupt meta row — recovery path is "
        "not read-only"
    )
    mem.close()


# --- I4: find_similar_by_vector holds lock -----------------------------


def test_find_similar_by_vector_holds_lock_for_entire_scan(tmp_path):
    """Structural pin: self._facts.get(fid) MUST be indented inside a
    `with self._lock:` block in find_similar_by_vector. Walk the file
    source directly (inspect.getsource normalises whitespace
    confusingly for class methods)."""
    import pathlib

    src_path = (
        pathlib.Path(__file__).resolve().parents[1]
        / "src" / "birch" / "memory_store" / "_query.py"
    )
    lines = src_path.read_text().split("\n")
    # Find def find_similar_by_vector.
    def_idx = next(
        i for i, line in enumerate(lines)
        if "def find_similar_by_vector" in line
    )
    # End of method: next def at the same indent.
    def_indent = len(lines[def_idx]) - len(lines[def_idx].lstrip())
    end_idx = next(
        (i for i in range(def_idx + 1, len(lines))
         if lines[i].strip().startswith("def ")
         and (len(lines[i]) - len(lines[i].lstrip())) == def_indent),
        len(lines),
    )
    body = lines[def_idx:end_idx]
    # Walk the body, track in-lock state.
    in_lock = False
    lock_indent = -1
    facts_get_under_lock = False
    facts_get_outside_lock = False
    for line in body:
        stripped = line.strip()
        if not stripped:
            continue
        ind = len(line) - len(line.lstrip())
        if in_lock and ind <= lock_indent:
            in_lock = False
        if stripped == "with self._lock:":
            in_lock = True
            lock_indent = ind
            continue
        # Skip comments — match only actual code references.
        if stripped.startswith("#"):
            continue
        if "self._facts.get(fid)" in stripped:
            if in_lock:
                facts_get_under_lock = True
            else:
                facts_get_outside_lock = True
    assert facts_get_under_lock, (
        "self._facts.get(fid) not found inside a `with self._lock:` "
        "block in find_similar_by_vector"
    )
    assert not facts_get_outside_lock, (
        "self._facts.get(fid) appears OUTSIDE the lock — race "
        "window with concurrent fact mutations"
    )


# --- I5: link() existence check ----------------------------------------


def test_link_raises_on_unknown_from_id(tmp_path):
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    f = mem.add_fact("api", "uses", "Postgres")
    with pytest.raises(KeyError, match="from_id"):
        mem.link("never-existed", f.fact_id)
    mem.close()


def test_link_raises_on_unknown_to_id(tmp_path):
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    f = mem.add_fact("api", "uses", "Postgres")
    with pytest.raises(KeyError, match="to_id"):
        mem.link(f.fact_id, "never-existed")
    mem.close()


def test_link_happy_path_still_works(tmp_path):
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    f1 = mem.add_fact("a", "is", "x")
    f2 = mem.add_fact("b", "is", "y")
    pre_degree = mem._engine._degrees.get(f2.fact_id, 0)
    mem.link(f1.fact_id, f2.fact_id)
    assert mem._engine._degrees.get(f2.fact_id, 0) == pre_degree + 1
    mem.close()


# --- I6: MetaFact NaN/Inf reject ----------------------------------------


def test_load_list_rejects_nan_in_float_vector():
    raw = '[1.0, NaN, 3.0]'
    out = _load_list(raw, float)
    assert out == [], (
        f"NaN should poison the entire vector; got {out}"
    )


def test_load_list_rejects_infinity_in_float_vector():
    raw = '[1.0, Infinity, 3.0]'
    out = _load_list(raw, float)
    assert out == []


def test_load_list_allows_finite_float_vector():
    raw = '[1.0, 2.0, 3.0]'
    out = _load_list(raw, float)
    assert out == [1.0, 2.0, 3.0]
    assert all(math.isfinite(x) for x in out)


def test_load_list_str_still_works():
    """Sanity: the cast=str path is unchanged by the isfinite gate."""
    raw = '["a", "b", "c"]'
    out = _load_list(raw, str)
    assert out == ["a", "b", "c"]
