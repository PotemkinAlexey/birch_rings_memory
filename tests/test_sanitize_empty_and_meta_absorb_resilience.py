"""Three Vader findings shipped together:

  1. ``_sanitize_for_llm`` can strip a validated input down to "".
     The validators (``_validate_text`` / ``_validate_spo_strings``)
     check non-emptiness BEFORE sanitisation, and ``str.strip()``
     doesn't strip zero-width Unicode. A payload of pure ZWSP /
     ZWNJ / ZWJ / BOM passes validation, then the sanitiser
     collapses it to ``""``. Without re-validation the write path
     persists an empty subject / predicate / object — a legitimate-
     looking but informationless row. Fix: post-sanitisation
     non-empty re-check via ``_check_non_empty_after_sanitize``.

  2. ``_absorb_dead`` meta-path was missing the try/except that the
     fact-path got in an earlier round. One transient ``absorb_meta``
     failure (numpy alloc, OOM, corrupted vector) used to abort the
     whole sweep — and the sweep runs from ``session_close``, so a
     bad MetaFact could break session closure entirely. Symmetric
     try/except / log / continue keeps the body live for safety.

  3. ``hawking_emit`` / ``hawking_emit_metas`` were doing manual
     ``_singularity.pop`` + ``idx.remove`` + ``_prune_empty_*_bucket``
     directly even though ``forget_fact`` / ``forget_meta`` exist as
     the documented single owner of bucket lifecycle. Encapsulation
     hygiene only — the manual code was correct, just bypassed the
     helper. Fix: route both emission paths through forget_*.
"""
from __future__ import annotations

from unittest.mock import patch

from birch import server as srv
from birch.black_hole import BlackHole
from birch.fact import FactPassport
from birch.memory_store import MemoryStore
from birch.meta_fact import MetaFact

# --- I1: sanitize-empty re-validation -------------------------------


def test_check_non_empty_after_sanitize_passes_clean_input():
    assert srv._check_non_empty_after_sanitize(
        {"subject": "api", "predicate": "uses", "object": "redis"},
    ) is None


def test_check_non_empty_after_sanitize_catches_empty():
    err = srv._check_non_empty_after_sanitize(
        {"subject": "", "predicate": "uses", "object": "redis"},
    )
    assert err is not None
    assert err["error"] == "field_empty_after_sanitization"
    assert err["bad_fields"] == ["subject"]


def test_check_non_empty_after_sanitize_lists_all_offenders():
    err = srv._check_non_empty_after_sanitize(
        {"subject": "", "predicate": "   ", "object": "redis"},
    )
    assert err["error"] == "field_empty_after_sanitization"
    assert sorted(err["bad_fields"]) == ["predicate", "subject"]


def test_zwsp_only_subject_collapses_then_rejected():
    """End-to-end: a subject of pure zero-width Unicode passes the
    initial SPO type check (str.strip() doesn't strip ZWSP), then
    the sanitiser collapses it to "" — without the re-validation
    fix, an empty subject would land in storage."""
    zwsp_only = "​‌‍﻿"   # ZWSP+ZWNJ+ZWJ+BOM
    # Sanity: the input passes the pre-sanitise SPO validator.
    assert srv._validate_spo_strings(zwsp_only, "p", "o") is None
    # And sanitisation collapses it.
    assert srv._sanitize_for_llm(zwsp_only) == ""
    # The post-sanitisation check catches it.
    after = {"subject": "", "predicate": "p", "object": "o"}
    err = srv._check_non_empty_after_sanitize(after)
    assert err["error"] == "field_empty_after_sanitization"


def test_record_fact_wires_sanitize_empty_check():
    """Source-level guard: record_fact must call
    _check_non_empty_after_sanitize between sanitisation and the
    actual write. Regression guard so the wiring can't drift."""
    import pathlib
    src = pathlib.Path(srv.__file__).read_text()
    # Coarse: find record_fact body, ensure the check is in it.
    import re
    m = re.search(r"^def record_fact\(", src, re.MULTILINE)
    assert m is not None
    body_start = m.start()
    next_def = re.compile(r"^(def |@)", re.MULTILINE).search(
        src, m.end(),
    )
    body = src[body_start:next_def.start() if next_def else len(src)]
    assert "_check_non_empty_after_sanitize" in body
    # And it appears AFTER the sanitisation calls.
    sanitize_pos = body.find("_sanitize_for_llm(subject)")
    check_pos = body.find("_check_non_empty_after_sanitize")
    assert sanitize_pos != -1 and check_pos != -1
    assert sanitize_pos < check_pos, (
        "_check_non_empty_after_sanitize must run AFTER sanitisation "
        "in record_fact — not before"
    )


def test_set_fact_wires_sanitize_empty_check():
    import pathlib
    import re
    src = pathlib.Path(srv.__file__).read_text()
    m = re.search(r"^def set_fact\(", src, re.MULTILINE)
    assert m is not None
    next_def = re.compile(r"^(def |@)", re.MULTILINE).search(
        src, m.end(),
    )
    body = src[m.start():next_def.start() if next_def else len(src)]
    assert "_check_non_empty_after_sanitize" in body


def test_record_facts_handles_sanitize_empty_per_item():
    import pathlib
    import re
    src = pathlib.Path(srv.__file__).read_text()
    m = re.search(r"^def record_facts\(", src, re.MULTILINE)
    assert m is not None
    next_def = re.compile(r"^(def |@)", re.MULTILINE).search(
        src, m.end(),
    )
    body = src[m.start():next_def.start() if next_def else len(src)]
    assert "field_empty_after_sanitization" in body


# --- I2: meta absorb resilience -------------------------------------


def test_absorb_dead_continues_when_meta_absorb_raises(tmp_path):
    """A transient absorb_meta failure must not abort the sweep
    (or, by extension, session_close which calls _absorb_dead).
    The bad MetaFact stays live with original layer; other live
    MetaFacts in the same sweep still get absorbed cleanly."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    # Inject two live MetaFacts with sub-threshold gravity.
    bad = MetaFact(
        meta_id="bad", vector=[0.1, 0.2, 0.3],
        gravity_score=0.05, layer=1,
        source_texts=["a b c"], source_fact_ids=["x"],
    )
    good = MetaFact(
        meta_id="good", vector=[0.4, 0.5, 0.6],
        gravity_score=0.05, layer=0,
        source_texts=["d e f"], source_fact_ids=["y"],
    )
    mem._meta_facts["bad"] = bad
    mem._meta_facts["good"] = good
    mem._meta_index.add("bad", bad.vector)
    mem._meta_index.add("good", good.vector)
    mem._engine.register(bad)
    mem._engine.register(good)

    # Patch absorb_meta to raise on `bad` but pass `good` through.
    original = mem._hole.absorb_meta

    def selective_absorb(meta):
        if meta.meta_id == "bad":
            raise RuntimeError("simulated meta absorb failure")
        return original(meta)

    with patch.object(mem._hole, "absorb_meta", side_effect=selective_absorb):
        absorbed = mem._absorb_dead()

    # The good one IS in the absorbed list and got moved.
    assert "good" in absorbed
    assert "good" not in mem._meta_facts
    # The bad one is NOT in the absorbed list, stays live with
    # its original layer.
    assert "bad" not in absorbed
    assert "bad" in mem._meta_facts
    assert mem._meta_facts["bad"].layer == 1
    mem.close()


# --- I3: forget_* used by hawking_emit ------------------------------


def test_hawking_emit_uses_forget_fact_for_lifecycle():
    """Source-level guard: hawking_emit should commit via
    forget_fact, not via the inline pop + index remove pair."""
    import pathlib

    import birch.black_hole as bh_mod
    text = pathlib.Path(bh_mod.__file__).read_text()
    import re
    # hawking_emit body slice.
    m = re.search(r"^    def hawking_emit\(", text, re.MULTILINE)
    assert m is not None
    next_def = re.compile(
        r"^    def ", re.MULTILINE
    ).search(text, m.end())
    body = text[m.start():next_def.start() if next_def else len(text)]
    assert "self.forget_fact(fid)" in body
    # And the old triple is gone from this method's body.
    assert "_singularity.pop(fid)" not in body, (
        "hawking_emit should delegate to forget_fact, not pop directly"
    )


def test_hawking_emit_metas_uses_forget_meta():
    import pathlib
    import re

    import birch.black_hole as bh_mod
    text = pathlib.Path(bh_mod.__file__).read_text()
    m = re.search(r"^    def hawking_emit_metas\(", text, re.MULTILINE)
    assert m is not None
    next_def = re.compile(
        r"^    def ", re.MULTILINE
    ).search(text, m.end())
    body = text[m.start():next_def.start() if next_def else len(text)]
    assert "self.forget_meta(mid)" in body
    assert "_meta_singularity.pop(mid)" not in body


def test_hawking_emit_still_works_after_refactor():
    """End-to-end sanity: a fact-then-emit round trip still returns
    the body and prunes its bucket."""
    hole = BlackHole()
    f = FactPassport(
        subject="a", predicate="b", object="c", fact_id="f",
        vector=[1.0, 0.0, 0.0],
    )
    hole.absorb(f)
    assert 3 in hole._indices
    emitted = hole.hawking_emit([1.0, 0.0, 0.0])
    assert len(emitted) == 1
    assert emitted[0].fact_id == "f"
    # Bucket pruned via forget_fact's lazy-prune.
    assert 3 not in hole._indices
    assert "f" not in hole._singularity
