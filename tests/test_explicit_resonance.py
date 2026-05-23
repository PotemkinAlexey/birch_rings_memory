"""Explicit resonance override on session_close — user-suggested feature.

The heuristic resonance scorer reads the message trajectory and
classifies as resonant / neutral / toxic. It mis-classifies declarative
technical summaries that contain grumpy-sounding vocabulary (failure
mode, stale snapshot, no repeats) as toxic even when the session was
a clean closure. The model writing to birch knows the actual outcome
better than the heuristic does — let it say so.

Two new params on session_close: ``sentiment`` (enum shortcut) and
``r_override`` (direct float). Both bypass the heuristic; response
carries ``scoring_source`` for transparency.
"""
from __future__ import annotations

import pytest

from birch.memory_store import MemoryStore


def _toxic_sounding_session(mem: MemoryStore, sid: str) -> None:
    """Plant text that the heuristic reads as failure / blocked /
    confused — useful baseline to show the override flips the label."""
    mem.session_start(sid)
    mem.session_message("this is broken, doesn't work, terrible")
    mem.session_message("still failing, I'm stuck, what a mess")


def test_default_heuristic_path_unchanged(tmp_path):
    """Calling session_close with neither sentiment nor r_override
    runs the original heuristic and labels the path 'heuristic'."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    _toxic_sounding_session(mem, "s")
    summary = mem.session_close(session_id="s")
    assert summary["scoring_source"] == "heuristic"
    # Don't assert the exact label — heuristic is what it is.
    assert summary["label"] in {"resonant", "neutral", "toxic"}
    mem.close()


def test_sentiment_resonant_overrides_toxic_text(tmp_path):
    """Toxic-looking text + sentiment='resonant' → resonant label."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    _toxic_sounding_session(mem, "s")
    summary = mem.session_close(session_id="s", sentiment="resonant")
    assert summary["scoring_source"] == "sentiment"
    assert summary["label"] == "resonant"
    assert summary["r"] == 0.7
    mem.close()


def test_sentiment_toxic_overrides_resonant_text(tmp_path):
    """Happy-looking text + sentiment='toxic' → toxic label.

    The reverse direction confirms the override is truly authoritative,
    not just additive to the heuristic.
    """
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    mem.session_start("s")
    mem.session_message("perfect, exactly what I needed, thanks!")
    mem.session_message("great, this works beautifully")
    summary = mem.session_close(session_id="s", sentiment="toxic")
    assert summary["scoring_source"] == "sentiment"
    assert summary["label"] == "toxic"
    assert summary["r"] == -0.7
    mem.close()


def test_sentiment_neutral_lands_in_neutral_band(tmp_path):
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    _toxic_sounding_session(mem, "s")
    summary = mem.session_close(session_id="s", sentiment="neutral")
    assert summary["scoring_source"] == "sentiment"
    assert summary["label"] == "neutral"
    assert summary["r"] == 0.0
    mem.close()


def test_sentiment_aliases_map_to_same_values(tmp_path):
    """'positive' is an alias for 'resonant'; 'negative' for 'toxic'."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    _toxic_sounding_session(mem, "a")
    pos = mem.session_close(session_id="a", sentiment="positive")
    assert pos["r"] == 0.7
    assert pos["label"] == "resonant"

    _toxic_sounding_session(mem, "b")
    neg = mem.session_close(session_id="b", sentiment="negative")
    assert neg["r"] == -0.7
    assert neg["label"] == "toxic"
    mem.close()


def test_sentiment_invalid_value_raises(tmp_path):
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    _toxic_sounding_session(mem, "s")
    with pytest.raises(ValueError, match="sentiment must be one of"):
        mem.session_close(session_id="s", sentiment="happyish")
    mem.close()


def test_r_override_sets_exact_value(tmp_path):
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    _toxic_sounding_session(mem, "s")
    summary = mem.session_close(session_id="s", r_override=0.85)
    assert summary["scoring_source"] == "r_override"
    assert summary["r"] == 0.85
    assert summary["label"] == "resonant"
    mem.close()


def test_r_override_clamps_to_unit_interval(tmp_path):
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    _toxic_sounding_session(mem, "s1")
    high = mem.session_close(session_id="s1", r_override=5.0)
    assert high["r"] == 1.0

    _toxic_sounding_session(mem, "s2")
    low = mem.session_close(session_id="s2", r_override=-9.0)
    assert low["r"] == -1.0
    mem.close()


def test_r_override_beats_sentiment_when_both_set(tmp_path):
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    _toxic_sounding_session(mem, "s")
    summary = mem.session_close(
        session_id="s", sentiment="toxic", r_override=0.5,
    )
    assert summary["scoring_source"] == "r_override"
    assert summary["r"] == 0.5
    mem.close()


def test_explicit_override_still_propagates_to_fact_gravity(tmp_path):
    """The override path doesn't just label — it actually drives the
    gravity propagation. A resonant override on a session that touched
    facts lifts those facts' utility EWMA the same way a heuristic
    resonant would."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    fact = mem.add_fact("api", "runs on", "Go")
    baseline_utility = fact.recent_utility

    mem.session_start("s")
    mem.session_message("looking at the api")
    mem.query("api Go", session_id="s")
    summary = mem.session_close(session_id="s", sentiment="resonant")
    assert summary["scoring_source"] == "sentiment"

    # Touched fact's utility EWMA must have moved up — same code path
    # the heuristic would have driven, just with the R value coming
    # from the explicit override.
    touched = mem.list_facts(subject="api")[0]
    assert touched.recent_utility > baseline_utility
    mem.close()
