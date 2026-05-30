"""Configurable cosine/gravity thresholds regressions.

The shippable item: hard-coded cosine thresholds were scattered
across the codebase — a mine under embedding-model swaps. They are
now centralised and overridable via env.

Deferred (not bugs):

  - O(N·d) vector index → known FAISS roadmap, not at scale yet.
  - SPO temporal collapse → by design (Birch vs Vertical Brain
    boundary), atomic mutable triples is the contract.
  - Async collapse race → wrong as stated; collapse_singularity
    holds the RLock for its duration so there's no race, only a
    latency window during long passes on huge singularities.
"""
from __future__ import annotations

import importlib
import os
import sys


def _reload_thresholds(env: dict[str, str]) -> object:
    """Re-import the thresholds module with a patched environment.

    Thresholds class attributes are read once at import time, so to
    test env overrides we have to drop the module from sys.modules
    and re-import. The fixture sets env vars, calls this, returns
    the fresh class.
    """
    # Snapshot + patch the env, then restore on the way out.
    saved = {k: os.environ.get(k) for k in env}
    for k, v in env.items():
        os.environ[k] = v
    try:
        sys.modules.pop("birch.thresholds", None)
        mod = importlib.import_module("birch.thresholds")
        return mod.Thresholds
    finally:
        for k, original in saved.items():
            if original is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = original
        # Re-import once more under the restored env so other tests
        # see the defaults again.
        sys.modules.pop("birch.thresholds", None)
        importlib.import_module("birch.thresholds")


def test_thresholds_defaults():
    """With no env vars set, defaults match the documented working set."""
    sys.modules.pop("birch.thresholds", None)
    from birch.thresholds import Thresholds
    assert Thresholds.ABSORPTION == 0.10
    assert Thresholds.HAWKING_FACT == 0.95
    assert Thresholds.HAWKING_META == 0.85
    assert Thresholds.AUTO_LINK == 0.80
    assert Thresholds.COLLAPSE == 0.92
    assert Thresholds.ECHO == 0.68
    assert Thresholds.FIND_SIMILAR_DEFAULT == 0.85


def test_thresholds_env_override():
    """BIRCH_HAWKING_META=0.78 picks up via env."""
    fresh = _reload_thresholds({"BIRCH_HAWKING_META_THRESHOLD": "0.78"})
    assert fresh.HAWKING_META == 0.78
    # Untouched defaults stay put.
    assert fresh.ABSORPTION == 0.10


def test_thresholds_env_override_multiple():
    fresh = _reload_thresholds({
        "BIRCH_ECHO_THRESHOLD": "0.55",
        "BIRCH_COLLAPSE_THRESHOLD": "0.97",
        "BIRCH_AUTO_LINK_THRESHOLD": "0.65",
    })
    assert fresh.ECHO == 0.55
    assert fresh.COLLAPSE == 0.97
    assert fresh.AUTO_LINK == 0.65


def test_thresholds_env_bad_value_falls_back():
    """A malformed env value falls back to the default — no crash."""
    fresh = _reload_thresholds({"BIRCH_HAWKING_META_THRESHOLD": "not-a-number"})
    assert fresh.HAWKING_META == 0.85


def test_thresholds_as_dict_shape():
    sys.modules.pop("birch.thresholds", None)
    from birch.thresholds import Thresholds
    d = Thresholds.as_dict()
    assert set(d) == {
        "absorption", "hawking_fact", "hawking_meta",
        "auto_link", "collapse", "echo", "find_similar_default",
        "salience_neighbor", "salience_protection", "salience_pin_protection",
        "salience_decay",
    }
    assert all(isinstance(v, float) for v in d.values())


def test_thresholds_surface_in_memory_stats(tmp_path):
    """stats["thresholds"] lets an operator confirm which values the
    process actually picked up."""
    sys.modules.pop("birch.thresholds", None)
    from birch.memory_store import MemoryStore
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    stats = mem.stats
    assert "thresholds" in stats
    assert "absorption" in stats["thresholds"]
    assert "hawking_meta" in stats["thresholds"]
    assert stats["thresholds"]["absorption"] == 0.10
    mem.close()
