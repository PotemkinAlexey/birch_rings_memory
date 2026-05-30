"""Cosine-similarity and gravity thresholds — one place, env-overridable.

Different embedding models have wildly different cosine distributions.
On the deterministic mock provider used in CI, vectors are sparse
hash-bucket sketches with very compressed similarities; on
``nomic-embed-text`` they spread closer to [-1, 1]; on a future
sentence-transformer they spread differently again. A threshold like
0.85 is statistically near-impossible on one and trivially hit on
another. Hard-coding a single number across all providers is a mine.

This module owns every threshold the system uses, exposes them as
classmethod-style attributes on ``Thresholds``, and lets each be
overridden via a ``BIRCH_<NAME>`` environment variable at process
start. Code reads thresholds via ``Thresholds.ABSORPTION``, etc.,
not by importing a constant — that way an operator who pins
``BIRCH_HAWKING_META=0.78`` for their embedding model gets it
applied everywhere it matters.

Per-provider defaults can layer on top via ``BIRCH_EMBED_PROVIDER``,
but the open-source default tracks the mock + nomic working set since
that's what the test suite and quickstart exercise.
"""
from __future__ import annotations

import math
import os


def _env_float(
    name: str,
    default: float,
    lo: float = 0.0,
    hi: float = 1.0,
) -> float:
    """Read a float from BIRCH_<name>; fall back to ``default`` on miss,
    parse error, non-finite value, or out-of-range value. Logs nothing
    — startup-quiet by intent.

    Every threshold this module owns is a cosine or a gravity score,
    both naturally in [0, 1]. Out-of-range values silently fall back
    to the default so an operator typo (BIRCH_HAWKING_FACT_THRESHOLD=2,
    =-999, or =nan) can't accidentally make every body Hawking-emit
    or poison every threshold comparison. NaN check is explicit even
    though range-check would catch it (NaN comparisons all return
    False) — clearer intent + faster reject."""
    raw = os.environ.get(f"BIRCH_{name}")
    if raw is None:
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(value):
        return default
    if not lo <= value <= hi:
        return default
    return value


class Thresholds:
    """Read-once snapshot of every tunable threshold.

    Resolved at import time. To change live, set env var BEFORE
    importing birch (e.g. via the MCP launcher shell), or override
    on a per-MemoryStore basis through constructor kwargs that take
    precedence over these defaults.
    """

    # Gravity floor — bodies below this after a tick fall into the
    # black hole. Embedding-independent (it's gravity, not cosine),
    # but still configurable since "what counts as dead" is a policy
    # knob.
    ABSORPTION: float = _env_float("ABSORPTION_THRESHOLD", 0.10)

    # Hawking emission for live single facts. Intentionally near-exact
    # so we only resurrect a dead body on a near-perfect query match.
    HAWKING_FACT: float = _env_float("HAWKING_FACT_THRESHOLD", 0.95)

    # Hawking emission for MetaFacts. A centroid lives between its
    # sources, so the cosine drift means 0.95 almost never fires.
    # 0.85 is the working default for nomic; on mock-with-tight-clusters
    # it can be lower, on a model with wider spread it should be higher.
    HAWKING_META: float = _env_float("HAWKING_META_THRESHOLD", 0.85)

    # Auto-link graph edge threshold — facts at similarity ≥ this get
    # an undirected graph edge for the `graph` feature of the formula.
    AUTO_LINK: float = _env_float("AUTO_LINK_THRESHOLD", 0.80)

    # Collapse pass — bodies at similarity ≥ this in the singularity
    # are union-found into one MetaFact. Tighter than HAWKING_META on
    # purpose: we'd rather under-bundle than glue unrelated topics.
    COLLAPSE: float = _env_float("COLLAPSE_THRESHOLD", 0.92)

    # Echo detection — K-means bundle of past session matches if the
    # new session's centroid is at similarity ≥ this. Above the noise
    # floor for the working model but well below "same conversation".
    ECHO: float = _env_float("ECHO_THRESHOLD", 0.68)

    # find_similar default — paraphrase candidate cut-off when an
    # agent calls find_similar without specifying min_similarity.
    FIND_SIMILAR_DEFAULT: float = _env_float(
        "FIND_SIMILAR_THRESHOLD", 0.85,
    )

    # Salience / irreplaceability — a live fact with no other live fact in its
    # namespace at cosine ≥ this is "unique"; losing it loses knowledge no
    # neighbour can replace, so it earns retention against disuse-absorption.
    # A cost-of-loss signal orthogonal to frequency.
    SALIENCE_NEIGHBOR: float = _env_float("SALIENCE_NEIGHBOR_THRESHOLD", 0.85)

    # How much full EARNED irreplaceability lowers the absorption floor: a fully
    # unique, proven fact's floor is ABSORPTION·(1 − SALIENCE_PROTECTION).
    # 0 disables the bottom-up (earned) half only — declared pins are governed
    # separately by SALIENCE_PIN_PROTECTION, so this knob does NOT silently turn
    # off record_fact(salient=True).
    SALIENCE_PROTECTION: float = _env_float("SALIENCE_PROTECTION", 0.9)

    # How much a full declared pin (encode_salience = 1.0) lowers the floor.
    # Independent of SALIENCE_PROTECTION so an operator can tune / disable the
    # earned heuristic without losing the explicit top-down channel. 0 disables
    # declared pins.
    SALIENCE_PIN_PROTECTION: float = _env_float("SALIENCE_PIN_PROTECTION", 0.9)

    # Encoding-salience use-it-or-lose-it decay: a declared pin loses this much
    # (× session confidence) each time the fact is surfaced into a session that
    # ends non-positive. ~3 useless surfacings fully erode a full pin. A pin
    # that never surfaces is never decayed (it never got its chance) — that is
    # what the per-namespace pin budget, not decay, backstops.
    SALIENCE_DECAY: float = _env_float("SALIENCE_DECAY", 0.34)

    @classmethod
    def as_dict(cls) -> dict[str, float]:
        """Snapshot for diagnostics / memory_stats. Lets an operator
        confirm which thresholds the process actually picked up."""
        return {
            "absorption": cls.ABSORPTION,
            "hawking_fact": cls.HAWKING_FACT,
            "hawking_meta": cls.HAWKING_META,
            "auto_link": cls.AUTO_LINK,
            "collapse": cls.COLLAPSE,
            "echo": cls.ECHO,
            "find_similar_default": cls.FIND_SIMILAR_DEFAULT,
            "salience_neighbor": cls.SALIENCE_NEIGHBOR,
            "salience_protection": cls.SALIENCE_PROTECTION,
            "salience_pin_protection": cls.SALIENCE_PIN_PROTECTION,
            "salience_decay": cls.SALIENCE_DECAY,
        }
