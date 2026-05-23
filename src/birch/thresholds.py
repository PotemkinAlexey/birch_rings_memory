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

import os


def _env_float(
    name: str,
    default: float,
    lo: float = 0.0,
    hi: float = 1.0,
) -> float:
    """Read a float from BIRCH_<name>; fall back to ``default`` on miss,
    parse error, or out-of-range value. Logs nothing — startup-quiet
    by intent.

    Every threshold this module owns is a cosine or a gravity score,
    both naturally in [0, 1]. Out-of-range values silently fall back
    to the default so an operator typo (BIRCH_HAWKING_FACT_THRESHOLD=2
    or =-999) can't accidentally make every body Hawking-emit or
    nothing collapse. Strict clamp would hide intent; default fallback
    preserves the working contract."""
    raw = os.environ.get(f"BIRCH_{name}")
    if raw is None:
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
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
        }
