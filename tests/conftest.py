"""Shared pytest fixtures and markers.

The default embedding provider under pytest is the deterministic mock — see
``birch.resonance.embeddings._select_provider``. Tests that depend on *real
semantic* similarity (echo detection between paraphrased sessions, query
matches across distinct vocabularies, MetaFact bundle relevance) cannot
assert their guarantees on top of a hash-bucket embedding; mark them with
``@needs_real_embeddings`` and they will skip under the mock provider and
run when ``BIRCH_EMBED_PROVIDER=ollama`` is set.

Everything else — gravity arithmetic, lifecycle invariants, SQLite
round-trip, multi-process coherence, set_fact / supersede_fact behaviour,
forecast_stability, adaptive weights, query filtering, conflict detection
— is provider-agnostic and runs identically under either provider.
"""
from __future__ import annotations

import os

import pytest


def _active_provider() -> str:
    explicit = os.environ.get("BIRCH_EMBED_PROVIDER", "").strip().lower()
    if explicit in {"ollama", "mock"}:
        return explicit
    # Same default as embeddings._select_provider() under pytest.
    return "mock"


needs_real_embeddings = pytest.mark.skipif(
    _active_provider() != "ollama",
    reason=(
        "needs a real semantic embedding provider; set "
        "BIRCH_EMBED_PROVIDER=ollama with a running Ollama endpoint to run"
    ),
)
