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
import urllib.request

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


_OLLAMA_PROBE: bool | None = None


def ollama_available() -> bool:
    """Best-effort probe: is an Ollama endpoint reachable right now?

    Cached for the process so we probe at most once. Honours OLLAMA_URL.
    Used by the ``embed_provider`` fixture to pick the *best available*
    provider — real embeddings when an endpoint answers, mock otherwise —
    so a test runs end-to-end either way instead of skipping.
    """
    global _OLLAMA_PROBE
    if _OLLAMA_PROBE is not None:
        return _OLLAMA_PROBE
    # An explicit env choice wins over probing: respect mock/ollama as set.
    explicit = os.environ.get("BIRCH_EMBED_PROVIDER", "").strip().lower()
    if explicit == "mock":
        _OLLAMA_PROBE = False
        return False
    if explicit == "ollama":
        _OLLAMA_PROBE = True
        return True
    base = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
    try:
        with urllib.request.urlopen(base, timeout=0.75) as resp:
            _OLLAMA_PROBE = 200 <= resp.status < 500
    except Exception:
        _OLLAMA_PROBE = False
    return _OLLAMA_PROBE


@pytest.fixture
def embed_provider(monkeypatch):
    """Use real Ollama embeddings when an endpoint is reachable, else mock.

    Yields the active provider name ("ollama" | "mock"). The choice is forced
    via BIRCH_EMBED_PROVIDER for the duration of the test (the embeddings
    module reads it live on every call), then restored. Lets a test exercise
    the real semantic path opportunistically without ever skipping.
    """
    provider = "ollama" if ollama_available() else "mock"
    monkeypatch.setenv("BIRCH_EMBED_PROVIDER", provider)
    return provider
