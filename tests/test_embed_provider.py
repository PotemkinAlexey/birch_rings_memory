"""Mock embedding provider + embedding error handling.

These tests run with no external dependencies — the mock provider is
deterministic and the Ollama path uses an unreachable URL to exercise
the error branch.
"""
from __future__ import annotations

import math
import os
from unittest import mock

import pytest

from birch.resonance.embeddings import (
    _MOCK_DIM,
    EmbeddingError,
    _mock_embed,
    embed,
    embed_batch,
)


def test_mock_embed_is_deterministic():
    a = _mock_embed("api runs on Go")
    b = _mock_embed("api runs on Go")
    assert a == b


def test_mock_embed_l2_normalised():
    v = _mock_embed("anything at all")
    norm = math.sqrt(sum(x * x for x in v))
    assert abs(norm - 1.0) < 1e-9


def test_mock_embed_dimension_is_stable():
    short = _mock_embed("x")
    longer = _mock_embed("the api runs on Go and uses Postgres for storage")
    assert len(short) == len(longer) == _MOCK_DIM


def test_mock_embed_distinct_inputs_distinct_outputs():
    a = _mock_embed("api runs on Go")
    b = _mock_embed("unrelated topic entirely different words")
    # Tiny chance of collision but with hash-bucket scheme it's practically zero.
    assert a != b


def test_mock_embed_token_overlap_increases_similarity():
    a = _mock_embed("api runs on Go")
    b = _mock_embed("api runs on Rust")        # shares 3 of 4 tokens
    c = _mock_embed("unrelated topic entirely different words")
    sim_ab = sum(x * y for x, y in zip(a, b))
    sim_ac = sum(x * y for x, y in zip(a, c))
    assert sim_ab > sim_ac


def test_embed_under_pytest_uses_mock_provider():
    """The provider auto-selects mock when running under pytest (no env)."""
    # No env override expected for the suite. embed() should return a
    # _MOCK_DIM vector, not a 768-d Ollama vector.
    v = embed("test text")
    assert len(v) == _MOCK_DIM


def test_embed_batch_under_pytest_uses_mock_provider():
    out = embed_batch(["a", "b", "c"])
    assert len(out) == 3
    assert all(len(v) == _MOCK_DIM for v in out)


def test_explicit_ollama_provider_fails_clearly_when_unreachable():
    """Set BIRCH_EMBED_PROVIDER=ollama against an unreachable URL → clear error."""
    with mock.patch.dict(os.environ, {
        "BIRCH_EMBED_PROVIDER": "ollama",
        "OLLAMA_URL": "http://127.0.0.1:1",  # nothing listens here
    }):
        # _BASE_URL was cached at import — patch the module constants too.
        import birch.resonance.embeddings as emb
        with mock.patch.object(emb, "_BASE_URL", "http://127.0.0.1:1"), \
             mock.patch.object(emb, "_BATCH_ENDPOINT", "http://127.0.0.1:1/api/embed"), \
             mock.patch.object(emb, "_LEGACY_ENDPOINT", "http://127.0.0.1:1/api/embeddings"):
            with pytest.raises(EmbeddingError) as exc_info:
                embed("hello")
    msg = str(exc_info.value)
    assert "Ollama" in msg or "ollama" in msg or "BIRCH_EMBED_PROVIDER" in msg
