"""Embedding client — provider-switched, deterministic offline by default for CI.

Two providers are supported:

- ``ollama`` (the default for runtime) — wraps the local Ollama HTTP API
  at ``/api/embed`` with a fallback to the legacy ``/api/embeddings``.
- ``mock`` (the default for tests and CI) — a deterministic stdlib-only
  hash embedding. Same text → same vector; texts that share tokens land
  closer in cosine; no network, no external process, no flakiness.

Pick the provider with ``BIRCH_EMBED_PROVIDER=ollama|mock``. If the
variable is unset, we default to ``mock`` whenever pytest is running and
``ollama`` otherwise — so contributor checkouts and CI work out of the
box without an embedding endpoint, but ``birch.server`` running for a
real agent keeps talking to Ollama.

Failures from the Ollama side carry a clear ``EmbeddingError`` rather
than leaking raw urllib / socket / JSON exceptions so the MCP server can
surface a useful diagnostic to the agent.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import socket
import sys
import urllib.error
import urllib.request

_BASE_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
_MODEL = os.environ.get("BIRCH_EMBED_MODEL", "nomic-embed-text")

_BATCH_ENDPOINT = f"{_BASE_URL}/api/embed"
_LEGACY_ENDPOINT = f"{_BASE_URL}/api/embeddings"

# Dimensionality for the mock provider — small enough to keep tests fast,
# wide enough that distinct token sets don't collide at the bit level.
_MOCK_DIM = 64


class EmbeddingError(RuntimeError):
    """Raised when the configured embedding provider cannot serve a request.

    Carries the original cause and the provider/endpoint identity, so a
    caller can render an actionable diagnostic instead of a stack trace.
    """


def _select_provider() -> str:
    """Resolve the active provider, defaulting to ``mock`` under pytest."""
    explicit = os.environ.get("BIRCH_EMBED_PROVIDER", "").strip().lower()
    if explicit in {"ollama", "mock"}:
        return explicit
    if "pytest" in sys.modules:
        return "mock"
    return "ollama"


# ── Mock provider — deterministic, stdlib-only ──────────────────────────────


def _mock_embed(text: str) -> list[float]:
    """Hash-bucket embedding: each token contributes to deterministic bins.

    The contract is small but useful:
      - same text → identical vector (exact match works);
      - texts sharing tokens land closer than texts that do not;
      - all vectors are L2-normalised so cosine == dot product;
      - dimensionality is fixed at _MOCK_DIM independent of input.
    """
    vec = [0.0] * _MOCK_DIM
    tokens = text.lower().split() or [text.lower()]
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        # Use the first _MOCK_DIM bytes — sha256 gives 32, so wrap when needed.
        for i in range(_MOCK_DIM):
            byte = digest[i % len(digest)]
            vec[i] += (byte / 255.0) - 0.5
    norm = math.sqrt(sum(v * v for v in vec))
    if norm < 1e-12:
        # Degenerate (empty / all-zero) — return a deterministic unit vector.
        out = [0.0] * _MOCK_DIM
        out[0] = 1.0
        return out
    return [v / norm for v in vec]


# ── Ollama provider ─────────────────────────────────────────────────────────


def _post(url: str, body: dict, timeout: float = 30.0) -> dict:
    """POST JSON to ``url``; wrap every failure mode in EmbeddingError."""
    payload = json.dumps(body).encode()
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        # Only let 404 escape untouched — that's the signal the caller
        # uses to fall back to the legacy endpoint. Every other HTTP
        # code (400 bad request, 500 model error, 503 etc.) is a real
        # provider failure that should surface as a typed EmbeddingError
        # at the MCP boundary, not as a raw stacktrace.
        if exc.code == 404:
            raise
        raise EmbeddingError(
            f"Ollama HTTP {exc.code} at {url}: {exc.reason}"
        ) from exc
    except (urllib.error.URLError, socket.timeout, ConnectionError) as exc:
        raise EmbeddingError(
            f"cannot reach Ollama at {_BASE_URL}: {exc}. "
            "Set BIRCH_EMBED_PROVIDER=mock for offline use, "
            "or start Ollama with the configured model."
        ) from exc
    except OSError as exc:
        raise EmbeddingError(f"network error talking to {url}: {exc}") from exc
    try:
        data = json.loads(raw)
    except ValueError as exc:
        snippet = raw[:120].decode("utf-8", errors="replace")
        raise EmbeddingError(
            f"non-JSON response from {url}: {snippet!r}"
        ) from exc
    if not isinstance(data, dict):
        raise EmbeddingError(
            f"unexpected response shape from {url}: "
            f"expected dict, got {type(data).__name__}"
        )
    return data


def _validate_vector(vec: object, where: str) -> list[float]:
    """Embedding-shape contract for the boundary between the HTTP client
    and the rest of birch.

    Two failure modes downstream — wrong-shape vectors crash
    ``np.asarray(dtype=float32)`` with a raw ``ValueError``, and
    non-numeric items crash later in ``VectorIndex.add`` after the
    record was already partially committed. Catching both here turns
    them into a single typed ``EmbeddingError`` the MCP layer can wrap.
    """
    if not isinstance(vec, list) or not vec:
        raise EmbeddingError(f"{where} is empty or wrong shape")
    try:
        return [float(x) for x in vec]
    except (TypeError, ValueError) as exc:
        raise EmbeddingError(
            f"{where} contains non-numeric values"
        ) from exc


def _ollama_embed_batch(texts: list[str]) -> list[list[float]]:
    try:
        data = _post(_BATCH_ENDPOINT, {"model": _MODEL, "input": texts})
        embeddings = data.get("embeddings")
        if isinstance(embeddings, list):
            if len(embeddings) != len(texts):
                raise EmbeddingError(
                    f"Ollama batch returned {len(embeddings)} embeddings "
                    f"for {len(texts)} inputs"
                )
            return [
                _validate_vector(v, f"Ollama batch embedding #{i}")
                for i, v in enumerate(embeddings)
            ]
    except urllib.error.HTTPError:
        # 404 from older Ollama builds — fall through to per-prompt legacy.
        pass
    return [_ollama_embed(t) for t in texts]


def _ollama_embed(text: str) -> list[float]:
    try:
        data = _post(_BATCH_ENDPOINT, {"model": _MODEL, "input": text})
        embeddings = data.get("embeddings")
        if isinstance(embeddings, list) and embeddings:
            return _validate_vector(
                embeddings[0], f"Ollama embedding ({_BATCH_ENDPOINT})",
            )
    except urllib.error.HTTPError:
        pass
    data = _post(_LEGACY_ENDPOINT, {"model": _MODEL, "prompt": text})
    embedding = data.get("embedding")
    if embedding is None:
        raise EmbeddingError(
            f"missing 'embedding' field in response from {_LEGACY_ENDPOINT}"
        )
    return _validate_vector(
        embedding, f"Ollama embedding ({_LEGACY_ENDPOINT})",
    )


# ── Public surface ──────────────────────────────────────────────────────────


def embed(text: str) -> list[float]:
    """Embed a single text using the active provider."""
    if _select_provider() == "mock":
        return _mock_embed(text)
    return _ollama_embed(text)


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed many texts in one round-trip when the provider supports it."""
    if not texts:
        return []
    if _select_provider() == "mock":
        return [_mock_embed(t) for t in texts]
    return _ollama_embed_batch(texts)
