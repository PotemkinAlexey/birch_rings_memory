"""Embedding client — wraps Ollama nomic-embed-text.

Uses the ``/api/embed`` endpoint which accepts a single string or a
list of strings and returns a list of embeddings in one call. We fall
back to the legacy ``/api/embeddings`` (single-prompt only) when the
new endpoint is unavailable, so older Ollama builds still work.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request


_BASE_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
_MODEL = os.environ.get("BIRCH_EMBED_MODEL", "nomic-embed-text")

_BATCH_ENDPOINT = f"{_BASE_URL}/api/embed"
_LEGACY_ENDPOINT = f"{_BASE_URL}/api/embeddings"


def _post(url: str, body: dict, timeout: float = 30.0) -> dict:
    payload = json.dumps(body).encode()
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts in one round-trip when possible."""
    if not texts:
        return []
    try:
        data = _post(_BATCH_ENDPOINT, {"model": _MODEL, "input": texts})
        if isinstance(data, dict) and "embeddings" in data:
            return data["embeddings"]
    except urllib.error.HTTPError:
        # 404 from older Ollama builds — fall through to legacy per-prompt.
        pass
    return [embed(t) for t in texts]


def embed(text: str) -> list[float]:
    """Embed a single text. Prefers the batch endpoint for consistency."""
    try:
        data = _post(_BATCH_ENDPOINT, {"model": _MODEL, "input": text})
        if isinstance(data, dict) and "embeddings" in data and data["embeddings"]:
            return data["embeddings"][0]
    except urllib.error.HTTPError:
        pass
    data = _post(_LEGACY_ENDPOINT, {"model": _MODEL, "prompt": text})
    return data["embedding"]
