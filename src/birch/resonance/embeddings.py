"""Embedding client — wraps Ollama nomic-embed-text."""
from __future__ import annotations

import json
import urllib.request


_OLLAMA_URL = "http://localhost:11434/api/embeddings"
_MODEL = "nomic-embed-text"


def embed(text: str) -> list[float]:
    payload = json.dumps({"model": _MODEL, "prompt": text}).encode()
    req = urllib.request.Request(
        _OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())["embedding"]


def embed_batch(texts: list[str]) -> list[list[float]]:
    return [embed(t) for t in texts]
