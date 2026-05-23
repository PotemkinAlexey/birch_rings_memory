"""MemoryStore — unified entry point for the BirchKM memory system.

This package was split out of the historical single-file
``birch/memory_store.py`` module. The public surface is unchanged —
``from birch.memory_store import MemoryStore`` and
``from birch.memory_store import QueryResult`` continue to work.

``embed`` and ``embed_batch`` are re-exported here so existing tests
that do ``monkeypatch.setattr(birch.memory_store, "embed", ...)`` or
``mock.patch("birch.memory_store.embed", ...)`` keep their effect.
"""
from __future__ import annotations

# Re-export the embedding entry points so tests that patch
# ``birch.memory_store.embed`` / ``birch.memory_store.embed_batch``
# still find the names on this package. The mixin modules call into
# these names through thin wrappers that re-resolve via this package,
# so a runtime patch propagates to every call site.
from ..resonance.embeddings import embed, embed_batch  # noqa: F401

# Re-exports that previously lived at module top.
from ._base import (
    _ABSORPTION_THRESHOLD,
    _META_HAWKING_THRESHOLD,
    MemoryStore,
)
from ._models import QueryResult, SessionContext

__all__ = [
    "MemoryStore",
    "QueryResult",
    "SessionContext",
    "embed",
    "embed_batch",
    "_ABSORPTION_THRESHOLD",
    "_META_HAWKING_THRESHOLD",
]
