"""Late-binding proxies for the embedding entry points.

The legacy single-file ``birch/memory_store.py`` imported ``embed`` /
``embed_batch`` at module top, which let tests do
``monkeypatch.setattr(birch.memory_store, "embed", fake)`` and have
every call site pick up the substitution. Splitting the module into a
package breaks that — the mixin modules would each capture their own
binding at import time.

These thin wrappers re-resolve the names via the package on every
call, so a runtime patch of ``birch.memory_store.embed`` propagates to
every mixin method that imports ``embed`` from here.
"""
from __future__ import annotations

from typing import Any


def embed(*args: Any, **kwargs: Any) -> Any:
    from birch import memory_store as _pkg
    return _pkg.embed(*args, **kwargs)


def embed_batch(*args: Any, **kwargs: Any) -> Any:
    from birch import memory_store as _pkg
    return _pkg.embed_batch(*args, **kwargs)
