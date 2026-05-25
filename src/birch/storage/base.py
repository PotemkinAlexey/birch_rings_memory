"""StorageBackend protocol — implement this to plug in any storage engine."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Protocol, runtime_checkable

from ..adaptive_gravity import AdaptiveWeights
from ..fact import FactPassport
from ..meta_fact import MetaFact


@runtime_checkable
class StorageBackend(Protocol):
    """
    Structural protocol for BirchKM persistence.

    Any class that implements these methods is a valid backend —
    no inheritance required. Implement only what you need; unused
    operations can raise NotImplementedError.
    """

    def save_fact(self, fact: FactPassport) -> None: ...

    def save_facts(self, facts: list[FactPassport]) -> None:
        """Persist many facts inside a single transaction.

        Default implementation loops save_fact; backends are encouraged
        to override with a real batch insert.
        """
        for f in facts:
            self.save_fact(f)

    def delete_fact(self, fact_id: str) -> None: ...

    def load_facts(self) -> list[FactPassport]: ...

    def save_edge(self, from_id: str, to_id: str) -> None: ...

    def load_edges(self) -> list[tuple[str, str]]: ...

    def delete_edges_for_fact(self, fact_id: str) -> None:
        """Drop every edge incident to ``fact_id`` (default: no-op).

        Backends that persist the edge graph should override; the default
        keeps minimal backends working unchanged. See ``MemoryStore.delete_fact``
        which calls this so orphan edges do not accumulate.
        """
        return None

    def save_echo_session(
        self,
        session_id: str,
        centroids: list[list[float]],
        r_score: float,
        recorded_at: float,
        fact_weights: dict[str, float] | None = None,
        echo_penalty: float = 0.0,
    ) -> None: ...

    def load_echo_sessions(self, *, cleanup: bool = True) -> list[dict]: ...

    def delete_echo_session(self, session_id: str) -> None: ...

    def save_open_session(
        self,
        session_id: str,
        messages: list[str],
        vectors: list[list[float]],
        facts: dict[str, float],
        started_at: float,
    ) -> None: ...

    def delete_open_session(self, session_id: str) -> None: ...

    def load_open_sessions(self, *, cleanup: bool = True) -> list[dict]: ...

    # ── MetaFact persistence ────────────────────────────────────────────────

    def save_meta_fact(self, meta: MetaFact) -> None: ...

    def save_meta_facts(self, metas: list[MetaFact]) -> None:
        """Persist many MetaFacts in one transaction.

        Default implementation loops save_meta_fact; backends are encouraged
        to override with a real batch insert (SQLite uses executemany).
        """
        for m in metas:
            self.save_meta_fact(m)

    def delete_meta_fact(self, meta_id: str) -> None: ...

    def load_meta_facts(self, *, cleanup: bool = True) -> list[MetaFact]: ...

    # ── Cross-process coordination (optional) ───────────────────────────────
    #
    # A backend shared by concurrent processes should implement both so
    # callers can detect another process' writes and serialize their own.
    # Backends used single-process only may omit them — MemoryStore probes
    # with getattr and degrades to a plain in-memory store.

    def data_version(self) -> int:
        """Return a counter that changes when another connection commits."""
        ...

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Reentrant exclusive transaction; nested writes share one commit."""
        yield

    # ── Adaptive gravity weights (optional) ─────────────────────────────────

    def save_adaptive_weights(self, weights: AdaptiveWeights) -> None:
        """Persist the learned pre-resonance gravity weights."""
        ...

    def load_adaptive_weights(self) -> AdaptiveWeights | None:
        """Return persisted weights, or None when nothing has been learned."""
        ...

    def close(self) -> None: ...
