"""StorageBackend protocol — implement this to plug in any storage engine."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

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

    def save_echo_session(
        self,
        session_id: str,
        centroids: list[list[float]],
        r_score: float,
        recorded_at: float,
        fact_weights: dict[str, float] | None = None,
        echo_penalty: float = 0.0,
    ) -> None: ...

    def load_echo_sessions(self) -> list[dict]: ...

    def save_open_session(
        self,
        session_id: str,
        messages: list[str],
        vectors: list[list[float]],
        facts: dict[str, float],
        started_at: float,
    ) -> None: ...

    def delete_open_session(self, session_id: str) -> None: ...

    def load_open_sessions(self) -> list[dict]: ...

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

    def load_meta_facts(self) -> list[MetaFact]: ...

    def close(self) -> None: ...
