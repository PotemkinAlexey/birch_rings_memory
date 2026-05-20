"""StorageBackend protocol — implement this to plug in any storage engine."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..fact import FactPassport


@runtime_checkable
class StorageBackend(Protocol):
    """
    Structural protocol for BirchKM persistence.

    Any class that implements these methods is a valid backend —
    no inheritance required. Implement only what you need; unused
    operations can raise NotImplementedError.
    """

    def save_fact(self, fact: FactPassport) -> None: ...

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
        fact_ids: list[str] | None = None,
        echo_penalty: float = 0.0,
    ) -> None: ...

    def load_echo_sessions(self) -> list[dict]: ...

    def close(self) -> None: ...
