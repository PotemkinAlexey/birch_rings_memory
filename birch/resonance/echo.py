"""Echo Validation — delayed resonance signal via cross-session topic matching."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from .centroid import centroid as _centroid, dispersion as _dispersion


import math


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


@dataclass
class StoredSession:
    session_id: str
    topic_vector: list[float]       # centroid of all session message embeddings
    r_score: float                  # resonance at close time
    timestamp: float = field(default_factory=time.time)
    echo_penalty: float = 0.0      # applied retroactively if echo detected


@dataclass
class EchoResult:
    matched_session_id: str | None
    similarity: float
    penalty: float          # 0.0 or negative — applied to matched session
    label: str              # "echo" | "clean" | "no_history"


# Similarity threshold above which we consider it a return to the same problem
_ECHO_THRESHOLD = 0.80


class EchoStore:
    """In-memory store of closed sessions for echo detection."""

    def __init__(self) -> None:
        self._sessions: dict[str, StoredSession] = {}

    def record(
        self,
        session_id: str,
        all_vectors: list[list[float]],
        r_score: float,
    ) -> None:
        """Store session using centroid of all message vectors — O(dim) memory."""
        topic_vector = _centroid(all_vectors) if len(all_vectors) > 1 else all_vectors[0]
        self._sessions[session_id] = StoredSession(
            session_id=session_id,
            topic_vector=topic_vector,
            r_score=r_score,
        )

    def detect_echo(self, new_topic_vector: list[float]) -> EchoResult:
        """
        Check if the new session is returning to a previously unresolved problem.

        Finds the most similar past session. If similarity > threshold and
        the past session had R < 0.5 (wasn't strongly resonant), it's an echo.
        If similarity > threshold and past session had R >= 0.5, user is
        returning despite success — still flag but lighter penalty.
        """
        if not self._sessions:
            return EchoResult(None, 0.0, 0.0, "no_history")

        best_id, best_sim = max(
            self._sessions.items(),
            key=lambda kv: _cosine(kv[1].topic_vector, new_topic_vector),
        )
        best_sim = _cosine(self._sessions[best_id].topic_vector, new_topic_vector)

        if best_sim < _ECHO_THRESHOLD:
            return EchoResult(None, round(best_sim, 4), 0.0, "clean")

        past = self._sessions[best_id]

        # Past session seemed resonant but user returned — strongest signal of false closure
        if past.r_score > 0.35:
            penalty = -0.8
        # Past session was already weak/toxic — confirm it stays bad
        else:
            penalty = -0.6

        # Apply penalty retroactively — guarantee score drops into toxic zone
        past.echo_penalty = penalty
        new_score = past.r_score + penalty
        # Echo is definitive evidence of failure: floor at -0.2 minimum
        past.r_score = min(-0.2, max(-1.0, new_score))

        return EchoResult(
            matched_session_id=best_id,
            similarity=round(best_sim, 4),
            penalty=penalty,
            label="echo",
        )

    def get(self, session_id: str) -> StoredSession | None:
        return self._sessions.get(session_id)
