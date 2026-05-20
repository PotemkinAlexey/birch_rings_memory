"""Echo Validation — delayed resonance signal via cross-session topic matching."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .cluster import ClusterBundle, bundle as _bundle, nearest_similarity


@dataclass
class StoredSession:
    session_id: str
    bundle: ClusterBundle           # K centroids representing session topics
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
_ECHO_THRESHOLD = 0.68


class EchoStore:
    """In-memory store of closed sessions for echo detection."""

    def __init__(self, default_k: int = 2) -> None:
        self._sessions: dict[str, StoredSession] = {}
        self._k = default_k

    def record(
        self,
        session_id: str,
        all_vectors: list[list[float]],
        r_score: float,
        k: int | None = None,
    ) -> ClusterBundle:
        """
        Store session as a bundle of K centroids.

        K is auto-reduced if session has fewer messages than K.
        Returns the computed bundle (useful for inspection/testing).
        """
        b = _bundle(all_vectors, k=k or self._k)
        self._sessions[session_id] = StoredSession(
            session_id=session_id,
            bundle=b,
            r_score=r_score,
        )
        return b

    def detect_echo(self, new_topic_vector: list[float]) -> EchoResult:
        """
        Check if the new session is returning to a previously unresolved problem.

        Matches against the nearest centroid in each session's bundle —
        a multi-topic session won't miss an echo just because the overall
        centroid drifted away from the problematic sub-topic.
        """
        if not self._sessions:
            return EchoResult(None, 0.0, 0.0, "no_history")

        best_id = max(
            self._sessions,
            key=lambda sid: nearest_similarity(new_topic_vector, self._sessions[sid].bundle),
        )
        best_sim = nearest_similarity(new_topic_vector, self._sessions[best_id].bundle)

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
        past.r_score = min(-0.2, max(-1.0, new_score))

        return EchoResult(
            matched_session_id=best_id,
            similarity=round(best_sim, 4),
            penalty=penalty,
            label="echo",
        )

    def get(self, session_id: str) -> StoredSession | None:
        return self._sessions.get(session_id)
