"""Echo Validation — delayed resonance signal via cross-session topic matching."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .cluster import ClusterBundle, nearest_similarity
from .cluster import bundle as _bundle


@dataclass
class StoredSession:
    session_id: str
    bundle: ClusterBundle           # K centroids representing session topics
    r_score: float                  # resonance at close time
    fact_weights: dict[str, float] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    echo_penalty: float = 0.0      # applied retroactively if echo detected

    @property
    def fact_ids(self) -> list[str]:
        return list(self.fact_weights.keys())


@dataclass
class EchoResult:
    matched_session_id: str | None
    similarity: float
    penalty: float          # 0.0 or negative — applied to matched session
    label: str              # "echo" | "clean" | "no_history"
    fact_weights: dict[str, float] = field(default_factory=dict)

    @property
    def fact_ids(self) -> list[str]:
        return list(self.fact_weights.keys())


# Similarity threshold above which we consider it a return to the same
# problem. Sourced from the env-overridable Thresholds module — an
# operator with a tighter-clustering embedding model can lower it, or
# raise it on a sparser model where 0.68 happens too often.
from ..thresholds import Thresholds  # noqa: E402

_ECHO_THRESHOLD = Thresholds.ECHO

# TTL defaults (in seconds). The store is unbounded by default; ``expire()``
# is opportunistic — call it from session lifecycle or from a background
# job, not on every detect_echo.
TTL_RESOLVED = 7 * 24 * 3600       # resonant sessions older than a week
TTL_PENALIZED = 14 * 24 * 3600     # echo-applied sessions older than 2 weeks
TTL_DEFAULT = 30 * 24 * 3600       # everything else older than 30 days


class EchoStore:
    """In-memory store of closed sessions for echo detection."""

    def __init__(self, default_k: int = 2) -> None:
        self._sessions: dict[str, StoredSession] = {}
        self._k = default_k
        # Process-lifetime counters surfaced by MemoryStore.stats. They
        # answer three questions you need to debug an echo system:
        # "is detect_echo finding anything at all" (detected),
        # "is the penalty actually being applied" (applied),
        # "is the second-hit idempotency hot" (ignored). Reset on
        # process restart by design — stats are per-instance, not
        # historical, same contract as collapse_counter.
        self.total_echoes_detected = 0
        self.total_echoes_applied = 0
        self.total_echoes_ignored = 0

    def record(
        self,
        session_id: str,
        all_vectors: list[list[float]],
        r_score: float,
        k: int | None = None,
        fact_weights: dict[str, float] | None = None,
    ) -> ClusterBundle:
        """
        Store session as a bundle of K centroids.

        K is auto-reduced if session has fewer messages than K.
        ``fact_weights`` maps fact_id → relevance weight ∈ [0, 1]; it
        is what future echoes use to apply a retroactive penalty to
        gravity, scaled by how relevant each fact actually was.
        """
        b = _bundle(all_vectors, k=k or self._k)
        self._sessions[session_id] = StoredSession(
            session_id=session_id,
            bundle=b,
            r_score=r_score,
            fact_weights=dict(fact_weights or {}),
        )
        return b

    def detect_echo(
        self,
        new_topic_vector: list[float],
        exclude_session_id: str | None = None,
    ) -> EchoResult:
        """
        Check if the new session is returning to a previously unresolved problem.

        Matches against the nearest centroid in each session's bundle —
        a multi-topic session won't miss an echo just because the overall
        centroid drifted away from the problematic sub-topic.

        ``exclude_session_id`` skips a known session from the match pool —
        typically used when ``check_echo`` is called explicitly with a
        currently-open session id that should not match itself.
        """
        candidates = [
            sid for sid in self._sessions if sid != exclude_session_id
        ]
        if not candidates:
            return EchoResult(None, 0.0, 0.0, "no_history")

        best_id = max(
            candidates,
            key=lambda sid: nearest_similarity(
                new_topic_vector, self._sessions[sid].bundle),
        )
        best_sim = nearest_similarity(new_topic_vector, self._sessions[best_id].bundle)

        if best_sim < _ECHO_THRESHOLD:
            return EchoResult(None, round(best_sim, 4), 0.0, "clean")

        past = self._sessions[best_id]
        self.total_echoes_detected += 1

        # Idempotent: do not stack penalties if echo was already applied.
        if past.echo_penalty != 0.0:
            self.total_echoes_ignored += 1
            return EchoResult(
                matched_session_id=best_id,
                similarity=round(best_sim, 4),
                penalty=0.0,
                label="echo",
                fact_weights=dict(past.fact_weights),
            )

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
        self.total_echoes_applied += 1

        return EchoResult(
            matched_session_id=best_id,
            similarity=round(best_sim, 4),
            penalty=penalty,
            label="echo",
            fact_weights=dict(past.fact_weights),
        )

    def get(self, session_id: str) -> StoredSession | None:
        return self._sessions.get(session_id)

    def expire(
        self,
        now: float | None = None,
        ttl_resolved: float = TTL_RESOLVED,
        ttl_penalized: float = TTL_PENALIZED,
        ttl_default: float = TTL_DEFAULT,
    ) -> list[str]:
        """Drop sessions that no longer need echo tracking.

        Three tiers:
          - resolved (r_score > 0.35) — already a win, no need to keep
            looking for echoes.
          - penalized (echo_penalty != 0) — already converted into a
            gravity correction; no further action it can drive.
          - everything else — capped at the long default TTL so the
            store never grows unboundedly.

        Returns the list of dropped session_ids.
        """
        now = now if now is not None else time.time()
        dropped: list[str] = []
        for sid, s in list(self._sessions.items()):
            age = now - s.timestamp
            # Penalty tier wins: once a session has been echoed it is locked
            # into the toxic floor and cannot drift back into "resolved".
            if s.echo_penalty != 0.0:
                if age > ttl_penalized:
                    dropped.append(sid)
                continue
            if s.r_score > 0.35:
                if age > ttl_resolved:
                    dropped.append(sid)
                continue
            if age > ttl_default:
                dropped.append(sid)
        for sid in dropped:
            del self._sessions[sid]
        return dropped

    def __len__(self) -> int:
        return len(self._sessions)
