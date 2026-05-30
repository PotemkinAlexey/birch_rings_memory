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


def echo_penalty_for(prior_r: float) -> float:
    """Echo penalty magnitude, scaled by confidence the revisit signals failure.

        penalty = -(0.6 + 0.2·clamp(prior_r,0,1)) · clamp(1 - prior_r, 0, 1)

    A strongly-resonant prior is ambiguous (continued use vs false closure) →
    near-zero penalty; a weak/toxic prior → full penalty. Continuous in prior_r
    (no step at 0.35). Single source of truth for both the immediate path and
    the deferred close-time apply. See ARCHITECTURE.md for the rationale.
    """
    p = max(0.0, min(1.0, prior_r))
    base = 0.6 + 0.2 * p
    confidence = max(0.0, min(1.0, 1.0 - prior_r))
    return round(-base * confidence, 4)


class EchoStore:
    """In-memory store of closed sessions for echo detection."""

    def __init__(self, default_k: int = 2) -> None:
        self._sessions: dict[str, StoredSession] = {}
        self._k = default_k
        # Per-process counters surfaced by MemoryStore.stats (reset on restart):
        # detected, applied, ignored (idempotent second hit).
        self.total_echoes_detected = 0
        self.total_echoes_applied = 0
        self.total_echoes_ignored = 0
        # Deferred path: peeked echoes session_close withheld because the
        # current session ended resonant (productive revisit). A high
        # cancelled:applied ratio = apply-on-open would have fired on reuse.
        self.total_echoes_cancelled = 0

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

    def _best_match(
        self, vec: list[float], exclude_session_id: str | None,
    ) -> tuple[str | None, float]:
        candidates = [
            sid for sid in self._sessions if sid != exclude_session_id
        ]
        if not candidates:
            return None, 0.0
        best_id = max(
            candidates,
            key=lambda sid: nearest_similarity(vec, self._sessions[sid].bundle),
        )
        return best_id, nearest_similarity(vec, self._sessions[best_id].bundle)

    def peek_echo(
        self,
        new_topic_vector: list[float],
        exclude_session_id: str | None = None,
    ) -> EchoResult:
        """
        Read-only echo detection — find a matching past session, mutate nothing.

        Open-time half of the deferred path: record the candidate now, decide at
        ``session_close`` once this session's own outcome is known (see
        ``apply_echo``). ``penalty`` is the would-be magnitude (informational;
        0.0 if already penalised). Matches the nearest centroid in each session's
        bundle, so a multi-topic session won't miss a drifted sub-topic.
        """
        best_id, best_sim = self._best_match(new_topic_vector, exclude_session_id)
        if best_id is None:
            return EchoResult(None, 0.0, 0.0, "no_history")
        if best_sim < _ECHO_THRESHOLD:
            return EchoResult(None, round(best_sim, 4), 0.0, "clean")
        past = self._sessions[best_id]
        would_be = 0.0 if past.echo_penalty != 0.0 else echo_penalty_for(past.r_score)
        return EchoResult(
            matched_session_id=best_id,
            similarity=round(best_sim, 4),
            penalty=would_be,
            label="echo",
            fact_weights=dict(past.fact_weights),
        )

    def apply_echo(self, matched_session_id: str, scale: float = 1.0) -> EchoResult:
        """
        Commit the retroactive penalty to a previously-peeked match.

        Close-time half of the deferred path. Idempotent: an already-penalised
        session is a no-op, so re-applied echoes never stack. ``scale`` ∈ [0, 1]
        attenuates by the current session's severity (a neutral return penalises
        less than a toxic one); the immediate path uses 1.0. No toxic floor.
        """
        past = self._sessions.get(matched_session_id)
        if past is None:
            # Matched session expired or was dropped between peek and apply.
            return EchoResult(None, 0.0, 0.0, "no_history")
        self.total_echoes_detected += 1
        if past.echo_penalty != 0.0:
            self.total_echoes_ignored += 1
            return EchoResult(
                matched_session_id, 0.0, 0.0, "echo", dict(past.fact_weights),
            )
        scale = max(0.0, min(1.0, scale))
        penalty = round(echo_penalty_for(past.r_score) * scale, 4)
        past.echo_penalty = penalty
        past.r_score = max(-1.0, min(1.0, past.r_score + penalty))
        self.total_echoes_applied += 1
        return EchoResult(
            matched_session_id, 0.0, penalty, "echo", dict(past.fact_weights),
        )

    def detect_echo(
        self,
        new_topic_vector: list[float],
        exclude_session_id: str | None = None,
    ) -> EchoResult:
        """
        Immediate echo detection — peek and apply in one shot.

        For the explicit ``check_echo`` tool and legacy direct callers. The
        streaming and record_session paths use ``peek_echo`` + ``apply_echo``
        so the penalty is gated on the current session's outcome.

        ``exclude_session_id`` skips a known session from the match pool.
        """
        peek = self.peek_echo(new_topic_vector, exclude_session_id)
        if peek.label != "echo" or peek.matched_session_id is None:
            return peek
        applied = self.apply_echo(peek.matched_session_id)
        # apply_echo does not recompute similarity; carry the real one over.
        return EchoResult(
            matched_session_id=applied.matched_session_id,
            similarity=peek.similarity,
            penalty=applied.penalty,
            label=applied.label,
            fact_weights=applied.fact_weights,
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
