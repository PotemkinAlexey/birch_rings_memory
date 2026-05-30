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
    """Retroactive penalty for a past session that a new session echoes.

    The single source of truth for echo penalty *magnitude*, shared by the
    immediate path (``detect_echo`` / explicit ``check_echo`` / one-shot
    ``record_session``) and the deferred, outcome-gated path applied at
    ``session_close``.

    The penalty is scaled by *confidence that the revisit signals failure*:

        confidence = clamp(1 - prior_r, 0, 1)
        penalty    = base(prior_r) * confidence

    Rationale (the "prior_R gate"): returning to a topic that previously
    closed *strongly resonant* is ambiguous — it is at least as likely to be
    "I'm still using what we built" as "the fix never worked". So a high
    prior_r yields a near-zero penalty: we do not punish a genuinely useful
    past session on suspicion alone. A revisit to a topic that closed *weak
    or toxic* is unambiguous — it was already failing and is failing again —
    so confidence is ~1 and the penalty lands at full magnitude.

    This is deliberately the cheap, always-on hedge. The principled fix for
    the streaming case is to wait for the *current* session's outcome before
    applying anything at all (see session_close's deferred echo path); this
    function just makes sure that whenever a penalty IS applied, its size is
    proportional to the evidence rather than a flat -0.8.
    """
    base = -0.8 if prior_r > 0.35 else -0.6
    confidence = max(0.0, min(1.0, 1.0 - prior_r))
    return round(base * confidence, 4)


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
        # Deferred path only: a candidate echo (peeked at session_open) that
        # session_close declined to apply because the *current* session ended
        # resonant — i.e. the revisit was productive, not a return-to-failure.
        # This is the metric that proves the open→close deferral is earning
        # its keep: a high cancelled:applied ratio means the old apply-on-open
        # heuristic was firing on continued-use, not false closure.
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
        Read-only echo detection — find a matching past session WITHOUT
        applying any penalty or mutating any state.

        This is the open-time half of the deferred echo path: a new session
        records the candidate it *might* be echoing, then waits. The penalty
        decision is taken later, at ``session_close``, once the current
        session's own outcome is known (see ``apply_echo``). Returning to a
        topic is not, by itself, evidence the past closure was false — the
        evidence is whether *this* conversation also ends badly. Peeking lets
        us hold that judgement until the evidence exists.

        ``penalty`` here is the *would-be* penalty if applied right now; it is
        informational only (0.0 if the match was already penalised). Matching
        is against the nearest centroid in each session's bundle, so a
        multi-topic session won't miss an echo on a drifted sub-topic.
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

    def apply_echo(self, matched_session_id: str) -> EchoResult:
        """
        Apply the retroactive penalty to a previously-peeked match.

        The close-time half of the deferred path: the caller (session_close)
        has decided — based on the *current* session's outcome — that this
        revisit really does signal a false closure, and now commits the
        penalty. Idempotent: a session already penalised is a no-op
        (penalty 0.0), so a re-applied echo never stacks.

        No forced toxic floor — the penalty is confidence-scaled in
        ``echo_penalty_for`` and the score lands wherever the evidence puts it.
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
        penalty = echo_penalty_for(past.r_score)
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

        For callers that have no future outcome to wait for: the explicit
        ``check_echo`` MCP tool, the one-shot ``record_session`` path, and
        legacy direct callers. The streaming session_open path should use
        ``peek_echo`` (at open) + ``apply_echo`` (at close) instead, so the
        penalty is gated on the current session's actual outcome.

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
