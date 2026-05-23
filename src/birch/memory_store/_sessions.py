"""SessionsMixin — session lifecycle and per-session attribution."""
from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Optional

from ..fact import FactPassport
from ..gravity import pre_resonance_features
from ..meta_fact import MetaFact
from ..resonance.detector import ResonanceResult, compute_resonance
from ._embed_proxy import embed
from ._models import SessionContext

if TYPE_CHECKING:  # pragma: no cover
    from ..gravity import GravityEngine
    from ..resonance.echo import EchoStore
    from ..storage import StorageBackend

_logger = logging.getLogger(__name__)


class SessionsMixin:
    """Session-management methods. See ``MemoryStore`` for the assembled API."""

    _lock: "threading.RLock"
    _storage: "Optional[StorageBackend]"
    _facts: "dict[str, FactPassport]"
    _meta_facts: "dict[str, MetaFact]"
    _engine: "GravityEngine"
    _echo: "EchoStore"
    _sessions: "dict[str, SessionContext]"
    _current_session_id: "Optional[str]"
    _mutation_version: int

    if TYPE_CHECKING:
        _sync: Callable[[], None]
        _txn: Callable[[], Any]
        _absorb_dead: Callable[[], list[str]]
        _maybe_trigger_collapse_locked: Callable[[int], None]
        _bump_mutation_locked: Callable[[], None]

    # ── Back-compat shims for the legacy single-session API ────────────────
    #
    # The public methods now accept an explicit session_id. When omitted
    # they fall back to ``_current_session_id`` so old single-user code
    # (``mem.session_start("s"); mem.add_fact(...); mem.session_close()``)
    # keeps working. These shims expose the current session's state under
    # the original attribute names that some tests still touch directly.

    @property
    def _session_id(self) -> Optional[str]:
        return self._current_session_id

    @property
    def _session_messages(self) -> list[str]:
        ctx = self._sessions.get(self._current_session_id or "")
        return ctx.messages if ctx else []

    @property
    def _session_vectors(self) -> list[list[float]]:
        ctx = self._sessions.get(self._current_session_id or "")
        return ctx.vectors if ctx else []

    @property
    def _session_facts(self) -> dict[str, float]:
        ctx = self._sessions.get(self._current_session_id or "")
        return ctx.facts if ctx else {}

    @_session_facts.setter
    def _session_facts(self, value) -> None:
        ctx = self._sessions.get(self._current_session_id or "")
        if ctx is not None:
            ctx.facts = {fid: float(w) for fid, w in dict(value).items()}

    @property
    def _session_fact_ids(self) -> list[str]:
        return list(self._session_facts.keys())

    @_session_fact_ids.setter
    def _session_fact_ids(self, value) -> None:
        if isinstance(value, dict):
            self._session_facts = value
        else:
            self._session_facts = {fid: 1.0 for fid in value}

    def _resolve_sid(self, session_id: Optional[str]) -> Optional[str]:
        return session_id if session_id is not None else self._current_session_id

    @staticmethod
    def _attribute_to(ctx: SessionContext, fact_id: str, weight: float) -> None:
        clipped = max(0.0, min(1.0, float(weight)))
        if clipped > ctx.facts.get(fact_id, 0.0):
            ctx.facts[fact_id] = clipped

    def _attribute_fact(self, fact_id: str, weight: float) -> None:
        """Tag a fact to the current (legacy) session if any."""
        ctx = self._sessions.get(self._current_session_id or "")
        if ctx is None:
            return
        self._attribute_to(ctx, fact_id, weight)

    # EWMA half-life for recent_utility: ~7 events to half-life. Slow
    # enough that a single outlier session does not swing the prior, fast
    # enough that a meaningful streak shows up within a week.
    _RECENT_UTILITY_ALPHA: float = 0.15

    def _apply_recent_utility_locked(
        self,
        body_weights: dict[str, float],
        r: float,
    ) -> None:
        """Caller must hold self._lock. Update the EWMA recent_utility of
        every touched body (FactPassport or MetaFact) by the per-body
        realised value ``target = clamp((r·weight + 1) / 2, 0, 1)``.

        Used by both ``session_close`` (positive/negative R from the just-
        closed session) and ``check_echo`` (retroactive negative penalty
        from an echo). Without sharing the EWMA path, the resonance signal
        would change but recent_utility would not — gravity formula would
        see contradictory inputs.
        """
        alpha = self._RECENT_UTILITY_ALPHA
        for bid, attr_weight in body_weights.items():
            body = self._facts.get(bid) or self._meta_facts.get(bid)
            if body is None:
                continue
            per_body_r = r * float(attr_weight)
            target = max(0.0, min(1.0, (per_body_r + 1.0) / 2.0))
            body.recent_utility = (
                (1.0 - alpha) * body.recent_utility + alpha * target
            )

    def _persist_session_locked(self, ctx: Optional[SessionContext]) -> None:
        """Caller must hold self._lock AND be inside a write transaction.

        After mutating ``ctx.facts`` (attribution from a write or read), flush
        the open-session row so the per-fact relevance map survives a crash
        before ``session_close``. Without this, an agent that records facts
        and dies mid-session loses the attribution mapping and the eventual
        resonance signal will not reach those bodies.
        """
        if ctx is None or self._storage is None:
            return
        if not hasattr(self._storage, "save_open_session"):
            return
        self._storage.save_open_session(
            ctx.session_id, ctx.messages, ctx.vectors, ctx.facts, time.time(),
        )

    # ── Session lifecycle ────────────────────────────────────────────────────

    def session_start(self, session_id: str) -> bool:
        """Open a session context. Safe to call concurrently.

        Idempotent: if the session already exists, the existing
        context is preserved (messages, vectors, fact attribution
        intact) and just promoted to the current session. An agent
        that calls session_open twice with the same id — typically
        on retry after a failed embed — does NOT lose its in-flight
        trajectory or the gravity signal it accumulated. Previously
        a second session_start silently overwrote the context with
        an empty one, which lost the conversation's resonance
        attribution before close.

        Returns ``True`` if a new SessionContext was created, ``False``
        if an existing one was promoted (idempotent reopen). The MCP
        wrapper surfaces this as ``already_open`` so retry-aware agents
        can distinguish a fresh open from a recovered in-flight one.
        """
        with self._lock:
            with self._txn():
                self._sync()
                if session_id in self._sessions:
                    # Already open — preserve in-flight state. Just
                    # promote to current and skip re-persistence
                    # (storage row is already there, and overwriting
                    # would clobber any attribution accumulated since
                    # the first session_start).
                    self._current_session_id = session_id
                    _logger.info(
                        "session_start: %r already open — preserving "
                        "in-flight context (idempotent open)",
                        session_id,
                    )
                    return False
                ctx = SessionContext(session_id=session_id)
                self._sessions[session_id] = ctx
                self._current_session_id = session_id
                if self._storage and hasattr(self._storage, "save_open_session"):
                    self._storage.save_open_session(
                        session_id, ctx.messages, ctx.vectors, ctx.facts, time.time()
                    )
                return True

    def session_message(self, text: str, session_id: Optional[str] = None) -> None:
        """Record a user message in the named session (or the current one)."""
        # Embed outside the lock — slow HTTP call, must not serialize agents.
        vec = embed(text)
        with self._lock:
            with self._txn():
                self._sync()
                sid = self._resolve_sid(session_id)
                if sid is None or sid not in self._sessions:
                    raise KeyError(f"unknown session: {sid!r}")
                ctx = self._sessions[sid]
                ctx.messages.append(text)
                ctx.vectors.append(vec)
                if self._storage and hasattr(self._storage, "save_open_session"):
                    self._storage.save_open_session(
                        sid, ctx.messages, ctx.vectors, ctx.facts, time.time()
                    )

    # Sentiment shortcut → R value mapping. Discrete choices spelled
    # out so a calling agent doesn't have to memorise which float means
    # what. 0.7 / -0.7 land cleanly inside the resonant / toxic bands
    # (threshold 0.35 / -0.20 per detector.classify) without saturating.
    _SENTIMENT_MAP: ClassVar[dict[str, float]] = {
        "resonant": 0.7,
        "positive": 0.7,
        "neutral": 0.0,
        "toxic": -0.7,
        "negative": -0.7,
    }

    def session_close(
        self,
        session_id: Optional[str] = None,
        sentiment: Optional[str] = None,
        r_override: Optional[float] = None,
    ) -> dict:
        """
        Close session: compute resonance, propagate R to facts,
        record echo bundle, tick gravity, absorb dead facts.

        Operates on the named session if provided; otherwise on the
        most recently opened one.

        Resonance scoring — three paths, caller's choice:
          - ``r_override`` (float in [-1, 1]): use this exact value.
            Wins over everything else. Right when the caller already
            knows the realised value precisely.
          - ``sentiment`` (``"resonant"`` / ``"neutral"`` / ``"toxic"``,
            or aliases ``"positive"`` / ``"negative"``): discrete
            shortcut. Maps to ±0.7 / 0.0 — lands cleanly inside the
            label bands without saturating. Right when an agent
            summarises declaratively ("round closed, 6 fixes, all
            tests pass") and knows the heuristic will mis-classify
            grumpy-sounding tech vocabulary as toxic.
          - Neither: fall back to ``compute_resonance`` on the message
            text. The original contract, unchanged when sentiment +
            r_override are both None.

        Response carries ``scoring_source`` (``"heuristic"`` /
        ``"sentiment"`` / ``"r_override"``) for transparency.
        """
        with self._lock:
            self._sync()
            sid = self._resolve_sid(session_id)
            if sid is None or sid not in self._sessions:
                return {}
            ctx = self._sessions[sid]
            if not ctx.messages:
                self._pop_session_locked(sid)
                return {}
            messages_snapshot = list(ctx.messages)
            vectors_snapshot = list(ctx.vectors)
            facts_snapshot = dict(ctx.facts)

        # Resolve the resonance score. Override paths skip the
        # behavioural / semantic / repetition computation entirely.
        scoring_source = "heuristic"
        if r_override is not None:
            r_value = float(max(-1.0, min(1.0, r_override)))
            label = (
                "resonant" if r_value > 0.35
                else ("toxic" if r_value < -0.20 else "neutral")
            )
            result = ResonanceResult(
                behavioral_score=0.0, semantic_score=0.0,
                repetition_score=0.0, r=r_value, label=label,
            )
            scoring_source = "r_override"
        elif sentiment is not None:
            if sentiment not in self._SENTIMENT_MAP:
                raise ValueError(
                    f"sentiment must be one of "
                    f"{sorted(self._SENTIMENT_MAP)}, got {sentiment!r}"
                )
            r_value = self._SENTIMENT_MAP[sentiment]
            label = (
                "resonant" if r_value > 0.35
                else ("toxic" if r_value < -0.20 else "neutral")
            )
            result = ResonanceResult(
                behavioral_score=0.0, semantic_score=0.0,
                repetition_score=0.0, r=r_value, label=label,
            )
            scoring_source = "sentiment"
        else:
            # Heuristic path — pure computation on the snapshot, run
            # outside the lock so other agents can keep querying.
            result = compute_resonance(
                messages_snapshot,
                start_vector=vectors_snapshot[0],
                end_vector=vectors_snapshot[-1],
                all_vectors=vectors_snapshot,
            )

        with self._lock:
            with self._txn():
                # Reload under the write lock — tick recomputes gravity for
                # every fact, so it must run on the authoritative state.
                self._sync()

                # Snapshot pre-resonance features for facts about to receive
                # their first ever resonance — these are the training events
                # for the adaptive weights. Resonance is the teacher; the
                # weights learn what predicted realised value *before* it.
                now_ts = time.time()
                max_deg = max(self._engine._degrees.values(), default=1)
                training_features: list[
                    tuple[float, float, float, float, float]
                ] = []
                for fid in facts_snapshot:
                    # Body may be a live FactPassport or a live MetaFact
                    # (Hawking-emitted); both share the same gravity surface
                    # and must train the formula symmetrically.
                    body = (self._facts.get(fid)
                            or self._meta_facts.get(fid))
                    if body is not None and body.resonance_count == 0:
                        training_features.append(pre_resonance_features(
                            body,
                            graph_degree=self._engine._degrees.get(fid, 0),
                            max_degree=max_deg,
                            now=now_ts,
                        ))

                # Propagate R to facts used in this session, weighted by how
                # relevant each fact was to the session's queries.
                self._engine.apply_session_resonance(facts_snapshot, result.r)

                # Touch every body the session actually consulted (so
                # access_count / last_accessed reflect the read), then
                # update recent_utility EWMA via the shared helper —
                # symmetric with check_echo which now applies the same
                # EWMA update on retroactive negative penalty.
                for fid in facts_snapshot:
                    body = (self._facts.get(fid)
                            or self._meta_facts.get(fid))
                    if body is not None:
                        body.touch()
                self._apply_recent_utility_locked(facts_snapshot, result.r)

                self._echo.record(
                    sid,
                    vectors_snapshot,
                    result.r,
                    fact_weights=facts_snapshot,
                )
                # Opportunistic TTL sweep on session close. Drops stale resolved
                # and already-penalised echo sessions so the store stays bounded
                # without a separate cron job.
                expired = self._echo.expire()
                if self._storage:
                    session_obj = self._echo.get(sid)
                    if session_obj:
                        self._storage.save_echo_session(
                            sid,
                            session_obj.bundle.centroids,
                            session_obj.r_score,
                            time.time(),
                            fact_weights=session_obj.fact_weights,
                            echo_penalty=session_obj.echo_penalty,
                        )
                    if expired and hasattr(self._storage, "delete_echo_session"):
                        for stale_sid in expired:
                            self._storage.delete_echo_session(stale_sid)

                migrations = self._engine.tick()
                absorbed = self._absorb_dead()

                if self._storage:
                    self._storage.save_facts(list(self._facts.values()))
                    if self._meta_facts and hasattr(self._storage, "save_meta_facts"):
                        self._storage.save_meta_facts(list(self._meta_facts.values()))

                # Train the adaptive weights: one regularised SGD step per
                # session toward (R + 1) / 2, using the mean of first-resonance
                # features as the predictor. One signal in, one step out — the
                # weights drift slowly toward what predicts value FOR YOU.
                if training_features:
                    n = len(training_features)
                    mean_f = sum(f[0] for f in training_features) / n
                    mean_a = sum(f[1] for f in training_features) / n
                    mean_g = sum(f[2] for f in training_features) / n
                    mean_u = sum(f[3] for f in training_features) / n
                    mean_s = sum(f[4] for f in training_features) / n
                    target = max(0.0, min(1.0, (result.r + 1.0) / 2.0))
                    # Multi-process safety: another process may have stepped
                    # the weights between our boot and this commit. Reload
                    # the authoritative row UNDER the write txn so our SGD
                    # step composes on top of their learning, not on top of
                    # our stale in-memory copy. Without this, the last
                    # writer silently overwrites every concurrent process's
                    # train_count.
                    if (self._storage
                            and hasattr(self._storage, "load_adaptive_weights")):
                        fresh = self._storage.load_adaptive_weights()
                        if (fresh is not None
                                and fresh.train_count
                                    >= self._engine.weights.train_count):
                            self._engine.weights = fresh
                    self._engine.weights.update(
                        mean_f, mean_a, mean_g, mean_u, mean_s, target=target,
                    )
                    if self._storage and hasattr(self._storage, "save_adaptive_weights"):
                        self._storage.save_adaptive_weights(self._engine.weights)

                # Counter-triggered collapse. Held inside the lock so the
                # collapse counter and the trigger decision are consistent.
                self._maybe_trigger_collapse_locked(len(absorbed))
                self._bump_mutation_locked()

                summary = {
                    "session_id": sid,
                    "r": result.r,
                    "label": result.label,
                    "migrations": migrations,
                    "absorbed": absorbed,
                    # Tells the caller which path resolved R: heuristic
                    # text scan (the original contract) vs an explicit
                    # sentiment label vs a direct numeric override.
                    "scoring_source": scoring_source,
                }
                self._pop_session_locked(sid)

        return summary

    def _pop_session_locked(self, sid: str) -> None:
        """Caller must hold self._lock."""
        self._sessions.pop(sid, None)
        if self._current_session_id == sid:
            self._current_session_id = (
                next(reversed(self._sessions)) if self._sessions else None
            )
        if self._storage and hasattr(self._storage, "delete_open_session"):
            self._storage.delete_open_session(sid)

    def abort_session(self, session_id: str) -> bool:
        """Drop an open session without computing resonance or touching gravity.

        For failure-path cleanup — when ``session_open(first_message=...)``
        or ``record_session`` partially started a session and a downstream
        embed failed, the caller can abort instead of leaking an orphan
        open session that waits for TTL cleanup or blocks future opens
        of the same id. Idempotent: aborting an unknown session returns
        False without raising.

        Distinct from ``session_close``: no R computed, no migrations,
        no EWMA, no adaptive-weight training. Pure cleanup.
        """
        with self._lock:
            with self._txn():
                self._sync()
                if session_id not in self._sessions:
                    return False
                self._pop_session_locked(session_id)
                return True
