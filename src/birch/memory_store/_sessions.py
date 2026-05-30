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
    _closing_sessions: "set[str]"
    _mutation_version: int

    if TYPE_CHECKING:
        _sync: Callable[[], None]
        _txn: Callable[[], Any]
        _reload: Callable[[], None]
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
    def _attribute_to(
        ctx: SessionContext, fact_id: str, weight: float,
    ) -> None:
        """Last-line attribution gate.

        ``max/min`` are not finite-aware: ``min(1.0, float("nan"))``
        returns ``1.0`` on some Python builds and ``nan`` on others
        depending on argument order — relying on clamp semantics for
        NaN is a quiet bug factory. Reject NaN / Infinity / non-
        numeric weights here so the attribution dict (which is
        persisted to ``open_sessions.facts`` and then drives
        ``apply_session_resonance``) can never carry a poisoned
        value, regardless of which call site wired up the weight.
        Legitimate-but-out-of-range values clamp to [0.0, 1.0].
        """
        import math as _math

        try:
            raw = float(weight)
        except (TypeError, ValueError):
            return
        if not _math.isfinite(raw):
            return
        clipped = max(0.0, min(1.0, raw))
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
        """Caller holds self._lock. EWMA-update recent_utility of every touched
        body toward ``target = clamp((r·weight + 1)/2, 0, 1)``. Shared by
        session_close and the echo path so resonance and utility move together.
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

    def _apply_echo_gravity_locked(self, result) -> list[str]:
        """Caller holds self._lock inside a write txn. Propagate an echo penalty
        into gravity: drop the matched past session's facts' resonance, mirror
        it into recent_utility, persist the affected bodies + echo-session row.
        Returns the penalised body ids ([] when there's no applicable penalty).
        Shared by the immediate (check_echo) and deferred (session_close) paths.
        """
        if not (result.label == "echo" and result.penalty != 0.0
                and result.fact_weights):
            return []
        self._engine.apply_session_resonance(result.fact_weights, result.penalty)
        # Mirror into recent_utility so the formula doesn't see resonance drop
        # while utility stays put. Same helper session_close uses forward.
        self._apply_recent_utility_locked(result.fact_weights, result.penalty)
        penalized_body_ids = list(result.fact_weights.keys())
        if self._storage:
            affected_facts = [
                self._facts[bid] for bid in penalized_body_ids
                if bid in self._facts
            ]
            affected_metas = [
                self._meta_facts[bid] for bid in penalized_body_ids
                if bid in self._meta_facts
            ]
            if affected_facts:
                self._storage.save_facts(affected_facts)
            if affected_metas and hasattr(self._storage, "save_meta_facts"):
                self._storage.save_meta_facts(affected_metas)
            past = (
                self._echo.get(result.matched_session_id)
                if result.matched_session_id
                else None
            )
            if past:
                self._storage.save_echo_session(
                    past.session_id,
                    past.bundle.centroids,
                    past.r_score,
                    time.time(),
                    fact_weights=past.fact_weights,
                    echo_penalty=past.echo_penalty,
                )
        self._bump_mutation_locked()
        return penalized_body_ids

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
            try:
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
            except Exception:
                # Storage save_open_session raised after the in-memory
                # session dict already had the new SessionContext.
                # Re-anchor every cache to disk truth before propagating
                # so callers don't see an in-memory session that disk
                # never accepted — symmetric with the other write paths
                # (commit 0f23a62).
                self._reload()
                raise

    def session_message(self, text: str, session_id: Optional[str] = None) -> None:
        """Record a user message in the named session (or the current one)."""
        # Embed outside the lock — slow HTTP call, must not serialize agents.
        vec = embed(text)
        with self._lock:
            try:
                with self._txn():
                    self._sync()
                    sid = self._resolve_sid(session_id)
                    if sid is None or sid not in self._sessions:
                        raise KeyError(f"unknown session: {sid!r}")
                    # Closing-session gate. session_close snapshots
                    # ctx state and then releases the lock for the
                    # heavy compute_resonance call. A push that
                    # lands during that window used to:
                    #   1. append to ctx.messages (in-memory)
                    #   2. persist to disk via save_open_session
                    #   3. get silently dropped when session_close
                    #      pops the ctx after writeback — R was
                    #      computed over the pre-push snapshot.
                    # Agent saw push succeed but the message never
                    # influenced R / echo bundle / future sessions.
                    # Reject explicitly so the caller can retry on
                    # a fresh session_open instead.
                    if sid in self._closing_sessions:
                        raise RuntimeError(
                            f"session_closing: session {sid!r} is "
                            f"mid-close — push would be silently "
                            f"dropped. Wait for close to complete or "
                            f"open a new session."
                        )
                    ctx = self._sessions[sid]
                    # Dim preflight BEFORE mutation. Embedding provider
                    # silently returning a different dim (model swap
                    # mid-session, mock-vs-real flip, partial response)
                    # used to produce ragged ctx.vectors that crashed
                    # the repetition detector later. Loader already
                    # validates dim consistency on read; the live
                    # write path was the missing symmetric guard.
                    if ctx.vectors and len(vec) != len(ctx.vectors[0]):
                        from ..vector_index import DimensionMismatchError
                        raise DimensionMismatchError(
                            f"session_message: vector dim {len(vec)} "
                            f"does not match session's existing dim "
                            f"{len(ctx.vectors[0])}. Embedding model "
                            f"likely changed mid-session."
                        )
                    ctx.messages.append(text)
                    ctx.vectors.append(vec)
                    if self._storage and hasattr(self._storage, "save_open_session"):
                        self._storage.save_open_session(
                            sid, ctx.messages, ctx.vectors, ctx.facts, time.time()
                        )
            except Exception:
                # ctx.messages / ctx.vectors were appended in-memory
                # BEFORE the storage write; on failure the in-memory
                # trajectory contains a message that disk doesn't have.
                # session_close would then compute resonance over a
                # message that never landed, and restart would diverge
                # from pre-restart behaviour. Re-anchor.
                self._reload()
                raise

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
        # Validate r_override / sentiment BEFORE marking the session
        # as closing. If validation raises, the closing flag was
        # never set and session_message can still target this sid
        # (which is the correct behaviour — close didn't really
        # start). Resolving the inputs here also means the rest of
        # the function only deals with a pre-computed ResonanceResult,
        # not the three branches.
        precomputed: Optional["ResonanceResult"] = None
        scoring_source = "heuristic"
        if r_override is not None:
            # NaN / Infinity / non-numeric reject at the core
            # boundary so library users get the same contract as
            # MCP callers (server.py validates the same way before
            # ever reaching this code). max(-1, min(1, NaN)) silently
            # propagates NaN through Python's min/max, which then
            # poisons every downstream comparison.
            import math as _math
            try:
                r_raw = float(r_override)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"r_override must be a finite float in [-1, 1], "
                    f"got {r_override!r}"
                ) from exc
            if not _math.isfinite(r_raw):
                raise ValueError(
                    "r_override must be a finite float in [-1, 1], "
                    "got NaN or Infinity"
                )
            r_value = max(-1.0, min(1.0, r_raw))
            label = (
                "resonant" if r_value > 0.35
                else ("toxic" if r_value < -0.20 else "neutral")
            )
            precomputed = ResonanceResult(
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
            precomputed = ResonanceResult(
                behavioral_score=0.0, semantic_score=0.0,
                repetition_score=0.0, r=r_value, label=label,
            )
            scoring_source = "sentiment"

        with self._lock:
            self._sync()
            sid = self._resolve_sid(session_id)
            if sid is None or sid not in self._sessions:
                return {}
            ctx = self._sessions[sid]
            if not ctx.messages:
                # Empty session has no R / EWMA / echo work to do —
                # just drop it. But _pop_session_locked mutates
                # self._sessions BEFORE calling storage.delete_open_session,
                # so a storage failure used to leave in-memory pop
                # diverged from disk. Wrap in the standard rollback
                # guard symmetric with abort_session.
                try:
                    with self._txn():
                        self._sync()
                        if sid in self._sessions:
                            self._pop_session_locked(sid)
                except Exception:
                    self._reload()
                    raise
                return {}
            messages_snapshot = list(ctx.messages)
            vectors_snapshot = list(ctx.vectors)
            facts_snapshot = dict(ctx.facts)
            # Deferred-echo marker peeked at open; resolved after R is known.
            pending_echo = ctx.pending_echo
            # Trajectory invariant check BEFORE marking closing:
            # messages_snapshot is non-empty here (the empty case
            # returned above), so vectors_snapshot[0] / [-1] will
            # be dereferenced by compute_resonance. A corrupted
            # ctx where len(messages) != len(vectors) (e.g. an
            # open_sessions row that survived an older loader gap)
            # would IndexError into the closing flag and brick the
            # sid. Raise BEFORE setting the flag so the sid stays
            # usable; caller can abort_session() to clean up.
            if len(messages_snapshot) != len(vectors_snapshot):
                raise ValueError(
                    f"session trajectory corrupted: "
                    f"len(messages)={len(messages_snapshot)} != "
                    f"len(vectors)={len(vectors_snapshot)}; "
                    f"abort_session({sid!r}) to clean up"
                )
            # Mark closing — under the lock, after snapshot, before
            # we release for compute_resonance. session_message will
            # now reject pushes to this sid until close completes
            # or fails. The try/finally below guarantees the flag
            # clears on every exit path — including ValueError from
            # an invariant check above, or storage failure mid-write
            # — so a partial failure never permanently bricks the
            # session id.
            self._closing_sessions.add(sid)

        try:
            # Heuristic path runs lock-free so other agents can keep
            # querying. r_override / sentiment paths were precomputed
            # at function entry (before the closing flag was set, so
            # bad inputs there don't leak into _closing_sessions).
            if precomputed is not None:
                result = precomputed
            else:
                result = compute_resonance(
                    messages_snapshot,
                    start_vector=vectors_snapshot[0],
                    end_vector=vectors_snapshot[-1],
                    all_vectors=vectors_snapshot,
                )

            with self._lock:
                try:
                    with self._txn():
                        # Reload under the write lock — tick recomputes gravity for
                        # every fact, so it must run on the authoritative state.
                        self._sync()

                        # Snapshot-drift detection. compute_resonance ran lock-free
                        # above so other agents could keep querying — but in that
                        # window another thread could session_push() new messages
                        # OR query() with this same session_id (which attributes
                        # new fact_ids to ctx.facts). Without revalidation those
                        # post-snapshot attributions would be silently dropped:
                        # facts_snapshot is a frozen copy, and below we propagate
                        # R only to its keys. After session_close pops the ctx
                        # the new attributions vanish along with it.
                        #
                        # Fix: merge the current ctx.facts state into the
                        # snapshot — giving new fact attributions the same R
                        # as the snapshot's, because they're part of the same
                        # session. Same-session writes during close are rare
                        # but always "still part of this conversation".
                        #
                        # Scope of merge — facts ONLY. messages_snapshot and
                        # vectors_snapshot are intentionally NOT updated:
                        # result.r was already computed lock-free over the
                        # snapshot trajectory and re-running compute_resonance
                        # under the writeback lock would (a) hold the lock
                        # for the heavy compute and (b) potentially churn R
                        # if new messages just barely tip the heuristic.
                        # Late messages are REJECTED entirely by the
                        # _closing_sessions gate in session_message (set
                        # right after the snapshot above) — they never
                        # make it to ctx.messages, so there is nothing
                        # to "drop". Only late fact attribution (from a
                        # concurrent query() on this sid) still merges
                        # via the drift_facts union below, because the
                        # gate is set per-method (session_message rejects;
                        # query() does not).
                        live_ctx = self._sessions.get(sid)
                        if live_ctx is not None:
                            drift_facts = {
                                fid: w
                                for fid, w in live_ctx.facts.items()
                                if fid not in facts_snapshot
                            }
                            if drift_facts:
                                # Track in the snapshot dict so all downstream
                                # propagation (gravity, EWMA, echo bundle) sees
                                # the union without separate code paths.
                                facts_snapshot.update(drift_facts)

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

                        # Gravity moves by effective_r = R · confidence, not raw
                        # R, so a low-confidence (conflicted/single-signal)
                        # session barely nudges it. Every gravity-moving step
                        # below uses effective_r; summary keeps raw r/label.
                        # r_override / sentiment carry confidence 1.0.
                        confidence = getattr(result, "confidence", 1.0)
                        effective_r = result.r * confidence

                        # Propagate to attributed facts (weighted by relevance).
                        self._engine.apply_session_resonance(facts_snapshot, effective_r)

                        # Touch consulted bodies, then EWMA-update recent_utility.
                        for fid in facts_snapshot:
                            body = (self._facts.get(fid)
                                    or self._meta_facts.get(fid))
                            if body is not None:
                                body.touch()
                        self._apply_recent_utility_locked(facts_snapshot, effective_r)

                        # Deferred echo (peeked at open) decided here on this
                        # session's outcome — before tick() so the migration
                        # sees the penalty. Cancel on a CONFIDENTLY resonant
                        # outcome (effective_r > 0.35, not the raw label: a
                        # damped-to-neutral session is no evidence of a
                        # productive revisit). Otherwise apply, scaled by
                        # severity below, so the cancel/apply boundary is
                        # continuous in effective_r.
                        echo_outcome = "none"
                        if pending_echo:
                            matched_id = pending_echo.get("matched_session_id")
                            if effective_r > 0.35:
                                self._echo.total_echoes_cancelled += 1
                                echo_outcome = "cancelled"
                            elif matched_id:
                                # severity ramps 0 (resonant threshold) → 1
                                # (fully toxic): a neutral return penalises less
                                # than a toxic one.
                                severity = max(
                                    0.0, min(1.0, (0.35 - effective_r) / 1.35))
                                echo_result = self._echo.apply_echo(
                                    matched_id, scale=severity)
                                applied_ids = self._apply_echo_gravity_locked(
                                    echo_result)
                                echo_outcome = (
                                    "applied" if applied_ids else "noop"
                                )

                        self._echo.record(
                            sid,
                            vectors_snapshot,
                            effective_r,  # damped value is the echo prior
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
                            target = max(0.0, min(1.0, (effective_r + 1.0) / 2.0))
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
                            # which path resolved R: heuristic / sentiment / r_override
                            "scoring_source": scoring_source,
                            # none / applied / cancelled / noop
                            "echo_outcome": echo_outcome,
                            # confidence ∈ [0,1] and effective_r = R·confidence
                            # (what moved gravity); raw r/label above unchanged.
                            "confidence": round(confidence, 3),
                            "effective_r": round(effective_r, 3),
                        }
                        self._pop_session_locked(sid)

                except Exception:
                    # Storage rolled back mid-write; in-memory caches
                    # may be partially mutated (apply_session_resonance,
                    # body.touch, _apply_recent_utility_locked, _echo.record,
                    # _engine.tick, _absorb_dead, weights.update,
                    # _pop_session_locked). Re-anchor every cache to disk
                    # truth before propagating — symmetric with add_fact /
                    # add_facts / query / collapse_singularity.
                    self._reload()
                    raise
            return summary
        finally:
            # The outer try/finally guarantees the closing flag clears
            # on EVERY exit path — successful close, storage-failure
            # _reload, AND any unexpected exception (including a
            # KeyboardInterrupt). Without this finally the sid could
            # be permanently bricked into "session_closing" with no
            # way to recover short of process restart. Holding the
            # lock briefly for the discard so the flag clears
            # atomically with any concurrent session_message check.
            with self._lock:
                self._closing_sessions.discard(sid)

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
            try:
                with self._txn():
                    self._sync()
                    if session_id not in self._sessions:
                        return False
                    self._pop_session_locked(session_id)
                    return True
            except Exception:
                # _pop_session_locked drops from self._sessions in
                # memory BEFORE the storage delete_open_session call.
                # On storage failure the in-memory pop is irreversible
                # without _reload — disk still has the session row,
                # so without recovery the next restart would resurrect
                # an "aborted" session.
                self._reload()
                raise
