"""SQLite implementation of StorageBackend."""
from __future__ import annotations

import json
import logging
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from ..adaptive_gravity import AdaptiveWeights
from ..fact import FactPassport
from ..meta_fact import MetaFact

_logger = logging.getLogger(__name__)


def _safe_loads(value: Any, default: Any) -> Any:
    """Parse a JSON cell tolerantly.

    A row corrupted by a partial write, manual SQL fix, or an older
    schema migration must not bring the entire MemoryStore startup
    down. Return ``default`` and log instead — the cost of one ignored
    row is far less than the cost of an unbootable memory server.
    """
    if value is None or value == "":
        return default
    try:
        return json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        snippet = (value[:120] if isinstance(value, str) else repr(value)[:120])
        _logger.warning(
            "SQLite load: corrupted JSON cell, using default: %s | snippet=%r",
            exc, snippet,
        )
        return default


def _finite_float(
    value: Any, default: float, *, lo: float | None = None,
    hi: float | None = None,
) -> float:
    """Coerce a JSON-loaded cell to a finite float with optional clamp.

    Same robustness contract as ``_safe_vector``: a corrupted cell
    (``None``, non-numeric, NaN, Infinity) must not poison downstream
    math. Gravity / utility / stability are bounded in [0, 1] and feed
    straight into the adaptive_gravity SGD — a NaN here cascades into
    NaN weights on the next session_close, which would silently freeze
    learning. Defaulting drops the bad cell back to the field default;
    optional ``lo``/``hi`` clamp legitimate but out-of-range values.
    """
    import math as _math

    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not _math.isfinite(out):
        return default
    if lo is not None:
        out = max(lo, out)
    if hi is not None:
        out = min(hi, out)
    return out


def _nonnegative_int(value: Any, default: int) -> int:
    """Coerce a JSON-loaded cell to a non-negative int.

    Counters (access_count, resonance_count) live in ints; a non-int
    cell (string, float-with-fractional-part, None) would raise inside
    ``int()`` and take the whole loader down. A negative value is
    operationally meaningless — clamp to zero rather than propagate.
    """
    try:
        out = int(value)
    except (TypeError, ValueError):
        return default
    return max(0, out)


def _layer(value: Any, default: int = 1) -> int:
    """Coerce layer field to a known layer id.

    Valid layers: -1 (singularity), 0 (surface), 1 (kinetic), 2 (core).
    An unknown value (5, "core", None, NaN) used to load as-is and
    silently break the live-vs-singularity split — a fact at layer=99
    would slip past every ``layer >= 0`` predicate and out of every
    ``layer == -1`` singularity scan. Default to kinetic (1, the
    creation default) when corrupted.
    """
    try:
        out = int(value)
    except (TypeError, ValueError):
        return default
    if out not in (-1, 0, 1, 2):
        return default
    return out


def _safe_centroids(value: Any) -> list[list[float]]:
    """Parse a JSON-encoded list of centroid vectors with shape checks.

    EchoStore.detect_echo cosines a query centroid against every stored
    centroid — a ragged shape (mixed dims), a non-list item, or a NaN
    cell would crash the cosine path far from the loader. The contract
    here is the same as _safe_vector lifted one nesting level: return
    an empty list on any non-conformant input so the caller's tolerant
    drop-corrupt-row path takes over. An empty list signals "nothing
    useful to match against" — the echo loader treats it as corruption
    and drops the session.
    """
    import math as _math

    raw = _safe_loads(value, [])
    if not isinstance(raw, list) or not raw:
        return []
    out: list[list[float]] = []
    expected_dim: int | None = None
    for v in raw:
        if not isinstance(v, list):
            return []
        try:
            coerced = [float(x) for x in v]
        except (TypeError, ValueError):
            return []
        if not all(_math.isfinite(x) for x in coerced):
            return []
        if expected_dim is None:
            expected_dim = len(coerced)
        elif len(coerced) != expected_dim:
            return []
        out.append(coerced)
    return out


def _safe_vector(value: Any) -> list[float]:
    """Parse a JSON-encoded vector tolerantly with shape validation.

    Valid JSON that is not a list of numbers (e.g. ``{"x": 1}``, ``"abc"``,
    ``[1, "oops", 3]``) must not break downstream code that assumes
    ``list[float]``. Returns an empty list in any non-conformant case so
    the fact loads but is unsearchable (caller's fallback path).
    """
    import math

    raw = _safe_loads(value, [])
    if not isinstance(raw, list):
        return []
    try:
        out = [float(x) for x in raw]
    except (TypeError, ValueError):
        return []
    # NaN / Infinity in stored vectors poison every downstream cosine
    # comparison (similarity becomes NaN, sort order undefined). If
    # the on-disk cell contains a non-finite value, treat the whole
    # vector as missing so the fact loads but is unsearchable.
    if not all(math.isfinite(x) for x in out):
        return []
    return out

_SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    fact_id         TEXT PRIMARY KEY,
    subject         TEXT NOT NULL,
    predicate       TEXT NOT NULL,
    object          TEXT NOT NULL,
    vector          TEXT,
    gravity_score   REAL DEFAULT 0.5,
    layer           INTEGER DEFAULT 1,
    created_at      REAL,
    ttl             REAL,
    source_session  TEXT,
    deprecated_by   TEXT,
    access_count    INTEGER DEFAULT 0,
    last_accessed   REAL,
    resonance_sum   REAL DEFAULT 0.0,
    resonance_count INTEGER DEFAULT 0,
    recent_utility  REAL DEFAULT 0.5,
    forecast_stability REAL DEFAULT 0.5
);

CREATE TABLE IF NOT EXISTS edges (
    from_id TEXT NOT NULL,
    to_id   TEXT NOT NULL,
    PRIMARY KEY (from_id, to_id)
);

CREATE TABLE IF NOT EXISTS echo_sessions (
    session_id   TEXT PRIMARY KEY,
    centroids    TEXT,
    r_score      REAL,
    recorded_at  REAL,
    fact_ids     TEXT,
    echo_penalty REAL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS open_sessions (
    session_id  TEXT PRIMARY KEY,
    messages    TEXT,
    vectors     TEXT,
    facts       TEXT,
    started_at  REAL
);

CREATE TABLE IF NOT EXISTS meta_facts (
    meta_id         TEXT PRIMARY KEY,
    vector          TEXT,
    weight          INTEGER DEFAULT 1,
    source_texts    TEXT,
    source_fact_ids TEXT,
    summary         TEXT,
    gravity_score   REAL DEFAULT 0.30,
    created_at      REAL,
    layer           INTEGER DEFAULT -1,
    access_count    INTEGER DEFAULT 0,
    last_accessed   REAL,
    resonance_sum   REAL DEFAULT 0.0,
    resonance_count INTEGER DEFAULT 0,
    recent_utility  REAL DEFAULT 0.5,
    forecast_stability REAL DEFAULT 0.5
);

CREATE TABLE IF NOT EXISTS adaptive_weights (
    id              INTEGER PRIMARY KEY CHECK (id = 1),
    w_freshness     REAL NOT NULL,
    w_access        REAL NOT NULL,
    w_graph         REAL NOT NULL,
    w_utility       REAL NOT NULL DEFAULT 0.10,
    w_stability     REAL NOT NULL DEFAULT 0.05,
    train_count     INTEGER NOT NULL DEFAULT 0,
    updated_at      REAL
);
"""


class SQLiteBackend:
    """Write-through SQLite backend.

    Safe for concurrent processes sharing one database file: WAL mode allows
    N readers alongside a single writer, and ``transaction()`` serializes
    writers via ``BEGIN IMMEDIATE``. Callers detect another process' commits
    through ``data_version()`` and reload their caches.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        # WAL: concurrent readers + one writer. busy_timeout: a writer that
        # finds the lock held waits instead of failing with "database is locked".
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        # Reentrancy depth for transaction(); nested save_* calls defer their
        # commit to the outermost block.
        self._txn_depth = 0
        self._conn.executescript(_SCHEMA)
        self._migrate_echo_sessions()
        self._migrate_recent_utility()
        self._migrate_forecast_stability()
        self._conn.commit()

    # ── Cross-process coordination ───────────────────────────────────────────

    def data_version(self) -> int:
        """Return SQLite's ``PRAGMA data_version``.

        The value changes whenever the database is modified by *another*
        connection — including other processes and out-of-band edits — but
        not for commits on this connection. A caller that caches state in
        memory compares this against the value seen at its last load: a
        change means the cache is stale and must be reloaded.
        """
        return int(self._conn.execute("PRAGMA data_version").fetchone()[0])

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Reentrant exclusive transaction.

        The outermost ``with`` issues ``BEGIN IMMEDIATE`` (grabbing the write
        lock up front so no other writer can interleave) and commits on exit;
        nested blocks are no-ops on the transaction boundary. ``save_*`` calls
        made inside defer their commit to the outermost block via
        ``_maybe_commit``.
        """
        outermost = self._txn_depth == 0
        if outermost:
            self._conn.execute("BEGIN IMMEDIATE")
        self._txn_depth += 1
        try:
            yield
        except BaseException:
            self._txn_depth -= 1
            if outermost:
                self._conn.rollback()
            raise
        else:
            self._txn_depth -= 1
            if outermost:
                self._conn.commit()

    def _maybe_commit(self) -> None:
        """Commit only when not inside an open transaction() block."""
        if self._txn_depth == 0:
            self._conn.commit()

    def _migrate_echo_sessions(self) -> None:
        """Forward-compatible schema migration for pre-existing DBs."""
        cols = {row["name"] for row in self._conn.execute("PRAGMA table_info(echo_sessions)")}
        if "fact_ids" not in cols:
            self._conn.execute("ALTER TABLE echo_sessions ADD COLUMN fact_ids TEXT")
        if "echo_penalty" not in cols:
            self._conn.execute("ALTER TABLE echo_sessions ADD COLUMN echo_penalty REAL DEFAULT 0")

    def _migrate_recent_utility(self) -> None:
        """Add recent_utility / w_utility columns to older DBs in place."""
        fact_cols = {
            row["name"] for row in self._conn.execute("PRAGMA table_info(facts)")
        }
        if "recent_utility" not in fact_cols:
            self._conn.execute(
                "ALTER TABLE facts ADD COLUMN recent_utility REAL DEFAULT 0.5"
            )
        meta_cols = {
            row["name"] for row in self._conn.execute("PRAGMA table_info(meta_facts)")
        }
        if "recent_utility" not in meta_cols:
            self._conn.execute(
                "ALTER TABLE meta_facts ADD COLUMN recent_utility REAL DEFAULT 0.5"
            )
        weight_cols = {
            row["name"]
            for row in self._conn.execute("PRAGMA table_info(adaptive_weights)")
        }
        if "w_utility" not in weight_cols:
            self._conn.execute(
                "ALTER TABLE adaptive_weights "
                "ADD COLUMN w_utility REAL NOT NULL DEFAULT 0.10"
            )

    def _migrate_forecast_stability(self) -> None:
        """Add forecast_stability / w_stability columns to older DBs."""
        fact_cols = {
            row["name"] for row in self._conn.execute("PRAGMA table_info(facts)")
        }
        if "forecast_stability" not in fact_cols:
            self._conn.execute(
                "ALTER TABLE facts "
                "ADD COLUMN forecast_stability REAL DEFAULT 0.5"
            )
        meta_cols = {
            row["name"] for row in self._conn.execute("PRAGMA table_info(meta_facts)")
        }
        if "forecast_stability" not in meta_cols:
            self._conn.execute(
                "ALTER TABLE meta_facts "
                "ADD COLUMN forecast_stability REAL DEFAULT 0.5"
            )
        weight_cols = {
            row["name"]
            for row in self._conn.execute("PRAGMA table_info(adaptive_weights)")
        }
        if "w_stability" not in weight_cols:
            self._conn.execute(
                "ALTER TABLE adaptive_weights "
                "ADD COLUMN w_stability REAL NOT NULL DEFAULT 0.05"
            )

    # ── Facts ────────────────────────────────────────────────────────────────

    _FACT_INSERT = (
        "INSERT OR REPLACE INTO facts "
        "(fact_id, subject, predicate, object, vector, gravity_score, layer, "
        " created_at, ttl, source_session, deprecated_by, access_count, "
        " last_accessed, resonance_sum, resonance_count, recent_utility, "
        " forecast_stability) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
    )

    @staticmethod
    def _fact_row(fact: FactPassport) -> tuple:
        # Write-side sanitisation symmetric with load_facts(). A
        # poisoned in-memory FactPassport (e.g. library user called
        # ``fact.gravity_score = float("nan")`` directly) used to
        # write radioactive scalars to disk that the next ``_reload``
        # would then "heroically" clean — better never to persist
        # them. ``allow_nan=False`` on the vector dump enforces the
        # same contract for the JSON cell; a NaN there raises
        # ValueError, the surrounding ``_txn`` rolls back, and
        # ``_reload`` restores the pre-write in-memory snapshot.
        ttl = fact.ttl
        if ttl is not None:
            ttl = _finite_float(ttl, None)  # type: ignore[arg-type]
        return (
            fact.fact_id, fact.subject, fact.predicate, fact.object,
            json.dumps(fact.vector, allow_nan=False),
            _finite_float(fact.gravity_score, 0.5, lo=0.0, hi=1.0),
            _layer(fact.layer, 1),
            _finite_float(fact.created_at, time.time()),
            ttl,
            fact.source_session, fact.deprecated_by,
            _nonnegative_int(fact.access_count, 0),
            _finite_float(fact.last_accessed, time.time()),
            _finite_float(fact.resonance_sum, 0.0),
            _nonnegative_int(fact.resonance_count, 0),
            _finite_float(fact.recent_utility, 0.5, lo=0.0, hi=1.0),
            _finite_float(
                fact.forecast_stability, 0.5, lo=0.0, hi=1.0,
            ),
        )

    def save_fact(self, fact: FactPassport) -> None:
        self._conn.execute(self._FACT_INSERT, self._fact_row(fact))
        self._maybe_commit()

    def save_facts(self, facts: list[FactPassport]) -> None:
        """One transaction, one commit — orders of magnitude faster on bulk dumps."""
        if not facts:
            return
        rows = [self._fact_row(f) for f in facts]
        self._conn.executemany(self._FACT_INSERT, rows)
        self._maybe_commit()

    def delete_fact(self, fact_id: str) -> None:
        self._conn.execute("DELETE FROM facts WHERE fact_id = ?", (fact_id,))
        self._maybe_commit()

    def load_facts(self) -> list[FactPassport]:
        rows = self._conn.execute("SELECT * FROM facts").fetchall()
        out: list[FactPassport] = []
        # Scalar sanitization mirrors _safe_vector's contract: a corrupt
        # cell (NaN, Infinity, wrong type, junk-typed via SQLite's loose
        # typing) must not crash the loader or poison adaptive-gravity
        # SGD downstream. Every numeric scalar goes through the helpers;
        # vector already had it.
        for r in rows:
            try:
                ttl = r["ttl"]
                if ttl is not None:
                    ttl = _finite_float(ttl, None)  # type: ignore[arg-type]
                created_at_raw = r["created_at"]
                # created_at is positional in the dataclass — None would
                # accept the field default, but a NaN cell would propagate.
                if created_at_raw is None:
                    created_at = None
                else:
                    import time as _time
                    created_at = _finite_float(created_at_raw, _time.time())
                last_accessed_raw = r["last_accessed"]
                if last_accessed_raw is None:
                    last_accessed = None
                else:
                    import time as _time
                    last_accessed = _finite_float(
                        last_accessed_raw, _time.time(),
                    )
                kwargs = dict(
                    subject=r["subject"], predicate=r["predicate"],
                    object=r["object"], fact_id=r["fact_id"],
                    vector=_safe_vector(r["vector"]),
                    gravity_score=_finite_float(
                        r["gravity_score"], 0.5, lo=0.0, hi=1.0,
                    ),
                    layer=_layer(r["layer"], 1),
                    ttl=ttl,
                    source_session=r["source_session"],
                    deprecated_by=r["deprecated_by"],
                    access_count=_nonnegative_int(r["access_count"], 0),
                    resonance_sum=_finite_float(r["resonance_sum"], 0.0),
                    resonance_count=_nonnegative_int(
                        r["resonance_count"], 0,
                    ),
                    recent_utility=_finite_float(
                        r["recent_utility"]
                        if "recent_utility" in r.keys() else 0.5,
                        0.5, lo=0.0, hi=1.0,
                    ),
                    forecast_stability=_finite_float(
                        r["forecast_stability"]
                        if "forecast_stability" in r.keys() else 0.5,
                        0.5, lo=0.0, hi=1.0,
                    ),
                )
                # Only pass created_at / last_accessed when non-None so
                # the dataclass default_factory (time.time()) fires for
                # legacy rows missing the cell.
                if created_at is not None:
                    kwargs["created_at"] = created_at
                if last_accessed is not None:
                    kwargs["last_accessed"] = last_accessed
                out.append(FactPassport(**kwargs))
            except Exception as exc:
                _logger.warning(
                    "SQLite load_facts: skipping corrupted row fact_id=%r: %s",
                    r["fact_id"] if "fact_id" in r.keys() else "?", exc,
                )
        return out

    # ── Edges ────────────────────────────────────────────────────────────────

    def save_edge(self, from_id: str, to_id: str) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO edges (from_id, to_id) VALUES (?,?)",
            (from_id, to_id),
        )
        self._maybe_commit()

    def load_edges(self) -> list[tuple[str, str]]:
        rows = self._conn.execute("SELECT from_id, to_id FROM edges").fetchall()
        return [(r["from_id"], r["to_id"]) for r in rows]

    def delete_edges_for_fact(self, fact_id: str) -> None:
        """Drop every edge whose endpoint is ``fact_id``."""
        self._conn.execute(
            "DELETE FROM edges WHERE from_id = ? OR to_id = ?",
            (fact_id, fact_id),
        )
        self._maybe_commit()

    def prune_orphan_edges(self) -> int:
        """Drop edge rows whose endpoints are no longer in the facts
        table. Returns the number of rows deleted.

        The schema has no FOREIGN KEY constraint on edges → facts (the
        graph is FactPassport-only, but MetaFact polymorphism makes
        retroactive FKs awkward), so a missed cleanup path leaves
        orphan rows on disk. The in-memory load already filters
        orphans, so this is on-disk hygiene only — call from a
        maintenance / doctor path, not the hot path. Idempotent.
        """
        cur = self._conn.execute(
            "DELETE FROM edges WHERE "
            "from_id NOT IN (SELECT fact_id FROM facts) "
            "OR to_id NOT IN (SELECT fact_id FROM facts)"
        )
        removed = cur.rowcount or 0
        self._maybe_commit()
        return removed

    # ── Echo sessions ─────────────────────────────────────────────────────────

    def save_echo_session(
        self,
        session_id: str,
        centroids: list[list[float]],
        r_score: float,
        recorded_at: float,
        fact_weights: dict[str, float] | None = None,
        echo_penalty: float = 0.0,
    ) -> None:
        # The on-disk column is still named ``fact_ids`` for backward
        # compatibility, but its JSON payload is now a {fact_id: weight}
        # dict. The loader accepts both shapes.
        payload = dict(fact_weights or {})
        # allow_nan=False on both JSON cells + scalar sanitisation on
        # the numeric fields. Write-side defence: a poisoned centroid
        # (NaN, Infinity) raises ValueError on dumps, the surrounding
        # txn rolls back, _reload restores the pre-write snapshot.
        # Cleaner than persisting radioactive JSON and having the
        # loader drop the row on next boot.
        self._conn.execute(
            "INSERT OR REPLACE INTO echo_sessions "
            "(session_id, centroids, r_score, recorded_at, fact_ids, echo_penalty) "
            "VALUES (?,?,?,?,?,?)",
            (
                session_id,
                json.dumps(centroids, allow_nan=False),
                _finite_float(r_score, 0.0, lo=-1.0, hi=1.0),
                _finite_float(recorded_at, time.time()),
                json.dumps(payload, allow_nan=False),
                _finite_float(echo_penalty, 0.0, lo=0.0, hi=1.0),
            ),
        )
        self._maybe_commit()

    def delete_echo_session(self, session_id: str) -> None:
        self._conn.execute(
            "DELETE FROM echo_sessions WHERE session_id = ?", (session_id,)
        )
        self._maybe_commit()

    def load_echo_sessions(self, *, cleanup: bool = True) -> list[dict]:
        rows = self._conn.execute("SELECT * FROM echo_sessions").fetchall()
        out = []
        for r in rows:
            try:
                raw_fact_ids = r["fact_ids"] if "fact_ids" in r.keys() else None
                parsed = _safe_loads(raw_fact_ids, {})
                if isinstance(parsed, list):
                    # Legacy rows stored just the ids — uniform weight 1.0.
                    fact_weights = {fid: 1.0 for fid in parsed}
                elif isinstance(parsed, dict):
                    # Coerce + finite-check each weight. NaN / Infinity
                    # in a stored weight would poison resonance
                    # propagation downstream (apply_session_resonance
                    # does r * weight; recent_utility EWMA same). Drop
                    # the bad pair rather than entire session.
                    import math as _math
                    fact_weights = {}
                    for fid, w in parsed.items():
                        try:
                            wf = float(w)
                        except (TypeError, ValueError):
                            continue
                        if not _math.isfinite(wf):
                            continue
                        fact_weights[fid] = max(0.0, min(1.0, wf))
                else:
                    fact_weights = {}
                centroids = _safe_centroids(r["centroids"])
                if not centroids:
                    # Empty/None centroids OR a non-conformant shape
                    # (ragged dim, non-list inner, NaN cell, non-numeric
                    # item) means corruption — the session would crash
                    # echo matching downstream. _safe_centroids
                    # collapses every bad shape to [], so a single
                    # check covers all of them.
                    raise ValueError("centroids missing or malformed")
                out.append({
                    "session_id": r["session_id"],
                    "centroids": centroids,
                    "r_score": r["r_score"],
                    "recorded_at": r["recorded_at"],
                    "fact_weights": fact_weights,
                    "echo_penalty": (
                        r["echo_penalty"]
                        if "echo_penalty" in r.keys() else 0.0
                    ),
                })
            except Exception as exc:
                _logger.warning(
                    "SQLite load_echo_sessions: dropping corrupted row "
                    "session_id=%r: %s",
                    r["session_id"] if "session_id" in r.keys() else "?", exc,
                )
                try:
                    if "session_id" in r.keys():
                        if cleanup:
                            self.delete_echo_session(r["session_id"])
                except Exception:   # pragma: no cover — best effort cleanup
                    pass
        return out

    # ── Open sessions ────────────────────────────────────────────────────────

    def save_open_session(
        self,
        session_id: str,
        messages: list[str],
        vectors: list[list[float]],
        facts: dict[str, float],
        started_at: float,
    ) -> None:
        # allow_nan=False on every JSON cell. A NaN sneaked into
        # vectors (mid-session embed model swap, mock-vs-real flip)
        # would otherwise persist to disk as the literal token
        # ``NaN`` — Python's json.loads accepts it on read, then the
        # loader's _safe_loads drops the row. Catch at write so the
        # session never leaves a poisoned trail on disk; the
        # surrounding session_message try/except + _reload restores
        # the in-memory snapshot.
        self._conn.execute(
            "INSERT OR REPLACE INTO open_sessions VALUES (?,?,?,?,?)",
            (
                session_id,
                json.dumps(messages, allow_nan=False),
                json.dumps(vectors, allow_nan=False),
                json.dumps(facts, allow_nan=False),
                _finite_float(started_at, time.time()),
            ),
        )
        self._maybe_commit()

    def delete_open_session(self, session_id: str) -> None:
        self._conn.execute(
            "DELETE FROM open_sessions WHERE session_id = ?", (session_id,)
        )
        self._maybe_commit()

    def load_open_sessions(self, *, cleanup: bool = True) -> list[dict]:
        # ORDER BY started_at so _reload restores `_current_session_id`
        # deterministically — without the ORDER BY, the last row from
        # `SELECT *` was driver-dependent and "current" after a cross-
        # process reload was effectively arbitrary. Most-recently-
        # started wins; matches the "this is the session you just
        # opened" intuition of the back-compat shims.
        rows = self._conn.execute(
            "SELECT * FROM open_sessions ORDER BY started_at ASC"
        ).fetchall()
        out: list[dict] = []
        for r in rows:
            try:
                messages = _safe_loads(r["messages"], [])
                vectors = _safe_loads(r["vectors"], [])
                facts = _safe_loads(r["facts"], {})
                # Shape validation. _safe_loads gives us "valid JSON",
                # but the consumer (_load_from_storage) calls .items()
                # on facts and indexes vectors element-wise — a row
                # where facts came back as a list, or vectors as a
                # dict, would crash the consumer instead of the loader.
                # Drop the row at the loader so a malformed cell never
                # reaches the consumer.
                if not isinstance(messages, list):
                    raise ValueError("messages must be a list")
                if not isinstance(vectors, list):
                    raise ValueError("vectors must be a list")
                if not isinstance(facts, dict):
                    raise ValueError("facts must be a dict")
                # Coerce facts values to float here rather
                # than at the consumer. The consumer used to do
                # {k: float(v) for ...} and would crash with raw
                # ValueError if any value were non-numeric. Doing it
                # at the loader gives the row a chance to be dropped
                # cleanly instead of taking startup down.
                import math as _math
                coerced_facts: dict[str, float] = {}
                for k, v in facts.items():
                    wf = float(v)
                    if not _math.isfinite(wf):
                        # NaN / Infinity in a stored attribution weight
                        # would poison resonance_sum / recent_utility
                        # downstream (apply_session_resonance does
                        # r * weight). Reject the whole row — caller's
                        # tolerant-load contract drops it.
                        raise ValueError(
                            "facts weight contains NaN or Infinity"
                        )
                    coerced_facts[str(k)] = max(0.0, min(1.0, wf))
                # Vectors must be list[list[float]] with consistent dim.
                # The consumer (compute_resonance → score_repetition →
                # centroid) takes dim from vectors[0] and indexes every
                # vector by it — a corrupted ragged shape would crash
                # the session_close path far from the loader. Drop the
                # row here instead.
                import math
                coerced_vectors: list[list[float]] = []
                expected_dim: int | None = None
                for v in vectors:
                    if not isinstance(v, list):
                        raise ValueError(
                            "vectors must be list[list[float]]"
                        )
                    coerced = [float(x) for x in v]
                    # NaN / Infinity poisons every downstream resonance
                    # / centroid / cosine calculation. Reject at loader.
                    if not all(math.isfinite(x) for x in coerced):
                        raise ValueError(
                            "vectors contain NaN or Infinity"
                        )
                    if expected_dim is None:
                        expected_dim = len(coerced)
                    elif len(coerced) != expected_dim:
                        raise ValueError(
                            "vectors must share a single dimension"
                        )
                    coerced_vectors.append(coerced)
                out.append({
                    "session_id": r["session_id"],
                    "messages": messages,
                    "vectors": coerced_vectors,
                    "facts": coerced_facts,
                    "started_at": r["started_at"],
                })
            except Exception as exc:
                # Drop the corrupted row so the crashed session does not
                # block startup. delete the on-disk record too, otherwise
                # the next open would try it again.
                _logger.warning(
                    "SQLite load_open_sessions: dropping corrupted row "
                    "session_id=%r: %s",
                    r["session_id"] if "session_id" in r.keys() else "?", exc,
                )
                try:
                    if cleanup:
                        self.delete_open_session(r["session_id"])
                except Exception:   # pragma: no cover — best effort cleanup
                    pass
        return out

    # ── MetaFacts ────────────────────────────────────────────────────────────

    _META_INSERT = (
        "INSERT OR REPLACE INTO meta_facts "
        "(meta_id, vector, weight, source_texts, source_fact_ids, summary, "
        " gravity_score, created_at, layer, access_count, last_accessed, "
        " resonance_sum, resonance_count, recent_utility, "
        " forecast_stability) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
    )

    @staticmethod
    def _meta_row(meta: MetaFact) -> tuple:
        # to_dict() now uses allow_nan=False on every JSON cell so a
        # poisoned vector / lineage raises at the boundary. Scalar
        # fields are clamped through the same helpers as load — a
        # MetaFact whose runtime gravity_score got NaN'd via direct
        # attribute mutation is sanitised before persistence rather
        # than being heroically cleaned by the next load.
        d = meta.to_dict()
        return (
            d["meta_id"], d["vector"],
            _nonnegative_int(d["weight"], 1),
            d["source_texts"], d["source_fact_ids"], d["summary"],
            _finite_float(d["gravity_score"], 0.30, lo=0.0, hi=1.0),
            _finite_float(d["created_at"], time.time()),
            _layer(d["layer"], -1),
            _nonnegative_int(d["access_count"], 0),
            _finite_float(d["last_accessed"], time.time()),
            _finite_float(d["resonance_sum"], 0.0),
            _nonnegative_int(d["resonance_count"], 0),
            _finite_float(
                d["recent_utility"], 0.5, lo=0.0, hi=1.0,
            ),
            _finite_float(
                d["forecast_stability"], 0.5, lo=0.0, hi=1.0,
            ),
        )

    def save_meta_fact(self, meta: MetaFact) -> None:
        self._conn.execute(self._META_INSERT, self._meta_row(meta))
        self._maybe_commit()

    def save_meta_facts(self, metas: list[MetaFact]) -> None:
        if not metas:
            return
        rows = [self._meta_row(m) for m in metas]
        self._conn.executemany(self._META_INSERT, rows)
        self._maybe_commit()

    def delete_meta_fact(self, meta_id: str) -> None:
        self._conn.execute("DELETE FROM meta_facts WHERE meta_id = ?", (meta_id,))
        self._maybe_commit()

    def load_meta_facts(self, *, cleanup: bool = True) -> list[MetaFact]:
        rows = self._conn.execute("SELECT * FROM meta_facts").fetchall()
        # Tolerant per row — same robustness contract as load_facts /
        # load_open_sessions / load_echo_sessions: one bad row must not
        # take MemoryStore startup down. MetaFact.from_dict tolerates
        # most JSON shape drift, but cell-level corruption (truncated
        # JSON, manual edits, format changes between versions) can
        # still raise. We skip + log + drop the row so it stops
        # blocking future boots.
        out: list[MetaFact] = []
        for r in rows:
            try:
                # Pre-validate the JSON-encoded cells. MetaFact.from_dict is
                # forgiving (returns [] on bad JSON), but a row whose JSON
                # cells are all garbage is corruption, not drift — drop it
                # rather than silently load a near-empty MetaFact.
                # Two corruption modes drop the row:
                #
                #   1. Cell is invalid JSON entirely ("{garbage").
                #   2. Cell is valid JSON but the wrong SHAPE (dict /
                #      string / number) where a list is required —
                #      MetaFact._load_list would silently coerce these
                #      to [], producing a body with no lineage. The
                #      original cell wasn't [], so this is corruption,
                #      not "saved-as-empty".
                #
                # An originally-empty list ("[]") is preserved as-is:
                # the round-trip contract (save X → load X) must hold
                # for callers that legitimately save empty MetaFacts
                # in tests / migrations.
                bad = False
                for cell in ("vector", "source_texts", "source_fact_ids"):
                    raw = r[cell] if cell in r.keys() else None
                    if not isinstance(raw, str) or not raw:
                        continue
                    try:
                        parsed = json.loads(raw)
                    except (TypeError, ValueError):
                        bad = True
                        break
                    if not isinstance(parsed, list):
                        bad = True
                        break
                if bad:
                    raise ValueError("corrupted JSON cell")
                meta = MetaFact.from_dict(dict(r))
                # Round-11: round-trip for legitimately empty MetaFacts
                # is preserved (tests/migrations), but a persisted body
                # with no lineage at all is operationally suspicious —
                # nothing in the compactor path produces one. Log so an
                # operator can spot manual-edit / migration drift, but
                # don't drop (would break the round-trip contract).
                if not meta.source_fact_ids and not meta.source_texts:
                    _logger.warning(
                        "SQLite load_meta_facts: meta_id=%r has empty "
                        "lineage (no source_fact_ids, no source_texts)",
                        meta.meta_id,
                    )
                out.append(meta)
            except Exception as exc:
                meta_id = r["meta_id"] if "meta_id" in r.keys() else "?"
                _logger.warning(
                    "SQLite load_meta_facts: dropping corrupted row "
                    "meta_id=%r: %s",
                    meta_id, exc,
                )
                try:
                    if "meta_id" in r.keys():
                        if cleanup:
                            self.delete_meta_fact(r["meta_id"])
                except Exception:   # pragma: no cover — best effort cleanup
                    pass
        return out

    # ── Adaptive gravity weights ─────────────────────────────────────────────

    def save_adaptive_weights(self, weights: AdaptiveWeights) -> None:
        """Persist the learned pre-resonance gravity weights (singleton row)."""
        import time as _time

        self._conn.execute(
            "INSERT OR REPLACE INTO adaptive_weights "
            "(id, w_freshness, w_access, w_graph, w_utility, w_stability, "
            " train_count, updated_at) "
            "VALUES (1, ?, ?, ?, ?, ?, ?, ?)",
            (
                weights.w_freshness,
                weights.w_access,
                weights.w_graph,
                weights.w_utility,
                weights.w_stability,
                weights.train_count,
                _time.time(),
            ),
        )
        self._maybe_commit()

    def load_adaptive_weights(self) -> AdaptiveWeights | None:
        """Return the persisted weights, or None if the store has none yet.

        Tolerant: every other loader (facts / metas / open_sessions /
        echo_sessions) drops corrupt rows and lets startup continue
        with a degraded view. adaptive_weights used to be the lone
        loader that crashed init on a malformed row — non-numeric
        cell, NaN sneaked past sanitize(), corrupt train_count.
        Now we log and fall back to ``None`` (= compute_gravity
        gets the hand-tuned prior via AdaptiveWeights.from_prior),
        symmetric with the rest of the storage philosophy.
        """
        import math as _math
        try:
            row = self._conn.execute(
                "SELECT w_freshness, w_access, w_graph, w_utility, "
                "w_stability, train_count FROM adaptive_weights "
                "WHERE id = 1"
            ).fetchone()
        except sqlite3.Error as exc:
            _logger.warning(
                "load_adaptive_weights: SQLite read failed (%s) — "
                "falling back to prior weights", exc,
            )
            return None
        if row is None:
            return None
        try:
            fields = {
                "w_freshness": float(row["w_freshness"]),
                "w_access": float(row["w_access"]),
                "w_graph": float(row["w_graph"]),
                "w_utility": float(
                    row["w_utility"]
                    if row["w_utility"] is not None else 0.10
                ),
                "w_stability": float(
                    row["w_stability"]
                    if row["w_stability"] is not None else 0.05
                ),
                "train_count": int(row["train_count"]),
            }
        except (TypeError, ValueError) as exc:
            _logger.warning(
                "load_adaptive_weights: corrupted cell (%s) — "
                "falling back to prior weights", exc,
            )
            return None
        # Reject NaN / Infinity BEFORE sanitize. sanitize clamps to
        # [0, BUDGET] but NaN survives min/max in Python and would
        # poison every gravity computation thereafter.
        if not all(
            _math.isfinite(v) for k, v in fields.items()
            if k != "train_count"
        ):
            _logger.warning(
                "load_adaptive_weights: non-finite weight on disk — "
                "falling back to prior weights",
            )
            return None
        weights = AdaptiveWeights(**fields)  # type: ignore[arg-type]
        # Sanitise: a row that was corrupted, manually edited, or written
        # by an old version with a different invariant must not be served
        # to compute_gravity as-is. Clamp non-negative and renormalise to
        # BUDGET so gravity stays in [0, 1].
        weights.sanitize()
        return weights

    def close(self) -> None:
        self._conn.close()
