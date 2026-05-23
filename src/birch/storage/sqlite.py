"""SQLite implementation of StorageBackend."""
from __future__ import annotations

import json
import logging
import sqlite3
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


def _safe_vector(value: Any) -> list[float]:
    """Parse a JSON-encoded vector tolerantly with shape validation.

    Valid JSON that is not a list of numbers (e.g. ``{"x": 1}``, ``"abc"``,
    ``[1, "oops", 3]``) must not break downstream code that assumes
    ``list[float]``. Returns an empty list in any non-conformant case so
    the fact loads but is unsearchable (caller's fallback path).
    """
    raw = _safe_loads(value, [])
    if not isinstance(raw, list):
        return []
    try:
        return [float(x) for x in raw]
    except (TypeError, ValueError):
        return []

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
        return (
            fact.fact_id, fact.subject, fact.predicate, fact.object,
            json.dumps(fact.vector),
            fact.gravity_score, fact.layer, fact.created_at, fact.ttl,
            fact.source_session, fact.deprecated_by,
            fact.access_count, fact.last_accessed,
            fact.resonance_sum, fact.resonance_count,
            fact.recent_utility,
            fact.forecast_stability,
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
        for r in rows:
            try:
                out.append(FactPassport(
                    subject=r["subject"], predicate=r["predicate"], object=r["object"],
                    fact_id=r["fact_id"],
                    vector=_safe_vector(r["vector"]),
                    gravity_score=r["gravity_score"], layer=r["layer"],
                    created_at=r["created_at"], ttl=r["ttl"],
                    source_session=r["source_session"], deprecated_by=r["deprecated_by"],
                    access_count=r["access_count"], last_accessed=r["last_accessed"],
                    resonance_sum=r["resonance_sum"],
                    resonance_count=r["resonance_count"],
                    recent_utility=(
                        r["recent_utility"]
                        if "recent_utility" in r.keys() and r["recent_utility"] is not None
                        else 0.5
                    ),
                    forecast_stability=(
                        r["forecast_stability"]
                        if "forecast_stability" in r.keys()
                        and r["forecast_stability"] is not None
                        else 0.5
                    ),
                ))
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
        self._conn.execute(
            "INSERT OR REPLACE INTO echo_sessions "
            "(session_id, centroids, r_score, recorded_at, fact_ids, echo_penalty) "
            "VALUES (?,?,?,?,?,?)",
            (
                session_id,
                json.dumps(centroids),
                r_score,
                recorded_at,
                json.dumps(payload),
                echo_penalty,
            ),
        )
        self._maybe_commit()

    def delete_echo_session(self, session_id: str) -> None:
        self._conn.execute(
            "DELETE FROM echo_sessions WHERE session_id = ?", (session_id,)
        )
        self._maybe_commit()

    def load_echo_sessions(self) -> list[dict]:
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
                    fact_weights = {fid: float(w) for fid, w in parsed.items()}
                else:
                    fact_weights = {}
                centroids = _safe_loads(r["centroids"], None)
                if not isinstance(centroids, list) or not centroids:
                    # Empty/None centroids means corruption — the session
                    # would be useless for echo matching anyway. Drop.
                    raise ValueError("centroids missing or not a list")
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
        self._conn.execute(
            "INSERT OR REPLACE INTO open_sessions VALUES (?,?,?,?,?)",
            (
                session_id,
                json.dumps(messages),
                json.dumps(vectors),
                json.dumps(facts),
                started_at,
            ),
        )
        self._maybe_commit()

    def delete_open_session(self, session_id: str) -> None:
        self._conn.execute(
            "DELETE FROM open_sessions WHERE session_id = ?", (session_id,)
        )
        self._maybe_commit()

    def load_open_sessions(self) -> list[dict]:
        rows = self._conn.execute("SELECT * FROM open_sessions").fetchall()
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
                coerced_facts: dict[str, float] = {}
                for k, v in facts.items():
                    coerced_facts[str(k)] = float(v)
                out.append({
                    "session_id": r["session_id"],
                    "messages": messages,
                    "vectors": vectors,
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
        d = meta.to_dict()
        return (
            d["meta_id"], d["vector"], d["weight"],
            d["source_texts"], d["source_fact_ids"], d["summary"],
            d["gravity_score"], d["created_at"], d["layer"],
            d["access_count"], d["last_accessed"],
            d["resonance_sum"], d["resonance_count"],
            d["recent_utility"],
            d["forecast_stability"],
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

    def load_meta_facts(self) -> list[MetaFact]:
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
        """Return the persisted weights, or None if the store has none yet."""
        row = self._conn.execute(
            "SELECT w_freshness, w_access, w_graph, w_utility, w_stability, "
            "train_count FROM adaptive_weights WHERE id = 1"
        ).fetchone()
        if row is None:
            return None
        weights = AdaptiveWeights(
            w_freshness=row["w_freshness"],
            w_access=row["w_access"],
            w_graph=row["w_graph"],
            w_utility=(
                row["w_utility"] if row["w_utility"] is not None else 0.10
            ),
            w_stability=(
                row["w_stability"] if row["w_stability"] is not None else 0.05
            ),
            train_count=int(row["train_count"]),
        )
        # Sanitise: a row that was corrupted, manually edited, or written
        # by an old version with a different invariant must not be served
        # to compute_gravity as-is. Clamp non-negative and renormalise to
        # BUDGET so gravity stays in [0, 1].
        weights.sanitize()
        return weights

    def close(self) -> None:
        self._conn.close()
