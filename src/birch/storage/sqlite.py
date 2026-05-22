"""SQLite implementation of StorageBackend."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from ..fact import FactPassport
from ..meta_fact import MetaFact

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
    resonance_count INTEGER DEFAULT 0
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
    resonance_count INTEGER DEFAULT 0
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

    # ── Facts ────────────────────────────────────────────────────────────────

    @staticmethod
    def _fact_row(fact: FactPassport) -> tuple:
        return (
            fact.fact_id, fact.subject, fact.predicate, fact.object,
            json.dumps(fact.vector),
            fact.gravity_score, fact.layer, fact.created_at, fact.ttl,
            fact.source_session, fact.deprecated_by,
            fact.access_count, fact.last_accessed,
            fact.resonance_sum, fact.resonance_count,
        )

    def save_fact(self, fact: FactPassport) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO facts VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            self._fact_row(fact),
        )
        self._maybe_commit()

    def save_facts(self, facts: list[FactPassport]) -> None:
        """One transaction, one commit — orders of magnitude faster on bulk dumps."""
        if not facts:
            return
        rows = [self._fact_row(f) for f in facts]
        self._conn.executemany(
            "INSERT OR REPLACE INTO facts VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            rows,
        )
        self._maybe_commit()

    def delete_fact(self, fact_id: str) -> None:
        self._conn.execute("DELETE FROM facts WHERE fact_id = ?", (fact_id,))
        self._maybe_commit()

    def load_facts(self) -> list[FactPassport]:
        rows = self._conn.execute("SELECT * FROM facts").fetchall()
        return [
            FactPassport(
                subject=r["subject"], predicate=r["predicate"], object=r["object"],
                fact_id=r["fact_id"],
                vector=json.loads(r["vector"]) if r["vector"] else [],
                gravity_score=r["gravity_score"], layer=r["layer"],
                created_at=r["created_at"], ttl=r["ttl"],
                source_session=r["source_session"], deprecated_by=r["deprecated_by"],
                access_count=r["access_count"], last_accessed=r["last_accessed"],
                resonance_sum=r["resonance_sum"], resonance_count=r["resonance_count"],
            )
            for r in rows
        ]

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
            raw_fact_ids = r["fact_ids"] if "fact_ids" in r.keys() else None
            try:
                parsed = json.loads(raw_fact_ids) if raw_fact_ids else {}
            except (TypeError, ValueError):
                parsed = {}
            if isinstance(parsed, list):
                # Legacy rows stored just the ids — treat them as uniform weight 1.0.
                fact_weights = {fid: 1.0 for fid in parsed}
            elif isinstance(parsed, dict):
                fact_weights = {fid: float(w) for fid, w in parsed.items()}
            else:
                fact_weights = {}
            out.append({
                "session_id": r["session_id"],
                "centroids": json.loads(r["centroids"]),
                "r_score": r["r_score"],
                "recorded_at": r["recorded_at"],
                "fact_weights": fact_weights,
                "echo_penalty": r["echo_penalty"] if "echo_penalty" in r.keys() else 0.0,
            })
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
        return [
            {
                "session_id": r["session_id"],
                "messages": json.loads(r["messages"]),
                "vectors": json.loads(r["vectors"]),
                "facts": json.loads(r["facts"]),
                "started_at": r["started_at"],
            }
            for r in rows
        ]

    # ── MetaFacts ────────────────────────────────────────────────────────────

    _META_INSERT = (
        "INSERT OR REPLACE INTO meta_facts "
        "(meta_id, vector, weight, source_texts, source_fact_ids, summary, "
        " gravity_score, created_at, layer, access_count, last_accessed, "
        " resonance_sum, resonance_count) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)"
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
        # MetaFact.from_dict is tolerant of the JSON-string column shape.
        return [MetaFact.from_dict(dict(r)) for r in rows]

    def close(self) -> None:
        self._conn.close()
