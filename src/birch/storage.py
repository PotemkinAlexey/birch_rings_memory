"""SQLite write-through storage backend for BirchKM facts."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Optional

from .fact import FactPassport


_SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    fact_id       TEXT PRIMARY KEY,
    subject       TEXT NOT NULL,
    predicate     TEXT NOT NULL,
    object        TEXT NOT NULL,
    vector        TEXT,
    gravity_score REAL DEFAULT 0.5,
    layer         INTEGER DEFAULT 1,
    created_at    REAL,
    ttl           REAL,
    source_session TEXT,
    deprecated_by TEXT,
    access_count  INTEGER DEFAULT 0,
    last_accessed REAL,
    resonance_sum REAL DEFAULT 0.0,
    resonance_count INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS edges (
    from_id TEXT NOT NULL,
    to_id   TEXT NOT NULL,
    PRIMARY KEY (from_id, to_id)
);

CREATE TABLE IF NOT EXISTS echo_sessions (
    session_id TEXT PRIMARY KEY,
    centroids  TEXT,
    r_score    REAL,
    recorded_at REAL
);
"""


class Storage:
    def __init__(self, db_path: str | Path) -> None:
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ── Facts ────────────────────────────────────────────────────────────────

    def save_fact(self, fact: FactPassport) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO facts VALUES
               (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                fact.fact_id,
                fact.subject,
                fact.predicate,
                fact.object,
                json.dumps(fact.vector),
                fact.gravity_score,
                fact.layer,
                fact.created_at,
                fact.ttl,
                fact.source_session,
                fact.deprecated_by,
                fact.access_count,
                fact.last_accessed,
                fact.resonance_sum,
                fact.resonance_count,
            ),
        )
        self._conn.commit()

    def delete_fact(self, fact_id: str) -> None:
        self._conn.execute("DELETE FROM facts WHERE fact_id = ?", (fact_id,))
        self._conn.commit()

    def load_facts(self) -> list[FactPassport]:
        rows = self._conn.execute("SELECT * FROM facts").fetchall()
        facts = []
        for r in rows:
            f = FactPassport(
                subject=r["subject"],
                predicate=r["predicate"],
                object=r["object"],
                fact_id=r["fact_id"],
                vector=json.loads(r["vector"]) if r["vector"] else [],
                gravity_score=r["gravity_score"],
                layer=r["layer"],
                created_at=r["created_at"],
                ttl=r["ttl"],
                source_session=r["source_session"],
                deprecated_by=r["deprecated_by"],
                access_count=r["access_count"],
                last_accessed=r["last_accessed"],
                resonance_sum=r["resonance_sum"],
                resonance_count=r["resonance_count"],
            )
            facts.append(f)
        return facts

    # ── Edges ────────────────────────────────────────────────────────────────

    def save_edge(self, from_id: str, to_id: str) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO edges (from_id, to_id) VALUES (?,?)",
            (from_id, to_id),
        )
        self._conn.commit()

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
    ) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO echo_sessions VALUES (?,?,?,?)",
            (session_id, json.dumps(centroids), r_score, recorded_at),
        )
        self._conn.commit()

    def load_echo_sessions(self) -> list[dict]:
        rows = self._conn.execute("SELECT * FROM echo_sessions").fetchall()
        return [
            {
                "session_id": r["session_id"],
                "centroids": json.loads(r["centroids"]),
                "r_score": r["r_score"],
                "recorded_at": r["recorded_at"],
            }
            for r in rows
        ]

    def close(self) -> None:
        self._conn.close()
