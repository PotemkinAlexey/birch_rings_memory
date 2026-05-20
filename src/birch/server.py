"""BirchKM MCP server — exposes memory tools to Claude agents."""
from __future__ import annotations

import os
import time
import uuid
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from .memory_store import MemoryStore

_DB_PATH = os.environ.get("BIRCH_DB", str(Path.home() / ".birch" / "memory.db"))
Path(_DB_PATH).parent.mkdir(parents=True, exist_ok=True)

_store = MemoryStore(db_path=_DB_PATH)
_active_sessions: dict[str, str] = {}   # agent_id → session_id

mcp = FastMCP("BirchKM")


@mcp.tool()
def query_memory(text: str, top_k: int = 5) -> list[dict]:
    """
    Search memory for facts relevant to the given text.

    Returns up to top_k facts ranked by semantic similarity.
    Each result has: subject, predicate, object, similarity, layer, gravity_score.
    Call this before answering to get relevant context from past sessions.
    """
    results = _store.query(text, top_k=top_k, hawking=True)
    return [
        {
            "fact_id": r.fact.fact_id,
            "subject": r.fact.subject,
            "predicate": r.fact.predicate,
            "object": r.fact.object,
            "similarity": r.similarity,
            "layer": r.fact.layer,
            "gravity_score": round(r.fact.gravity_score, 3),
            "source": r.source,
        }
        for r in results
    ]


@mcp.tool()
def record_fact(subject: str, predicate: str, object: str) -> dict:
    """
    Store a new fact in memory.

    Use subject-predicate-object triples, e.g.:
      subject="mailer service", predicate="runs on", object="Go"
      subject="user", predicate="prefers", object="dark mode"
      subject="project", predicate="uses", object="PostgreSQL"

    Returns the fact_id of the stored fact.
    """
    fact = _store.add_fact(subject, predicate, object)
    return {
        "fact_id": fact.fact_id,
        "layer": fact.layer,
        "gravity_score": round(fact.gravity_score, 3),
    }


@mcp.tool()
def record_session(messages: list[str], agent_id: str = "default") -> dict:
    """
    Score a completed session and update memory gravity.

    Pass all user messages from the session in order.
    BirchKM will:
      - Score the session (resonant / toxic / neutral)
      - Propagate the score to facts used in this session
      - Detect echo if the user returned to an unresolved problem
      - Absorb dead facts into the black hole

    Returns: label, r_score, migrations, absorbed count.
    """
    session_id = f"{agent_id}-{int(time.time())}-{uuid.uuid4().hex[:4]}"
    _store.session_start(session_id)
    for msg in messages:
        _store.session_message(msg)
    summary = _store.session_close()
    return {
        "session_id": session_id,
        "label": summary.get("label"),
        "r_score": round(summary.get("r", 0.0), 3),
        "migrations": len(summary.get("migrations", [])),
        "absorbed": len(summary.get("absorbed", [])),
        "stats": _store.stats,
    }


@mcp.tool()
def memory_stats() -> dict:
    """Return current memory layer distribution and black hole status."""
    return _store.stats


if __name__ == "__main__":
    mcp.run()
