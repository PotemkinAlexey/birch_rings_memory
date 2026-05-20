"""BirchKM MCP server — exposes memory tools to Claude agents."""
from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP

from .memory_store import MemoryStore

_DB_PATH = os.environ.get("BIRCH_DB", str(Path.home() / ".birch" / "memory.db"))
Path(_DB_PATH).parent.mkdir(parents=True, exist_ok=True)

_store = MemoryStore(db_path=_DB_PATH)

mcp = FastMCP("BirchKM")


@mcp.tool()
def query_memory(
    text: str,
    top_k: int = 5,
    session_id: Optional[str] = None,
) -> list[dict]:
    """
    Search memory for facts relevant to the given text.

    Returns up to top_k facts ranked by semantic similarity.
    Each result has: subject, predicate, object, similarity, layer, gravity_score, source.

    Pass session_id to attribute retrieved facts to an open session so their
    gravity is updated when the session closes. Omit for read-only lookups.

    source values:
      "surface"  — hot fact, high gravity, trust directly
      "kinetic"  — working memory, moderate use
      "core"     — cold archive, verify before relying on it
      "hawking"  — returned from black hole, treat with suspicion
    """
    results = _store.query(text, top_k=top_k, hawking=True, session_id=session_id)
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
def record_fact(
    subject: str,
    predicate: str,
    object: str,
    session_id: Optional[str] = None,
) -> dict:
    """
    Store a new fact in memory as a subject-predicate-object triple.

    Good triples:
      subject="mailer service", predicate="runs on",      object="Go"
      subject="user",           predicate="prefers",      object="dark mode"
      subject="deploy pipeline",predicate="fails when",   object="migrations run first"

    Identical triples (case-insensitive, whitespace-normalised) are deduplicated —
    the existing fact is touched and returned. Check already_existed in the response
    to know if you created a new fact or confirmed an existing one.

    Pass session_id to attribute this fact to an open session so its gravity
    is updated when the session closes.
    """
    already_existed = _store.fact_exists(subject, predicate, object)
    fact = _store.add_fact(subject, predicate, object, session_id=session_id)
    return {
        "fact_id": fact.fact_id,
        "already_existed": already_existed,
        "layer": fact.layer,
        "gravity_score": round(fact.gravity_score, 3),
    }


@mcp.tool()
def record_session(messages: list[str], agent_id: str = "default") -> dict:
    """
    Score a completed session and update memory gravity.

    Pass all user messages from the session in order. Do not include
    your own responses — the resonance engine scores user-side signals only.

    BirchKM will:
      - Score the session R in [-1, +1] (resonant / neutral / toxic)
      - Propagate R to all facts touched during the session
      - Detect echo if the user returned to an unresolved problem
      - Absorb dead facts into the black hole

    Returns: label, r_score, migrations, absorbed count, current stats.
    Call once per session, at the end.
    """
    session_id = f"{agent_id}-{int(time.time())}-{uuid.uuid4().hex[:4]}"
    _store.session_start(session_id)
    for msg in messages:
        _store.session_message(msg, session_id=session_id)
    summary = _store.session_close(session_id=session_id)
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
    """
    Return current memory state — layer distribution and black hole status.

    Interpret:
      black_hole_mass rising  — facts are failing; review what is being stored
      surface count dropping  — active knowledge declining; system needs fresh input
      hawking_emissions > 0   — dead facts resurface; store may have stale info
    """
    return _store.stats


if __name__ == "__main__":
    mcp.run()
