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
    min_similarity: float = 0.0,
    layers: Optional[list[str]] = None,
) -> dict:
    """
    Search memory for facts relevant to the given text.

    Returns up to top_k hits ranked by semantic similarity. Every item includes
    kind, body_id, similarity, source, layer, gravity_score.

    kind == "fact" — also subject, predicate, object, fact_id (same as body_id).
    kind == "meta" — also meta_id, weight, source_texts, source_fact_ids, summary.

    Pass session_id to attribute retrieved bodies to an open session so their
    gravity is updated when the session closes. Omit for read-only lookups.

    min_similarity: drop results below this cosine threshold (0.0 = return all).
    layers: restrict to specific layers, e.g. ["surface", "kinetic"].
            Omit to search all layers. Valid values: "surface", "kinetic", "core".

    source values:
      "surface" / "kinetic" / "core" — live FactPassport layers
      "hawking"      — single fact recovered from the black hole
      "hawking_meta" — MetaFact bundle recovered from the black hole
    """
    layer_map = {"surface": 0, "kinetic": 1, "core": 2}
    if layers:
        layer_ints = [layer_map[name] for name in layers if name in layer_map]
        min_layer = min(layer_ints) if layer_ints else 0
        max_layer = max(layer_ints) if layer_ints else 2
    else:
        min_layer, max_layer = 0, 2
    results = _store.query(
        text,
        top_k=top_k,
        hawking=True,
        session_id=session_id,
        min_similarity=min_similarity,
        min_layer=min_layer,
        max_layer=max_layer,
    )
    hits = [r.to_mcp_dict() for r in results]
    if not session_id:
        hint = (
            "pass session_id to attribute these reads to a session "
            "so gravity updates on session_close"
        )
    else:
        hint = "call session_close when the conversation ends to propagate resonance to gravity"
    return {"results": hits, "_hint": hint}


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
    if not session_id:
        hint = (
            "open a session with session_open and pass session_id here "
            "so gravity updates when the session closes"
        )
    else:
        hint = "call session_close when the conversation ends to propagate resonance to gravity"
    return {
        "fact_id": fact.fact_id,
        "already_existed": already_existed,
        "layer": fact.layer,
        "gravity_score": round(fact.gravity_score, 3),
        "_hint": hint,
    }


@mcp.tool()
def record_facts(
    facts: list[dict],
    session_id: Optional[str] = None,
) -> dict:
    """
    Store multiple facts in one batch — one Ollama round-trip, one SQLite transaction.

    Each item in facts must have "subject", "predicate", "object".
    Optional per-item "session_id" overrides the top-level session_id.

    Example:
      facts=[
        {"subject": "API", "predicate": "written in", "object": "Go"},
        {"subject": "API", "predicate": "deployed on", "object": "Kubernetes"},
      ]

    Returns one result per input fact, in the same order.
    Duplicates are touched and returned with already_existed=true.
    """
    triples = [
        (f["subject"], f["predicate"], f["object"])
        for f in facts
    ]
    # Check existence before batch insert (no per-item round-trip needed).
    existed = [_store.fact_exists(s, p, o) for s, p, o in triples]
    # Resolve per-item session_id override, falling back to top-level.
    sid = session_id
    results_facts = _store.add_facts(triples, session_id=sid)
    if not session_id:
        hint = (
            "open a session with session_open and pass session_id here "
            "so gravity updates when the session closes"
        )
    else:
        hint = "call session_close when the conversation ends to propagate resonance to gravity"
    return {
        "facts": [
            {
                "fact_id": fp.fact_id,
                "already_existed": existed[i],
                "layer": fp.layer,
                "gravity_score": round(fp.gravity_score, 3),
            }
            for i, fp in enumerate(results_facts)
        ],
        "_hint": hint,
    }


@mcp.tool()
def supersede_fact(old_fact_id: str, new_fact_id: str) -> dict:
    """
    Mark a fact as superseded by a newer one — the canonical "we now know better".

    The old fact's deprecated_by pointer is set (lineage preserved), the SPO
    slot is freed for the new claim, and the body is sent to the singularity
    immediately. The row stays in storage, so the deprecated fact can still
    fuel singularity collapse (MetaFact compression) and be Hawking-emitted
    if a future query reopens the topic.

    Use this whenever you record a replacement fact for an older one — it is
    the right path for "stale / wrong / outdated" data. Do NOT use
    delete_fact for that; delete_fact is a destructive primitive for
    secrets and accidental writes.

    Returns {"superseded": true, "old_id", "new_id", "absorbed": [...]}.
    """
    return _store.supersede_fact(old_fact_id, new_fact_id)


@mcp.tool()
def retire_fact(fact_id: str) -> dict:
    """
    Send a no-longer-relevant fact to the singularity with no replacement.

    Use when the fact's topic is simply over (a project ended, a feature
    was removed) and there is no newer fact replacing it. The body is
    marked expired and absorbed into the black hole in the same call. The
    row stays in storage, so the fact can still feed singularity collapse
    and Hawking emission.

    Prefer supersede_fact when you DO have a replacement — that preserves
    the "we used to think X, now Y" lineage. Use delete_fact only for
    truly destructive removal.

    Returns {"retired": true, "fact_id", "absorbed": [...]}.
    """
    return _store.retire_fact(fact_id)


@mcp.tool()
def forecast_memory(horizon_ticks: int = 50) -> dict:
    """
    Run the galaxy forward and write a per-fact stability prediction back.

    For every live fact the galaxy simulates a body in orbit around the
    central black hole. After ``horizon_ticks`` integrator steps each body
    has either survived (its orbital radius gives a stability score in
    [0, 1]) or crossed the event horizon (stability = 0). The score lands
    on FactPassport.forecast_stability and is consumed by the adaptive
    gravity formula via ``w_stability`` — facts the galaxy predicts to
    fall earn a smaller gravity contribution from this feature, facts
    predicted safe earn more.

    The feature complements the local pre-resonance signals (freshness,
    access, graph, recent_utility) with a forecast no local feature can
    produce: it sees the body's *future*. Whether the forecast helps is
    not assumed — ``w_stability`` is learned by the same SGD pass that
    learns the other adaptive weights, so a useless forecast just sits
    at its prior and contributes nothing.

    Call at session start (or once per day) on a real store. Pure numpy,
    O(n²) per step in fact count.

    Returns a summary: how many facts were forecasted and updated, and a
    coarse distribution across {safe / kinetic / near_horizon / predicted_fall}.
    """
    return _store.run_forecast(horizon_ticks=horizon_ticks)


@mcp.tool()
def delete_fact(fact_id: str) -> dict:
    """
    Hard-delete a fact — destructive primitive, the data is GONE.

    Removes the fact from all live layers, the vector index, the SPO dedup
    index, AND storage. Does NOT send it to the black hole; there is no
    lineage, no MetaFact compression, no Hawking rescue. Use only for:

      - Secrets / sensitive data that must not exist (GDPR-style removal)
      - Facts you just recorded by accident in the same session

    For "stale / wrong / outdated" data, prefer supersede_fact (when there
    is a replacement) or retire_fact (when there is not) — both preserve
    the body in the singularity so the brain can still learn from it.

    Returns {"deleted": true} if found, {"deleted": false} if not found.
    """
    deleted = _store.delete_fact(fact_id)
    return {"deleted": deleted, "fact_id": fact_id}


@mcp.tool()
def list_facts(
    subject: Optional[str] = None,
    predicate: Optional[str] = None,
    limit: int = 50,
) -> list[dict]:
    """
    List live facts, optionally filtered by subject and/or predicate substring.

    Matching is case-insensitive. Results are sorted by gravity_score descending.
    Use this to audit memory without a semantic query — e.g. list_facts(subject="birch")
    returns everything stored about birch, regardless of how you'd phrase the question.
    """
    facts = _store.list_facts(subject=subject, predicate=predicate, limit=limit)
    return [
        {
            "fact_id": f.fact_id,
            "subject": f.subject,
            "predicate": f.predicate,
            "object": f.object,
            "layer": f.layer,
            "gravity_score": round(f.gravity_score, 3),
            "source": {0: "surface", 1: "kinetic", 2: "core"}.get(f.layer, "kinetic"),
        }
        for f in facts
    ]


@mcp.tool()
def session_open(session_id: Optional[str] = None, agent_id: str = "default") -> dict:
    """
    Open a named memory session for tracking facts and messages over time.

    Returns the session_id to pass to subsequent session_push / session_close
    calls and to record_fact / query_memory so gravity updates are attributed.

    If session_id is omitted, a unique one is generated.
    """
    sid = session_id or f"{agent_id}-{int(time.time())}-{uuid.uuid4().hex[:4]}"
    _store.session_start(sid)
    return {
        "session_id": sid,
        "_hint": (
            "pass this session_id to record_fact, record_facts, query_memory, "
            "and session_push — then call session_close when done"
        ),
    }


@mcp.tool()
def session_push(text: str, session_id: str) -> dict:
    """
    Append a user message to an open session.

    Call this for each user message during the session. The text is embedded
    and stored so the resonance engine can score the session trajectory on close.
    Do NOT push your own (assistant) responses — only user-side text.
    """
    _store.session_message(text, session_id=session_id)
    return {
        "session_id": session_id,
        "ok": True,
        "_hint": (
            "call session_close when the conversation ends to score resonance "
            "and update gravity"
        ),
    }


@mcp.tool()
def session_close(session_id: str) -> dict:
    """
    Close a session: score resonance, update fact gravity, detect echo.

    Call once when the conversation ends. BirchKM will:
      - Score R in [-1, +1] from the session messages
      - Propagate R to all facts touched during the session
      - Detect echo (return to unresolved problem) and apply retroactive penalty
      - Absorb dead facts into the black hole
      - Trigger gravitational collapse if thresholds are met

    Returns: label, r_score, migrations, absorbed count, current stats.
    """
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
