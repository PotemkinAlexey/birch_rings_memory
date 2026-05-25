"""QueryResult.similarity holds the raw cosine, not a 4-decimal rounded
value. Earlier code rounded at construction:

    QueryResult(fact=fact, similarity=round(sim, 4), ...)

That value then flowed into the resonance attribution feedback loop:

    attribution_pairs = [(r.body_id, r.similarity) for r in top]

So a fact returned at raw similarity 0.95004 contributed a *rounded*
weight of 0.9500 to the session's gravity update. Tiny, but the comment
right above said "round only on output, never on decision" — which was
not strictly true. The truncation happened *before* the feedback loop.

Fix: store raw at construction; `to_mcp_dict` already rounds for display.

This test pins three properties:

  1. QueryResult.similarity is the raw cosine when returned by query().
  2. to_mcp_dict() still serialises the rounded display value.
  3. Internal attribution / weighting consumers see raw precision.
"""
from __future__ import annotations

from unittest.mock import patch

from birch.memory_store import MemoryStore


def test_query_result_similarity_is_raw_not_rounded(tmp_path):
    """Construct a sim with > 4-decimal precision and verify the
    QueryResult stores it untouched."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    f = mem.add_fact("alpha", "beta", "gamma")
    fid = f.fact_id
    # Force a sim with full float precision — definitely not a
    # 4-decimal value.
    raw_sim = 0.9500499999  # 10 digits past the decimal
    with patch.object(
        mem._index, "all_similarities",
        return_value={fid: raw_sim},
    ):
        results = mem.query("anything", top_k=5)
    hit = next(r for r in results if r.fact and r.fact.fact_id == fid)
    # Stored value is raw — every digit preserved.
    assert hit.similarity == raw_sim, (
        f"QueryResult.similarity={hit.similarity!r} should be raw "
        f"{raw_sim!r}; rounding at construction silently truncates "
        "the attribution weight in the gravity feedback loop"
    )
    mem.close()


def test_to_mcp_dict_rounds_for_display():
    """The display serialiser still emits the 4-decimal rounded
    value — wire format stays compact for MCP consumers."""
    from birch.fact import FactPassport
    from birch.memory_store import QueryResult

    f = FactPassport(subject="a", predicate="b", object="c", fact_id="f")
    r = QueryResult(fact=f, similarity=0.9500499999, source="surface")
    out = r.to_mcp_dict()
    assert out["similarity"] == 0.95   # round(0.9500499999, 4) == 0.95
    # And the in-memory object is untouched.
    assert r.similarity == 0.9500499999


def test_attribution_uses_raw_similarity_in_gravity_loop(tmp_path):
    """End-to-end: a query with raw sim 0.9999 (clearly above 4-decimal
    granularity) attributes the raw weight to the open session, not
    the rounded one. The session's ctx.facts dict is the feedback
    loop's input, so checking what landed there is the load-bearing
    assertion."""
    mem = MemoryStore(db_path=str(tmp_path / "m.db"))
    f = mem.add_fact("alpha", "beta", "gamma")
    fid = f.fact_id
    mem.session_start("s1")
    raw_sim = 0.9999777    # 7 digits past decimal
    with patch.object(
        mem._index, "all_similarities",
        return_value={fid: raw_sim},
    ):
        mem.query("anything", top_k=5, session_id="s1")
    # The session's attribution weight came from the QueryResult's
    # similarity field; verify it kept full precision (clipped to
    # [0, 1] by _attribute_to, but no rounding loss).
    attribution = mem._sessions["s1"].facts.get(fid)
    assert attribution is not None
    assert attribution == raw_sim, (
        f"attribution weight {attribution!r} != raw {raw_sim!r} — "
        "the resonance feedback loop must see full precision, not "
        "the 4-decimal display value"
    )
    mem.close()


def test_no_remaining_round_at_queryresult_construction():
    """Source-level audit: no QueryResult(...similarity=round(...)...)
    pattern in the codebase. Regression guard so a future contributor
    doesn't accidentally re-introduce the truncation."""
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parents[1] / "src" / "birch"
    offenders: list[str] = []
    pattern = re.compile(r"similarity\s*=\s*round\s*\(", re.MULTILINE)
    for py in root.rglob("*.py"):
        # _models.py legitimately rounds inside to_mcp_dict — that's
        # the ONE place where round() should appear on similarity.
        text = py.read_text()
        for m in pattern.finditer(text):
            # Get a few lines of context to filter out the to_mcp_dict
            # legitimate use.
            start = text.rfind("\n", 0, m.start()) + 1
            end = text.find("\n", m.end())
            line = text[start:end]
            # echo.py uses a different EchoResult dataclass — its
            # similarity isn't fed into gravity attribution.
            if "EchoResult" in text or py.name == "echo.py":
                continue
            # _models.py to_mcp_dict's `"similarity": round(self.similarity, 4)`
            # is the one legitimate site.
            if "self.similarity" in line:
                continue
            offenders.append(f"{py.relative_to(root)}: {line.strip()}")
    assert not offenders, (
        f"Found round() on similarity at construction sites: "
        f"{offenders}. QueryResult.similarity must store the raw "
        f"cosine — round only inside to_mcp_dict for display."
    )
