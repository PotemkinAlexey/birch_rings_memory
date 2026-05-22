"""Diagnose — read the settled galaxy as a report on the real store.

After a replay the galaxy's shape is signal, not decoration:

  - facts left on a low, decayed orbit are at risk of being forgotten;
  - friends-of-friends groups are emergent topics — clusters the
    dynamics found with no tags at all;
  - the MetaFacts that formed are the compactions the store wants.

This is the rabbit hole paying rent: the galaxy stops being a picture
and becomes an instrument pointed at the real BirchKM store.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..fact import FactPassport
from .collapse import friends_of_friends
from .engine import CORE, Body, Galaxy
from .replay import build_history, replay


@dataclass
class GalaxyReport:
    """What the settled galaxy says about the store."""

    at_risk: list[tuple[str, float]]   # (label, radius) — live facts near the hole
    topics: list[list[str]]            # emergent FoF clusters — member labels
    metafacts: list[list[str]]         # collapsed groups — source labels
    absorbed: list[str]                # labels of facts let go into the hole
    live: int                          # bodies still orbiting
    total: int                         # facts the store started with


def diagnose(
    galaxy: Galaxy,
    *,
    absorbed_ids: list[str],
    fact_labels: dict[str, str],
    topic_linking: float = 2.6,
) -> GalaxyReport:
    """Read a settled galaxy as a report. ``fact_labels`` maps fact_id -> label."""

    def label_of(body: Body) -> str:
        return body.label or fact_labels.get(body.fact_id, body.fact_id)

    # Live ordinary facts that have decayed into the core ring — these are
    # the ones the dynamics have pushed toward the black hole.
    at_risk = sorted(
        ((label_of(b), round(b.radius, 2)) for b in galaxy.bodies
         if b.kind == "fact" and galaxy.ring_of(b) == CORE),
        key=lambda item: item[1],
    )

    # Emergent topics: friends-of-friends groups of two or more bodies.
    topics: list[list[str]] = []
    for group in friends_of_friends(galaxy.bodies, topic_linking):
        if len(group) >= 2:
            topics.append([label_of(b) for b in group])
    topics.sort(key=len, reverse=True)

    # MetaFacts that collapsed during the replay — clumps the store wants
    # compacted.
    metafacts = [
        [fact_labels.get(src, src) for src in b.source_ids]
        for b in galaxy.bodies if b.kind == "meta"
    ]

    absorbed = [fact_labels.get(a, a) for a in absorbed_ids]
    return GalaxyReport(
        at_risk=at_risk,
        topics=topics,
        metafacts=metafacts,
        absorbed=absorbed,
        live=len(galaxy.bodies),
        total=len(fact_labels),
    )


def run_diagnosis(
    facts: list[FactPassport],
    sessions: list[dict],
    *,
    steps: int = 1400,
    attention_mass: float = 40.0,
    now: float | None = None,
) -> GalaxyReport:
    """Build the history, replay it, and diagnose the settled galaxy."""
    history = build_history(facts, sessions, steps=steps, now=now)
    galaxy = Galaxy(attention_mass=attention_mass)
    absorbed = replay(galaxy, history)
    labels = {f.fact_id: f"{f.subject} {f.predicate} {f.object}" for f in facts}
    return diagnose(galaxy, absorbed_ids=absorbed, fact_labels=labels)


def format_report(report: GalaxyReport) -> str:
    """Render a GalaxyReport as a human-readable block of text."""
    lines = [
        f"BirchKM galaxy diagnosis — {report.total} facts, "
        f"{report.live} bodies still orbiting",
        "",
        f"At risk of forgetting — {len(report.at_risk)} facts decayed into "
        f"the core ring:",
    ]
    for label, radius in report.at_risk[:12]:
        lines.append(f"  r={radius:5.2f}  {label}")
    if len(report.at_risk) > 12:
        lines.append(f"  ... and {len(report.at_risk) - 12} more")

    lines += ["", f"Emergent topics — {len(report.topics)} clusters the "
              f"dynamics found:"]
    for i, topic in enumerate(report.topics[:8], 1):
        lines.append(f"  topic {i} ({len(topic)} facts): {topic[0]}")
        for label in topic[1:4]:
            lines.append(f"           {label}")
        if len(topic) > 4:
            lines.append(f"           ... +{len(topic) - 4} more")

    lines += ["", f"MetaFacts the store wants — {len(report.metafacts)} "
              f"clumps collapsed:"]
    for i, group in enumerate(report.metafacts[:8], 1):
        lines.append(f"  metafact {i}: {len(group)} facts — e.g. {group[0]}")

    lines += ["", f"Let go into the black hole: {len(report.absorbed)} facts"]
    return "\n".join(lines)
