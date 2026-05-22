"""python -m birch.galaxy — replay the real BirchKM store and diagnose it.

Read-only: loads facts and session history from the store, replays them as
births, resonance kicks and a moving attention mass, writes an animated GIF
and a final still to ~/.birch/galaxy/, then prints a diagnosis — what the
settled galaxy says about the store. The store itself is never modified.
"""
from __future__ import annotations

import os
from pathlib import Path

from ..storage.sqlite import SQLiteBackend
from .engine import Galaxy
from .render import render, render_animation
from .replay import build_history
from .report import diagnose, format_report

_DB = os.environ.get("BIRCH_DB", str(Path.home() / ".birch" / "memory.db"))
_OUT = Path.home() / ".birch" / "galaxy"
_STEPS = 1400


def main() -> None:
    backend = SQLiteBackend(_DB)
    facts = backend.load_facts()
    sessions = backend.load_echo_sessions()
    backend.close()
    print(f"loaded {len(facts)} facts and {len(sessions)} sessions from {_DB}")

    history = build_history(facts, sessions, steps=_STEPS)
    galaxy = Galaxy(attention_mass=40.0)
    _OUT.mkdir(parents=True, exist_ok=True)

    gif, absorbed = render_animation(galaxy, history, str(_OUT / "galaxy.gif"))
    print("wrote", gif)
    still = render(
        galaxy, str(_OUT / "galaxy_final.png"),
        title="BirchKM memory galaxy — end of history",
    )
    print("wrote", still)

    labels = {f.fact_id: f"{f.subject} {f.predicate} {f.object}" for f in facts}
    report = diagnose(galaxy, absorbed_ids=absorbed, fact_labels=labels)
    print()
    print(format_report(report))
    print(f"\nHawking emissions during the replay: {galaxy.hawking_count}")


if __name__ == "__main__":
    main()
