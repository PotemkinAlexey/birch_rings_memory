"""python -m birch.galaxy — replay the real BirchKM store as a galaxy.

Read-only: loads facts and session history from the store, replays them
as births and resonance kicks, writes an animated GIF and a final still
to ~/.birch/galaxy/. The store itself is never modified.
"""
from __future__ import annotations

import os
from pathlib import Path

from ..storage.sqlite import SQLiteBackend
from .engine import Galaxy
from .render import render, render_animation
from .replay import build_history

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
    galaxy = Galaxy()
    _OUT.mkdir(parents=True, exist_ok=True)

    gif = render_animation(galaxy, history, str(_OUT / "galaxy.gif"))
    print(f"replayed {_STEPS} steps; {len(galaxy.absorbed)} facts absorbed")
    print("final rings:", galaxy.ring_counts())
    print("wrote", gif)

    still = render(
        galaxy, str(_OUT / "galaxy_final.png"),
        title="BirchKM memory galaxy — end of history",
    )
    print("wrote", still)


if __name__ == "__main__":
    main()
