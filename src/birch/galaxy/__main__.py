"""python -m birch.galaxy — replay the real BirchKM store and diagnose it.

  python -m birch.galaxy            one replay: GIF, final still, diagnosis
  python -m birch.galaxy --3d       replay in 3-D: a rotating-camera GIF
  python -m birch.galaxy --watch    live: re-render whenever the store changes

Read-only. Loads facts and session history from the store, replays them as
births, resonance kicks and a moving attention mass, and writes its output
to ~/.birch/galaxy/. The store itself is never modified.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from ..storage.sqlite import SQLiteBackend
from .engine import Galaxy
from .render import render, render_3d, render_animation
from .replay import build_history, replay
from .report import diagnose, format_report

_DB = os.environ.get("BIRCH_DB", str(Path.home() / ".birch" / "memory.db"))
_OUT = Path.home() / ".birch" / "galaxy"
_STEPS = 1400
_POLL_SECONDS = 3.0


def _load() -> tuple[list, list]:
    backend = SQLiteBackend(_DB)
    facts = backend.load_facts()
    sessions = backend.load_echo_sessions()
    backend.close()
    return facts, sessions


def main(dim: int = 2) -> None:
    facts, sessions = _load()
    print(f"loaded {len(facts)} facts and {len(sessions)} sessions from {_DB}")

    history = build_history(facts, sessions, steps=_STEPS, dim=dim)
    galaxy = Galaxy(attention_mass=40.0, dim=dim)
    _OUT.mkdir(parents=True, exist_ok=True)

    if dim == 3:
        absorbed = replay(galaxy, history)
        print("wrote", render_3d(galaxy, str(_OUT / "galaxy_3d.gif")))
    else:
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


def _db_mtime() -> float:
    """Latest modification time across the DB and its WAL/SHM sidecars."""
    latest = 0.0
    for suffix in ("", "-wal", "-shm"):
        path = Path(_DB + suffix)
        if path.exists():
            latest = max(latest, path.stat().st_mtime)
    return latest


def watch() -> None:
    """Re-render the galaxy as a still every time the store changes."""
    print(f"watching {_DB} — re-renders on every change, Ctrl-C to stop")
    _OUT.mkdir(parents=True, exist_ok=True)
    last = -1.0
    try:
        while True:
            mtime = _db_mtime()
            if mtime != last:
                last = mtime
                facts, sessions = _load()
                history = build_history(facts, sessions, steps=_STEPS)
                galaxy = Galaxy(attention_mass=40.0)
                absorbed = replay(galaxy, history)
                still = render(
                    galaxy, str(_OUT / "galaxy_live.png"),
                    title="BirchKM memory galaxy — live",
                )
                stamp = time.strftime("%H:%M:%S")
                print(f"[{stamp}] {len(facts)} facts -> {galaxy.ring_counts()}"
                      f" · {len(absorbed)} absorbed · wrote {still}")
            time.sleep(_POLL_SECONDS)
    except KeyboardInterrupt:
        print("\nstopped watching")


if __name__ == "__main__":
    if "--watch" in sys.argv:
        watch()
    else:
        main(dim=3 if "--3d" in sys.argv else 2)
