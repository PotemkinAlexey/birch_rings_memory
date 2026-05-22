"""python -m birch.galaxy — render the real BirchKM store as a galaxy.

Read-only: loads facts from the store, simulates, writes PNG snapshots to
~/.birch/galaxy/. The store itself is never modified.
"""
from __future__ import annotations

import os
from pathlib import Path

from ..storage.sqlite import SQLiteBackend
from .loader import build_galaxy
from .render import render

_DB = os.environ.get("BIRCH_DB", str(Path.home() / ".birch" / "memory.db"))
_OUT = Path.home() / ".birch" / "galaxy"
_STEPS = 1000


def main() -> None:
    backend = SQLiteBackend(_DB)
    facts = backend.load_facts()
    backend.close()
    print(f"loaded {len(facts)} facts from {_DB}")

    galaxy = build_galaxy(facts)
    _OUT.mkdir(parents=True, exist_ok=True)

    print("initial rings:", galaxy.ring_counts())
    print("wrote", render(
        galaxy, str(_OUT / "galaxy_before.png"),
        title="BirchKM memory galaxy — initial",
    ))

    absorbed = galaxy.run(_STEPS)
    print(f"ran {_STEPS} steps, {len(absorbed)} facts absorbed")
    print("final rings:", galaxy.ring_counts())
    print("wrote", render(
        galaxy, str(_OUT / "galaxy_after.png"),
        title=f"BirchKM memory galaxy — after {_STEPS} steps",
    ))


if __name__ == "__main__":
    main()
