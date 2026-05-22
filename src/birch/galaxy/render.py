"""Render — draw a Galaxy as a picture.

matplotlib is an optional extra (``pip install birch-km[galaxy]``); it is
imported lazily so the rest of the package stays dependency-light.
"""
from __future__ import annotations

from .engine import ABSORBED, CORE, KINETIC, SURFACE, Galaxy

# ring -> plot colour
_RING_COLOUR = {
    SURFACE: "#f4c430",   # gold — hot, far, safe
    KINETIC: "#4a90d9",   # blue — the working ring
    CORE: "#8a8a8a",      # grey — cold, near the hole
}


def render(galaxy: Galaxy, path: str, title: str = "BirchKM memory galaxy") -> str:
    """Save a PNG snapshot of the galaxy to ``path``; returns ``path``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 9), facecolor="#0b0b14")
    ax.set_facecolor("#0b0b14")

    # Ring guide circles.
    for radius in (galaxy.horizon, galaxy.r_core, galaxy.r_surface):
        ax.add_patch(
            plt.Circle((0, 0), radius, fill=False, color="#333344",
                       linestyle="--", linewidth=0.8)
        )
    # The central black hole.
    ax.add_patch(plt.Circle((0, 0), galaxy.horizon, color="black", zorder=3))
    ax.scatter([0], [0], s=12, color="#ff5555", zorder=4)

    # Bodies, coloured by ring, sized by mass.
    for ring, colour in _RING_COLOUR.items():
        members = [b for b in galaxy.bodies if galaxy.ring_of(b) == ring]
        if not members:
            continue
        ax.scatter(
            [b.pos[0] for b in members],
            [b.pos[1] for b in members],
            s=[12 + 26 * b.mass for b in members],
            c=colour, alpha=0.82, edgecolors="none", label=ring,
        )

    span = galaxy.r_surface * 1.6
    ax.set_xlim(-span, span)
    ax.set_ylim(-span, span)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    counts = galaxy.ring_counts()
    subtitle = (
        f"step {galaxy.steps}   ·   surface {counts[SURFACE]}   "
        f"kinetic {counts[KINETIC]}   core {counts[CORE]}   "
        f"absorbed {counts[ABSORBED]}"
    )
    ax.set_title(f"{title}\n{subtitle}", color="#dddddd", fontsize=12)
    ax.legend(loc="upper right", facecolor="#0b0b14",
              labelcolor="#dddddd", edgecolor="#333344")

    fig.savefig(path, dpi=110, facecolor="#0b0b14", bbox_inches="tight")
    plt.close(fig)
    return path
