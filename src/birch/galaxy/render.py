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

    # Ordinary facts, coloured by ring, sized by mass.
    for ring, colour in _RING_COLOUR.items():
        members = [b for b in galaxy.bodies
                   if b.kind == "fact" and galaxy.ring_of(b) == ring]
        if not members:
            continue
        ax.scatter(
            [b.pos[0] for b in members],
            [b.pos[1] for b in members],
            s=[12 + 26 * b.mass for b in members],
            c=colour, alpha=0.82, edgecolors="none", label=ring,
        )

    # MetaFacts — collapsed clumps — drawn as bright stars.
    metas = [b for b in galaxy.bodies if b.kind == "meta"]
    if metas:
        ax.scatter(
            [b.pos[0] for b in metas],
            [b.pos[1] for b in metas],
            s=[min(420.0, 70 + 22 * b.mass) for b in metas],
            c="#ffffff", marker="*", edgecolors="#b070ff",
            linewidths=1.0, zorder=5, label="metafact",
        )

    # The attention mass — the user's current focus.
    if galaxy.attention_pos is not None:
        ax_x = float(galaxy.attention_pos[0])
        ax_y = float(galaxy.attention_pos[1])
        ax.scatter([ax_x], [ax_y], s=900, facecolors="none", edgecolors="#ff44cc",
                   linewidths=2.2, zorder=6, label="attention")
        ax.scatter([ax_x], [ax_y], s=40, c="#ff44cc", zorder=6)

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


def render_animation(
    galaxy: Galaxy,
    history: object,
    path: str,
    frames: int = 150,
) -> str:
    """Replay ``history`` and write an animated GIF of the galaxy forming.

    ``history`` is a replay.History; typed loosely to keep render.py free
    of a hard import cycle with replay.py.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    from .replay import replay

    total_steps = getattr(history, "steps", 1)
    interval = max(1, total_steps // frames)
    snaps: list[tuple] = []

    def capture(step: int, gal: Galaxy) -> None:
        if step % interval == 0 or step == total_steps - 1:
            attn = (None if gal.attention_pos is None
                    else (float(gal.attention_pos[0]), float(gal.attention_pos[1])))
            snaps.append((
                step,
                len(gal.absorbed),
                attn,
                [(float(b.pos[0]), float(b.pos[1]), gal.ring_of(b),
                  b.mass, b.kind) for b in gal.bodies],
            ))

    replay(galaxy, history, on_step=capture)  # type: ignore[arg-type]

    fig, ax = plt.subplots(figsize=(9, 9), facecolor="#0b0b14")
    span = galaxy.r_surface * 1.6

    def draw(frame_idx: int) -> None:
        ax.clear()
        ax.set_facecolor("#0b0b14")
        step, absorbed, attn, bodies = snaps[frame_idx]
        for radius in (galaxy.horizon, galaxy.r_core, galaxy.r_surface):
            ax.add_patch(plt.Circle((0, 0), radius, fill=False, color="#333344",
                                    linestyle="--", linewidth=0.8))
        ax.add_patch(plt.Circle((0, 0), galaxy.horizon, color="black", zorder=3))
        ax.scatter([0], [0], s=12, color="#ff5555", zorder=4)
        if attn is not None:
            ax.scatter([attn[0]], [attn[1]], s=850, facecolors="none",
                       edgecolors="#ff44cc", linewidths=2.0, zorder=6)
            ax.scatter([attn[0]], [attn[1]], s=34, c="#ff44cc", zorder=6)
        for ring, colour in _RING_COLOUR.items():
            pts = [(x, y, m) for (x, y, r, m, k) in bodies
                   if k == "fact" and r == ring]
            if pts:
                ax.scatter([p[0] for p in pts], [p[1] for p in pts],
                           s=[12 + 26 * p[2] for p in pts], c=colour,
                           alpha=0.82, edgecolors="none")
        metas = [(x, y, m) for (x, y, r, m, k) in bodies if k == "meta"]
        if metas:
            ax.scatter([p[0] for p in metas], [p[1] for p in metas],
                       s=[min(420.0, 70 + 22 * p[2]) for p in metas],
                       c="#ffffff", marker="*", edgecolors="#b070ff",
                       linewidths=1.0, zorder=5)
        ax.set_xlim(-span, span)
        ax.set_ylim(-span, span)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            f"BirchKM memory galaxy — step {step}   ·   {len(bodies)} live   "
            f"·   {len(metas)} metafacts   ·   {absorbed} absorbed",
            color="#dddddd", fontsize=12,
        )

    # draw returns None (no blitting) — matplotlib's stub wants artists.
    anim = FuncAnimation(fig, draw, frames=len(snaps), interval=80)  # type: ignore[arg-type]
    anim.save(path, writer=PillowWriter(fps=12))
    plt.close(fig)
    return path
