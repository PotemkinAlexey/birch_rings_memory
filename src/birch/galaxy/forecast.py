"""Forecast — turn a settled galaxy into a per-fact stability prediction.

The galaxy started life as a telescope: a research model that ran beside the
live engine to make the physics literal. This module hands the galaxy a
*producer* role too. Given a snapshot of live facts, build the galaxy, run
it forward for a short horizon, and report — per fact — how far each body
ended up from the event horizon.

The result, ``forecast_stability ∈ [0, 1]``, lands on the FactPassport and
is consumed by the adaptive gravity formula as a 5th learnable feature:

    1.0   body finished safely on the surface ring or beyond
    0.0   body crossed the horizon (absorbed)
    0.5   no usable prediction (default neutral prior — what an untouched
          fact carries before ``run_forecast`` has been called)

This makes the galaxy more than a picture: the formula now has access to
a forecast — a signal local features cannot produce by themselves —
without giving up legibility or determinism. The adaptive weight
``w_stability`` learns whether this user's data actually rewards the
forecast or not; on day one the weight sits at its prior and the feature
contributes only the neutral 0.5 to gravity.
"""
from __future__ import annotations

from ..fact import FactPassport
from .engine import Galaxy
from .loader import build_galaxy


def forecast_stability(
    facts: list[FactPassport],
    *,
    horizon_ticks: int = 50,
    galaxy: Galaxy | None = None,
) -> dict[str, float]:
    """Run the galaxy forward and report per-fact stability.

    ``horizon_ticks`` is how many integrator steps to advance. 50 is a
    cheap default; larger horizons see more decay but cost more compute.

    Facts without an embedding land in the galaxy via a deterministic
    fallback direction, so every fact gets a forecast. The returned dict
    is keyed by ``fact_id``; callers that want the symmetric "stability
    by default" semantics for facts missing from the dict should treat
    the absence as 0.5 (the neutral prior on FactPassport).
    """
    if not facts:
        return {}

    gal = build_galaxy(facts, galaxy=galaxy)
    absorbed_during_run: set[str] = set()

    for _ in range(max(0, int(horizon_ticks))):
        absorbed_during_run.update(gal.step())

    out: dict[str, float] = {}
    # Live survivors: stability is how far they finished between the
    # horizon and the surface ring. r >= r_surface → 1.0 (very stable),
    # r at horizon → 0.0, anywhere in between → linear interpolation.
    span = max(1e-6, gal.r_surface - gal.horizon)
    for body in gal.bodies:
        normalised = (body.radius - gal.horizon) / span
        out[body.fact_id] = float(max(0.0, min(1.0, normalised)))
    # Bodies that crossed the horizon during the forecast → 0.0.
    for fid in absorbed_during_run:
        out[fid] = 0.0
    return out
