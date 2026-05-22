"""Galaxy — the research N-body model of BirchKM memory.

A laboratory, not the live engine. The MCP server still scores facts with
``compute_gravity``; nothing here touches it. Here a fact is a *body* in
orbit around the memory's black hole:

  - the central black hole is the sink;
  - a body's orbital radius is its ring — far is surface, near is core;
  - dynamical friction decays an unused orbit inward toward the hole;
  - resonance and access are thrust that boosts an orbit back out.

Run ``python -m birch.galaxy`` to render the real store as a galaxy.
"""
from .engine import Body, Galaxy

__all__ = ["Body", "Galaxy"]
