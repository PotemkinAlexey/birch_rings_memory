# BirchKM — Birch Knowledge Model

Relational memory system for AI agents inspired by the physics of Paul Birch's
megastructure. Facts live in layered orbits around a black hole sink — the more
useful a fact proves, the higher it floats; the more it misleads, the deeper it
sinks.

---

## Core idea

Standard AI memory systems are static libraries — you write facts in, retrieve
them by similarity, and nothing changes unless you explicitly update them.

BirchKM adds a **resonance feedback loop**: every time a session closes, the
system scores whether the conversation was productive (resonant) or circular
(toxic) — without any explicit user feedback. That score propagates back to the
facts that were used, changing their gravity. Facts that help float up. Facts
that confuse sink down. Facts that fall below the event horizon are absorbed by
the black hole and only return if a future query is close enough to trigger
Hawking emission.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  surface   (layer 0)  gravity > 0.70  hot cache      │
│  kinetic   (layer 1)  gravity 0.30–0.70  working mem │
│  core      (layer 2)  gravity < 0.30  cold archive   │
│─────────────────────────────────────────────────────│
│  black hole (layer -1)  gravity < 0.10               │
│             ↑  hawking_emit()  similarity ≥ 0.95     │
└─────────────────────────────────────────────────────┘
```

### Resonance pipeline

Every session is scored by three signals — no LLM calls required:

| Signal | Mechanism | What it catches |
|---|---|---|
| **Behavioral** | Pattern match on final messages | "works", "got it" vs "still broken" |
| **Semantic** | Cosine shift + specificity delta start→end | Did the conversation narrow down to a solution? |
| **Repetition** | Dispersion around session centroid | Circular rephrasing with no progress |

The combined **R score** lives in `[-1.0, +1.0]`:
- `R > 0.35` → resonant — facts used get a gravity bonus
- `R < -0.15` → toxic — facts used get a gravity penalty
- in between → neutral

### Echo validation (delayed signal)

Each closed session is stored as a **K-means bundle** of centroids (not a
single vector) — so multi-topic sessions don't lose sub-topic structure.

When a new session opens, BirchKM checks: does this look like returning to an
unresolved problem? If `similarity ≥ 0.80` to any centroid in a past session,
an **echo penalty** is applied retroactively — the past session's R drops into
toxic territory, pulling down the gravity of facts it used.

### Gravity engine

```
gravity = 0.55 × behavioral_access
        + 0.25 × avg_resonance (normalized)  
        + 0.20 × graph_degree
```

`access_score` decays exponentially with time (half-life ~14h) so facts that
aren't queried gradually lose buoyancy without any manual cleanup.

Layer migration happens automatically on every `tick()`:
- `gravity > 0.70` → promote one layer up
- `gravity < 0.30` → demote one layer down
- `gravity < 0.10` → absorbed by black hole

### Hawking emission

Facts in the black hole are not permanently lost. A query with cosine
similarity `≥ 0.95` to an absorbed fact triggers emission — the fact returns
to `kinetic` layer with `gravity = 0.30`. The threshold is intentionally high:
only an almost-exact match justifies the retrieval cost.

---

## Modules

| Module | Responsibility |
|---|---|
| `birch/fact.py` | `FactPassport` — subject/predicate/object triple + gravity metadata |
| `birch/gravity.py` | `GravityEngine` — computes scores, triggers layer migration |
| `birch/black_hole.py` | `BlackHole` — irreversible sink + Hawking emission |
| `birch/memory_store.py` | `MemoryStore` — unified API over all layers |
| `birch/resonance/behavioral.py` | Pattern-based closure signal |
| `birch/resonance/semantic.py` | Cosine shift + specificity delta |
| `birch/resonance/repetition.py` | Centroid dispersion detector |
| `birch/resonance/detector.py` | Combines all signals into R score |
| `birch/resonance/echo.py` | Cross-session echo detection + retroactive penalty |
| `birch/resonance/centroid.py` | `centroid()` + `dispersion()` utilities |
| `birch/resonance/cluster.py` | K-means++ bundle for session storage |
| `birch/resonance/embeddings.py` | Ollama `nomic-embed-text` client |

---

## Quickstart

```bash
# Requires Ollama with nomic-embed-text
ollama pull nomic-embed-text

git clone https://github.com/PotemkinAlexey/birch_rings_memory.git
cd birch_rings_memory
python -m venv .venv && source .venv/bin/activate
```

```python
from birch.memory_store import MemoryStore

mem = MemoryStore()

# Add facts
f_go = mem.add_fact("модуль рассылок", "работает на", "Go")
f_db = mem.add_fact("база данных", "использует", "PostgreSQL")
mem.link(f_go.fact_id, f_db.fact_id)

# Run a session
mem.session_start("session_1")
mem.session_message("как настроить модуль рассылок")
mem.session_message("как подключить к PostgreSQL")
mem.session_message("всё заработало, спасибо!")
summary = mem.session_close()
print(summary)  # R score, migrations, absorbed facts

# Query
results = mem.query("модуль рассылок Go", top_k=3)
for r in results:
    print(r.source, r.similarity, r.fact)

# Check for echo before starting a new session
echo = mem.check_echo("рассылки снова не работают")
if echo["echo"]:
    print(f"Warning: returning to unresolved problem (sim={echo['similarity']:.2f})")
```

---

## Experiment results

```
Resonance detector (8 sessions, EN + RU):
  Baseline (patterns only):         7/8
  + semantic embeddings:            7/8
  + repetition detector:            8/8  ← circular sessions with no keywords

Echo validation (4 paired sessions):
  false_resolution (looked ok, came back):   ✓ detected, R retroactively toxic
  genuine_resolution (different topic):      ✓ no echo
  stuck_then_returns (was toxic, came back): ✓ detected
  multi_topic_echo (postgres sub-topic):     ✓ bundle caught what centroid missed
```

---

## What makes this different from GraphRAG / Mem0

Standard systems treat memory as a static index — facts stay where you put
them until you explicitly change them.

BirchKM memory is **kinetic**: facts compete for space based on how useful they
actually proved. The system learns from session outcomes without any explicit
user feedback, and corrects false-positive "success" scores when the same
problem resurfaces in a later session.

The black hole is not a metaphor — it is the mechanism that prevents stale,
misleading facts from accumulating silently in the retrieval index.

---

## Requirements

- Python 3.9+
- [Ollama](https://ollama.com) with `nomic-embed-text` (for embeddings)
- No other dependencies

---

## Status

Proof of concept. Resonance pipeline and gravity engine are functional.
Persistence (SQLite / Redis backend) and a real graph store (Neo4j) are the
natural next steps.

## License

Apache 2.0
