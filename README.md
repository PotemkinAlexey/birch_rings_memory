# BirchKM — Birch Knowledge Model

Kinetic memory system for AI agents inspired by the physics of Paul Birch's
megastructure. Facts live in layered orbits around a black hole sink — the more
useful a fact proves, the higher it floats; the more it misleads, the deeper it sinks.

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
┌──────────────────────────────────────────────────────────────┐
│  surface   (layer 0)  gravity > 0.70   hot cache             │
│  kinetic   (layer 1)  gravity 0.30–0.70  working memory      │
│  core      (layer 2)  gravity < 0.30   cold archive          │
│──────────────────────────────────────────────────────────────│
│  black hole (layer -1)  gravity < 0.10                       │
│              ↑  hawking_emit()  similarity ≥ 0.95            │
└──────────────────────────────────────────────────────────────┘
```

### Resonance pipeline

Every session is scored by three signals — no LLM calls required:

| Signal | Mechanism | What it catches |
|---|---|---|
| **Behavioral** | Pattern match on final messages | "works", "got it", "found it" vs "still broken" |
| **Semantic** | Cosine shift + specificity delta start→end | Did the conversation narrow to a solution? |
| **Repetition** | Dispersion around session centroid | Circular rephrasing with no progress |

The combined **R score** lives in `[-1.0, +1.0]`:
- `R > 0.35` → resonant — facts used get a gravity bonus
- `R < -0.15` → toxic — facts used get a gravity penalty
- in between → neutral

### Echo validation (delayed signal)

Each closed session is stored as a **K-means bundle** of centroids (not a
single vector) — so multi-topic sessions don't lose sub-topic structure.

When a new session opens, BirchKM checks: does this look like returning to an
unresolved problem? If `similarity ≥ 0.68` to any centroid in a past session,
an **echo penalty** is applied retroactively — the past session's R score drops
into toxic territory, pulling down the gravity of facts it used.

### Gravity engine

```
gravity = 0.35 × access_score   (log-scaled, decays with time, half-life ~14h)
        + 0.45 × avg_resonance  (normalized from [-1, +1] to [0, 1])
        + 0.20 × graph_degree   (relative to max degree in the graph)
```

Layer migration happens automatically on every `session_close()`:
- `gravity > 0.70` → promote one layer up (toward surface)
- `gravity < 0.30` → demote one layer down (toward core)
- `gravity < 0.10` → absorbed by black hole

### Hawking emission

Facts in the black hole are not permanently lost. A query with cosine
similarity `≥ 0.95` to an absorbed fact triggers emission — the fact returns
to `kinetic` layer with `gravity = 0.30`. The threshold is intentionally high:
only an almost-exact match justifies retrieval.

---

## Modules

| Module | Responsibility |
|---|---|
| `fact.py` | `FactPassport` — subject/predicate/object triple + gravity metadata |
| `gravity.py` | `GravityEngine` — computes scores, triggers layer migration |
| `black_hole.py` | `BlackHole` — irreversible sink + Hawking emission |
| `memory_store.py` | `MemoryStore` — unified API over all layers |
| `storage/base.py` | `StorageBackend` — Protocol for pluggable persistence |
| `storage/sqlite.py` | `SQLiteBackend` — default write-through implementation |
| `server.py` | MCP server — exposes memory as tools for Claude agents |
| `resonance/behavioral.py` | Pattern-based closure signal |
| `resonance/semantic.py` | Cosine shift + specificity delta |
| `resonance/repetition.py` | Centroid dispersion detector |
| `resonance/detector.py` | Combines all signals into R score |
| `resonance/echo.py` | Cross-session echo detection + retroactive penalty |
| `resonance/cluster.py` | K-means++ bundle for session storage |
| `resonance/embeddings.py` | Ollama `nomic-embed-text` client |

---

## Quickstart

```bash
# Requires Ollama with nomic-embed-text
ollama pull nomic-embed-text

git clone https://github.com/PotemkinAlexey/birch_rings_memory.git
cd birch_rings_memory
python -m pip install -e .
```

### In-memory (no persistence)

```python
from birch.memory_store import MemoryStore

mem = MemoryStore()
f_go = mem.add_fact("mailer service", "runs on", "Go")
f_db = mem.add_fact("database", "uses", "PostgreSQL")
mem.link(f_go.fact_id, f_db.fact_id)

mem.session_start("s1")
mem.session_message("how to configure the mailer service")
mem.session_message("how to connect it to PostgreSQL")
mem.session_message("everything works, thanks!")
summary = mem.session_close()
# {"label": "resonant", "r": 0.71, "migrations": [...], "absorbed": []}

results = mem.query("mailer service Go", top_k=3)
for r in results:
    print(r.source, r.similarity, r.fact)
```

### With SQLite persistence

```python
mem = MemoryStore(db_path="~/.birch/memory.db")
```

Memory survives process restarts. Facts, edges, gravity scores and echo
session bundles are all persisted automatically.

### With a custom backend

```python
from birch.storage import StorageBackend
from birch.memory_store import MemoryStore

class RedisBackend:          # no inheritance required
    def save_fact(self, fact): ...
    def load_facts(self): ...
    # ... implement StorageBackend protocol

mem = MemoryStore(storage=RedisBackend(...))
```

---

## Connecting to Claude agents (MCP)

See [AGENTS.md](AGENTS.md) for full setup. Quick version:

```bash
# Start the MCP server
BIRCH_DB=~/.birch/memory.db python -m birch.server
```

Add to `~/.claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "birch-km": {
      "command": "/path/to/your/venv/bin/python",
      "args": ["-m", "birch.server"],
      "env": { "BIRCH_DB": "/Users/you/.birch/memory.db" }
    }
  }
}
```

Claude then has four tools: `query_memory`, `record_fact`, `record_session`, `memory_stats`.

---

## What makes this different from GraphRAG / Mem0

Standard systems treat memory as a static index — facts stay where you put
them until you explicitly change them.

BirchKM memory is **kinetic**: facts compete for space based on how useful
they actually proved. Three properties distinguish it:

1. **No explicit feedback required** — resonance is inferred from session
   structure (behavioral patterns + semantic shift + topic dispersion).
2. **Retroactive correction** — if the user returns to an unresolved problem,
   the echo system penalizes the past session's R score, pulling down the
   gravity of facts that gave a false sense of resolution.
3. **Lossy by design** — the black hole is not an edge case. It is the
   mechanism that prevents stale, misleading facts from accumulating silently.

---

## Test results

```
Resonance detector (8 sessions):
  Baseline (patterns only):    6/8
  Full (+ embeddings):         8/8  ← hard cases need semantic shift

Echo validation (4 paired sessions):
  false_resolution              ✓ echo detected, R retroactively toxic
  genuine_resolution            ✓ no echo
  stuck_then_returns            ✓ echo detected
  multi_topic_echo              ✓ bundle caught what single centroid would miss

Gravity engine:
  hot fact promotes to surface  ✓
  cold fact demotes to core     ✓
  graph degree helps buoyancy   ✓
```

---

## Requirements

- Python 3.9+
- [Ollama](https://ollama.com) with `nomic-embed-text`
- `mcp[cli]` (installed automatically)

---

## Status

Working proof of concept. Resonance pipeline, gravity engine, black hole,
SQLite persistence, and MCP server are all functional. Next natural steps:
vector index (hnswlib) for large fact stores, multi-agent shared memory.

## License

Apache 2.0
