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
│   ├── FactPassports   ↑ hawking_emit()       sim ≥ 0.95      │
│   └── MetaFacts       ↑ hawking_emit_metas() sim ≥ 0.85      │
│                                                              │
│           gravitational collapse: clusters of dead facts     │
│           fuse into a single dense MetaFact (cosine ≥ 0.92)  │
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

Resonance reaches a fact **weighted by the fact's relevance to the session**.
A fact returned by `query_memory` at cosine 0.95 absorbs almost the full
session R; a fact returned at 0.10 absorbs almost none. Facts explicitly
added or re-confirmed via `record_fact` are pinned at weight 1.0.

Layer migration happens automatically on every `session_close()`:
- `gravity > 0.70` → promote one layer up (toward surface)
- `gravity < 0.30` → demote one layer down (toward core)
- `gravity < 0.10` → absorbed by black hole

### Hawking emission

Facts in the black hole are not permanently lost. A query with cosine
similarity `≥ 0.95` to an absorbed fact triggers emission — the fact leaves
the singularity, returns to `kinetic` layer with `gravity = 0.30`, and is
persisted back to storage. The threshold is intentionally high: only an
almost-exact match justifies retrieval.

MetaFacts emit at a looser threshold (`0.85`) because a centroid drifts
between its sources; their gravity on emission is set by their own
weight: `gravity = 0.30 + 0.10 · log10(weight)`, capped at `0.70`. A
bundle of fifty dead facts ends up around `0.47` — dense, but not
straight into the surface layer.

### Memory consolidation (gravitational collapse)

The black hole would grow linearly with every session close. **Singularity
Compactor** runs gravitational collapse over the singularity: a single
numpy `matmul` computes pairwise cosine over all absorbed fact vectors,
path-compressing Union-Find groups every transitive cluster above the
collapse threshold (default `0.92`), and each group becomes one
`MetaFact`. The originals are removed from both `_singularity` and the
index; their texts and ids live on inside the MetaFact (`source_texts`,
`source_fact_ids`) for lineage.

`MemoryStore` runs collapse opportunistically on `session_close` — when
the singularity has both grown past `COLLAPSE_FACT_MASS_TRIGGER` (default
`100`) and accumulated at least `COLLAPSE_DELTA_TRIGGER` new absorptions
(default `50`) since the last pass, a job is submitted to a
single-worker `ThreadPoolExecutor`. Set `collapse_async=False` on
construction for predictable test runs or deployments that don't want
background threads.

Live MetaFacts participate in the full feedback loop: they accumulate
`access_count`, `resonance_sum`, and migrate between layers like any
fact. When their gravity drops back below `0.10`, they are re-absorbed
by the singularity (a future pass may compact MetaFacts with MetaFacts).

### Echo TTL

Closed sessions don't live forever in `EchoStore`. A TTL sweep runs on
every `session_close()` with three tiers:

| Tier | Trigger | Default TTL |
|------|---------|-------------|
| Penalty | `echo_penalty != 0` (already converted to gravity correction) | 14 days |
| Resolved | `r_score > 0.35` and no penalty | 7 days |
| Default | everything else | 30 days |

The penalty tier wins precedence — once a session has been echoed its
`r_score` is locked into the toxic floor and cannot drift back into
"resolved". Drops propagate to disk via `delete_echo_session`.

### Numpy vector index

Live facts and absorbed facts each live in a numpy-backed `VectorIndex`:
an L2-normalised `(n, d)` matrix kept in sync with every add/remove. A
query is a single matrix–vector dot product plus an `argpartition` for
top-K, so retrieval stays in milliseconds well past tens of thousands
of facts. MetaFacts have their own pair of indices (live + singularity)
so polymorphic `query()` does four scans without ID collisions.

---

## Modules

| Module | Responsibility |
|---|---|
| `fact.py` | `FactPassport` — subject/predicate/object triple + gravity metadata |
| `meta_fact.py` | `MetaFact` — dense centroid bundle with lineage + feedback-loop fields |
| `gravity.py` | `GravityEngine` — computes scores, triggers layer migration |
| `black_hole.py` | `BlackHole` — irreversible sink + Hawking emission (facts + metas) |
| `singularity_compactor.py` | `collapse_singularity()` — Union-Find collapse + center of mass |
| `vector_index.py` | `VectorIndex` — numpy-backed cosine search |
| `memory_store.py` | `MemoryStore` — unified API, per-session contexts, RLock, collapse orchestration |
| `storage/base.py` | `StorageBackend` — Protocol for pluggable persistence |
| `storage/sqlite.py` | `SQLiteBackend` — default write-through implementation, batched commits |
| `server.py` | MCP server — exposes memory as tools for Claude agents |
| `resonance/behavioral.py` | Pattern-based closure signal |
| `resonance/semantic.py` | Cosine shift + specificity delta |
| `resonance/repetition.py` | Centroid dispersion detector |
| `resonance/detector.py` | Combines all signals into R score |
| `resonance/echo.py` | Cross-session echo detection + retroactive penalty + TTL sweep |
| `resonance/cluster.py` | K-means++ bundle for session storage |
| `resonance/embeddings.py` | Ollama batch embedding client |

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
mem.add_fact("mailer service", "runs on", "Go")
mem.add_fact("database", "uses", "PostgreSQL")

# Open a session before querying so retrieved facts are attributed to it.
# Resonance propagates to facts the agent actually read, weighted by
# how relevant each fact was (cosine similarity at query time).
mem.session_start("s1")
mem.session_message("how to configure the mailer service")

results = mem.query("mailer service Go", top_k=3)
# → returns the "mailer service runs on Go" fact; gravity will respond
#   to this session's outcome because the fact is now attributed to s1.

mem.session_message("how to connect it to PostgreSQL")
mem.session_message("everything works, thanks!")
summary = mem.session_close()
# {"label": "resonant", "r": 0.71, ...}
# The "mailer service" fact's gravity rises because it was used in a
# resonant session.  A repeated toxic session would pull it back down.

# Query results are polymorphic — either a FactPassport or a MetaFact.
for r in results:
    if r.kind == "fact":
        print(r.source, r.similarity, r.fact.subject, r.fact.predicate, r.fact.object)
    else:  # r.kind == "meta"
        print(r.source, r.similarity, "META", r.meta.weight, r.meta.source_texts[:2])
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

class RedisBackend:                          # no inheritance required
    # Facts
    def save_fact(self, fact): ...
    def save_facts(self, facts): ...         # default loops save_fact
    def delete_fact(self, fact_id): ...
    def load_facts(self): ...
    # Edges
    def save_edge(self, from_id, to_id): ...
    def load_edges(self): ...
    # Echo sessions (closed sessions with K-means topic bundle)
    def save_echo_session(self, session_id, centroids,
                          r_score, recorded_at,
                          fact_weights=None, echo_penalty=0.0): ...
    def load_echo_sessions(self): ...
    def delete_echo_session(self, session_id): ...
    # Open sessions (for crash recovery)
    def save_open_session(self, session_id, messages,
                          vectors, facts, started_at): ...
    def delete_open_session(self, session_id): ...
    def load_open_sessions(self): ...
    # MetaFacts (compressed bundles produced by collapse)
    def save_meta_fact(self, meta): ...
    def save_meta_facts(self, metas): ...    # default loops save_meta_fact
    def delete_meta_fact(self, meta_id): ...
    def load_meta_facts(self): ...
    def close(self): ...

mem = MemoryStore(storage=RedisBackend(...))
```

### Concurrent sessions

`MemoryStore` is thread-safe and supports multiple open sessions at once.
Pass `session_id` to every call so each agent's facts attribute to the
right context — `_current_session_id` is only safe under sequential use.

```python
mem = MemoryStore(db_path="~/.birch/memory.db")
mem.session_start("agent_A")
mem.session_start("agent_B")

mem.session_message("how do I configure the mailer", session_id="agent_A")
mem.session_message("nothing works again",          session_id="agent_B")

mem.query("mailer service Go", top_k=3, session_id="agent_A")
mem.query("legacy script",     top_k=3, session_id="agent_B")

mem.session_close(session_id="agent_A")   # resonant — gravity goes up
mem.session_close(session_id="agent_B")   # toxic    — gravity goes down
```

Embedding HTTP calls happen outside the internal lock, so concurrent agents
do not serialize on Ollama. The MCP server already threads `session_id`
through `record_session`, so concurrent calls are safe by default.

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
they actually proved. Four properties distinguish it:

1. **No explicit feedback required** — resonance is inferred from session
   structure (behavioral patterns + semantic shift + topic dispersion).
2. **Retroactive correction** — if the user returns to an unresolved problem,
   the echo system penalizes the past session's R score, pulling down the
   gravity of facts that gave a false sense of resolution.
3. **Lossy by design** — the black hole is not an edge case. It is the
   mechanism that prevents stale, misleading facts from accumulating silently.
4. **Self-consolidating** — clusters of dead facts collapse into dense
   MetaFacts. Lineage (`source_fact_ids`, `source_texts`) is preserved,
   and surviving MetaFacts re-enter the live layers via Hawking emission
   like any other body.

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
- `mcp[cli]` and `numpy>=1.21` (installed automatically)

Environment knobs:

- `OLLAMA_URL` — defaults to `http://localhost:11434`
- `BIRCH_EMBED_MODEL` — defaults to `nomic-embed-text`
- `BIRCH_DB` — SQLite path consumed by `python -m birch.server`

---

## Status

Working proof of concept. All of the following are functional and covered
by the test suite: resonance pipeline, gravity engine, black hole with
polymorphic singularity (facts + MetaFacts), SQLite persistence, numpy
vector index, per-session concurrency, auto-linking on `add_fact`,
counter-triggered background collapse with lineage, EchoStore TTL, and
the MCP server.

Next natural steps: a persistent similarity index for very large stores
(FAISS / hnswlib), an LLM-driven `MetaFact.summary` writer that runs
async after collapse, recursive collapse (MetaFacts colliding with
MetaFacts), and shared multi-agent memory across processes.

## License

Apache 2.0
