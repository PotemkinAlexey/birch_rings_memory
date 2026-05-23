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
gravity = w_freshness × freshness            (learned; prior 0.28; ~2-week half-life)
        + w_access    × access               (learned; prior 0.14; log-scaled, ~3-day decay)
        + w_graph     × graph                (learned; prior 0.09; degree / max-degree)
        + w_utility   × recent_utility       (learned; prior 0.09; EWMA of closure-weighted R)
        + w_stability × forecast_stability   (learned; prior 0.05; galaxy forward forecast)
        + 0.35        × resonance            (fixed: observation, not prediction)
```

The **freshness** term is a grace period — a new fact rides high and is not
archived to the cold core before it has had a chance to prove itself.
**Resonance** reaches a fact weighted by its relevance to the session: a
fact returned by `query_memory` at cosine 0.95 absorbs almost the full
session R, one returned at 0.10 almost none; facts added or re-confirmed
via `record_fact` are pinned at weight 1.0. Resonance contributes 0 until a
session has actually scored the fact.

**`recent_utility`** is an EWMA of closure-weighted resonance updated for
every fact a session touched: `(1-α)·prev + α·target` with α=0.15 and
target `(R · attribution_weight + 1) / 2`. It captures *how the sessions
this fact participated in tended to end*, independently of how often it
was touched. Default 0.5 is a Bayesian neutral prior — an untouched fact
gets a soft floor and does not need to "prove itself" to escape junk
status. Positive context is emergent: the user never labels anything, the
gravity formula learns the attribution.

**`forecast_stability`** is the galaxy's forecast — built by running the
N-body model forward `horizon_ticks` steps and reading off how far each
body finished from the event horizon. 1.0 = safely on the surface ring,
0.0 = crossed the horizon during the simulation, 0.5 = no forecast yet
(neutral prior). It is the only adaptive feature that reads from the
*future*; the others are all derived from a fact's current local state.
Triggered explicitly via the `forecast_memory` MCP tool, not on every
session_close (the simulation is O(n²)). The galaxy was the telescope;
this turns it into a producer of features the live formula consumes.

**The five pre-resonance weights are learned, not hand-set.** Freshness,
access, graph, utility and stability each carry a weight that starts at
the values above (the prior) and adapts to the user's own resonance
feedback — one regularised SGD step per closed session, fit against
`(R+1)/2` as ground truth. The resonance weight stays fixed at 0.35:
resonance is observation, not prediction. A budget renormalisation keeps
the five learned weights summing to 0.65 so the formula stays in
`[0, 1]`. `memory_stats` exposes the live weights and the training count
so the formula stays legible — at zero data behaviour is identical to
the hand-tuned prior, and as the store is used the weights drift toward
what predicts value *for you*. A feature that turns out to have no
predictive power simply has its weight stay near its prior; useful ones
climb. The forecast feature is a clean example: `w_stability` only grows
if galaxy predictions correlate with realised session outcomes.

Layer migration happens on every `session_close()`. Each tick a fact steps
one layer **toward the layer its gravity belongs in** — surface above 0.70,
core below 0.30, kinetic between — so a fact stranded in the core climbs
back out once its gravity recovers. Below 0.10 it is absorbed by the
black hole.

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
| `fact.py` | `FactPassport` — subject/predicate/object triple + gravity metadata (incl. `recent_utility`) |
| `meta_fact.py` | `MetaFact` — dense centroid bundle with lineage + feedback-loop fields |
| `adaptive_gravity.py` | `AdaptiveWeights` — four learned pre-resonance weights, regularised SGD with budget renormalisation |
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
| `galaxy/` | N-body research model — facts as orbiting bodies (see below); now also a feature producer via `galaxy/forecast.py` |

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

**Across processes.** Each MCP client spawns its own `birch.server`, so
several `MemoryStore` instances share one SQLite file. The in-memory state
is treated as a cache, not the source of truth: `SQLiteBackend` runs in WAL
mode and exposes `data_version()`, every operation reloads from disk when
another process has written, and every write runs inside an exclusive
transaction. A lone process never reloads and stays hot; a stale process
can no longer clobber another's gravity ticks.

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

Claude then has thirteen tools:

| Tool | What it does |
|---|---|
| `query_memory` | Semantic search — returns facts and MetaFacts ranked by similarity |
| `record_fact` | Store one subject-predicate-object triple |
| `record_facts` | Store many triples in one batch (one Ollama round-trip) |
| `supersede_fact` | Replace an old fact with a newer one — old body goes to the singularity with `deprecated_by` set; lineage preserved, MetaFact / Hawking still possible |
| `retire_fact` | Send a no-longer-relevant fact to the singularity (no replacement) — `ttl=now`, same singularity benefits as supersede |
| `delete_fact` | Hard-delete — data is GONE, no singularity, no lineage. Use only for secrets / accidental writes; prefer supersede or retire for stale data |
| `list_facts` | List live facts by subject/predicate, sorted by gravity — audit without a query |
| `session_open` | Open a named session so reads and writes can be attributed to it |
| `session_push` | Append a user message to an open session |
| `session_close` | Close a session — score resonance, update gravity, detect echo |
| `record_session` | Score a completed session in one call (open + push messages + close) |
| `forecast_memory` | Run the galaxy forward and write a per-fact stability prediction back into the live store; feeds the adaptive gravity formula via `w_stability` |
| `memory_stats` | Report layer distribution, black hole status, and live adaptive weights |

`query_memory` returns polymorphic hits. Every item has `kind`, `body_id`, `similarity`,
`source`, `layer`, `gravity_score`. Fact hits (`kind: "fact"`) include `subject`,
`predicate`, `object`. MetaFact hits (`kind: "meta"`, `source: "hawking_meta"`) include
`weight`, `source_texts`, `source_fact_ids`, and `summary`. See [AGENTS.md](AGENTS.md).

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

## The galaxy — an N-body research model

`birch/galaxy/` is a research model that sits beside the live engine — the
MCP server still scores facts with the gravity formula above; the galaxy
makes the metaphor literal. Facts become bodies in orbit around the black
hole, and the physics is simulated rather than scored:

- a body's orbital **radius** is its ring — far is surface, near is core;
- **dynamical friction** decays an unused orbit inward;
- a closed session is an orbital **kick** — resonance is thrust;
- a cold, gravitationally bound clump undergoes a **Jeans collapse** into a
  MetaFact;
- the current topic places a moving **attention mass** that bends the disk.

Run it on the real store:

```bash
pip install -e ".[galaxy]"      # adds matplotlib
python -m birch.galaxy
```

It replays the store's whole history, writes an animated GIF, and prints a
diagnosis — which facts are spiralling toward the black hole (forgetting
risk), which friends-of-friends clusters are emergent topics, and which
clumps the store wants compacted into MetaFacts. See [ARCHITECTURE.md](ARCHITECTURE.md).

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
by the test suite: resonance pipeline, **adaptive gravity engine**
(pre-resonance weights learned from session resonance — no hand-set
magic numbers), black hole with polymorphic singularity (facts +
MetaFacts), SQLite persistence, numpy vector index, per-session
concurrency, **cross-process safety** (WAL + `data_version` cache
invalidation), auto-linking on `add_fact`, counter-triggered background
collapse with lineage, EchoStore TTL, the MCP server, and the `galaxy`
N-body research model. CI runs ruff and mypy on every push.

Next natural steps: a persistent similarity index for very large stores
(FAISS / hnswlib), an LLM-driven `MetaFact.summary` writer that runs async
after collapse, and recursive collapse (MetaFacts colliding with MetaFacts).

## License

Apache 2.0
