# BirchKM — Architecture

## Inspiration

Paul Birch's megastructure is a rotating shell around a black hole. Matter in
the accretion disk self-organizes by kinetic energy — hot, fast-moving matter
stays near the surface; cold, slow matter sinks toward the core; matter that
loses all momentum crosses the event horizon.

BirchKM applies this as a memory metaphor: facts are particles. Sessions are
collisions. Resonance is kinetic energy. The black hole is not a metaphor — it
is the deletion mechanism.

---

## Layers

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

All facts enter at `kinetic` (layer 1) with `gravity = 0.5`. From there they
drift based on usage. Layer migration is automatic — it happens every time
`session_close()` calls `tick()`.

---

## Fact

The atomic unit is a `FactPassport` — a subject–predicate–object triple plus
metadata:

```
subject       "mailer service"
predicate     "runs on"
object        "Go"
─────────────────────────────
fact_id       full uuid4
vector        [768-dim float]   nomic-embed-text embedding of "s p o"
gravity_score 0.5               starts neutral, drifts with sessions
layer         1                 0=surface 1=kinetic 2=core -1=black_hole
created_at    unix timestamp
ttl           optional expiry
access_count  incremented on query hit
last_accessed unix timestamp
resonance_sum cumulative R from sessions that used this fact
resonance_count number of contributing sessions
source_session session_id that created this fact
deprecated_by  fact_id of the successor (if superseded)
```

### SPO deduplication

`MemoryStore` keeps a `_spo_index: dict[(subject, predicate, object), fact_id]`
keyed on case- and whitespace-normalised triples. A repeated `add_fact` for an
existing triple returns the original `FactPassport`, bumps `access_count`, and
re-attributes the fact to the calling session — instead of creating a parallel
record that would dilute its gravity. The index is rebuilt from storage on
startup and is updated on deprecation and on black-hole absorption.

---

## Gravity engine

Gravity is recomputed every `tick()`:

```
gravity = 0.35 × access_score
        + 0.45 × resonance_score
        + 0.20 × graph_score
```

### access_score

```
access_score = log1p(access_count) / log1p(100)
             × exp(−0.05 × age_hours)
```

Logarithmic so a single hot fact doesn't dominate. Exponential decay with
half-life ~14h — facts not queried gradually lose buoyancy without manual
cleanup.

### resonance_score

```
resonance_score = (avg_resonance + 1.0) / 2.0
avg_resonance   = resonance_sum / resonance_count
```

Normalized from `[-1, +1]` to `[0, 1]`. A fact that only appeared in toxic
sessions converges toward 0.0; a fact that only appeared in resonant sessions
converges toward 1.0.

### graph_score

```
graph_score = degree(fact) / max_degree_in_graph
```

Facts with more `link()` connections are harder to sink. Encodes the intuition
that well-connected knowledge is more structural and harder to invalidate.

### Layer migration (per tick)

```
gravity > 0.70  →  promote one layer up (layer - 1, floor 0)
gravity < 0.30  →  demote one layer down (layer + 1, ceiling 2)
gravity < 0.10  →  absorbed by black hole (layer = -1, removed from live index)
```

---

## Resonance pipeline

Resonance is computed from user messages alone — no LLM calls, no explicit
feedback. Three signals are combined:

### Behavioral (weight 0.55)

Pattern match on the final messages of the session:

```
POSITIVE  →  +1.0   "works", "got it", "found it", "figured it out",
                     "thanks", "solved", "fixed", "perfect", "done"
NEGATIVE  →  −0.8   "doesn't work", "still not", "again", "error",
                     "failed", "broken", "not working"
UNFINISHED → −0.5   "...", "wait", "stop"
FOLLOWUP   →  0.0   "got it, and how to..." — positive keyword, but
                     session continues; suppress positive signal
```

Check order: NEGATIVE before POSITIVE — prevents "не работает" matching
"работает". Consensus amplification: if all tail signals agree, the last
score is multiplied by 1.2.

### Semantic (weight 0.25)

Measures whether the conversation narrowed from vague to specific:

```
cosine(start_vector, end_vector) > 0.75
AND specificity_delta < 0.05
→  stuck score = −0.4   (high similarity + no specificity gain = circular)
```

`specificity_delta = token_count(end) / token_count(start) − 1`

A productive session moves from short, vague messages ("I have a performance
problem") to longer, specific ones ("missing index on foreign key, added it").

### Repetition (weight 0.20)

Centroid dispersion of all session vectors:

```
dispersion < 0.05  →  −0.8   (tight loop, almost identical messages)
dispersion < 0.12  →  −0.4   (low variety)
else               →   0.0
```

`dispersion = mean cosine distance from session centroid`

### Combined R score

```
R = 0.55 × behavioral + 0.25 × semantic + 0.20 × repetition
```

Thresholds:
- `R > 0.35` → resonant
- `R < -0.15` → toxic
- in between → neutral

R is propagated to all facts accessed during the session via
`apply_session_resonance()`, which updates `resonance_sum` and
`resonance_count`.

### Per-fact weighting

`apply_session_resonance()` accepts a `dict[fact_id → weight]`. The weight
is the maximum cosine similarity at which the fact appeared inside the
session (1.0 for facts created or re-confirmed via `record_fact`, similarity
for facts returned by `query`). The contribution is:

```
fact.resonance_sum   += R × weight   # scaled contribution accumulates
fact.resonance_count += 1            # counts sessions, not weighted sessions
```

`avg_resonance = resonance_sum / resonance_count` is therefore the mean of
per-session weighted contributions `(R × w)` — not a true weighted average
of R scores, but it is bounded to `[-1, +1]` and correctly discounts
low-similarity matches: a fact at cosine 0.10 contributes 10× less to
`resonance_sum` per session than one at cosine 1.0, while both increment
`resonance_count` by 1. Legacy callers that pass a flat `list[fact_id]`
still work — every weight defaults to 1.0.

---

## Echo validation

Each closed session is stored as a **K-means++ bundle** of K centroids rather
than a single centroid. This preserves sub-topic structure — a session that
covered both "auth" and "deployment" won't miss an echo on either sub-topic.

```
EchoStore
  session_id → StoredSession
                  bundle:        ClusterBundle (K centroids, cosine K-means++)
                  r_score:       float         (retroactively mutable)
                  fact_weights:  dict[fact_id → relevance]
                  echo_penalty:  float
```

### Detection

When a new session opens, `check_echo(first_message)` computes:

```
similarity = max cosine(new_vector, centroid)   for each centroid in bundle
```

If `similarity ≥ 0.68`:
- Past session seemed resonant (`r_score > 0.35`): `penalty = -0.8`
- Past session was already weak: `penalty = -0.6`
- New `r_score = min(-0.2, max(-1.0, old_r + penalty))`

The penalty is **retroactive and idempotent**:

- `EchoStore` records `echo_penalty` per matched session and refuses to
  stack a second hit on the same session.
- `MemoryStore.check_echo` calls `apply_session_resonance(fact_weights, R')`
  with the *delta* between new and old `r_score`, so the past session's
  facts absorb the correction exactly once, weighted by how relevant each
  fact was to that session.

Facts that looked good because the user appeared satisfied now get a
negative signal scaled by their actual involvement.

---

## Black hole

Facts absorbed by the black hole are removed from the live index and from
the gravity engine. They are stored in `BlackHole._singularity` keyed by
`fact_id`, with their vectors mirrored into a private `VectorIndex` for
fast similarity scans.

### Hawking emission

```python
def hawking_emit(query_vector):
    # all_similarities returns {fid: cosine} for every fact in the index.
    sims = self._index.all_similarities(query_vector)
    to_emit = [fid for fid, sim in sims.items() if sim >= 0.95]
    emitted = []
    for fid in to_emit:
        rec = self._singularity.pop(fid)
        self._index.remove(fid)
        self._total_emissions += 1    # process-level counter, never decrements
        emitted.append(rec.fact)
    return emitted
```

Emitted facts leave the singularity, return to `kinetic` layer with
`gravity = 0.30`, are re-registered in the gravity engine, and are
persisted back to storage so a restart will not reanimate them. The
threshold 0.95 is intentionally high: Hawking emission should be rare
and only triggered by near-exact recall.

---

## Persistence

`MemoryStore` accepts any object satisfying the `StorageBackend` protocol:

```python
class StorageBackend(Protocol):
    def save_fact(self, fact: FactPassport) -> None: ...
    def save_facts(self, facts: list[FactPassport]) -> None: ...   # batched
    def delete_fact(self, fact_id: str) -> None: ...
    def load_facts(self) -> list[FactPassport]: ...
    def save_edge(self, from_id: str, to_id: str) -> None: ...
    def load_edges(self) -> list[tuple[str, str]]: ...
    def save_echo_session(
        self,
        session_id: str,
        centroids: list[list[float]],
        r_score: float,
        recorded_at: float,
        fact_weights: dict[str, float] | None = None,
        echo_penalty: float = 0.0,
    ) -> None: ...
    def load_echo_sessions(self) -> list[dict]: ...
    def close(self) -> None: ...
```

`save_facts` has a default implementation that loops `save_fact`; concrete
backends are encouraged to override it for batched commits. `SQLiteBackend`
does so via `executemany` inside a single transaction, which makes startup
re-hydration and large `tick()` cascades materially faster.

`SQLiteBackend` is the default. Write-through: every mutation hits the DB
immediately. On startup, all facts, edges, and echo session bundles are
loaded into memory and into a numpy `VectorIndex`. Retrieval is in-memory:
a single matrix–vector dot product against the index, plus a parallel scan
of the black hole's index for Hawking candidates. SQLite is only for
durability.

The `echo_sessions` table is migrated on startup — older rows that stored
`fact_ids` as a JSON list are read back transparently, and new rows store
the per-fact weight map as a JSON dict in the same column.

---

## Concurrency

`MemoryStore` is designed for several agents sharing one process.

```
MemoryStore
  _sessions: dict[session_id → SessionContext]
  _current_session_id: Optional[str]      # legacy convenience for single-agent use
  _lock: threading.RLock
```

A `SessionContext` carries one session's messages, vectors, accumulated
`fact_weights`, and the echo result returned at `session_start`. Every
mutating public method (`add_fact`, `session_message`, `session_close`,
`query`, `check_echo`, `link`, `deprecate`, `stats`) accepts an optional
`session_id` and acquires `_lock` for the critical section. Embedding HTTP
calls are made *outside* the lock so multiple agents don't serialize on
Ollama.

`_current_session_id` is still set by `session_start`/`session_close` so
single-threaded callers can omit `session_id` exactly as before. The MCP
server (`server.py`) always passes `session_id` explicitly, so any two
agents talking to the same server stay isolated.

For visibility, `MemoryStore.stats` includes `active_sessions`.

---

## Module map

```
src/birch/
  fact.py                   FactPassport dataclass
  gravity.py                GravityEngine — score computation + migration
  black_hole.py             BlackHole — sink + Hawking emission (numpy index)
  vector_index.py           VectorIndex — numpy L2-normalised cosine search
  memory_store.py           MemoryStore — unified API, per-session contexts, RLock
  server.py                 MCP server (FastMCP), threads session_id through
  storage/
    base.py                 StorageBackend protocol + save_facts batch hook
    sqlite.py               SQLiteBackend — executemany commits, schema migrations
  resonance/
    detector.py             compute_resonance() — combines all signals
    behavioral.py           pattern match on message closure
    semantic.py             cosine shift + specificity delta
    repetition.py           centroid dispersion
    echo.py                 EchoStore — cross-session echo, per-fact weight map
    centroid.py             centroid() + dispersion() utilities
    cluster.py              K-means++ ClusterBundle
    embeddings.py           Ollama batch (/api/embed) + single fallback
```

---

## Data flow

```
user message
    │
    ▼
embed()  ──────────────────────────────────────────────┐
    │                                                   │
    ▼                                                   │
session_message(session_id)                             │
  appends to ctx.messages                               │
  appends to ctx.vectors                                │
    │                                                   │
    ▼                                                   │
session_close(session_id)                               │
  compute_resonance(messages, vectors)                  │
    ├── score_behavioral(messages)                      │
    ├── score_semantic_shift(start_vec, end_vec)        │
    └── score_repetition(all_vecs)                      │
  → R score                                             │
    │                                                   │
    ├── apply_session_resonance(ctx.fact_weights, R)    │
    ├── EchoStore.record(session_id, vecs, R,           │
    │                    fact_weights=ctx.fact_weights) │
    ├── GravityEngine.tick()  → layer migrations        │
    └── _absorb_dead()        → black hole              │
                                                        │
query(text, session_id)                                 │
    │◄──────────────────────────────────────────────────┘
    ├── embed(text)
    ├── VectorIndex.search  over live facts
    ├── _attribute_fact(fid, similarity) into ctx       │
    └── BlackHole.hawking_emit(vec)  if similarity ≥ 0.95
            └── persists re-registered fact via save_fact
```
