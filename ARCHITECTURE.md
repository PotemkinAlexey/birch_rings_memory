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

The black hole holds two kinds of bodies. They are kept in separate
dicts and separate vector indices so Hawking emission stays typed and
ID collisions are impossible:

```
BlackHole
  _singularity      : dict[str, SingularityRecord]      # FactPassports
  _meta_singularity : dict[str, MetaSingularityRecord]  # MetaFacts
  _index            : VectorIndex                       # fact vectors
  _meta_index       : VectorIndex                       # meta vectors
```

### Hawking emission — facts

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

### Hawking emission — MetaFacts

A MetaFact centroid lives between its sources, so the same 0.95
threshold almost never fires. `hawking_emit_metas(query, threshold)`
accepts a looser bound (`MemoryStore` calls it with `0.85`), and the
emitted MetaFact's gravity is set by its own weight, not by a fixed
constant:

```
gravity_on_emission = base + 0.10 · log10(weight)   # base=0.30, cap=0.70
```

`weight=1` → `0.30`. `weight=10` → `0.40`. `weight=100` → `0.50`. A
bundle of a thousand dead facts ends up at `0.60` — dense, but never
straight into the surface layer (`0.70`).

---

## MetaFact

A `MetaFact` is the residue of a `SingularityCompactor.collapse()` —
several FactPassports collapsed into a single centroid body:

```
meta_id          full uuid4
vector           [768-dim float]   L2-normalised weighted center of mass
weight           int               number of facts absorbed
source_texts     list[str]         "subject predicate object" per original
source_fact_ids  list[str]         lineage to the absorbed FactPassports
summary          str               (optional, future LLM pass)
gravity_score    0.30 by default   bumped on emission via log(weight)
layer            -1 in singularity, 1 after Hawking, etc.
access_count, last_accessed
resonance_sum, resonance_count
```

The feedback-loop surface is exposed through the same attributes and
methods as `FactPassport` — `touch()`, `apply_resonance(r)`,
`avg_resonance`, `is_deprecated`, `is_expired`, `fact_id`. The
`GravityEngine`, `BlackHole`, and `MemoryStore.session_close()` treat
a MetaFact like any other body via duck typing.

A MetaFact whose gravity falls back below `0.10` is re-absorbed by the
singularity — `BlackHole.absorb_meta()` handles it, and a future
collapse may eventually merge it with another MetaFact (not yet
implemented; recursive collapse is a planned pass).

---

## Memory consolidation

`SingularityCompactor.collapse_singularity(hole, threshold, min_group_size)`
runs gravitational collapse over the singularity's FactPassports:

1. Snapshot every absorbed vector into a `(n, d)` numpy matrix.
2. `M @ M.T` gives the full pairwise cosine matrix in one matmul.
3. Path-compressing Union-Find groups every transitive cluster at
   `cosine ≥ threshold` (default `0.92`).
4. Each group with at least `min_group_size` members becomes a MetaFact
   whose vector is the L2-normalised weighted center of mass.
5. The originals are removed from `_singularity` and from `_index`; the
   new MetaFact is `absorb_meta()`-ed back into the same hole.

Pre-existing MetaFacts in the singularity are not touched (recursive
collapse needs its own threshold tuning). FactPassports with empty
vectors are skipped.

### Orchestration

`MemoryStore` triggers collapse opportunistically after `_absorb_dead()`
runs at session close:

```
if hole.fact_mass         >= COLLAPSE_FACT_MASS_TRIGGER (100)  AND
   counter-since-last     >= COLLAPSE_DELTA_TRIGGER     (50):
    submit collapse to single-worker ThreadPoolExecutor
```

A second trigger while one collapse is inflight is dropped (not queued).
`collapse_async=False` runs the pass inline for predictable tests or
deployments without background threads.

`MemoryStore.close()` snapshots the executor and any inflight future
under the lock then releases the lock before waiting on them — the
worker itself wants the same lock from inside `collapse_singularity`,
so blocking on it while holding it would deadlock the shutdown.

For visibility, `MemoryStore.stats` includes `collapse_counter`,
`total_collapses`, and `last_collapse_at`.

---

## Persistence

`MemoryStore` accepts any object satisfying the `StorageBackend` protocol:

```python
class StorageBackend(Protocol):
    # Facts
    def save_fact(self, fact: FactPassport) -> None: ...
    def save_facts(self, facts: list[FactPassport]) -> None: ...   # batched
    def delete_fact(self, fact_id: str) -> None: ...
    def load_facts(self) -> list[FactPassport]: ...
    # Edges (auto-link graph + manual link())
    def save_edge(self, from_id: str, to_id: str) -> None: ...
    def load_edges(self) -> list[tuple[str, str]]: ...
    # Echo sessions (closed sessions with K-means topic bundle)
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
    def delete_echo_session(self, session_id: str) -> None: ...
    # Open sessions (for crash recovery)
    def save_open_session(self, session_id, messages,
                          vectors, facts, started_at) -> None: ...
    def delete_open_session(self, session_id: str) -> None: ...
    def load_open_sessions(self) -> list[dict]: ...
    # MetaFacts (compressed bundles produced by collapse)
    def save_meta_fact(self, meta: MetaFact) -> None: ...
    def save_meta_facts(self, metas: list[MetaFact]) -> None: ...   # batched
    def delete_meta_fact(self, meta_id: str) -> None: ...
    def load_meta_facts(self) -> list[MetaFact]: ...
    def close(self) -> None: ...
```

`save_facts` and `save_meta_facts` have default implementations that
loop the singular variant; concrete backends are encouraged to override
them for batched commits. `SQLiteBackend` does so via `executemany`
inside a single transaction, which makes startup re-hydration and large
`tick()` cascades materially faster.

`SQLiteBackend` is the default. Write-through: every mutation hits the DB
immediately. On startup, facts go into the live store and the numpy
`VectorIndex`; MetaFacts at `layer == -1` go back into the black hole's
singularity, MetaFacts at `layer >= 0` go into the live MetaFact store.
Retrieval is in-memory: four matrix–vector dot products (live facts,
live metas, fact singularity, meta singularity), sorted and merged.
SQLite is only for durability.

Schema migrations happen on open: the `echo_sessions` table picks up
`fact_ids` and `echo_penalty` columns via `ALTER TABLE`, while the new
`meta_facts` table is added by the idempotent `CREATE TABLE IF NOT EXISTS`
in the bundled schema. Older rows that stored `fact_ids` as a JSON list
inside the echo session column are read back transparently — new rows
store the per-fact weight map as a JSON dict in the same column.

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
  meta_fact.py              MetaFact dataclass + lineage + Hawking gravity helper
  gravity.py                GravityEngine — score computation + migration
  black_hole.py             BlackHole — polymorphic sink (facts + metas) + Hawking
  singularity_compactor.py  collapse_singularity() — Union-Find + center of mass
  vector_index.py           VectorIndex — numpy L2-normalised cosine search
  memory_store.py           MemoryStore — unified API, sessions, RLock, collapse orchestration
  server.py                 MCP server (FastMCP), threads session_id through
  storage/
    base.py                 StorageBackend protocol + save_*_batch hooks
    sqlite.py               SQLiteBackend — executemany commits, schema migrations
  resonance/
    detector.py             compute_resonance() — combines all signals
    behavioral.py           pattern match on message closure
    semantic.py             cosine shift + specificity delta
    repetition.py           centroid dispersion
    echo.py                 EchoStore — cross-session echo, per-fact weights, TTL sweep
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
    │     ↳ also scales MetaFact resonance when         │
    │       ctx attributed any meta_id                  │
    ├── EchoStore.record(session_id, vecs, R,           │
    │                    fact_weights=ctx.fact_weights) │
    ├── EchoStore.expire()    → drop stale sessions     │
    ├── GravityEngine.tick()  → layer migrations        │
    ├── _absorb_dead()        → black hole (facts+metas)│
    └── _maybe_trigger_collapse_locked(absorbed_count)  │
          ↳ when fact_mass ≥ 100 AND counter ≥ 50:      │
            submit collapse_singularity() to executor   │
                                                        │
collapse_singularity(hole, threshold=0.92)              │
    ├── one matmul over hole._index → pairwise cosine   │
    ├── Union-Find groups at cosine ≥ threshold         │
    ├── for each group: build MetaFact (center of mass) │
    │     drop originals from _singularity + _index     │
    │     absorb_meta(new) into _meta_singularity       │
    └── save_meta_facts(new_metas) + delete_fact(...)   │
                                                        │
query(text, session_id)                                 │
    │◄──────────────────────────────────────────────────┘
    ├── embed(text)
    ├── VectorIndex.search over live facts            → QueryResult(fact=…)
    ├── _meta_index.search over live MetaFacts        → QueryResult(meta=…)
    ├── BlackHole.hawking_emit(vec)        sim ≥ 0.95 → re-register fact
    ├── BlackHole.hawking_emit_metas(vec)  sim ≥ 0.85 → re-register meta
    └── _attribute_to(ctx, body_id, similarity)
          ↳ touches FactPassport or MetaFact symmetrically
```
