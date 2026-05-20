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
fact_id       uuid[:8]
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

---

## Echo validation

Each closed session is stored as a **K-means++ bundle** of K centroids rather
than a single centroid. This preserves sub-topic structure — a session that
covered both "auth" and "deployment" won't miss an echo on either sub-topic.

```
EchoStore
  session_id → StoredSession
                  bundle: ClusterBundle   (K centroids, cosine K-means++)
                  r_score: float          (retroactively mutable)
                  echo_penalty: float
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

The penalty is **retroactive** — the past session's gravity contribution is
revised downward. Facts that looked good because the user appeared satisfied
now get a negative signal.

---

## Black hole

Facts absorbed by the black hole are removed from the live index and from
the gravity engine. They are stored in `BlackHole._absorbed` by fact vector.

### Hawking emission

```python
def hawking_emit(query_vector):
    return [f for f in _absorbed if cosine(query_vector, f.vector) >= 0.95]
```

Emitted facts return to `kinetic` layer with `gravity = 0.30` and are
re-registered in the gravity engine. The threshold 0.95 is intentionally high:
Hawking emission should be rare and only triggered by near-exact recall.

---

## Persistence

`MemoryStore` accepts any object satisfying the `StorageBackend` protocol:

```python
class StorageBackend(Protocol):
    def save_fact(self, fact: FactPassport) -> None: ...
    def delete_fact(self, fact_id: str) -> None: ...
    def load_facts(self) -> list[FactPassport]: ...
    def save_edge(self, from_id: str, to_id: str) -> None: ...
    def load_edges(self) -> list[tuple[str, str]]: ...
    def save_echo_session(self, session_id, centroids, r_score, recorded_at) -> None: ...
    def load_echo_sessions(self) -> list[dict]: ...
    def close(self) -> None: ...
```

`SQLiteBackend` is the default. Write-through: every mutation hits the DB
immediately. On startup, all facts, edges, and echo session bundles are loaded
into memory. Retrieval is always in-memory (linear cosine scan); SQLite is
only for durability.

---

## Module map

```
src/birch/
  fact.py                   FactPassport dataclass
  gravity.py                GravityEngine — score computation + migration
  black_hole.py             BlackHole — sink + Hawking emission
  memory_store.py           MemoryStore — unified API
  server.py                 MCP server (FastMCP)
  storage/
    base.py                 StorageBackend protocol
    sqlite.py               SQLiteBackend
  resonance/
    detector.py             compute_resonance() — combines all signals
    behavioral.py           pattern match on message closure
    semantic.py             cosine shift + specificity delta
    repetition.py           centroid dispersion
    echo.py                 EchoStore — cross-session echo detection
    centroid.py             centroid() + dispersion() utilities
    cluster.py              K-means++ ClusterBundle
    embeddings.py           Ollama nomic-embed-text client
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
session_message()                                       │
  appends to _session_messages                          │
  appends to _session_vectors                           │
    │                                                   │
    ▼                                                   │
session_close()                                         │
  compute_resonance(messages, vectors)                  │
    ├── score_behavioral(messages)                      │
    ├── score_semantic_shift(start_vec, end_vec)        │
    └── score_repetition(all_vecs)                      │
  → R score                                             │
    │                                                   │
    ├── apply_session_resonance(fact_ids, R)            │
    ├── EchoStore.record(session_id, vecs, R)           │
    ├── GravityEngine.tick()  → layer migrations        │
    └── _absorb_dead()        → black hole              │
                                                        │
query(text)                                             │
    │◄──────────────────────────────────────────────────┘
    ├── embed(text)
    ├── cosine scan over live facts (by layer)
    └── BlackHole.hawking_emit(vec)  if similarity ≥ 0.95
```
