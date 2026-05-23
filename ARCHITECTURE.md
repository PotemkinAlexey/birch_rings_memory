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

Gravity is recomputed every `tick()` from six components:

```
gravity = w_freshness × freshness            (learned; prior 0.28; ~2-week half-life)
        + w_access    × access               (learned; prior 0.14; log-scaled, ~3-day decay)
        + w_graph     × graph                (learned; prior 0.09; degree / max-degree)
        + w_utility   × recent_utility       (learned; prior 0.09; EWMA of closure-weighted R)
        + w_stability × forecast_stability   (learned; prior 0.05; galaxy forward forecast)
        + 0.35        × resonance            (fixed: observation, not prediction)
```

### freshness

```
freshness = exp(−age_hours × ln2 / 336)        # age since created_at
```

A new fact is presumed relevant and rides high; it sinks as it ages
untouched. This is the **grace period** — a fact is not archived to the
cold core before it has had a chance to prove itself — expressed as a
smooth decay term rather than a hard cliff.

### access

```
access = log1p(access_count) / log1p(100)
       × exp(−idle_hours × ln2 / 72)            # idle since last_accessed
```

Logarithmic so one hot fact doesn't dominate; the ~3-day half-life lets an
un-revisited fact shed its access boost.

### resonance

```
resonance = (avg_resonance + 1.0) / 2.0    if resonance_count > 0
          = 0.0                            otherwise
avg_resonance = resonance_sum / resonance_count
```

Normalized from `[-1, +1]` to `[0, 1]`. Crucially it contributes **0** until
a session has actually scored the fact — an un-resonated fact is not propped
up by a neutral 0.5 baseline, which would otherwise hold junk above the
black-hole floor forever.

### graph

```
graph = degree(fact) / max_degree_in_graph
```

Facts with more `link()` connections are harder to sink. Encodes the
intuition that well-connected knowledge is more structural.

### recent_utility

```
recent_utility[t] = (1 − α) · recent_utility[t−1]  +  α · target
target            = clamp((R · attribution_weight + 1) / 2, 0, 1)
α                 = 0.15      # ≈ 7 sessions to half-life
default           = 0.5       # Bayesian neutral prior
```

Per-fact EWMA of closure-weighted resonance, updated at every
`session_close` for every fact the session touched (with the cosine
attribution weight as the multiplier). It captures **how recent sessions
that used this fact tended to end**, independent of how often the fact
was touched — `access` already covers that. A fact silently riding along
positive sessions climbs above the 0.5 prior and gets a small boost;
a fact that keeps appearing in toxic sessions sinks below 0.5 and gets
pulled toward the black hole even if its raw access count is high.

The 0.5 default is the soft floor: an untouched fact carries a neutral
prior, so it contributes `w_utility · 0.5` to gravity from day one and
does not need a session to "prove itself" before it stops looking like
junk.

### forecast_stability

```
forecast_stability = clip((radius_after_N_ticks − horizon) / (r_surface − horizon), 0, 1)
default            = 0.5      # no forecast run yet — neutral prior
absorbed           = 0.0      # crossed the horizon during the forecast
```

The N-body galaxy is built from current facts via the shared loader,
advanced `horizon_ticks` integrator steps (default 50), and the
finishing radius of every body becomes its stability score. Bodies that
crossed the event horizon during the run get 0.0; survivors get a
linear interpolation between the horizon and the surface ring.

It is the only adaptive feature that consults a fact's *future*: a
fact whose current local features (freshness, access, graph) look fine
but whose orbital trajectory will fall in 30 ticks gets a low stability
and is pulled toward the black hole *before* the local features
register the trouble. Conversely, a fact spiralling outward into a
stable orbit because of resonance kicks gets a high stability even if
it is still numerically near the core.

Updated explicitly via `MemoryStore.run_forecast(horizon_ticks)`
(exposed as the `forecast_memory` MCP tool), NOT on every
`session_close` — the simulation is O(n²·steps) and is meant to be a
periodic batch job, not a per-write hook. Each forecast pass writes
back to every live FactPassport's `forecast_stability` and persists the
update.

### Positive context is emergent, not assigned

We deliberately do not ask the user to label facts ("👍 / 👎", "scope =
work", "polarity = positive"). The system reads it instead:

- **Closure** is the user's natural endpoint — "thanks, exactly that"
  vs. "опять не работает". It already arrives, free, at the end of
  every session.
- **Resonance** turns that closure into a per-session R ∈ [−1, +1] from
  behavioural / semantic / repetition signals, with no LLM call.
- **`recent_utility`** is the per-fact EWMA of that signal — the slow,
  ambient memory of "has this fact tended to be in conversations that
  ended well *for me*".

The user never marks anything; the gravity formula learns the
attribution. Magic numbers that needed personalisation become weights
the system fits to the user, and the only "labels" are the ones the
user produced organically by closing a session.

### Adaptive weights — the formula learns

The five pre-resonance weights (freshness, access, graph, utility,
stability) are not hand-set magic numbers any more. They live in
``AdaptiveWeights`` and are learned from the user's own resonance
feedback:

- Behaviour at zero data is identical to the prior
  `(0.28, 0.14, 0.09, 0.09, 0.05)`, so flipping the switch is safe.
- Each `session_close` snapshots
  `(freshness, access, graph, recent_utility, forecast_stability)` for
  every fact about to receive its *first* resonance, averages those,
  and takes one regularised SGD step toward `(R + 1) / 2`. The
  non-circularity is the point: the weights learn what predicts
  realised value *before* a fact has been reacted to this session.
- A regularisation term pulls each weight back toward the prior every
  step; a budget renormalisation keeps the five learned weights summing
  to `0.65` so the formula stays in `[0, 1]`.
- The resonance weight stays fixed at `0.35`: resonance is observation,
  not prediction.
- A feature that turns out not to predict realised value just has its
  weight stay near the prior. `w_stability` only grows if galaxy
  forecasts actually correlate with session outcomes for this user; a
  useless feature gets weighted near zero, no manual tuning needed.

Weights persist in a singleton SQLite row and round-trip via the
`StorageBackend` protocol's `save_adaptive_weights` / `load_adaptive_weights`.
`memory_stats` returns them, so the user can read `freshness 0.36,
access 0.14, graph 0.07, utility 0.08 — trained on 47 sessions` and
see exactly what the formula has learned. Magic numbers that needed personalisation are
fit to the user; the only remaining constants are the learner's `lr`
and `reg` — standard, robust hyperparameters with safe defaults.

### Layer migration (per tick)

Each tick a fact steps **one layer toward the layer its gravity belongs in**:

```
target = surface  if gravity > 0.70
       = core      if gravity < 0.30
       = kinetic   otherwise
layer moves one step toward target
gravity < 0.10  →  absorbed by the black hole (layer = -1)
```

The one-step cap keeps movement gradual. Migrating toward the band — not
only on the extremes — means a fact stranded in the core climbs back out
once its gravity recovers into the kinetic range, instead of being trapped
there because promotion once required clearing 0.70 outright.

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

**Idempotency is per-stored-session, not per-recurrence.** A given
StoredSession can receive an echo penalty at most once. If the user
returns to the same unresolved topic a third time, the third visit will
not deliver additional penalty against that StoredSession — the cap
prevents penalty stacking against any single record. The matching
mechanism is still active, so a separate echo against a different past
session that touched the same problem can still fire. This is a design
choice biased toward safety over progressive punishment: better to
under-penalise than to compound a single misleading session into a
gravity dead zone. If a future use case needs progressive echo
correction, the natural extension is per-session `echo_count` +
`last_echo_at` with a decay rule, not removing the cap.

`detect_echo(exclude_session_id=...)` accepts an explicit skip — used
when `check_echo` is called against an open session that already has its
own StoredSession (would otherwise match itself).

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

### Intake — three ways a body enters the singularity

`_absorb_dead()` is the single chokepoint; it is called from
`session_close` and immediately by the explicit retirement operations.
A body crosses the event horizon when **one** of these is true:

| Trigger | Set by | Intent |
|---|---|---|
| `gravity_score < 0.10` | natural decay (the `tick()` formula) | the body proved itself unhelpful |
| `is_deprecated` (`deprecated_by` is set) | `supersede_fact(old, new)` | newer fact replaces this one |
| `is_expired` (`ttl <= now()`) | `retire_fact(fact_id)` | topic is over, no replacement |

`MemoryStore.delete_fact` is **not** an intake — it removes the row
from storage entirely, bypassing the singularity, losing the body to
both MetaFact compression and Hawking emission. It is reserved for
hard removal (secrets, accidental writes); the canonical agent-facing
paths for stale data are `supersede_fact` and `retire_fact`.

A symmetry property follows: a body that decays naturally and a body
explicitly retired end up in the same place with the same affordances —
they can be fused into a MetaFact, and they can be Hawking-emitted by a
sufficiently close future query. Retirement is a fast path to a state
gravity decay would reach on its own; it is not a separate fate.

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
    # Cross-process coordination (optional — getattr-probed)
    def data_version(self) -> int: ...
    @contextmanager
    def transaction(self) -> Iterator[None]: ...
    # Adaptive gravity weights (optional — getattr-probed)
    def save_adaptive_weights(self, weights: AdaptiveWeights) -> None: ...
    def load_adaptive_weights(self) -> AdaptiveWeights | None: ...
    def close(self) -> None: ...
```

`data_version` and `transaction` are the cross-process safety seam (see
*Concurrency → Across processes* below); a backend without them runs
single-process only and `MemoryStore` skips the reload-on-other-process
machinery via `getattr`. `save_adaptive_weights` /
`load_adaptive_weights` persist the singleton row of learned
pre-resonance weights; a backend without them falls back to the prior
on every restart, which is safe but loses the personalisation.

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

### Within a process

`MemoryStore` supports several agents sharing one process.

```
MemoryStore
  _sessions: dict[session_id → SessionContext]
  _current_session_id: Optional[str]      # legacy convenience for single-agent use
  _lock: threading.RLock
```

A `SessionContext` carries one session's messages, vectors, accumulated
`fact_weights`, and the echo result returned at `session_start`. Every
mutating public method (`add_fact`, `session_message`, `session_close`,
`query`, `check_echo`, `link`, `deprecate`, `supersede_fact`,
`retire_fact`, `delete_fact`, `stats`) accepts an optional `session_id`
where applicable and acquires `_lock` for the critical section. Embedding
HTTP calls are made *outside* the lock so multiple agents don't serialize
on Ollama.

`_current_session_id` is still set by `session_start`/`session_close` so
single-threaded callers can omit `session_id` exactly as before. The MCP
server (`server.py`) always passes `session_id` explicitly, so any two
agents talking to the same server stay isolated.

For visibility, `MemoryStore.stats` includes `active_sessions`.

### Across processes

Each MCP client spawns its own `birch.server`, so several `MemoryStore`
instances share one SQLite file. The in-memory state is therefore a
**cache, not the source of truth**:

- `SQLiteBackend` runs in WAL mode and exposes `data_version()` —
  SQLite's counter of commits made by *other* connections.
- At the start of every operation `MemoryStore` checks `data_version`;
  if another process has written, it reloads every cache from disk
  before proceeding.
- Every write runs inside a reentrant exclusive transaction
  (`BEGIN IMMEDIATE`), so reload + mutate + persist cannot interleave
  with another writer.
- A single active process never sees `data_version` move, so it never
  reloads — the common case stays hot.

This is what stops a process with a stale cache from clobbering another
process's gravity ticks — the failure mode a write-behind in-memory
store hits as soon as a second client connects.

---

## Galaxy — the N-body research model (and a feature producer)

`birch/galaxy/` started as a research model that sits *beside* the live
engine, not inside it — the MCP server still scores facts with
`compute_gravity`. It still does that, but it now also produces a
feature that the live formula consumes: `forecast_stability`. The
galaxy makes the metaphor literal: instead of a scoring formula, facts
are bodies in orbit and the physics is simulated directly.

- **engine.py** — a 2D N-body integrator: a central black hole, leapfrog
  integration, softened gravity, dynamical friction. A body's orbital
  radius is its ring (far = surface, near = core); crossing the event
  horizon is absorption.
- **loader.py / projection.py** — turn facts into bodies. A shared PCA
  basis (`Projector`) sets each body's angle from its embedding;
  freshness and earned value set its starting orbit and mass.
- **replay.py** — runs the galaxy along the store's real timeline: facts
  are born at their `created_at`, closed sessions become orbital kicks
  (resonance is thrust), and the current topic places a moving attention
  mass.
- **collapse.py** — friends-of-friends finds clumps; a bound, sub-virial
  group (`2·KE < |PE|`) collapses into a MetaFact — Jeans instability in
  place of a cosine Union-Find.
- **report.py** — reads the settled galaxy back as a diagnosis of the
  store: facts at forgetting-risk, emergent topic clusters, MetaFact
  candidates.
- **render.py** — matplotlib stills and animated GIFs.
- **forecast.py** — the producer arm. Builds the galaxy from current
  facts, advances `horizon_ticks` steps, and reports per-fact stability
  in `[0, 1]`. Wired into the live formula via the 5th adaptive feature
  `forecast_stability`; triggered by `MemoryStore.run_forecast` (MCP
  tool `forecast_memory`).

Run `python -m birch.galaxy` to replay and diagnose the real store. One
finding from the model: an off-centre attractor in a black-hole-dominated
disk cannot *gather* facts into a standing clump — there is no stable
off-centre point — so the attention mass ships as a gentle perturber.

---

## Module map

```
src/birch/
  adaptive_gravity.py       AdaptiveWeights — four learned pre-resonance weights, regularised SGD
  fact.py                   FactPassport dataclass (incl. recent_utility EWMA)
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
  galaxy/                   N-body research model (beside the live engine)
    engine.py               Galaxy — 2D N-body integrator, rings, absorption
    loader.py               build a Galaxy from facts
    projection.py           Projector — shared 2D PCA basis
    replay.py               replay the store's history as births + kicks
    collapse.py             friends-of-friends + Jeans collapse into MetaFacts
    report.py               diagnose the settled galaxy
    render.py               matplotlib stills and animated GIFs
    forecast.py             feature producer: per-fact stability for the adaptive formula
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
