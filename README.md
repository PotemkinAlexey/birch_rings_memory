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

**Confidence-damped gravity step.** R is also emitted with a **confidence** in
`[0, 1]` — how much the three signals agree *and* how broad the base is. It is
`agreement × corroboration`, where `agreement = |Σ contributions| / Σ|contributions|`
(1.0 when the signals pull the same way, → 0 as they cancel) and `corroboration`
rises with the participation ratio of the voting signals (a lone signal floors
at 0.75; a second balanced signal lifts it to 1.0). Gravity then moves by
`effective_r = R · confidence`, not raw R — so a session where behavioral reads
toxic on grumpy tech vocabulary while the semantic trajectory reads productive
(low agreement), or where one lone regex match carries the verdict (low
corroboration), barely nudges gravity instead of confidently mislabelling. This
keeps a noisy self-derived signal from compounding through the feedback loop.
Explicit caller signals (`sentiment` / `r_override`) carry confidence 1.0.
`session_close` reports `confidence` and `effective_r` alongside the raw `r`.

### Echo validation (deferred, outcome-gated)

Each closed session is stored as a **K-means bundle** of centroids (not a
single vector) — so multi-topic sessions don't lose sub-topic structure.

When a new session opens with a `first_message`, BirchKM checks: does this look
like returning to a past topic? If `similarity ≥ 0.68` to any centroid in a
past session, it **arms a pending echo marker** — but applies *nothing* yet.
Returning to a topic is not, by itself, evidence the past closure was false; the
evidence is whether *this* conversation also ends badly. So the decision waits
for `session_close`:

- this session ends **resonant** → the revisit was productive (continued use /
  a fix that stuck) → **cancel**, no penalty (tracked as `total_echoes_cancelled`);
- this session ends **neutral / toxic** → a genuine return-to-failure → **apply**
  the retroactive penalty to the matched past session's facts, scaled by this
  session's severity (a neutral return penalises less than a toxic one).

This replaces the old apply-on-open behaviour, which guessed "returned ⇒
unresolved" and penalised immediately — firing on continued use as often as on
real false closure. The penalty **magnitude** is itself evidence-proportional:
`base · clamp(1 − prior_r, 0, 1)`, so a revisit to a *strongly resonant* past
topic (ambiguous) is barely penalised while a weak/toxic prior takes the full
hit. There is no forced toxic floor — the score lands where the evidence puts
it. The one-shot `record_session` is also outcome-gated — it receives the whole
conversation up front, so it peeks at open and lets the close decide, just like
the streaming path. Only the explicit `check_echo` tool keeps immediate
(apply-now) semantics, for callers that deliberately want a detect-and-apply now.

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

**Contrastive attribution.** Cosine relevance is *topical*, not *causal*: a
genuinely useful fact retrieved at high similarity into a session that failed
for unrelated reasons would absorb a large negative hit just for being
on-topic. So each session's impulse is anchored on the fact's own resonance
history — its discriminative signal (does it ride resonant or toxic sessions on
net). A session whose outcome **contradicts** that established history is
attenuated in proportion to how established the fact is (`trust = n/(n+K)`,
`K = BIRCH_CONTRAST_K`, default 5): a new fact takes the full hit, a fact with a
long resonant track record resists a single incidental toxic session, and a
confirming session always applies in full so real shifts are still learned. The
trust decision reads a **separate, un-shrunk** track record (`raw_avg_resonance`),
not the gravity-side mean it has already shrunk — otherwise trust would feed on
its own past shrinking and freeze a fact's early reputation against later
decline. The raw mean is order-independent and flips sign the moment a fact
genuinely turns bad, at which point contradicting sessions stop being shrunk.
Symmetric — a consistently-toxic fact is not redeemed by one stray resonant
session — and bounded: it only ever shrinks a contradicting impulse, never
amplifies. The rule is **inert** on sign-consistent history (it changes nothing
when a fact's sessions agree), so it cannot distort the common case; `stats`
exposes `contrastive_attenuations` to show how often it actually fires.
Set `BIRCH_CONTRAST_K=0` to disable.

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

**Salience (irreplaceability) raises that floor for rare-but-critical facts.**
Gravity is mostly frequency-coupled (access, recency, utility), so a fact used
once a year would decay and be absorbed before its next use. The absorption
floor is therefore lowered for a fact that is *both* unique in its namespace
(no live neighbour at cosine ≥ `0.85`) *and* has proven useful
(`avg_resonance > 0`): `floor = 0.10 · (1 − protection · irreplaceability · value)`.
A once-a-year fact that was decisive each time and has no substitute is kept;
a redundant fact decays normally (the knowledge survives in its neighbours);
an unproven unique fact also decays (uniqueness alone isn't criticality — almost
everything is unique). Both factors are frequency-orthogonal — this is the
cost-of-loss signal the frequency terms can't see. `BIRCH_SALIENCE_PROTECTION=0`
disables it; `stats.salience_retained` counts what it kept.

That covers *proven* rare-critical facts. The **cold-start** case — critical but
not yet exercised (recorded today, decisive in 11 months) — has no outcome to
infer from, so it needs a **top-down** signal: `record_fact(salient=True)` pins
a fact (`encode_salience`), flooring it from the moment of writing regardless of
resonance. This is the only declared channel in an otherwise inferential system,
and it doesn't break the thesis — the thesis is "don't make the user *rate*
usefulness", and criticality-at-encoding is a different, un-inferrable signal
(the brain tags importance by attention at encoding too, not only by repetition).
It's kept honest: a pin **decays use-it-or-lose-it** (eroded only when the fact
surfaces into a non-positive session — a pin that keeps proving useless fades; a
truly dormant one is held), a per-namespace **budget** evicts the *highest-gravity*
pin under contention (the one needing it least — never the matured low-gravity
cold-start candidate), and **telemetry** (`pins_created / pins_active /
pins_resonated`, derived from persisted per-fact flags so it survives restarts)
lets a month of real traffic decide whether the channel earns its keep: if
pinned facts rarely go on to resonate, it's noise and should be cut.

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
Compactor** runs gravitational collapse over the singularity: bodies are
partitioned by vector dimension (so a model swap leaves old-dim and
new-dim bodies compacting independently rather than crashing on a
ragged numpy array), each dim-group gets its own numpy `matmul` for
pairwise cosine, path-compressing Union-Find groups every transitive
cluster above the collapse threshold (default `0.92`), and each group
becomes one `MetaFact`. The originals are removed from both
`_singularity` and the index; their texts and ids live on inside the
MetaFact (`source_texts`, `source_fact_ids`) for lineage.

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

Storage layout is a **preallocated buffer that grows geometrically** —
`add()` is amortised O(d) (write into the next free slot; double the
buffer when full) instead of O(n·d) per insert. `remove()` is O(d) via
swap-with-last (the public surface never promised insertion order; search
returns by score). After a mass-delete, the buffer auto-shrinks when
usage falls below `capacity / 4` so a long-running store does not sit on
peak allocation forever. On 10k facts × 768 dim the difference is a
~30 MB matrix copy per insert versus a single 3 KB overwrite.

### Boundary hardening

Every MCP tool input passes a typed validator family at the boundary:
`_validate_text` (length cap, non-empty), `_validate_spo_strings` (SPO
triple shape), `_validate_optional_id` (session_id type), `_validate_int`
/ `_validate_float` / `_validate_bool` (numeric / enum shapes). Failures
return structured `{"error": "...", "field": "...", "hint": "..."}`
responses instead of crashing inside core. Per-field length is capped at
`BIRCH_MAX_FIELD_LEN` chars (default 2000, tunable 128..200000) — defence
against an agent looping with a megabyte-scale paste accidentally paying
full embedding-provider cost and bloating SQLite rows.

Storage writes are symmetric defence-in-depth. `json.dumps(..., allow_nan
=False)` on every JSON cell (vectors, centroids, session payloads); every
numeric scalar passes a finite + clamp gate before persistence. NaN /
Infinity never reaches disk; the surrounding `_txn()` rolls back and
`_reload()` restores the in-memory snapshot on any write failure.

Public object methods — `apply_resonance`, `avg_resonance`,
`__post_init__` for `FactPassport` / `MetaFact` — self-defend against
NaN / Infinity inputs (library users can call them directly, bypassing
storage / engine gates). `compute_gravity` ends with a final
`math.isfinite` check before its `min/max` clamp; the cascade is
belt-and-suspenders, not a single point of failure.

### Indirect prompt-injection advisory

BirchKM stores data, not LLM instructions — but agents read retrieved
bodies straight into their context. `_sanitize_for_llm` strips ASCII C0
control codes (except TAB/LF/CR), DEL, and zero-width Unicode (ZWSP /
ZWNJ / ZWJ / BOM) at the write boundary so an "invisible bytes" payload
never reaches storage. Visible instruction markers (`<|im_start|>`,
`[INST]`, `<<SYS>>`, etc.) are NOT rewritten — aggressive replacement is
itself a content-filter bypass surface. Instead, `query_memory` attaches
a per-hit `has_instruction_markers` boolean and a top-level
`injection_warnings` list when retrieved bodies contain known markers,
so the consumer knows which results to wrap in structural delimiters
before feeding into downstream LLM context. The honest contract is
"consumer wrapping discipline is non-negotiable"; the advisory is the
safety net.

---

## Modules

| Module | Responsibility |
|---|---|
| `fact.py` | `FactPassport` — subject/predicate/object triple + gravity metadata (incl. `recent_utility`) |
| `meta_fact.py` | `MetaFact` — dense centroid bundle with lineage + feedback-loop fields |
| `adaptive_gravity.py` | `AdaptiveWeights` — five learned pre-resonance weights (incl. `w_stability` for `forecast_stability`), regularised SGD with budget renormalisation |
| `gravity.py` | `GravityEngine` — computes scores, triggers layer migration |
| `black_hole.py` | `BlackHole` — irreversible sink + Hawking emission (facts + metas) |
| `singularity_compactor.py` | `collapse_singularity()` — Union-Find collapse + center of mass |
| `vector_index.py` | `VectorIndex` — numpy-backed cosine search |
| `memory_store/` | `MemoryStore` package — split for navigability after it crossed 2500 LOC. Composition root `_base.py` plus five mixin files (`_sessions.py`, `_facts.py`, `_query.py`, `_singularity.py`, `_stats.py`) + `_models.py` (`QueryResult`, `SessionContext`) + `_embed_proxy.py` (late-binding embed lookup). Public import `from birch.memory_store import MemoryStore` unchanged |
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

Claude then has nineteen tools:

| Tool | What it does |
|---|---|
| `query_memory` | Semantic search — returns facts and MetaFacts ranked by similarity, plus `conflicts` for any (subject, predicate) with multiple competing values |
| `record_fact` | Store one SPO triple; response includes `similar_existing` paraphrase hints. Use when several `object`s can coexist on the same (subject, predicate) |
| `record_facts` | Store many triples in one batch (one Ollama round-trip) |
| `set_fact` | Slot-replace upsert: writes the new fact AND auto-supersedes any live fact sharing `(subject, predicate)`. Use for HEAD / version / single-valued scalars |
| `find_similar` | Read-only paraphrase search — surface candidates before writing or for planning `set_fact` / `supersede_fact` cleanup |
| `supersede_fact` | Mark `old_id` superseded by `new_id` — old body goes to the singularity with `deprecated_by` set; lineage preserved, MetaFact / Hawking still possible |
| `retire_fact` | Send a no-longer-relevant fact to the singularity (no replacement) — `ttl=now`, same singularity benefits as supersede |
| `delete_fact` | Legacy hard-delete — handles ONLY live FactPassports. Kept for backward compat; prefer `delete_body` for polymorphic ids from `query_memory` |
| `delete_body` | Polymorphic hard-delete — handles live FactPassports, live MetaFacts, singularity FactPassports, and singularity MetaFacts under a single `body_id` (the kind `query_memory` returns). Same destructive contract as `delete_fact` |
| `list_facts` | List live facts with filters (`subject_prefix`, `min_gravity`, `layer`, `exclude_deprecated`); sorted by gravity — audit without a query |
| `explain_fact` | Decompose a body's gravity into per-feature contributions — debug "why is this gravity so low". Polymorphic: handles live FactPassports, live MetaFacts, and both singularity kinds (the four locations `query_memory` can return) |
| `explain_body` | Polymorphic alias for `explain_fact` with the body-named contract — use when you got a `body_id` from `query_memory` |
| `session_open` | Open a named session so reads and writes can be attributed to it |
| `session_push` | Append a user message to an open session |
| `session_close` | Close a session — score resonance, update gravity, detect echo. Optional `sentiment` / `r_override` to declare R when the heuristic would misclassify (e.g. declarative technical summaries) |
| `check_echo` | Explicit **apply-now** cross-session topic match; on a hit, retroactive penalty propagates to the past session's R and the gravity of every fact it touched. `session_open(first_message=...)` and `record_session` do NOT call this — they peek a deferred marker and let `session_close` decide by outcome; use `check_echo` only when you want detect-and-apply immediately |
| `record_session` | Score a completed session in one call (open + push messages + close) |
| `forecast_memory` | Run the galaxy forward and write a per-body stability prediction back into the live store (covers FactPassports + MetaFacts); feeds the adaptive gravity formula via `w_stability` |
| `memory_stats` | Report layer distribution, black hole status, live adaptive weights, echo counters, thresholds |

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

- Python 3.11+
- [Ollama](https://ollama.com) with `nomic-embed-text`
- `mcp[cli]` and `numpy>=1.21` (installed automatically)

Environment knobs:

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_URL` | `http://localhost:11434` | Embedding provider endpoint |
| `BIRCH_EMBED_MODEL` | `nomic-embed-text` | Embedding model name (keys the vector cache) |
| `BIRCH_EMBED_PROVIDER` | `ollama` | `ollama` or `mock` (offline / CI) |
| `BIRCH_DB` | unset | SQLite path consumed by `python -m birch.server` |
| `BIRCH_MAX_FIELD_LEN` | `2000` | Per-field text cap at the MCP boundary (DoS/billing defence) |
| `BIRCH_RECORD_FACTS_BATCH_CAP` | `500` | Max items per `record_facts` batch |
| `BIRCH_EMBED_RETRIES` | `2` | Embedding HTTP retry count on transient failures |
| `BIRCH_EMBED_RETRY_BACKOFF_S` | `0.5` | Initial backoff between embedding retries |
| `BIRCH_HAWKING_FACT` / `BIRCH_HAWKING_META` / `BIRCH_ABSORPTION` / `BIRCH_AUTO_LINK` / `BIRCH_COLLAPSE` / `BIRCH_ECHO` / `BIRCH_FIND_SIMILAR_DEFAULT` | per-threshold default | Cosine thresholds — pin to your embedding model's distribution. See `birch/thresholds.py` |

`memory_stats.thresholds` echoes every threshold the process actually
picked up, so a misconfigured env knob is visible at runtime.

---

## Status

Working proof of concept. All of the following are functional and covered
by the test suite (741 passing, 20 skipped): resonance pipeline,
**adaptive gravity engine** (pre-resonance weights learned from session
resonance — no hand-set magic numbers), black hole with polymorphic
singularity and atomic absorption (facts + MetaFacts; mixed-dim safe),
SQLite persistence with tolerant loaders + `allow_nan=False` write
defence, **preallocated numpy vector index** (amortised O(d) add via
geometric growth + swap-with-last remove), per-session concurrency with
closing-session race protection, **cross-process safety** (WAL +
`data_version` cache invalidation + rollback-recovery on every write
path), auto-linking on `add_fact`, counter-triggered background collapse
with lineage, EchoStore TTL, the MCP server with full input-validator
family + per-field length caps + invisible-character strip + prompt-
injection advisory on retrieval, and the `galaxy` N-body research
model. CI runs ruff and mypy on every push.

Next natural steps: a persistent similarity index for very large stores
(FAISS / hnswlib), an LLM-driven `MetaFact.summary` writer that runs async
after collapse, and recursive collapse (MetaFacts colliding with MetaFacts).

## See also

**[MemoryBricks](https://github.com/PotemkinAlexey/memorybricks)** —
umbrella architecture combining BirchKM (this repo, the *dynamics*
layer) with [Vertical Brain](https://github.com/PotemkinAlexey/vertical-brain-for-ai)
(governance layer) under a unified MCP surface. Pattern is documented
in [`docs/STRUCTURED_LIVING_MEMORY.md`](docs/STRUCTURED_LIVING_MEMORY.md).

To talk to both backends from one MCP client, install
[`memorybricks-mcp`](https://github.com/PotemkinAlexey/memorybricks/tree/main/packages/memorybricks-mcp) —
it spawns `birch-mcp` and the VB server as children and exposes
`recall` / `remember` / `forget` that route between facts and chunks
in one call. The native Birch tools above stay available alongside.

## License

[PolyForm Noncommercial License 1.0.0](LICENSE) — free to use, modify, and
distribute for **noncommercial** purposes (personal, research, education,
nonprofit, government). **Commercial use is not granted** by this license;
contact the author for a separate commercial license.

Copyright (c) 2026 Alexey Potemkin.
