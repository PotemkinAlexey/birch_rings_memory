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

### Confidence and the damped step

`compute_resonance` also returns a `confidence` ∈ `[0, 1]`:

```
agreement     = |Σ contributions| / Σ|contributions|     # signs pull together?
participation = 1 / Σ pᵢ²   (pᵢ = |cᵢ| / Σ|c|)            # how many signals vote?
corroboration = min(1, 0.75 + 0.25 · (participation − 1)) # lone signal → 0.75, two → 1.0
confidence    = agreement × corroboration
```

`agreement` falls when signals conflict (behavioral toxic vs semantic
productive); `corroboration` falls when a single signal carries the verdict
with the others silent (a lone signal trivially "agrees" with itself, so
`agreement` alone can't see it). Gravity is then moved by
`effective_r = R · confidence`, not raw `R` — a conflicted or single-signal
session barely nudges gravity, keeping a noisy self-derived signal from
compounding through the loop. `effective_r` is what flows into
`apply_session_resonance`, the `recent_utility` EWMA, the weight-training
target, and the echo prior; raw `r` / `label` are reported unchanged.
`sentiment` / `r_override` closes carry `confidence = 1.0` (explicit signal).

R (as `effective_r`) is propagated to all facts accessed during the session via
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

### Contrastive attribution (outlier-robust)

Cosine weight is *topical* relevance, not *causal* responsibility: a useful
fact retrieved at high similarity into a session that failed for unrelated
reasons would absorb a large negative `R × w`. Before applying the impulse,
`contrastive_impulse(fact, R, w)` anchors it on the fact's own history:

```
base = R × w
if resonance_count == 0 or R == 0:        return base      # no history → full
if sign(raw_avg_resonance) == sign(R):    return base      # confirms history → full
consistency = min(1, |raw_avg_resonance| / 0.35)           # how one-signed
trust       = (resonance_count / (resonance_count + K)) · consistency
return base × (1 − trust)                                  # contradicts → shrunk
```

Armor scales by **both** tenure and consistency. ``trust = n/(n+K)`` alone
would armor a long-but-mushy history (``|raw_avg| ≈ 0``, weak evidence of the
true sign) as hard as a strongly one-signed one — but a contradicting session
against a near-zero history is more likely the fact's real mixed nature than an
outlier, so it stays responsive (full armor only once ``|raw_avg|`` reaches the
resonant band, 0.35). ``K = BIRCH_CONTRAST_K`` (default 5).

**The prior is read from a separate, un-shrunk accumulator.** Each body keeps
*two* running sums over the same `resonance_count` sessions:

```
raw_resonance_sum   += R × w                 # true track record (never shrunk)
resonance_sum       += contrastive_impulse   # gravity input (shrunk)
avg_resonance     = resonance_sum     / resonance_count   # → gravity
raw_avg_resonance = raw_resonance_sum / resonance_count   # → trust decision
```

The shrink decision reads `raw_avg_resonance`, **not** `avg_resonance`. Reading
the gravity-side mean would make trust depend on impulses the rule itself
already shrank — a self-reference that turns the rule into an order-dependent
rich-get-richer attractor: a fact that became "established good" early would
protect its reputation using a trust score computed from that very protected
history, so late toxic sessions get shrunk and its real decline is masked. The
raw mean is order-independent and never shaped by past shrink decisions, so it
flips sign the moment a fact genuinely turns bad — at which point contradicting
sessions stop being shrunk and land in full. (Both sums round-trip through
storage; `_migrate_raw_resonance` backfills `raw := resonance_sum` for pre-fix
rows, which is exact since they were never shrunk.)

A new fact takes the full hit; a fact with a long resonant track record resists
a single incidental toxic session (and a consistently-toxic fact is not redeemed
by one stray resonant session — symmetric). Bounded: it only ever shrinks a
*contradicting* impulse, never amplifies, and is **inert** when a fact's session
signs are consistent (so it leaves the common case untouched — the drift
detector's utility correlation is bit-identical with it on and off). The engine
counts `contrastive_attenuations`; `K ≤ 0` disables it. This is the
outlier-robust increment of contrastive attribution; the fuller
population-baseline discriminative direction (down-weighting facts present in
every session equally) is future work.

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

### Detection — deferred and outcome-gated

Echo is split into a read-only **peek** at open and an **apply** at close.
`similarity = max cosine(new_vector, centroid)` over the matched bundle.

```
session_open(first_message):                       # peek_echo — read-only
    if similarity ≥ 0.68: arm pending marker on the session context
    apply nothing, mutate no gravity

session_close:                                     # decide on this session's outcome
    if no pending marker:        echo_outcome = none
    elif label == resonant:      cancel    (productive revisit) ; cancelled++
    else (neutral / toxic):      apply_echo(matched, scale = severity)
```

Returning to a topic is not by itself evidence of false closure — the evidence
is whether *this* session also failed. So nothing is applied until close, gated
on the current outcome. The penalty magnitude is evidence-proportional and
continuous:

```
base    = 0.6 + 0.2 · clamp(prior_r, 0, 1)         # 0.6 → 0.8, no step at 0.35
penalty = −base · clamp(1 − prior_r, 0, 1) · scale
severity = clamp((0.35 − effective_r) / 1.35, 0, 1)  # neutral return < toxic return
new r_score = clamp(old_r + penalty, −1, 1)          # no forced toxic floor
```

A revisit to a strongly-resonant prior is barely penalised (ambiguous — likely
continued use); a weak/toxic prior takes the full hit. There is **no** forced
toxic floor (the old `min(-0.2, …)` is gone) and no step at `prior_r = 0.35`.

The penalty is **retroactive and idempotent**:

- `EchoStore` records `echo_penalty` per matched session and refuses to
  stack a second hit on the same session.
- `apply_echo` mutates the matched session's `r_score`; `MemoryStore` then
  calls `apply_session_resonance(fact_weights, penalty)` so the past session's
  facts absorb the correction once, weighted by relevance (and passing through
  the same contrastive-attribution guard as any other impulse).

`record_session` is outcome-gated like the streaming path — it receives the
whole conversation up front, so it `peek_echo`s at open and lets the close
decide (resonant ⇒ cancel, else ⇒ apply). Only the explicit `check_echo` MCP
tool keeps the **immediate** peek+apply path (`detect_echo`), for callers that
deliberately want detect-and-apply now. Facts that looked good because the user
appeared satisfied, but whose topic genuinely came back unresolved, get a
negative signal scaled by their
actual involvement.

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
| `gravity_score < floor(fact)` | natural decay (the `tick()` formula) | the body proved itself unhelpful |
| `is_deprecated` (`deprecated_by` is set) | `supersede_fact(old, new)` | newer fact replaces this one |
| `is_expired` (`ttl <= now()`) | `retire_fact(fact_id)` | topic is over, no replacement |

The decay floor is normally `0.10`, but **salience (irreplaceability)** lowers
it for a fact that is *both* unique in its namespace and proven useful:

```
irreplaceability = 1 / (1 + same-namespace live neighbours at cosine ≥ SALIENCE_NEIGHBOR)
earned           = irreplaceability · clamp(avg_resonance, 0, 1)   # bottom-up, needs history
salience         = max(encode_salience, earned)                   # declared OR earned
floor(fact)      = ABSORPTION · (1 − SALIENCE_PROTECTION · salience)
```

Gravity is frequency-coupled (access, recency, utility), so a rare-but-critical
fact — used once a year, decisive each time, no substitute — would otherwise
decay below `0.10` and be lost before its next use. Salience is the
frequency-orthogonal cost-of-loss counter-signal: both factors are means /
neighbourhood properties, frozen on disuse. Uniqueness alone is deliberately
NOT enough (almost every fact is unique → absorption would halt and junk would
accumulate); coupling to proven value targets the genuinely critical. The
`is_deprecated` / `is_expired` lifecycle exits ignore salience — a superseded
fact is replaced regardless. `SALIENCE_PROTECTION=0` reverts to the flat floor.

**`encode_salience` is the top-down half** — the only declared signal in an
otherwise inferential system. `earned` salience needs an outcome, so it can't
protect a critical-but-never-yet-exercised fact (the cold-start case); that fact
is un-inferrable by construction and needs `record_fact(salient=True)`, which
sets `encode_salience = 1.0` and floors the fact from the moment of writing. The
thesis survives because the thesis is "don't make the user *rate* usefulness",
and criticality-at-encoding is an orthogonal, un-inferrable signal — not a
retrospective rating. Three disciplines keep the seam thin so it can't swallow
the bottom-up spirit:

- **Use-it-or-lose-it decay** (`session_close`): a pin loses `SALIENCE_DECAY ·
  confidence` only when its fact surfaced into a non-positive session. Usage-
  keyed, not wall-clock: a pin that keeps proving useless fades; a dormant one
  is held (it never got its chance); confidence-scaled so one noisy miss barely
  touches it. Decay never opens a transfer valley — a resonant surfacing leaves
  the pin intact while `earned` rises to take over.
- **Per-namespace budget** (`BIRCH_SALIENCE_PIN_BUDGET`, default 32): the only
  backstop against never-surfaced junk pins (indistinguishable from never-
  surfaced critical ones). Under contention it evicts the **highest-gravity**
  pin — the one needing protection least — which is anti-adversarial: a matured
  cold-start candidate sits at *low* gravity after months of decay, so it is the
  last thing evicted. The budget *bounds* hoarding, it doesn't *resolve* it
  (within the budget, never-surfaced pins are inherently indistinguishable) —
  size it as capacity planning, not a principled threshold.
- **Telemetry as the verdict** (`stats.pins_created / pins_active /
  pins_resonated / pins_evicted`): whether the declared channel is worth its
  cost is empirical, not arguable. A near-zero `pins_resonated / pins_created`
  over real traffic means people pin noise — bury the channel and accept the
  documented cold-start ceiling.

Ranking-boost (salience as a gravity term, not just an absorption floor)
remains a possible follow-up; this is the retention half.

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
  adaptive_gravity.py       AdaptiveWeights — five learned pre-resonance weights (w_freshness, w_access, w_graph, w_utility, w_stability), regularised SGD
  fact.py                   FactPassport dataclass (incl. recent_utility EWMA + forecast_stability)
  meta_fact.py              MetaFact dataclass + lineage + Hawking gravity helper
  gravity.py                GravityEngine — score computation + migration
  black_hole.py             BlackHole — polymorphic sink (facts + metas) + Hawking
  singularity_compactor.py  collapse_singularity() — Union-Find + center of mass
  vector_index.py           VectorIndex — numpy L2-normalised cosine search
  memory_store/             MemoryStore package — split for navigability after the unified API crossed 2500 LOC. Composition root `_base.py` (init + lifecycle + _reload atomicity + _sync + _txn) plus five mixin files:
                              _sessions.py    SessionsMixin — open/push/close/attribution/SGD training step
                              _facts.py       FactsMixin — CRUD, supersede/retire, set_fact, delete_body, explain_fact/explain_body, polymorphic body navigation
                              _query.py       QueryMixin — query, find_similar, check_echo
                              _singularity.py SingularityMixin — collapse + run_forecast (with snapshot revalidation)
                              _stats.py       StatsMixin — memory_stats
                            plus `_models.py` (QueryResult, SessionContext) and `_embed_proxy.py` (late-binding embed lookup so monkeypatch.setattr(birch.memory_store, "embed", ...) still propagates to mixin call sites after the split). Public import `from birch.memory_store import MemoryStore` is unchanged.
  server.py                 MCP server (FastMCP), 19 tools, threads session_id through. Family of boundary validators: _validate_text, _validate_spo_strings, _validate_id, _validate_optional_text, _validate_int, _validate_float, _env_int.
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


## Operational notes from external review

A few recurring concerns surfaced during external review were
deferred as design choices or future work — documented here so the
next reviewer (or contributor) doesn't re-raise them:

### Vector index O(N·d) at scale
`VectorIndex` is a numpy L2-normalised matrix; cosine search is one
matmul per query. Fine for personal-store scale (few hundred to few
thousand facts). At 100k+ facts the matmul becomes the bottleneck —
future work is to swap in HNSW (FAISS / LanceDB). Not blocking
because the personal-context-lakehouse use case rarely exceeds 10k.

### Vector index storage — preallocate-then-grow
The storage layout is a preallocated `_buffer` of shape `(_capacity,
_dim)` with `_size` live rows; `_buffer[_size:_capacity]` is headroom.
`add()` writes into the next free slot in O(d) and the buffer doubles
geometrically when `_size == _capacity`. `remove()` is O(d) via
swap-with-last (the public surface — `__len__`, `__contains__`,
`search`, `all_similarities` — never promised insertion order; search
returns by score and `all_similarities` returns a dict). After a
mass-delete with usage below `capacity / 4` the buffer reallocates
down to the smallest power-of-2 multiple of `_INITIAL_CAPACITY` that
fits the live size with 2× headroom — long-running stores don't sit
on peak allocation forever. Net `add` cost: amortised O(d) instead of
the old O(n·d) `np.vstack` strategy. On 10k facts × 768 dim that's a
~30 MB matrix copy per insert versus a single 3 KB overwrite. Public
contract is byte-for-byte unchanged; the 725-test suite pinned every
caller-visible behaviour through the rewrite.

### SPO temporal collapse is by design
``_spo_index`` keys on ``(subject, predicate, object)`` normalised
case + whitespace. A second identical triple touches the existing
fact instead of creating a new one. This **is** the contract:
Birch holds atomic mutable triples, gravity-ranked. Temporal
evolution lives in ``access_count``, ``created_at``,
``recent_utility`` (EWMA), ``resonance_sum/count``. If you need a
narrative log of "we used to think X, now Y, here's the why",
that's the Vertical Brain layer (see AGENTS.md boundary table) —
write a paragraph there, mutate the triple here.

### Async collapse holds the lock for its duration
``collapse_singularity`` acquires ``self._lock`` (RLock) and a
write txn for its entire pass. No race conditions — every other
``MemoryStore`` operation serialises behind it. The real concern is
**latency**: a long collapse on a huge singularity blocks
interactive queries. Mitigations already in place: per-dim
partitioning, bounded ``min_group_size``, the matmul
itself is fast for the few-thousand-body regime. If collapse
latency ever becomes user-visible, the next step is snapshot-under-
lock / compute-outside-lock / re-acquire-to-apply — over-engineered
for now.

### Hard-coded cosine thresholds (closed)
``Thresholds`` (``birch/thresholds.py``) centralises every cosine
and gravity threshold. Each is overridable via ``BIRCH_*`` env
vars — pin them to your embedding model's cosine distribution.
``memory_stats.thresholds`` echoes what the process actually
picked up. No more "0.85 is statistically impossible on this model"
landmines under provider swaps.

### Boundary validator family at the MCP edge
Every input that an agent can send through the MCP server passes a
typed validator at the boundary: ``_validate_text`` (non-empty string +
``BIRCH_MAX_FIELD_LEN`` cap), ``_validate_spo_strings`` (S/P/O shape +
per-field length), ``_validate_optional_id`` (session_id type),
``_validate_int`` / ``_validate_float`` / ``_validate_bool`` (numeric +
enum shapes). Failures return ``{"ok": False, "error": "...",
"field": "...", "hint": "..."}`` instead of crashing inside core.
Symmetric coverage across ``record_fact`` / ``record_facts`` (top-level
and per-item) / ``set_fact`` / ``query_memory`` / ``session_open`` /
``session_push`` / ``session_close``. The per-field length cap is the
DoS/billing primitive — a 10 MB paste never reaches the embedding
provider.

### Tolerant SQLite loaders and write-side allow_nan=False
Every loader (``load_facts`` / ``load_meta_facts`` /
``load_open_sessions`` / ``load_echo_sessions`` /
``load_adaptive_weights``) skips corrupt rows with a warning instead
of failing startup — drift from manual edits, partial writes, model
swaps, or schema migrations no longer takes the store down. Scalar
numeric fields run through ``_finite_float`` / ``_layer`` /
``_nonnegative_int`` so a NaN / Infinity / non-numeric cell loads as
the field default. Symmetric on write: ``_fact_row`` /
``_meta_row`` / ``save_echo_session`` / ``save_open_session`` use
``json.dumps(..., allow_nan=False)`` on every JSON cell and the same
sanitiser helpers on every scalar. Radioactive cells never reach disk;
the surrounding ``_txn()`` rolls back and ``_reload()`` restores the
pre-write in-memory snapshot.

### Self-defending body methods and final compute_gravity gate
``FactPassport.apply_resonance`` / ``MetaFact.apply_resonance`` reject
NaN / Infinity / non-numeric ``r`` (library users can call these
directly, bypassing engine + MCP gates). ``avg_resonance`` returns 0.0
when the computed mean is non-finite. ``compute_gravity`` ends with
``if not math.isfinite(gravity): return 0.0`` before its existing
``min/max`` clamp — Python's ``min/max`` is NOT NaN-aware
(``min(1.0, NaN)`` is platform-dependent), so the cascade is
belt-and-suspenders, not a single point of failure. ``__post_init__``
on both dataclasses normalises direct-construction values through the
same contract as the SQLite loader, closing the library-mode bypass.

### Atomic BlackHole.absorb + atomic absorb_meta
``absorb`` and ``absorb_meta`` are three-phase ops: preflight dim check
(raises ``DimensionMismatchError`` BEFORE any mutation), then set
layer + dict entry, then commit to the vector index. Any failure on
the third step rolls back the dict insert and layer mutation. Previous
order — set layer, put in dict, then add to index — left bodies
half-absorbed on dim mismatch: invisible to Hawking but still present
in singularity dict, while the live ``_facts`` hadn't been deleted
yet. In-memory mode (``storage=None``) couldn't recover via
``_reload``. ``_absorb_dead`` catches per-fact so one mismatched body
doesn't abort the whole sweep — it stays live and visible until the
operator runs collapse to bucket by dim.

### Closing-session race protection
``session_close`` snapshots ctx state then releases the lock for the
heavy ``compute_resonance`` call. A push that lands during that window
used to silently persist to disk and then get dropped when
``session_close`` popped the ctx — the agent saw ``push`` succeed but
the message never influenced R / echo / future sessions. Now the sid
is marked in ``_closing_sessions`` right after the snapshot;
``session_message`` rejects pushes to a closing sid with structured
``RuntimeError("session_closing")``. Flag clears on both success and
failure paths so the sid is never permanently bricked.

### Rollback-recovery on every write path
``add_fact`` / ``add_facts`` / ``set_fact`` / ``query`` / ``check_echo``
/ ``session_message`` / ``session_close`` / ``run_forecast`` /
``collapse_singularity`` all wrap their writeback in ``try: ...
except: self._reload(); raise``. SQLite txn rolls back disk truth on
failure; ``_reload`` re-anchors every in-memory cache to the
post-rollback disk state. Without this, the live ``_facts`` / ``_hole``
/ ``_engine`` / ``_echo`` caches would drift ahead of disk truth and a
restart would silently snap them back — silently corrupting adaptive
gravity training and layer migration.

### Prompt-injection advisory
BirchKM stores data; the consumer of ``query_memory`` is responsible
for wrapping retrieved bodies before feeding them into a downstream
LLM context. The boundary helps in two ways: ``_sanitize_for_llm``
strips invisible smuggling bytes (ASCII C0 except TAB/LF/CR, DEL,
zero-width Unicode) at the write boundary so they never reach storage;
``_has_instruction_markers`` detects visible markers
(``<|im_start|>``, ``[INST]``, ``<<SYS>>``, llama header IDs, etc.) at
the read boundary and attaches per-hit ``has_instruction_markers``
booleans + a top-level ``injection_warnings`` list to the
``query_memory`` response. Visible markers are NOT rewritten —
aggressive replacement is itself a content-filter bypass surface and
breaks legitimate discussion of prompt templates. Consumer wrapping
discipline remains non-negotiable; the advisory is the safety net.
