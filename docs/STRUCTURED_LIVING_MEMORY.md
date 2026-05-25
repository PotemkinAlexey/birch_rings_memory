# Structured Living Memory for AI Agents

> Architectural pattern. Product / umbrella repo: **MemoryBricks**.

A unified architectural target for **Vertical Brain** + **Birch Rings
Memory**, shipped under the umbrella name **MemoryBricks**. Not a
"merge" of two systems — a **layered model** where each keeps doing
what it's good at, and the composition produces a memory substrate
neither has alone.

This is a design document, not a shipped feature. The current code
runs both systems as separate MCP servers (per `~/.claude/MEMORY.md`:
"two-brain trial, on purpose, don't consolidate"). This document
explains why the trial is now ready for staged integration and what
the end-state looks like.

## Positioning

**MemoryBricks is a governed living-memory layer for AI agents:
structured by governance, ranked by observed usefulness.**

What MemoryBricks is **not**:

- Not a vector database (vector indexing is an implementation detail
  inside `dynamics`)
- Not a RAG framework (RAG is one use case among many; the contract
  is MCP-level, framework-neutral)
- Not a chatbot memory plugin (no LLM coupling at the storage layer)

What it **is**: a memory operating layer with two orthogonal axes —
**curation quality** (Bronze / Silver / Gold from governance) and
**emergent utility** (gravity / resonance / decay from dynamics).
Retrieval is `similarity × reputation`, scoped to the requesting
context (namespace / tenant / version).

---

## TL;DR (2-minute read)

**What:** A layered memory architecture for AI agents combining
Vertical Brain (governance) with Birch Rings Memory (dynamics). Not
a merge — each system keeps its role, the composition adds
reputation-weighted retrieval on top of a governed hierarchy.

**Vertical Brain handles:** namespaces (`WORK/DataArt/Databricks`),
Bronze/Silver/Gold curation layers, ACL, multi-tenant boundaries,
long-form chunk storage. *Where memory lives.*

**Birch Rings Memory handles:** gravity-ranked retrieval, resonance
feedback loop (sessions teach the system which facts actually help,
no labels required), aging/decay, clustering via centroids
(MetaFacts), black hole sink with Hawking emission. *How memory
evolves.*

**Why not just RAG:** every RAG system retrieves by similarity. Few
retrieve by **proven usefulness across sessions**. This architecture
adds reputation as a first-class signal — a fact that helped resolve
sessions 47/50 times is scored above an unused fact at the same
cosine. Within a scoped context (namespace / tenant / version), so
reputation never leaks across governance boundaries.

**Scoring target:**
```
final_score = similarity × namespace_relevance × reputation(body, context) × freshness × trust_layer
```

**First implementation step (Step 1, 1-2 weeks):** add `namespace`
field to Birch `FactPassport` / `MetaFact`. Birch becomes
hierarchy-aware without touching VB. No data migration, no model
retraining, just schema extension + per-namespace SPO dedup. See
**Implementation: ready-to-file issues** below for the full
copy-paste-ready issue body.

**Status:** Design v0.5 (named the product **MemoryBricks**),
validation gating Step 4 (full unified scoring) on Step 3 telemetry.
Premise-before-investment.

---

## One-line summary

> **Vertical Brain decides where memory should live and how it is governed.
> Birch decides how memory evolves, decays, clusters and gains reputation
> over time.**

VB without Birch = a well-organised library with shelves but no
reading-trail data — librarian's intuition decides what's important.

Birch without VB = an interesting physics model of memory dynamics
but no enterprise skeleton — no namespaces, no ACL, no governance.

Together = **Structured Living Memory**: memory that ranks itself by
emergent usefulness inside a governed hierarchy.

---

## The two-axis model

Memory items live at the intersection of two **orthogonal** dimensions:

| | **High utility (Birch)** | **Low utility (Birch)** |
|---|---|---|
| **Gold (VB)** | Curated AND used — the ideal | Curated but never retrieved — **promotion mismatch**, your model of "important" doesn't match reality. Visible signal to operator. |
| **Silver (VB)** | Promoted curation that actually pays off | Promoted but cooled — candidate for review |
| **Bronze (VB)** | Raw observation that keeps coming back — **candidate for promotion** (data-driven, not gut feeling) | Noise that decays naturally into the black hole |

The two axes are independent because they answer different questions:

- **Curation quality** = *"how confident am I that this is worth
  keeping?"* (manual, deliberate, slow-moving)
- **Emergent utility** = *"how often does retrieval actually benefit
  from this?"* (automatic, behaviour-driven, fast-moving)

Today neither system has both signals. VB has the first (manual
quality assignment). Birch has the second (resonance feedback +
gravity). The combination gives the system a real promotion signal
("this Bronze chunk has high gravity 3 weeks running, consider
Silver") and a real curation-debt signal ("this Gold rule has zero
gravity for 2 months, do you still need it?").

---

## Layer roles

### Vertical Brain (skeleton, governance, routing)

- **Hierarchical namespaces** (`WORK/DataArt/Databricks/...`) as the
  primary data-organisation axis
- **Bronze → Silver → Gold** as the curation pipeline
- **Locked context capsules** with isolation between projects
- **Tenant-prefix multi-tenancy** for enterprise deployment
- **ACL via `chunk_filter` + `metadata`** for per-row visibility
- **Routing decisions** — where does a new chunk live, who can read it
- **Long-form narrative** chunks (paragraphs, decisions, references)

Vertical Brain answers: *where is this knowledge, who owns it, and
which structural layer is it at right now?*

### Birch Rings Memory (dynamics, resonance, reputation)

- **Adaptive gravity** with 5 learned pre-resonance weights (freshness,
  access, graph, utility, stability)
- **Resonance feedback loop** — session outcomes propagate to facts
  used, no explicit labels required
- **Echo retroactive penalty** — unresolved problems pull down past
  facts that gave false sense of resolution
- **Singularity compactor** — clusters of dead facts collapse into
  dense MetaFact centroids
- **Black hole + Hawking emission** — irreversible-but-recoverable sink
- **N-body galaxy forecast** — predicted stability as a future-derived
  feature feeding the live ranking

Birch answers: *which knowledge actually helps, how confident should
we be in it, what's becoming useless, what should we forget?*

---

## Gold reframed as routing/control, not knowledge

**Critical reframe in this architecture:** Gold is not the highest tier
of knowledge content. Gold is the **rulebook for retrieval routing**.

- Gold aspect: *"queries about Databricks notebook orchestration should
  prefer the WORK/DataArt/Databricks namespace and weight Silver
  chunks at 2× similarity"*
- Bronze chunk: *"the notebook timed out because the cluster's idle
  timeout was 10 minutes"*

This reframe matters because it dissolves a structural conflict that
would otherwise break the integration:

- **VB Gold today** = high-confidence durable knowledge → would
  conflict with Birch's "everything decays" by needing immunity that
  contradicts the gravity model.
- **VB Gold reframed** = routing/control infrastructure → naturally
  immune to decay (it's infrastructure, not data), with its OWN
  separate "is this rule actually firing?" signal.

Gold immunity is then operationally clean: Gold doesn't accumulate
gravity scores; it accumulates `times_consulted_for_routing`. A Gold
rule with zero consults in 3 months is a deletion candidate, but it
doesn't compete with Bronze content for the same scoring slot.

---

## Reputation-based retrieval — the differentiator

The strongest practical claim of this architecture:

> **Retrieval should not be similarity-based alone. It should be
> similarity × reputation, where reputation emerges from actual
> usage patterns.**

Most practical RAG systems today still rely primarily on semantic
similarity and metadata filters. Some add a re-ranking stage
(cross-encoder), some hybrid lexical+vector, some graph traversal —
all are improvements on the relevance axis. What is rare across the
production landscape is an explicit, persistent **reputation signal
derived from observed usefulness** across sessions. A fact that's
been retrieved 50 times and helped resolve sessions 47 of those times
is statistically more trustworthy than a fact with similar cosine but
never used. Most current systems can't distinguish them because the
signal is never collected.

Target scoring formula:

```
final_score =
    semantic_similarity                 # cosine to query vector
  × namespace_relevance                 # 1.0 inside requested scope, dampened outside
  × reputation_weight(body, context)    # Birch gravity_score in this context — NOT global
  × freshness_weight                    # not too old
  × trust_level                         # VB Bronze/Silver/Gold modifier
```

The `reputation_weight` function takes `context` (requesting namespace,
tenant_id, optional version tag) as input. Same body, two contexts ⇒
two reputation values. See **Reputation is scoped, not global** below
for the rationale and implications.

Hallucination reduction follows architecturally: facts retrieved come
with self-reported confidence based on observed usefulness, not just
cosine luck. The LLM consumer can weight (or skip) low-reputation
results explicitly.

This is the **conference-talk-shaped** claim. "Structured living memory
with reputation-weighted retrieval reduces hallucination by X%" —
provable with a real benchmark, which neither VB nor Birch alone could
generate.

---

## Defining `positive_use` (the load-bearing definition)

The entire reputation mechanism hinges on a single operational
definition: **what counts as a fact "helping"?** Get this wrong and
the whole system optimises for the wrong thing.

**Initial definition (operational, not semantic):**

> `positive_use` is recorded when a retrieved memory participated in a
> session that ended in a resonant state by Birch's existing scoring
> (R > 0.35 by heuristic, or operator-declared `sentiment="resonant"`
> / `r_override` in that range).

Critical caveats this definition encodes:

1. **It is NOT a truth claim.** A frequently-used fact that ended
   resonant sessions is not "true". It is operationally useful in
   the population of sessions where it was retrieved. Truth verification
   is a separate concern (and one Birch deliberately doesn't try to
   solve — see Non-goals).

2. **It is NOT a popularity vote.** Resonance score has a behavioural
   component (closing patterns like "works", "got it") and a
   semantic-shift component (did the conversation narrow?). Both are
   independent of whether the user "liked" any specific fact.

3. **Attribution is similarity-weighted.** A fact returned at cosine
   0.95 absorbs nearly the full session R; one returned at 0.10
   barely moves. So a fact retrieved with high similarity to a
   resonant session's queries weights heavily; a tangential hit in
   the same session barely registers. This is Birch's existing
   per-fact attribution mechanism, extended to chunks.

**What `positive_use` is NOT (anti-definitions to pin):**

- NOT "the LLM answered confidently after seeing this fact"
- NOT "the user didn't ask a follow-up question"
- NOT "the user manually upvoted"
- NOT "the task succeeded externally"

These are all interesting signals, but each is either expensive to
collect (manual feedback), externally-coupled (task success requires
outside instrumentation), or noisy (LLM confidence is famously
miscalibrated). The operational definition above uses only signals
the system already has.

**Step 3 may discover the definition is wrong.** Cohort analysis
(does observed positive_use track manual VB Gold promotions?) is the
empirical test. If they diverge, refine the definition — don't
override the data.

---

## Validation hypothesis (how we'll know if this works)

Before any "reduces hallucination" claim, the architecture commits to
**falsifiable** outcomes. Each can be measured with the system's own
telemetry, no manual labelling required.

**Reputation-weighted retrieval should measurably reduce:**

| Metric | Definition | Current baseline |
|---|---|---|
| **Stale-fact retrieval rate** | Fraction of `recall()` results in the bottom quartile of `last_accessed` | All retrievals weighted equally by similarity |
| **Repeat retrieval of unhelpful facts** | Same `body_id` returned in 2+ consecutive sessions that scored toxic by Birch | No suppression after toxic session |
| **Irrelevant context injection** | `body_id`s retrieved with `similarity < 0.5` but admitted to top-k because of noise in cosine | Cosine top-k is the only filter |
| **Contradiction rate** | Pairs of retrieved facts with same `(subject, predicate)` and conflicting `object` (Birch already detects via `conflicts` hint) | Surfaced as advisory but not suppressed |

**Reputation-weighted retrieval should measurably increase:**

| Metric | Definition |
|---|---|
| **Hit precision after first session** | `body_id`s retrieved more than once that received `positive_use` on the previous round |
| **Time-to-resolution** | Sessions on a topic close in fewer messages because prior-helpful facts surface first |
| **Operator-promotion alignment** | Bronze chunks that achieved high gravity over a month → manually promoted by operator at higher rate than control |

**Hypothesis is REJECTED if Step 3 data shows:**

- No correlation between gravity-on-chunks and manual promotion
  decisions over 1-2 months → reputation signal doesn't track what
  operator cares about
- Stale-fact retrieval rate doesn't change vs current cosine-only
  baseline → the formula's weights don't move the needle
- Contradiction rate stays flat or rises → reputation rewards
  confident-wrong facts (see Failure mode below)

If any of these falsify, the architecture pivots, doesn't dig in.
Premise-before-investment is non-negotiable.

---

## Failure mode: false reinforcement

The single biggest risk in reputation-based retrieval: **a wrong fact
that gets used often becomes "trusted".** This is the mirror image of
similarity-only's problem — instead of "high cosine but wrong", it's
"high gravity but wrong".

Concrete scenario: a fact `("postgres", "default port", "5433")` is
slightly wrong (it's 5432). It's retrieved 20 times across sessions
that all scored resonant because the user was asking about Postgres
in contexts where the port didn't matter to the actual answer. Its
gravity climbs. Now a new user query DOES care about the port. The
reputation-weighted retrieval surfaces the wrong fact at high
confidence.

**Mitigations the architecture already provides:**

1. **Echo retroactive penalty** (Birch). If the user returns to the
   same topic unresolved, the past session's R is pulled into toxic
   territory and the penalty propagates to the facts that misled it.
   The wrong-port fact's gravity drops. Already in production.

2. **Contradiction detection** (Birch `conflicts` hint in
   `query_memory`). When two facts share `(subject, predicate)` with
   different `object`, the system surfaces the conflict to the
   consumer. Reputation doesn't silently overrule diversity.

3. **Source trust modifier** (VB). A chunk's `trust_level` (derived
   from Bronze/Silver/Gold layer, source attribution, immutability
   flag) multiplies into the final score. An immutable Gold-curated
   value beats a Bronze chunk no matter the reputation gap.

4. **Recent-utility EWMA decay** (Birch). Even high-gravity facts
   need recent positive use to stay confident; a fact that hasn't
   helped recently has reduced effective weight via the
   `recent_utility` term in the gravity formula.

**Mitigations the architecture must ADD:**

5. **Reputation cap.** No fact should be able to suppress contradicting
   evidence by reputation alone. If `conflicts` returns 2+ candidates,
   ALL of them must surface to the consumer regardless of reputation
   spread. (Today Birch already does this; the new
   `reputation_weight` multiplier in the unified scoring formula must
   NOT defeat the surfacing.)

6. **Negative-feedback amplification at low retrieval count.** A fact
   used 5 times with 1 toxic session should lose more reputation than
   a fact used 500 times with 1 toxic session. Variance-aware
   decay. Birch's current EWMA is roughly this but needs verification
   in chunk-scale data.

7. **Manual override path always available.** Operator-visible
   "blacklist this body_id" tool that pins gravity to floor regardless
   of usage. For the rare case where the system can't unlearn fast
   enough.

This risk is real and must be tested explicitly in Step 4. The
validation hypothesis includes the contradiction rate metric
specifically to catch it.

---

## Reputation is scoped, not global

A core architectural invariant: **a memory item's reputation must be
computed within its namespace / tenant / version context. High utility
in one context does NOT automatically imply high utility elsewhere.**

The same factual content can be:

- **Relevant in a pet project, dangerous in enterprise** — "deploy by
  pushing to main" works for a hobby repo, ships outages in production
- **Correct for one version, stale for another** — "the v2 API uses
  bearer tokens" was true until v3 switched to OAuth
- **High-trust in one tenant, untrusted in another** — internal
  guidelines for Team A don't apply to Team B, even verbatim
- **Useful in one investigation thread, misleading in another** — a
  fact that resolved a database performance question would
  systematically mislead a security-audit question on the same DB

A globally-scored reputation collapses all of these into one number,
which is the exact information loss VB's namespace hierarchy and ACL
are designed to prevent. The unified architecture must NOT undo that
by accident.

**Concrete implications for the design:**

1. **Reputation fields are per-context, not per-body.** A `FactPassport`
   (or `MetaFact`, or VB chunk) carries its scoring fields, but the
   gravity formula evaluates them *with namespace / tenant / version
   as input*. Same body in two namespaces ⇒ two independent
   reputation trajectories. (Step 1's `namespace` field on Birch
   facts is the schema-level enabler; Step 4 wires it into the
   formula.)

2. **The scoring formula's `reputation_weight` term is
   context-conditioned.** Restate the formula:

   ```
   final_score =
       semantic_similarity
     × namespace_relevance
     × reputation_weight(body, context)    # ← context-scoped, not global
     × freshness_weight
     × trust_level
   ```

   `context` includes at least: requesting namespace, tenant_id (if
   multi-tenant deployment), and an optional version tag. The
   reputation function may share data across contexts when explicitly
   allowed (e.g. "boost related-by-graph-edge bodies inside same
   tenant"), but never by default.

3. **Cross-namespace gravity propagation is the design surface for
   bounded sharing.** This was already in Open Questions; the
   constraint above narrows the answer space. Default is NO
   propagation across namespace boundaries; explicit opt-in (per
   graph-edge, per tenant-scope) enables it where the operator wants
   federated reputation. Never silent.

4. **Resonance feedback is recorded per-context.** When session_close
   propagates R to facts the session used, the propagation happens
   *within the session's namespace context*. A fact used in
   `WORK/A/B` getting positive R does not boost the same fact's
   reputation in `PERSONAL/notes`.

5. **SPO dedup becomes context-aware (Step 1 implication).** Birch's
   current `_spo_index` collapses identical triples globally. With
   namespace as a first-class field, dedup keys on
   `(namespace, subject, predicate, object)` instead — same triple
   in two namespaces is two facts with two reputations.

**Why this matters beyond engineering hygiene:** without scoped
reputation, the system becomes vulnerable to **cross-context
poisoning** — a malicious or accidental high-reputation fact in one
tenant could surface across all tenants because "it scored well
somewhere". This is the multi-tenant version of the false-reinforcement
risk above, and it has the same answer: the scoring layer must not
defeat the governance layer.

The validation hypothesis adds one more falsifiable metric:

| Metric | Definition |
|---|---|
| **Cross-context reputation drift** | Same body's reputation in namespace A vs B should track that namespace's actual usage patterns, not converge to a global mean. Measured by reputation variance for any body appearing in 2+ namespaces, expected to stay non-zero. |

If Step 4 shows that reputation converges to a global mean despite
context scoping, the implementation has a bug or the scoping
boundaries are too porous.

---

## What composition looks like, concretely

Five things change when the systems compose. Each is independently
shippable.

### 1. Namespace becomes a first-class field in Birch facts

Every `FactPassport` and `MetaFact` gets `namespace: str = ""`.
`query_memory(namespace_prefix=...)` filters scope, symmetric with VB's
`root_path`. Default `""` = root namespace = current Birch behaviour.

### 2. Unified MCP recall surface

One `recall(query, namespace=None, top_k=10)` tool that:
- queries Birch for atomic facts
- queries VB for paragraph chunks
- applies the unified scoring formula above
- merges results polymorphically (`kind: "fact" | "chunk"`)
- consumers branch on `kind` for rendering

The consumer doesn't know which store backed which hit. Data stays in
two separate backends; only the front-end is unified.

### 3. Usage tracking on VB chunks

Every chunk gets `access_count`, `last_accessed`, `last_positive_use`
(when reading session closed resonant). `recall()` updates these on
every hit. This generates the raw data needed to score VB chunks by
Birch's gravity formula, without yet running the full resonance loop.

After 1-2 months of this data: empirical evidence about whether usage
patterns on chunks track the assumed value (Bronze stays low,
promoted Silver climbs, etc.) or whether the assumption was wrong.

### 4. Cross-ref formalised as first-class

The existing AGENTS.md convention ("a Birch fact's `object` may be a
VB `chunk_id`") becomes a typed cross-reference. `query_memory` knows
to expand `kind: "vb_chunk"` references and inline the chunk content.

### 5. Gold-as-routing wired into recall

Gold aspects at the namespace of the query consulted FIRST, and treated
as scoring modifiers, not result candidates. `recall()` returns:
- `gold_directives: list[GoldRule]` — what governance says about this query
- `results: list[Hit]` — actual content, scored per directives

---

## Open questions (must be answered before Step 4)

The big-picture vision is sound. But several decisions need real data,
not armchair theorising:

**Where does the session boundary live for resonance?**
Birch has explicit `session_open/close`. VB doesn't. Three options:
- Adopt Birch's explicit sessions for all memory access → simple, but
  asks the consumer to manage session lifecycle for read-only queries
- Derive resonance from raw read patterns (e.g. "consumer returned to
  same query 3× in an hour without satisfaction") → harder, needs
  detector
- Hybrid — explicit for write sessions, derived for read patterns

**What gets gravity at what granularity?**
Atomic facts and paragraph chunks have different "usefulness signatures".
A fact being "used" = retrieved at high similarity. A chunk being
"used" = retrieved AND consumer didn't return for follow-up. The
resonance formula needs different parameters per type, or a normalised
signal that works for both.

**Cross-namespace gravity propagation.**
Birch has `auto_link` graph edges. Should an edge that crosses a
namespace boundary still propagate gravity? Arguments either way:
- Yes: thematic clusters cross workspaces, propagation reflects real
  knowledge relationships
- No: namespace boundaries should also be reputation boundaries (ACL
  doesn't leak; reputation shouldn't either)
- Hybrid: same-tenant cross-namespace propagates, cross-tenant doesn't

**Gold lifecycle without decay.**
If Gold is immune to gravity, what happens to a Gold rule that hasn't
fired in 6 months? Three options:
- Stays forever (operator must clean up)
- Auto-archive to "cold Gold" state (visible but de-prioritised in
  routing decisions)
- Decay applies but slower (50× freshness half-life vs Bronze)

**Migration story.**
Two existing live stores with months of accumulated data. Either:
- Tabula rasa (lose history, fastest path)
- Synthetic initial state (every existing Bronze chunk starts at gravity
  0.5, every existing Birch fact gets `namespace=""`)
- Side-by-side during transition with `recall()` querying both old and
  new stores

These don't have right answers yet. Step 3's data informs them.

---

## Staged migration plan

Five steps, each independently valuable, each reversible.

### Step 0 — This document (DONE)
Architectural decisions on paper. Naming, layer roles, scoring formula,
open questions enumerated.

### Step 1 — Namespace field in Birch (1-2 weeks)
- Add `namespace` field to `FactPassport` / `MetaFact` schema
- Migrate existing storage (default `""`)
- `query_memory(namespace_prefix=...)` filter
- `record_fact` accepts optional `namespace`
- Birch becomes hierarchy-aware without breaking VB

### Step 2 — Unified `recall()` MCP tool (1-2 weeks)
- New MCP tool in a new wrapper package (`structured_living_memory`?)
  that knows both birch and vb servers
- Queries both, merges results, returns polymorphic hits
- Cross-ref expansion (Birch fact whose `object` is a VB chunk_id
  inlines the chunk)
- No data migration. Just routing.

### Step 3 — Chunk usage tracking in VB (1-2 months)
- Add `access_count`, `last_accessed`, `last_positive_use` to VB chunk
  schema. Migration migrates existing chunks to 0/null defaults.
- `recall()` (and direct VB query) updates these fields on every hit
- Cohort analysis: do these signals correlate with VB's manual
  promotion decisions? If yes → Step 4 is justified. If no → revisit
  the synthesis premise.

### Step 4 — Full unified scoring (3-6 months, conditional on Step 3 data)
- Adaptive gravity formula extended to chunks
- Resonance loop for chunk reads
- Singularity compactor over Bronze chunks (auto-summarisation)
- Gold reframed to routing-only (deprecate any Gold content that's
  actually knowledge — promote to Silver)
- Full `final_score = sim × namespace × reputation × freshness × trust`
- Real benchmark publication

---

## Implementation: ready-to-file issues

Copy any of these bodies into a GitHub issue (in birch-km, in
vertical-brain, or in a new `structured-living-memory` umbrella repo
— see Naming). Each Step's issue is self-contained; no cross-step
dependencies for the issue scope itself, only for the runtime path.

---

### Issue: Step 0 — Create MemoryBricks umbrella repo

**Repo:** new — `memorybricks`
**Effort:** 1 day
**Blocks:** Step 2 (umbrella hosts the unified MCP wrapper); v1.0
release path
**Blocked by:** this document committed (so README can link to it)

**Background.** Naming and positioning decisions in
`STRUCTURED_LIVING_MEMORY.md` (this doc) settled on "MemoryBricks"
as the public-facing product / umbrella name. Step 0 creates that
repo so subsequent Steps have a place to land and a discoverable
home for the architecture.

**Scope.**
- Create `memorybricks` GitHub repo (private initially; public when
  Step 2 wrapper actually exists).
- `README.md` with positioning paragraph (see Naming section in
  this doc).
- `docs/STRUCTURED_LIVING_MEMORY.md` mirroring this document
  (initial copy; later this becomes the canonical home and Birch's
  copy becomes a stub linking here).
- `LICENSE` (Apache 2.0, matching Birch).
- `roadmap/` directory with one markdown file per future Step's
  issue body, ready to file as GitHub issues:
  - `roadmap/step-1-namespace-in-birch.md`
  - `roadmap/step-2-unified-recall-mcp.md`
  - `roadmap/step-3-vb-usage-tracking.md`
  - `roadmap/step-4-context-scoped-scoring.md`
- `packages/` directory empty but committed with `.gitkeep` —
  signals intended layout per the Naming section.
- CI skeleton: Python project structure with `ruff` + `pytest`
  hooks ready (the wrapper is Python; matches Birch / VB tooling).

**Acceptance criteria.**
- Repo accessible at `github.com/PotemkinAlexey/memorybricks` (or
  chosen org).
- README's first paragraph passes "ten-second pitch" — a stranger
  reads it and knows what the project is.
- All roadmap files are valid markdown, each linkable.
- Document referenced from this repo's README and from
  `vertical-brain`'s README (small "See also" link).

**Concrete commands** (run once, after this doc is committed):

```bash
gh repo create memorybricks --private \
    --description "Governed living-memory layer for AI agents"
git clone git@github.com:PotemkinAlexey/memorybricks.git
cd memorybricks
mkdir -p docs roadmap packages
cp ../birch_rings_memory/docs/STRUCTURED_LIVING_MEMORY.md docs/
# … create README.md per Naming section
# … split this doc's Implementation section into roadmap/*.md
git add -A
git commit -m "Initial commit: MemoryBricks umbrella for Structured Living Memory"
git push origin main
```

---

### Issue: Step 1 — Add `namespace` field to Birch FactPassport / MetaFact

**Repo:** `birch_rings_memory`
**Effort:** 1-2 weeks
**Blocks:** Step 2 (recall wrapper relies on namespace filter), Step 4
**Blocked by:** nothing

**Background.** First concrete move from the Structured Living Memory
plan (`LIVING-MEMORY.md`). Birch facts currently live in a flat space;
VB lives in a namespaced one. This issue extends Birch's schema with
a `namespace` field so it can later be queried by VB-style scope
without breaking any existing call site.

**Scope.**
- Add `namespace: str = ""` to `FactPassport` and `MetaFact`
  dataclasses. Default empty string = global scope (= today's
  behaviour).
- SQLite schema migration: add `namespace TEXT NOT NULL DEFAULT ''`
  column to `facts` and `meta_facts` tables. Migrate existing rows
  to `""`.
- `MemoryStore.add_fact` / `add_facts` / `set_fact` accept optional
  `namespace` kwarg, defaults to `""`.
- `MemoryStore.query` / `find_similar` / `list_facts` accept
  optional `namespace_prefix` filter, defaults to None (no filter).
- **SPO dedup becomes context-aware.** `_spo_index` key changes
  from `(subject, predicate, object)` to
  `(namespace, subject, predicate, object)`. Same triple in two
  namespaces = two independent facts with two independent gravity
  trajectories.
- MCP boundary: `record_fact`, `set_fact`, `query_memory`, etc. all
  gain optional `namespace` / `namespace_prefix` parameters.

**Acceptance criteria.**
- All 796 existing tests pass unchanged (default empty namespace
  preserves current behaviour).
- New tests:
  - Two facts with same SPO but different namespaces coexist
    independently
  - `query_memory(namespace_prefix="WORK/")` filters correctly
  - SQLite migration round-trips on a legacy DB (no namespace column)
- Schema documented in `ARCHITECTURE.md` storage block.

**Files affected.**
- `src/birch/fact.py`, `src/birch/meta_fact.py`
- `src/birch/storage/sqlite.py` (schema + loader + writer)
- `src/birch/memory_store/_facts.py` (SPO dedup change)
- `src/birch/memory_store/_query.py` (filter)
- `src/birch/server.py` (MCP boundary)
- `README.md`, `ARCHITECTURE.md`, `AGENTS.md` (documentation)

---

### Issue: Step 2 — Build unified `recall()` MCP wrapper

**Repo:** `memorybricks` (`packages/memorybricks-mcp/`)
**Effort:** 1-2 weeks
**Blocks:** Step 3 (chunk usage tracking needs read-path interception
which `recall()` is the natural place for)
**Blocked by:** Step 0 (umbrella repo exists); Step 1 (uses
namespace filter)

**Background.** Currently agents juggle two MCP servers: birch-km
for atomic facts, vertical-brain for chunks. Workflow friction
("which one do I write to / read from?"). This issue ships a thin
wrapper MCP server that exposes `recall()` / `remember()` /
`forget()` and routes to both backends.

**Scope.**
- New MCP server package depending on both `birch.server` and
  `vertical_brain.mcp.server` (or their client-mode equivalents).
- Tool `recall(query, namespace=None, top_k=10, ...)`:
  - Queries Birch facts (`query_memory`)
  - Queries VB chunks (`search` or `context_search`)
  - Merges results polymorphically: each hit has
    `kind: "fact" | "chunk"`, `body_id`, `similarity`, `source`,
    plus kind-specific fields
  - Returns unified ranked list
  - **No unified scoring yet** — just stable merge by similarity.
    Reputation weighting lands in Step 4.
- Tool `remember(content, kind: "fact" | "chunk", namespace, ...)`:
  routes to the right backend by `kind`. (Agents that already know
  which they want still use the underlying server's tool directly.)
- Tool `forget(body_id, kind)`: routes by `kind`.
- **Cross-reference expansion.** If a Birch fact's `object` matches
  a VB chunk_id, `recall()` inlines the chunk content (or summary)
  in the response. Formalises the existing AGENTS.md convention.

**Acceptance criteria.**
- Single MCP server config in `~/.claude/claude_desktop_config.json`
  surfaces `recall` / `remember` / `forget` alongside (not
  replacing) the individual birch / VB tools.
- Integration test: write 5 facts to Birch + 3 chunks to VB,
  `recall("query")` returns merged hits with `kind` discriminator.
- Cross-reference inlining works end-to-end.
- Documentation: how to migrate from "directly call birch /
  vertical-brain tools" to "use recall when scope is mixed".

**Files affected.**
- New repo / package
- Documentation updates in both birch-km and vertical-brain pointing
  at the wrapper

---

### Issue: Step 3 — Add usage tracking to Vertical Brain chunks

**Repo:** `vertical-brain-for-ai`
**Effort:** 1-2 months (instrumentation + cohort analysis)
**Blocks:** Step 4 (reputation scoring on chunks needs this data)
**Blocked by:** Step 2 (recall is the interception point); Step 1
indirectly (namespace as scope for usage tracking)

**Background.** VB chunks currently have no usage telemetry —
operators promote Bronze → Silver based on intuition. This issue
adds the minimal signals needed to make promotion data-driven AND
to feed Step 4's reputation scoring.

**Scope.**
- Add to `Chunk` schema:
  - `access_count: int = 0`
  - `last_accessed: float | None = None`
  - `last_positive_use: float | None = None` (timestamp of most
    recent session that closed resonant after retrieving this chunk;
    initially None for everything)
- SQLite schema migration adds the three columns
- `recall()` (and any direct VB query path) updates `access_count`
  and `last_accessed` on every hit
- `last_positive_use` updated by the resonance-feedback path from
  Birch (Step 4 wires this; Step 3 just adds the field)
- Cohort analysis tool / script: dump per-chunk usage statistics,
  correlate against operator's manual promotion history. Answer
  "do high-usage chunks track manual Silver promotions?"

**Acceptance criteria.**
- All existing VB tests pass.
- Two months of telemetry on a real store (maintainer's daily use).
- Analysis report: correlation coefficient between
  `access_count` and "got promoted to Silver/Gold". Reject Step 4
  premise if coefficient is near zero.
- Cohort tool can re-run on any future data drop.

**Files affected.**
- VB `Chunk` dataclass
- VB SQLite store migration
- VB query handlers
- New analysis script under `scripts/usage_cohort_analysis.py`

---

### Issue: Step 4 — Context-scoped unified reputation scoring

**Repo:** umbrella (`structured-living-memory`) or both backends
**Effort:** 3-6 months
**Blocks:** v1.0 release with the "Structured Living Memory" name
**Blocked by:** Step 3 (data); architectural decisions still open
(see Open Questions in LIVING-MEMORY.md — they need answers before
this issue can be scoped tightly)

**Background.** Full unified scoring formula:
```
final_score =
    semantic_similarity
  × namespace_relevance
  × reputation_weight(body, context)
  × freshness_weight
  × trust_level
```
Implements the conference-talk-shaped claim: reputation-weighted
retrieval reduces hallucination by measurable amounts.

**Scope.**
- Adaptive gravity formula extended to VB chunks (per-context, see
  "Reputation is scoped, not global" in LIVING-MEMORY.md)
- Resonance loop for chunk reads (session-anchored, similarity-
  weighted attribution)
- Singularity compactor over Bronze chunks (auto-summarisation)
- Gold reframed to routing-only (any current Gold chunk that's
  actually knowledge gets demoted to Silver as part of migration)
- Full benchmark publication: validation hypothesis metrics
  measured against current cosine-only baseline

**Acceptance criteria.**
- Validation hypothesis metrics show statistically significant
  improvement OR the architecture is rejected per the
  premise-before-investment principle
- Cross-context reputation drift stays non-zero (reputation is
  truly scoped, not converging to global mean)
- False reinforcement mitigations in place: reputation cap,
  variance-aware decay, manual override path
- v1.0 release of the umbrella package with full architecture
  shipped, documented, benchmarked

**Open questions to answer before scoping tightly:**
- Session boundary for resonance: explicit close vs derived from
  read patterns vs hybrid?
- Gravity granularity: per-fact vs per-chunk vs both with
  different parameters?
- Cross-namespace gravity propagation: default OFF (set by Step
  invariant); what opt-in mechanism for federated reputation?
- Gold lifecycle without decay: stays forever, auto-archive, or
  slow decay?
- Migration story: tabula rasa, synthetic initial state, or
  side-by-side transition?

These have to be answered with data from Step 3 + design discussion
before Step 4 is broken into implementable sub-issues.

---

### Optional Issue: Companion document in Vertical Brain

**Repo:** `vertical-brain-for-ai`
**Effort:** 1 day
**Blocked by:** This document (LIVING-MEMORY.md) committed

**Scope.** Create `docs/STRUCTURED_LIVING_MEMORY.md` in the VB repo
that:
- Links to this document as the canonical source on the dynamics layer
- Describes how `governance` (VB's role) interacts with `dynamics`
  (Birch's role) from the VB-author's perspective
- Adds the Gold-as-routing reframe to the VB Gold documentation
- Pins the staged migration plan from VB's side

This is symmetric mirror, not duplicate content — the source of
truth is the birch-km LIVING-MEMORY.md, VB just makes it
discoverable from its own side.

---

## Non-goals (explicit)

- **Replacing either current system tomorrow.** Both live in production
  for the maintainer's actual workflow. Migration is staged.
- **Following the biological metaphor as architecture spec.** "Nervous
  system" / "brain" / "resonance" are useful for *explaining* the
  shape; they are NOT specifications. The system has ACL, namespaces,
  tenant boundaries, audit, version migration — none of these exist in
  a biological brain. When metaphor and engineering reality conflict,
  engineering wins.
- **Premature unified scoring.** The formula above is a target. The
  weights (`reputation_weight × freshness_weight × trust_level`)
  require empirical fitting against real query → outcome data. Step 3
  generates that data; Step 4 fits the weights.
- **Productisation before validation.** "Structured Living Memory for
  AI Agents" is a strong concept name, but until Step 3 confirms the
  premise, it stays as internal terminology. No marketing site, no
  conference talk, no "v1.0 release" until reputation-vs-similarity
  data shows real signal.

---

## Naming

Five-level name structure, each layer with its own scope:

| Layer | Name | Why |
|---|---|---|
| Product / public-facing brand / umbrella repo | **MemoryBricks** | Concrete metaphor (memory composed of small, composable, managed blocks). Productisable; not academic. Distinguishes from "Mem0" / "memgraph" / generic "memory-something" naming noise. |
| Architectural pattern | **Structured Living Memory** | Functional description of the design — "structured" = governance, "living" = dynamics. Used in talks, papers, this document. Not the brand. |
| Governance layer | **Vertical Brain** (existing repo) | Already published. Stays as the implementation of the governance axis. Internal abstraction name: `governance`. |
| Dynamics layer | **Birch Rings Memory** (existing repo) | Already published. Stays as the implementation of the dynamics axis. Internal abstraction name: `dynamics`. |
| MCP tool surface | `recall` / `remember` / `forget` | Verb-based, intent-driven; consumer doesn't know which backend served what. |

**Suggested umbrella repo layout** (Step 0 in the implementation
plan):

```
memorybricks/
├── README.md                          # Positioning, quickstart, links
├── docs/
│   └── STRUCTURED_LIVING_MEMORY.md   # Symlink or mirror of this doc
├── packages/
│   ├── memorybricks-mcp/              # Unified recall/remember/forget MCP server (Step 2)
│   ├── memorybricks-governance/       # Thin client / adapter for Vertical Brain
│   └── memorybricks-dynamics/         # Thin client / adapter for Birch
├── roadmap/                           # Issue templates per Step
└── benchmarks/                        # Validation hypothesis measurements (Step 4)
```

`memorybricks` does NOT vendor Birch or VB; both stay independent
repos. The umbrella holds the unified MCP wrapper, the canonical
docs, the roadmap, and the eventual benchmark suite. Versions of
Birch and VB are pinned as dependencies; release of MemoryBricks
v1.0 pins specific Birch / VB versions.

**README first paragraph** (suggested):

```
MemoryBricks is an experimental governed living-memory layer for
AI agents. It combines Vertical Brain's namespace / governance
model with Birch Rings Memory's resonance, decay, clustering, and
reputation-weighted retrieval — exposed as a single MCP surface.

Not a vector DB. Not a RAG framework. Not a chatbot memory plugin.
A memory operating layer: structured by governance, ranked by
observed usefulness.
```

This naming is now mostly settled. The MemoryBricks brand name
should be tested with at least one external reader before the
umbrella repo is published — fast sanity check that the metaphor
("bricks composing into living memory") lands without explanation.

---

## Companion changes in Vertical Brain

If this document is adopted, VB will need a counterpart entry in its
own docs (likely `docs/LIVING-MEMORY.md` symmetric with this file).
That document should:

- Cross-link here as the canonical source on the dynamics layer
- Describe how `governance` (VB's role) interacts with `dynamics`
  (Birch's role) from the VB-author's perspective
- Add the Gold-as-routing reframe to the VB Gold documentation
- Pin the staged migration plan from VB's side (Step 1 adds
  `namespace` in Birch but VB doesn't need to change; Step 3 is where
  VB schema changes)

Open question: does VB own the unified `recall()` MCP tool, does
Birch, or does it live in a third repo? Probably the third option for
clean dependency ordering, but worth deciding before Step 2 starts.

---

## Status

**Document version:** 0.5 (2026-05-25)

**Next concrete action:** Read this with fresh eyes in 48 hours. If
the layer split still feels right, schedule Step 1. If anything reads
as "wishful thinking" or "wouldn't it be cool if...", revise.

Either way, this document supersedes the standalone
`~/.claude/MEMORY.md` note that says "two-brain trial, don't
consolidate" — that was correct guidance for the experimental phase;
this is the architectural plan that follows from it.

### Changelog

**0.5** (2026-05-25, product-name revision)

- **MemoryBricks** committed as the public product / umbrella repo
  name. Five-level Naming table now: MemoryBricks (brand) →
  Structured Living Memory (architectural pattern) → Vertical Brain
  (governance impl) + Birch Rings Memory (dynamics impl) → MCP
  surface verbs (recall/remember/forget).
- New **Positioning** section at top: explicit "not a vector DB / not
  a RAG framework / not a chatbot memory plugin" anti-claims +
  one-line affirmative pitch.
- New **Step 0** in the implementation issue list: create
  `memorybricks` umbrella repo with concrete `gh` commands,
  README seed, suggested package layout. Subsequent Steps' "Repo:"
  fields now point at `memorybricks` packages where applicable.
- Suggested README first paragraph included verbatim in Naming
  section — ready to paste into the new umbrella repo's README.

**0.4** (2026-05-25, usability revision)

- Added **TL;DR (2-minute read)** section at top — what / why VB /
  why Birch / why not RAG / first step / status. New reader gets
  the whole arc without scrolling.
- Added **Implementation: ready-to-file issues** section: each Step
  (1-4 + optional companion doc) is a copy-paste-ready GitHub issue
  body with scope, acceptance criteria, files affected, effort
  estimate, and blocked-by graph. Removes the "where do I start?"
  friction.
- File location: this document is intended to live at
  `docs/STRUCTURED_LIVING_MEMORY.md` from v0.4 onwards (was
  `LIVING-MEMORY.md` in repo root for v0.1-v0.3 review). Naming
  matches the conventional `docs/` layout and the "Structured
  Living Memory" architectural name.

**0.3** (2026-05-25, review-driven revision)

- New section **Reputation is scoped, not global**: pins an
  architectural invariant that reputation must be computed within
  namespace / tenant / version context, NOT collapsed to a global
  number. Lists 5 concrete design implications: per-context
  reputation fields, scoring formula update
  (`reputation_weight(body, context)` instead of bare `reputation_
  weight`), cross-namespace propagation default-OFF, per-context
  resonance feedback, and context-aware SPO dedup as a Step 1
  schema implication.
- Updated the scoring formula in the differentiator section to
  reflect context scoping.
- Added one validation metric: **cross-context reputation drift**
  (reputation of the same body in two namespaces should reflect
  per-namespace usage, not converge to global mean).
- Connected this back to the existing "Cross-namespace gravity
  propagation" open question — scoping is the constraint that
  narrows the answer space (default no-propagation, explicit opt-in
  for federated reputation).
- Names cross-context poisoning as the multi-tenant version of the
  false-reinforcement risk: scoring layer must not defeat governance
  layer.

**0.2** (2026-05-25, review-driven revision)

- Softened the "every RAG system" claim. Acknowledges existing
  rerankers, hybrid search, graph RAG; pins the specific gap as
  "persistent reputation signal across sessions", not "everyone's
  doing it wrong".
- New section **Defining `positive_use`**: explicit operational
  definition (recorded when retrieved memory participates in a
  Birch-resonant session, similarity-weighted attribution), plus
  anti-definitions for what it is NOT (truth claim, popularity vote,
  LLM-confidence proxy, manual upvote). The whole reputation
  mechanism hinges on this definition; pinning it early prevents
  drift.
- New section **Validation hypothesis**: falsifiable metrics that
  must move (stale-fact retrieval rate, repeat-of-unhelpful,
  irrelevant-context-injection, contradiction rate, hit precision,
  time-to-resolution, operator-promotion alignment) and explicit
  rejection criteria if Step 3 data doesn't bear them out.
- New section **Failure mode: false reinforcement**: the mirror-image
  risk (wrong fact that gets used often climbs reputation), mapped
  against existing mitigations (echo retroactive penalty,
  contradiction detection, source trust modifier, recent-utility
  EWMA decay) plus mitigations the architecture must add (reputation
  cap, variance-aware decay, manual override path).

**0.1** (2026-05-25, initial draft)

- Two-axis model (quality × utility)
- VB-as-skeleton / Birch-as-dynamics layer split
- Gold reframed as routing/control, not knowledge
- Five-step composition plan
- Open questions enumerated
- Naming + companion VB doc note
