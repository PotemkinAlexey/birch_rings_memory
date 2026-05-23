# BirchKM — Agent Contract

## Principles

**Not permitted** — do not store PII, credentials, secrets, or anything the user
has not explicitly asked to remember.

**Not certain** — do not present a retrieved fact as ground truth without citing
its `gravity_score`. Low gravity means the system itself doubts it.

**Storing** — use triples that capture a relationship, not prose. One fact per
`record_fact` call.

**Changing** — if a fact is superseded, `record_fact` the new version, then
call `supersede_fact(old_id, new_id)`. The old body goes to the singularity
with its `deprecated_by` pointer intact, so lineage is preserved and the
body can still feed MetaFact compression or be Hawking-emitted. If a fact's
topic is just over with no replacement, call `retire_fact(fact_id)` —
same singularity benefits, no successor required.

**Deleting** — only on explicit user order, and only with `delete_fact`,
which is the destructive primitive (no singularity, no lineage). Reserve
it for secrets / accidental writes / GDPR removal. For "stale / wrong /
outdated" data, the right operation is `supersede_fact` or `retire_fact`
— see the *Retiring a fact* table below.

---

## Session lifecycle

Call `query_memory` before composing your first response. No exceptions.
If it fails, tell the user and stop — do not respond from training data alone.

Call `record_session` at the end of every session that produced an answer or
attempted to solve a problem. If nothing worth persisting happened, tell the
user: **"No durable memory was created this session."**

Do not save everything for session end. If a fact is confirmed mid-session,
call `record_fact` immediately.

---

## Reading memory

### query_memory

Call with the user's first message or the core question of the session.
Do not paraphrase — pass the user's text as close to verbatim as possible;
the embedding does the semantic work.

Interpret the response by `source`:

| source | meaning | action |
|---|---|---|
| `surface` | hot fact, used often, high gravity | trust, cite directly |
| `kinetic` | working memory, moderate use | use, but check relevance |
| `core` | cold archive, rarely used | verify before relying on it |
| `hawking` | single FactPassport returned from the black hole | treat with suspicion; it sank for a reason |
| `hawking_meta` | a MetaFact bundle returned from the black hole | dense context, but it represents *aggregated dead facts* — verify against the user's actual question |

Interpret the response by `kind` — a query hit is either:

- `kind == "fact"` — one SPO triple. Read `subject`/`predicate`/`object`
  as you always have.
- `kind == "meta"` — a MetaFact: `weight` facts collapsed into one
  centroid. Read `weight`, `source_texts` (up to a handful of original
  SPO strings), and `summary` if non-empty. A MetaFact answers "this
  cluster of related dead facts came up before", not "here is one
  precise statement". Cite it as a bundle, not as a single source.

Interpret the response by `gravity_score`:

- `> 0.70` — system considers it reliable
- `0.30–0.70` — neutral; weight it accordingly
- `< 0.30` — weak signal; corroborate or discard

`similarity < 0.60` — do not use the fact in your answer; it is noise.

Do not summarize retrieved facts without stating the source layer and gravity.

Every fact returned by `query_memory` is attributed to the current session
proportionally to its similarity. A fact returned at cosine 0.95 absorbs
nearly the full session R; a fact returned at 0.10 absorbs almost none.
You can ask for a broad `top_k` without worrying that low-similarity noise
will significantly tilt the gravity of unrelated facts.

---

## Writing facts

### record_fact

Use subject–predicate–object triples that are:

- **Atomic** — one fact per call, independently meaningful
- **Relational** — captures a connection, not a description
- **Durable** — still true next session, not session-specific observations

Good triples:

```
("auth service",    "uses",         "JWT")
("user",            "prefers",      "dark mode")
("mailer",          "connects to",  "PostgreSQL on port 5432")
("deploy pipeline", "fails when",   "migrations run before health check")
```

Bad triples:

```
("user asked", "about", "authentication")    ← session artifact, not a fact
("JWT is",     "good",  "for stateless auth") ← opinion, not a relationship
("service",    "info",  "mailer runs on Go")  ← predicate is not a relationship
```

Search before writing. If `query_memory` returns a fact with `similarity > 0.85`
covering the same ground — do not duplicate. Facts compete by gravity; duplicates
dilute the signal.

BirchKM enforces this at the store level too: identical SPO triples
(case-insensitive, whitespace-normalised) collapse to a single fact, so a
duplicate `record_fact` returns the existing `fact_id` and touches it instead
of creating a new record. The returned `gravity_score` reflects the existing
fact's current value; `access_count` will have incremented by one. That is a
safety net, not a license — still search before writing, since paraphrases
(`"runs on" Go` vs `"is written in" Go`) will not be de-duped automatically.

### Retiring a fact — pick the right path

Three operations exist for "this fact should leave the live layers". They
are NOT interchangeable; pick by intent.

| Situation | Tool | What happens |
|---|---|---|
| There is a newer fact that *replaces* this one | `supersede_fact(old_id, new_id)` | Old fact's `deprecated_by` is set, body goes to the singularity. Lineage preserved; can feed MetaFact compression and Hawking emission. |
| The topic is over, no replacement | `retire_fact(fact_id)` | `ttl` set to now, body goes to the singularity. Same singularity benefits as supersede. |
| Hard removal (secrets / accidental write) | `delete_fact(fact_id)` | Row deleted from storage. **No lineage, no singularity, no Hawking rescue.** Use sparingly. |

The default for "we now know better" is **`supersede_fact`**, not
`delete_fact`. Hard delete loses the "we used to think X" record and
denies the body to singularity collapse, which is how BirchKM compresses
related stale facts into MetaFacts. Use `delete_fact` only when the data
must genuinely cease to exist.

If a fact's gravity decays naturally below `0.10`, the runtime absorbs
it into the singularity on the next tick — no tool call needed.

---

## Session scoring

### record_session

Pass **all user messages** from the session in order. Do not include your own
responses — the resonance engine scores user-side signals only.

Call once per session, at the end. Multiple calls for the same session corrupt
the gravity signal.

Interpret the response:

| label | meaning | implication |
|---|---|---|
| `resonant` | session produced value | facts used will gain gravity |
| `neutral` | unclear outcome | no gravity change |
| `toxic` | session was circular or stuck | facts used will lose gravity; check for echo |

If `label` is `toxic` — consider calling `query_memory` on the opening message
of the next session with the same topic before responding. A recurring toxic
pattern means the memory itself may be misleading you.

---

## Echo signal

The system detects when a user returns to an unresolved problem. Echo is
checked at `check_echo` time and also implicitly whenever a new session
opens on a topic close to a closed one (`similarity ≥ 0.68`). On a hit:

- The matched past session's R score is pulled into toxic territory.
- The retroactive penalty is propagated to the gravity of every fact that
  past session touched, scaled by each fact's per-session relevance weight.
- The store records that the penalty was applied, so a second echo on the
  same matched session does not stack penalties.

When `record_session` returns a `toxic` label on a topic you have seen before,
treat it as an echo:

- Do not repeat the same answer
- Retrieve memory with `query_memory` and inspect `gravity_score`; low-gravity
  facts near the topic are candidates for what misled the previous session
- Acknowledge to the user that this problem has come up before and was not
  resolved

---

## Hawking emission

A result with `source: "hawking"` was previously absorbed by the black
hole — its gravity fell below the absorption floor. It returned only
because your query matched it at `similarity ≥ 0.95`.

Do not use a Hawking fact as primary evidence. Use it as a lead:
- Present it to the user as a recovered memory
- Ask whether it is still relevant
- If confirmed, `record_fact` it again so it re-enters the live layers with
  fresh gravity

### MetaFacts and `source: "hawking_meta"`

A `MetaFact` is the residue of *several* facts that fell into the black
hole around the same semantic topic, fused by gravitational collapse
into one dense bundle. A query hit with `source: "hawking_meta"` is a
MetaFact that just emerged at similarity `≥ 0.85` (the threshold is
lower than for single facts because a centroid lives between its
sources).

Treat MetaFact hits as **aggregated leads, not citable statements**:

- Read `source_texts` (up to a handful of "subject predicate object"
  strings) for the constituent ideas.
- Read `weight` to know how many facts were merged. Higher weight =
  the cluster was strongly recurring.
- Read `summary` if non-empty — a future LLM pass may populate it
  with a one-line synthesis.
- Do not cite a MetaFact as a single source-of-truth. Ask the user
  whether the cluster is still relevant, or call `record_fact` on
  any specific statement you want to keep live.

---

## What not to store

- Session-specific observations ("user seemed frustrated") — these are not
  facts, they are session artifacts
- Ephemeral values — URLs, tokens, timestamps, version numbers that will
  change
- Anything the user has not explicitly asked to remember
- Redundant facts — if `query_memory` already has it at high similarity and
  gravity, don't duplicate

---

## memory_stats

Call at session start if you suspect memory is growing stale or the user
asks about memory health.

Interpret:

- `black_hole_mass` rising steadily — facts are failing; review what is
  being stored and whether sessions are being scored correctly
- `black_hole_meta_mass` rising while `black_hole_fact_mass` falls —
  consolidation is working: dead-fact clusters are being fused into
  MetaFacts. This is healthy compression, not a failure mode.
- `surface` count dropping — active knowledge is declining; the system may
  need fresh input
- `hawking_emissions` non-zero — dead bodies are being retrieved; the
  store may contain outdated information that keeps resurfacing
- `total_collapses` increasing without `black_hole_fact_mass` falling —
  collapses are running but not finding clusters. The collapse threshold
  may be too strict for your data.
- `active_sessions` > 0 after all agents have closed — a session was opened
  but `record_session` was never called; that context is leaking state

---

## Failure modes

| symptom | likely cause | action |
|---|---|---|
| All results have low similarity | query too vague, or memory is sparse | broaden query; prompt user for specifics |
| All results from `core` or `hawking` | facts not being touched across sessions | check that `record_session` is being called |
| `toxic` label on every session | same circular pattern repeating | inspect low-gravity facts near the topic; one may be wrong |
| `gravity_score` never rises | `record_session` not called, or called with wrong messages | verify session scoring workflow |
