# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html);
sub-1.0 minors can include behavioural changes.

## [Unreleased]

Resonance feedback loop made evidence-gated and self-damping, so a noisy,
self-derived R cannot compound through the loop.

### Added
- **Encoding salience (declarative pin) — the top-down half.** `record_fact(..., salient=True)`
  marks a fact critical at write time, flooring it against disuse-absorption
  *independent of resonance* — the one case bottom-up inference can't reach
  (rare-but-critical that hasn't been exercised yet; the cold-start gap of the
  retention feature below). It is NOT a utility rating — the thesis stays "don't
  make the user rate usefulness"; this is an orthogonal, un-inferrable
  criticality signal, and it is kept honest three ways: (a) **use-it-or-lose-it
  decay** — a pin erodes by `SALIENCE_DECAY · confidence` only when its fact
  surfaces into a non-positive session (a pin that never surfaces is never
  decayed; one that keeps surfacing uselessly fades), (b) a per-namespace
  **pin budget** (`BIRCH_SALIENCE_PIN_BUDGET`, default 32) that evicts the
  *highest-gravity* pin under contention — the one needing protection least,
  which is anti-adversarial to a matured low-gravity cold-start candidate, and
  (c) **telemetry to earn its keep**: `stats.pins_created / pins_active /
  pins_resonated / pins_evicted` — a near-zero `pins_resonated/pins_created`
  over real traffic is the signal to bury the channel. New persisted
  `FactPassport.encode_salience` (migrated, legacy rows default 0).
- **Salience / irreplaceability retention.** A frequency-orthogonal cost-of-loss
  signal: a fact that is *both* unique in its namespace (no live neighbour at
  cosine ≥ `BIRCH_SALIENCE_NEIGHBOR_THRESHOLD`) *and* has proven useful
  (`avg_resonance > 0`) earns a lowered disuse-absorption floor —
  `ABSORPTION·(1 − SALIENCE_PROTECTION·irreplaceability·value)` — so a
  rare-but-critical fact (used once a year, decisive each time, no substitute)
  is not forgotten just for being rarely touched. Uniqueness alone is NOT
  enough (almost every fact is unique → would halt absorption); coupling to
  proven value targets the genuinely critical and lets junk decay. Both factors
  are frequency-orthogonal. `BIRCH_SALIENCE_PROTECTION=0` disables it; new
  `stats.salience_retained` counts distinct facts kept this way.
- **Confidence-damped gravity step.** `compute_resonance` now emits a
  `confidence` in `[0, 1]` (`agreement × corroboration`: do the three signals
  pull the same way, and is the verdict backed by more than one signal).
  Gravity moves by `effective_r = R · confidence`, so conflicting or
  single-signal sessions barely nudge gravity. `session_close` reports
  `confidence` and `effective_r` (raw `r` / `label` unchanged for transparency).
- **Contrastive / outlier-robust attribution.** A session impulse that
  contradicts a fact's established resonance history is attenuated in
  proportion to how established the fact is (`trust = n/(n+K)`,
  `K = BIRCH_CONTRAST_K`, default 5) — so a useful fact is not sunk for being
  incidentally on-topic in a session that failed for unrelated reasons. Inert
  on sign-consistent history; bounded (only shrinks, never amplifies). New
  `stats.contrastive_attenuations` counter. The trust decision reads a
  separate, un-shrunk track record (`raw_resonance_sum` / `raw_avg_resonance`,
  new persisted field, backfilled `:= resonance_sum` for legacy rows) rather
  than the gravity-side mean it already shrank — without that split the rule
  is self-referential and order-dependent (an early "good" reputation freezes
  against later decline). Only the gravity input is shrunk; the track record
  stays raw and order-independent.
- **Adversarial drift detector** (`tests/test_gravity_drift.py`) — synthetic
  facts with fixed ground-truth utility (shuffled vs creation order); asserts
  final gravity correlates with utility, not appearance order. Runs against
  real Ollama when reachable, mock otherwise (`embed_provider` fixture).
- `session_close` / `memory_stats` surface `total_echoes_cancelled` and
  `contrastive_attenuations`; `session_close` surfaces `confidence`,
  `effective_r`, `echo_outcome`.

### Changed
- **Echo is now deferred and outcome-gated.** `session_open(first_message=...)`
  arms a *pending* echo marker instead of applying a penalty immediately; the
  decision is taken at `session_close` against the current session's outcome —
  resonant ⇒ cancel (productive revisit), non-resonant ⇒ apply. The applied
  penalty is scaled by the current session's severity (a neutral return
  penalises less than a toxic one). The explicit `check_echo` tool and the
  one-shot `record_session` keep immediate apply-on-detect semantics.
- **Echo penalty magnitude is evidence-proportional and continuous.**
  `base · clamp(1 − prior_r, 0, 1)` with `base` ramping 0.6→0.8 over `prior_r`
  — a revisit to a strongly-resonant prior is barely penalised; a weak/toxic
  prior takes the full hit. Removed the forced toxic floor (`min(-0.2, …)`) and
  the hard step at `prior_r = 0.35` that penalised a marginally-better session
  harder than a marginally-worse one.

### Fixed
- **Contrastive armor now scales by consistency, not just tenure.** `trust`
  used only `n/(n+K)`, so a long but near-zero ("mushy") history got the same
  outlier protection as a strongly one-signed one. It is now
  `(n/(n+K)) · min(1, |raw_avg|/0.35)` — a marginal history stays responsive to
  contradicting sessions (its mixed nature is signal, not noise); full armor
  only once the track record reaches the resonant band.
- **`record_session` is now outcome-gated too.** It received the whole
  conversation up front but still applied echo immediately (via `check_echo`)
  before scoring it — so a productive one-shot revisit could penalise the past
  session. It now `peek_echo`s and lets the close decide, consistent with the
  streaming path. Only the explicit `check_echo` tool stays apply-now.
- **Echo cancellation keyed on `effective_r`, not the raw label.** A
  barely-resonant session that confidence damping reduced to ~neutral (e.g.
  raw 0.36, confidence 0.05 → effective_r ≈ 0.02) used to cancel a pending echo
  as "resonant". Cancellation now requires `effective_r > 0.35`; the ambiguous
  middle falls through to apply, where severity (also from effective_r) keeps
  the penalty tiny — consistent across the whole spectrum.
- **`query_memory` backfill now honours `namespace_prefix`.** The
  post-revalidation backfill path applied layer/gravity/subject/similarity
  filters but not the namespace scope, so a rare post-race backfill could leak
  cross-namespace facts — breaking MemoryBricks "reputation scoped, not global".
- **MCP error-contract tests bound to the real server.** `test_mcp_contract.py`
  replicated validators inline and had silently drifted (asserted `invalid_top_k`
  while the server returns `invalid_int`; find_similar / list_facts mirrors
  drifted too). New `test_server_contract.py` calls the real `birch.server` tool
  functions and pins the envelopes they actually return; the inline file is
  trimmed to exception-path shape pins plus the source-token guard.

### Configuration
- `BIRCH_CONTRAST_K` (default `5.0`) — agreeing-history a fact needs to earn
  ~50% protection from a contradicting session; `<= 0` disables contrastive
  attribution.

### License
- Project license changed from **Apache-2.0** to **PolyForm Noncommercial
  1.0.0** — free for noncommercial use (personal, research, education,
  nonprofit, government); commercial use requires a separate license. Added a
  top-level `LICENSE` file; `pyproject.toml` now points `license` at it.

## [0.3.0] — 2026-05-25

First tagged release. The project predates this tag — 118 commits of
development collapsed into one release note grouped by theme rather than
chronologically. Future releases will be incremental from this baseline.

### Added

#### Core memory model
- `FactPassport` — atomic subject/predicate/object triple with gravity,
  layer, deprecation lineage, EWMA-tracked recent_utility, and forecast
  stability fields.
- `MetaFact` — dense centroid bundle representing a cluster of dead
  facts; carries source lineage (`source_fact_ids`, `source_texts`), its
  own gravity surface, and participates in the live feedback loop after
  Hawking emission.
- Three live layers (surface / kinetic / core) plus a singularity at
  layer `-1`; bodies migrate per `session_close()` tick based on gravity.

#### Resonance pipeline
- Behavioural detector — pattern match on closing user messages
  ("works", "got it" vs "still broken").
- Semantic detector — cosine shift + specificity delta between session
  start and end vectors.
- Repetition detector — centroid dispersion of message vectors.
- Combined R score in `[-1, +1]`; thresholds: resonant > 0.35, toxic
  < -0.15.
- `record_session` and `session_close` accept explicit `sentiment` or
  `r_override` when heuristics would misclassify (e.g. grumpy-sounding
  technical summaries).
- Per-fact attribution weighted by query-time cosine similarity — broad
  `top_k` no longer significantly tilts unrelated facts.

#### Adaptive gravity engine
- Five learned pre-resonance weights (`w_freshness`, `w_access`,
  `w_graph`, `w_utility`, `w_stability`) trained one regularised SGD
  step per closed session, target `(R+1)/2`.
- Budget renormalisation keeps the five learned weights summing to
  `0.65` so the formula stays in `[0, 1]` (resonance weight is fixed
  at `0.35`).
- Cross-process safety: weights reload from disk under the writeback
  lock before applying SGD — concurrent processes compose, last writer
  no longer silently overwrites.

#### Echo validation (cross-session retroactive penalty)
- Closed sessions stored as K-means++ bundle of centroids per session
  topic, not a single vector.
- New session opens trigger automatic echo check at `similarity ≥ 0.68`
  to past unresolved problems; matched session's R drops into toxic
  territory and the penalty propagates to gravity of every fact that
  session touched.
- Three-tier TTL sweep on every close (penalty / resolved / default).
- One-shot idempotency — second echo on the same matched session does
  not stack penalties.

#### Black hole + Hawking emission
- `BlackHole.absorb` / `absorb_meta` are atomic three-phase ops:
  preflight dim check (raises `DimensionMismatchError` before any
  mutation), then dict entry + layer mutation, then commit to vector
  index. Failure rolls everything back. `_absorb_dead` catches per-fact
  so one mismatched body doesn't abort the whole sweep.
- Hawking emission with peek-then-commit two-phase: facts are
  resurrected (state mutation + persistence) only after the live
  ranking confirms they made the returned set.
- Hawking emit thresholds: `0.95` for single FactPassports, `0.85` for
  MetaFacts (looser because centroids drift between sources).
- Emitted MetaFact gravity scales as `0.30 + 0.10 · log10(weight)`
  capped at `0.70` — bundles re-enter weighted by collapse density.

#### Singularity compactor
- Union-Find collapse with path compression; groups dead facts above
  `0.92` cosine threshold (configurable).
- Per-dimension partitioning so a model swap leaves old-dim and new-dim
  bodies compacting independently rather than crashing on ragged numpy.
- Counter-triggered background collapse via single-worker
  ThreadPoolExecutor; `collapse_async=False` for predictable tests.
- Background collapse errors surface via `memory_stats.last_collapse_
  error` instead of staying buried in the Future.

#### Galaxy — N-body research model + feature producer
- `birch/galaxy/` simulates facts as bodies in orbit around the black
  hole; emergent dynamics include orbital decay (`dynamical_friction`),
  session kicks, Jeans collapse for cold clumps, and attention-mass
  bending of the disk.
- `Galaxy.forecast_stability` runs forward `horizon_ticks` steps and
  reads per-body proximity to the event horizon back into the live
  store via the `forecast_memory` MCP tool. Feeds adaptive gravity
  through `w_stability` — the only feature derived from the future.
- `forecast_memory` response data-version-cached so subsequent calls
  with no intervening writes hit the cache.
- Snapshot revalidation: heavy N² simulation runs lock-free; writeback
  re-checks `(data_version, mutation_version)` and aborts cleanly with
  `forecast_snapshot_stale` if the universe moved.

#### Persistence
- Default `SQLiteBackend` with WAL mode, `BEGIN IMMEDIATE` for writers,
  `busy_timeout`, `check_same_thread=False`.
- `data_version` cache invalidation across processes — every operation
  reloads from disk when another process has written; a lone process
  never reloads and stays hot.
- `_mutation_version` counter so single-process query cache invalidates
  across local writes (collapse, absorb, supersede).
- Schema migration handled at first connect: legacy DBs without
  `recent_utility`, `forecast_stability`, `w_utility`, `w_stability`,
  `layer = -1` columns auto-upgrade.
- Tolerant loaders — every loader (`load_facts`, `load_meta_facts`,
  `load_open_sessions`, `load_echo_sessions`, `load_adaptive_weights`)
  skips corrupt rows with a warning instead of failing startup. Scalar
  numeric fields run through finite + clamp gates so a NaN cell loads
  as the field default.
- Write-side `allow_nan=False` on every JSON cell + same scalar
  sanitisation on every write so radioactive data never reaches disk.
- Pluggable `StorageBackend` Protocol — Redis / Postgres / in-memory
  custom backends supported without inheritance.

#### Numpy vector index
- L2-normalised `(n, d)` matrix with single matmul cosine search.
- Preallocated buffer with geometric growth (×2) — `add()` is
  amortised O(d) instead of the old O(n·d) `np.vstack` strategy. On
  10k facts × 768 dim that's a ~30 MB matrix copy per insert versus a
  single 3 KB overwrite.
- `remove()` uses swap-with-last (O(d)) plus auto-shrink when usage
  falls below `capacity / 4` so long-running stores don't sit on peak
  allocation.
- `VectorIndex.dim` public read-only property — replaces 8 callsites
  that reached into private `_dim`; encapsulation contract preserved
  for future internal evolution.
- `DimensionMismatchError` raised loudly on dim mismatch in a populated
  index; index resets dim on full-empty so a new model can establish
  a new dim without rebuilding the store.
- `top_k <= 0` guard against argpartition's undefined behaviour on
  edge inputs.

#### Concurrency safety
- All write paths (`add_fact`, `add_facts`, `set_fact`, `query`,
  `check_echo`, `session_message`, `session_close`, `run_forecast`,
  `collapse_singularity`) wrap their writeback in `try: ... except:
  self._reload(); raise`. SQLite txn rolls back disk truth on failure;
  `_reload` re-anchors every in-memory cache to the post-rollback
  disk state.
- Closing-session race protection — `session_close` marks sid in
  `_closing_sessions` right after snapshot; `session_message` rejects
  pushes to a closing sid with structured
  `RuntimeError("session_closing")`. Late messages never silently land
  in the closed bundle.

#### MCP server surface — 19 tools
- Lifecycle: `session_open`, `session_push`, `session_close`,
  `record_session`, `check_echo`.
- Writes: `record_fact`, `record_facts`, `set_fact`.
- Reads: `query_memory`, `find_similar`, `list_facts`, `explain_fact`,
  `explain_body`.
- Lifecycle ops: `supersede_fact`, `retire_fact`, `delete_fact`,
  `delete_body`.
- Operational: `forecast_memory`, `memory_stats`.
- Full typed input-validator family — `_validate_text`,
  `_validate_spo_strings`, `_validate_optional_id`, `_validate_int`,
  `_validate_float`, `_validate_bool`, `_validate_optional_text`.
  Boundary failures return structured `{"error": "...", "field": "...",
  "hint": "..."}` instead of crashing inside core.
- `EmbeddingError` wrapped at every MCP tool that touches `embed()`.
- `DimensionMismatchError` / `ValueError` / `TypeError` wrapped at
  forecast_memory.

#### Security / robustness boundaries
- `BIRCH_MAX_FIELD_LEN` env var (default 2000, tunable 128..200000)
  caps every S/P/O / query text / session message length. Defence
  against DoS / embedding-provider billing — a 10 MB paste never
  reaches the embedding call.
- `_sanitize_for_llm` strips ASCII C0 control codes (except TAB/LF/CR),
  DEL, and zero-width Unicode (ZWSP/ZWNJ/ZWJ/BOM) at the write
  boundary. Invisible-bytes smuggling vector closed.
- `_has_instruction_markers` detects visible LLM control sequences
  (`<|im_start|>`, `[INST]`, `<<SYS>>`, llama header IDs) on retrieval.
  `query_memory` attaches per-hit `has_instruction_markers` boolean +
  top-level `injection_warnings` list + `_injection_hint` so consumers
  wrap flagged bodies before LLM context. Detection-only — never
  rewrites stored data (aggressive rewriting is itself a content-filter
  bypass surface).
- Self-defending public methods: `FactPassport.apply_resonance` /
  `MetaFact.apply_resonance` reject NaN / Infinity / non-numeric;
  `avg_resonance` returns 0.0 on non-finite mean; `compute_gravity`
  ends with `math.isfinite` check before its `min/max` clamp
  (Python's `min/max` is NOT NaN-aware).
- `__post_init__` on both dataclasses normalises direct-construction
  values through the same contract as the loader — closes library-mode
  bypass.

#### Adaptive weight sanitisation
- `AdaptiveWeights.load` tolerates corrupt rows (non-numeric, NaN
  sneaked past sanitise); falls back to hand-tuned prior with a warning
  instead of crashing init.
- `record_facts` batch-size cap via `BIRCH_RECORD_FACTS_BATCH_CAP`
  (default 500).
- Embedding HTTP retries via `BIRCH_EMBED_RETRIES` (default 2) and
  `BIRCH_EMBED_RETRY_BACKOFF_S` (default 0.5).
- All cosine thresholds centralised in `birch/thresholds.py`, every
  one overridable via `BIRCH_*` env var. `memory_stats.thresholds`
  echoes what the process actually picked up.

#### Observability
- `memory_stats` exposes layer distribution, black-hole fact/meta mass,
  Hawking emission count, total live bodies, active sessions, collapse
  counter / total / last error, adaptive weight values + train_count,
  echo counters (detected / applied / ignored), every threshold value
  with `thresholds_are_import_time` flag.
- `explain_fact` / `explain_body` decompose a body's gravity into
  per-feature contributions (freshness / access / graph / utility /
  stability / resonance). Polymorphic — handles live FactPassports,
  live MetaFacts, singularity FactPassports, and singularity MetaFacts.

#### Test suite
- 741 tests pass, 20 skipped (sentinel-skipped slow paths).
- Multi-process chaos suite behind `@pytest.mark.chaos` marker; not
  in the default `pytest` run, opt-in via `pytest -m chaos`.
- Property-based invariants via `hypothesis` (gravity bounds, layer
  migration monotonicity).
- Source-level audit tests pin contracts that span modules
  (`._dim` direct-access ban, MCP wiring of validators).
- CI runs `ruff` + `mypy` on every push.

### Removed

- `deprecate(fact_id)` is now an alias for `supersede_fact` — the
  semantic difference disappeared once supersede became the canonical
  "we now know better" path; `delete_fact` remains the only true
  destructive primitive.
- Hand-tuned gravity weights — five pre-resonance weights are now
  learned, no magic numbers in the formula.

### Notes for downstream

- Public API stability target is "additive only for v0.x minors".
  Breaking changes (e.g. removed MCP tools, renamed result fields)
  will bump to v1.0 with a migration note.
- `VectorIndex` internal storage (`_buffer`, `_size`, `_capacity`)
  was rewritten for amortised O(d) `add` — public surface
  (`__len__`, `__contains__`, `dim`, `add`, `remove`, `search`,
  `similarity`, `all_similarities`) is unchanged.
- Two-brain trial pattern with vertical-brain remains: birch-km for
  atomic SPO + gravity ranking; vertical-brain for paragraph-scale
  narrative + curated Silver/Gold layers. See AGENTS.md boundary
  table.
