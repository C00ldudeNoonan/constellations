# Changelog

## v0.5.0 - 2026-08-04

### Complete the data-quality checks: chi-squared, golden sets, LLM-judge (issue #10)

- `drift` gains a `chi_squared` metric (Pearson goodness-of-fit statistic of the
  current counts against the baseline-expected counts). Unlike the bounded
  PSI/KS/JS metrics it scales with sample size, so its `max` is calibrated per
  corpus.
- Added the `golden` test type: compare a model's rows to a checked-in golden
  model referenced by `to: ref(...)`, joined on `key`, with optional per-column
  numeric `tolerance` and an `exhaustive` flag (fail on unexpected extra rows).
  Missing golden keys and column mismatches fail; `--store-failures` persists the
  offending keys and which columns diverged. The golden model is an ordinary
  model built first as a DAG dependency (same path as `relationships`).
- Added the optional `llm_judge` test type: samples up to `sample_size` rows
  (deterministically by `seed` — a stable sort precedes sampling so warehouse row
  order cannot change the sample), asks the profile's `llm:` provider whether each
  `column` value meets `criterion` (structured boolean verdict via the shared
  inference path), and fails when the pass rate falls below `min_pass_rate`. It is
  a sampled escape hatch for subjective quality, not a deterministic gate. The
  test runner threads the resolved profile and run budget through `build` and
  `test`, so judge calls honor `llm.provider_options` and the run-wide
  `llm.budget` caps just like `llm:` models, and a project declaring `llm_judge`
  without an `llm:` profile fails preflight before any model is built.
- `golden` numeric `tolerance` now also covers `DECIMAL`/`NUMERIC` columns, and
  duplicate keys are surfaced (a golden-side duplicate errors; model-side
  duplicates fail the check) rather than silently collapsing.
- With these, #10's deterministic distribution/embedding/golden checks are
  complete; only provider-specific extensions remain.

### LLM `model_assertion` relation extractor (issue #240)

- Added the `model_assertion` extractor to `extract_relations`, the first
  built-in transform to call an LLM. Per document it asks a governed inference
  provider which of the schema-controlled `relation_types` hold between the
  in-scope candidate mention pairs, with a `confidence`; assertions at/above
  `threshold` are `asserted`, below are `no_relation`, and a pair mapped to
  conflicting types is `ambiguous`. Out-of-allow-list types and hallucinated
  mention pairs are dropped, so the model can only assert governed relations.
- It runs through the same shared structured-inference core as the native `llm:`
  kind (`extract_fields_with_usage`, `output_cardinality: many`), so caching,
  retries, and credential resolution are shared, and the run-wide `llm.budget`
  caps (`max_api_calls`, token, and cost limits) are charged and enforced per
  call. Provider, model, and credentials resolve from the profile's `llm:` block
  only; a model using it must set `transform.uses_llm: true`, enforced at compile
  via a new transform `requires_llm(options)` hook, and the resolved provider
  identity folds into the model's code version so a provider/model change
  reprocesses the table. A malformed provider response fails the run rather than
  silently deleting a document's relations, and confidences outside `[0, 1]` (or
  non-finite) are dropped.
- Evidence text and raw provider responses never enter logs or artifacts — only
  the shaped relation grain (`relation_type`, `directed`, `status`, `confidence`,
  mention IDs/offsets, extractor identity) is published. Unit tests inject a
  deterministic fake; a driver test exercises the full path offline through the
  `deterministic` provider.

### Run-over-run drift checks (issue #10)

- Added a `drift` test type: compares a column's distribution against the same
  field in a baseline model referenced by `to: ref('baseline')`, failing when
  the divergence exceeds `max`.
- Metrics: `psi` (Population Stability Index, default), `ks` (Kolmogorov-Smirnov,
  numeric only), and `jensen_shannon`. Numeric columns are compared over
  baseline-quantile bins (`bins`, default 10); categoricals over their value
  proportions. `field` maps to a differently-named baseline column.
- The baseline is an ordinary model you snapshot and reference — an explicit,
  git-reviewable comparison rather than an implicit last-run store — and it
  becomes a DAG dependency of the tested model (same path as `relationships`),
  so it is built first. Deterministic and offline; a non-numeric/numeric
  mismatch and `ks` on a categorical column fail with actionable messages.
  Chi-squared and the optional LLM-judge tier remain tracked in #10.

### Distribution data checks (issue #10)

- Added three deterministic single-column distribution test types:
  - `column_stat` — a numeric column's `mean`/`min`/`max`/`sum`/`stddev`/
    `median`/`quantile` must fall within `[min, max]`.
  - `cardinality` — distinct-value count (`min`/`max`) and/or distinct ratio
    (`min_ratio`/`max_ratio`).
  - `outlier_rate` — fraction of numeric outliers (`method: iqr` with `k`·IQR,
    or `method: zscore`) must be `<= max_rate` (default 0); honors
    `--store-failures`.
- All compute in process, validate options at compile time, skip nulls/non-finite
  values, and fail actionably on a non-numeric column. The `arxiv_papers` example
  gains `column_stat`, `outlier_rate`, and `cardinality` checks. Run-over-run
  baselines and drift metrics (PSI/KS/chi-squared) remain tracked in #10.

### Embedding-quality data checks (issue #10)

- Added four deterministic, offline test types that operate on the vector column
  of an `embed` model, extending the quality-check framework:
  - `embedding_valid` — per-vector dimensionality, finiteness, L2-norm bounds
    (`min_norm`/`max_norm`), and zero-vector rate (`max_zero_rate`, default 0).
  - `embedding_variance` — collapse guard: mean per-dimension variance must be
    `>= min_variance`.
  - `embedding_duplicates` — exact-duplicate-vector rate `<= max_rate`
    (default 0), detected by hashing.
  - `embedding_outliers` — fraction of vectors beyond `z` (default 3) standard
    deviations of the centroid distance must be `<= max_rate` (default 0).
- Each check reads only the vector column (memory proportional to the embedding
  data, not the whole relation) and computes in process — no provider call. A
  zero/NaN embedding and representation collapse are common silent provider
  failures that `not_null` cannot see. Options are validated at compile time,
  and the `economic_entity_links_embeddings` example gains `embedding_valid` +
  `embedding_variance` checks.

### Removed mypy; ty is the sole static type checker, now project-wide (issue #49)

- Completed the mypy → ty migration: `ty` has been the required, blocking type
  checker since v0.3.0 and ran clean alongside mypy for two release cycles. With
  parity confirmed, mypy is removed — the `mypy` dev dependency, the
  `[tool.mypy]` configuration, the CI and release-workflow mypy steps, and the
  mypy contributor/release guidance are all gone.
- Extended ty coverage from `src/dbt_ml` to the whole project — `tests`,
  `examples`, and `scripts` are now type-checked too (previously neither checker
  looked at them). Resolved ~160 findings this surfaced with real annotations,
  narrowing, and `cast(Any, …)` at test-double boundaries rather than broad
  suppressions. Annotation-*presence* (Ruff `ANN`/`PYI`) stays enforced on
  package source only; `[tool.ty]` rules are unchanged.

### Vertex AI Gemini inference provider (issue #17)

- Added `VertexInferenceProvider` as a core built-in, so `provider: vertex` now
  covers **both** structured generation and embeddings (the Vertex embedding
  provider shipped earlier). The inference and embedding registries stay
  separate but share the `vertex` name.

### `dbt_ref` source: consume dbt-built tables (issue #177)

- A dbt-ml transform can now read a dbt-built table by declaring
  `source: dbt_ref('<dbt_model>')`, closing the bidirectional gap in the
  embedded dbt-duckdb integration (Architecture A). These models run in embedded
  mode via `dbt-ml codegen` + `dbt build` — the standalone runner rejects a
  `dbt_ref(...)` source with an actionable error, since only dbt can resolve it.

### Fixed

- BigQuery `search:` model builds no longer fail during state reconciliation
  with `BigQuery state page read failed` (issue #249). BigQuery requires the
  table alias *before* `FOR SYSTEM_TIME AS OF …`; the state-page reader and its
  absence-probe subquery emitted it after, a syntax error.

### Dependencies

- Raised `cryptography` to `>=50.0.0` (CVE-2026-69247, flagged by `pip-audit`)
  and re-locked.

## v0.4.0 - 2026-07-31

### Typed relation extraction over entity mentions (issues #220, #240)

- Added the `dbt_ml.text.transforms.extract_relations` transform, which emits a
  child table of relations between the entity mentions of a document (one row
  per related pair), anchored on the stable mention IDs from the NLP entity
  table.
- The output grain distinguishes three relationship kinds via a `method` column
  — `co_occurrence` (proximity), `rule` (deterministic typed rules), and
  `model_assertion` (learned/LLM) — so a consumer never mistakes co-occurrence
  for a semantic assertion. Rows also carry `relation_type`, `directed`,
  `status` (`asserted`/`ambiguous`/`no_relation`), `confidence`, subject/object
  mention IDs and offsets, participating labels, and extractor identity/version.
- Two deterministic, offline extractors ship:
  - `co_occurrence` — two mentions co-occur when they share a sentence
    (`scope: sentence`) or fall within `max_char_gap` characters
    (`scope: window`). Symmetric, so each unordered pair yields one
    `directed: false` row with a stable orientation and `relation_id`.
  - `rule` — directed, typed relations from operator-declared label rules
    (`subject_label`, `object_label`, `relation_type`); the distinct rule
    relation types are the schema-controlled set the model can emit, and the
    subject/object orientation follows the rule rather than text position.
  Both support an optional `labels` allow-list and a `max_pairs_per_document`
  guard that fails closed on a pathological document; evidence text is withheld
  unless `include_mention_text: true`.
- Learned/LLM extractors slot into the same extractor registry without touching
  transform execution (the seam pattern from entity-linking resolvers); the
  generic structured-LLM path (#144) remains the way to run one. A deterministic
  fake extractor in the test suite exercises typed/directed/scored/multi-status
  rows through the driver.
- Relations materialize `full` or `incremental`; the incremental path re-derives
  exactly a changed document's relation rows (issue #218). Added
  `document_relations` (co-occurrence) and `document_typed_relations` (rule)
  models to the `economic_nlp` example.

### Entity linking: fuzzy resolver, incremental materialization, governed-context composition (issue #217)

- Added a third entity-linking resolver, `resolver: fuzzy`, that matches mention
  text against alias text by deterministic offline string similarity. `metric`
  selects `trigram_dice` (character-trigram Dice, default) or `jaccard_token`
  (whitespace-token Jaccard); both score in `[0, 1]`. A required `threshold`,
  `ambiguity_margin`, and the `matched`/`ambiguous`/`unmatched` statuses behave
  as for `vector_similarity`, and the similarity is written to `match_score`.
  Matching is case- and width-insensitive by default (`normalize: false` scores
  raw surface forms). No optional extra, network access, or credentials.
- `link_entities` now supports `materialization: incremental` for every resolver
  (issue #218): documents in the `mentions` model are the parents and the
  `aliases` model is a whole-table reference input, so an unchanged corpus
  re-links nothing while any alias edit re-links every document. Both
  `economic_entity_links` and `economic_entity_links_embeddings` example projects
  now materialize incrementally.
- Added `dbt_ml.agent_context.project_entity_link` and `entity_link_method` to
  project a matched link into the agent-context `context_entity_links` grain
  (issue #145). The link's `canonical_id` becomes the row's `entity_key`, so a
  governed metric keyed on the same namespace/name/canonical value resolves to
  the identical `entity_id` — the cross-plane join key for combining documentary
  evidence with structured metrics (issues #132/#147). Unresolved mentions
  cannot be projected.

### Declarative `for_each` matrix model expansion (issue #57)

- Added a `for_each` key to `ModelConfig`. Declaring it on a model turns it
  into a template: dbt-ml expands it into one concrete `ModelConfig` per
  cartesian-product combination of the declared axes and removes the template
  from the model list.
- Variant names follow the pattern `<base>__<axis>_<slug>__…` using
  identifier-safe slugs (letters and digits only; dots, spaces, and other
  punctuation replaced with `_`; long values truncated with an 8-character
  SHA-256 suffix). Every variant automatically receives the base model name as
  a tag so it can be selected with `--select tag:<base_name>`.
- Placeholder syntax: write `${matrix.<axis>}` anywhere in a string value in
  the model config. An exact-match placeholder (`"${matrix.min_df}"`) is
  substituted type-preservingly — an integer axis value yields an integer, a
  list yields a list, and so on. A placeholder embedded in a longer string
  (`"prefix_${matrix.label}_suffix"`) is interpolated as a string. Typed
  fields such as `chunk_size` and `dimensions` accept placeholder strings in
  the template because expansion runs on raw YAML dicts before Pydantic
  validation.
- Expansion is a deterministic pass in `config/loader.py` before dependency
  resolution, so `transform.path` values that contain placeholders are resolved
  to real paths before SQL ref discovery runs, and everything downstream — the
  DAG, selectors, runner, incremental state, manifest, and docs — sees ordinary
  concrete models.
- Up to 256 variants per template; the config validator catches empty axes and
  non-identifier axis names at YAML parse time; slug collisions and the
  expansion limit produce `ConfigError` at load time.

### BigQuery Iceberg/BigLake managed tables (issue #163)

- Added `table_format: iceberg` to BigQuery `warehouse_options`, storing a model
  as a BigLake managed Apache Iceberg table in Cloud Storage. It pairs with a
  required `connection` (a Cloud Resource connection, or `DEFAULT`) and
  `storage_uri` (a `gs://` location).
- Iceberg targets are created with explicit column DDL derived from the model's
  output schema via a new polars→BigQuery type mapping (`List`/embedding-vector
  columns become `ARRAY<T>`, structs become `STRUCT<…>`); BigQuery's unsupported
  Iceberg column types (`JSON`, `GEOGRAPHY`, `BIGNUMERIC`, `INTERVAL`) are
  rejected before any warehouse call.
- Because BigQuery Iceberg tables support neither `CREATE OR REPLACE` nor a
  truncating load, `full` models are replaced by drop → create → append — not
  atomic, gated by a new `iceberg_table_format` capability rather than
  `atomic_full_replace`. `incremental` models `MERGE`/`insert_overwrite` in
  place, and `on_schema_change: append_new_columns` evolves the schema with
  `ALTER TABLE … ADD COLUMN`.
- Interaction matrix: time partitioning only (no `int64` range), and
  `kms_key_name` is rejected with Iceberg (customer-managed encryption on managed
  Iceberg is not yet supported). The live round-trip test is gated on
  `DBT_ML_BQ_TEST_PROJECT`, `DBT_ML_BQ_TEST_CONNECTION`, and
  `DBT_ML_BQ_TEST_STORAGE_URI`.

### Deterministic keyphrase extraction (issue #219)

- Added the `dbt_ml.text.transforms.extract_keyphrases` transform, which
  extracts a ranked keyphrase child table from the NLP token child table. No
  IDF, no learned model, no optional extra — the same token table and the same
  options always produce the same output.
- Candidates are contiguous lemma n-grams (configurable `min_phrase_length` and
  `max_phrase_length`, default 1–3) formed within sentence boundaries. Boundary
  tokens must not be stop words and must not carry a POS tag in the configurable
  `stop_pos` set; interior tokens are unrestricted, so phrases like "rate of
  return" are valid 3-grams. Score is normalized term frequency (occurrence count
  / total candidate n-gram count in the document); tie-breaking is alphabetic on
  `phrase_lemma` for deterministic rank assignment.
- Output is a child table with one row per `(document_id, phrase_lemma)`:
  `phrase_id` (stable hash), `rank`, `score`, `phrase_lemma`, `phrase_length`,
  `token_start`, `token_end`, `sentence_index`, plus the five NLP identity
  columns and `extractor`/`extractor_version`. Phrase text is opt-in
  (`include_phrase_text: true`) because it is a verbatim excerpt of the source
  document.
- Supports `declared_incremental_contract` with `parent_key="document_id"` and
  `child_key="phrase_id"` — re-extracting a changed document replaces exactly
  its keyphrase rows, consistent with `nlp_tokens` and `nlp_entities`.
- Added a `document_keyphrases` model to the `examples/economic_nlp` pipeline.

### Deterministic document tone/sentiment (issue #216)

- Added the `dbt_ml.text.transforms.document_tone` transform, which scores
  per-document tone by matching the token child table against an operator-owned
  tone lexicon (`term`, `category`, optional `weight`). It is deterministic and
  reads tables, not text, so it needs no optional extra and no LLM — a general
  sentiment score is never presented as an economic fact.
- Signals are lexicon categories selected by an explicit `emit:` list, so general
  polarity (`positive`/`negative`) and domain signals (`uncertainty`,
  `hawkish`/`dovish`, …) stay separate and the output schema is fixed at compile
  time. Each emitted category produces a normalized `<category>_score` and a
  `<category>_hits` count, plus `token_count`, `matched_token_count`, `coverage`,
  and a `status`.
- Scores and coverage divide by `token_count` and are `null` — never a misleading
  `0` — when a document falls below `min_tokens` (`status: insufficient_text`).
  Optional negation flips a matched term preceded by a negator within a bounded
  same-sentence window; negators are configurable for non-English lexicons.
- The lexicon's content is fingerprinted as `lexicon_version` so an edit is
  visible to downstream invalidation without retaining the lexicon; `scorer` and
  `scorer_version` identify the deterministic path so a future learned scorer is
  additive. Tokens whose `nlp_language` disagrees with the configured `language`
  fail the run. No document text or matched phrases reach the output;
  `include_fields` carries publisher/release-date metadata onto the tone row.
- Added a `document_tone` model and committed tone lexicon to the
  `examples/economic_nlp` pipeline.

### Incremental Python transforms with child-row deletion (issue #218)

- One-to-many Python transforms may now declare
  `declared_incremental_contract(options)` and run with
  `materialization: incremental`. The runner skips parents whose input and code
  version are unchanged, invokes the transform only on changed and new parents,
  and replaces a changed parent's children by deleting on the parent key and
  upserting on the child key — so an unchanged corpus performs no provider work,
  a shrunk parent leaves no orphan child rows, and a removed parent's rows and
  state are deleted.
- The `IncrementalContract` names the output `parent_key` and `child_key`, the
  `parent_source` dependency (and its key column) that defines the parents, and
  whole-table `reference_deps`. A change to any reference dependency (such as the
  `link_entities` alias table) invalidates every parent. The contract is
  required for incremental Python transforms and validated against `depends_on`
  at compile time, so a grain mismatch is a compile error rather than a
  wrong-key delete at build time.
- `nlp_tokens`, `nlp_entities`, and `link_entities` declare the contract and can
  now materialize incrementally. State advances only after a successful
  publication; a failed publication reprocesses the affected parents on retry
  and never advances their state past the failure. The active warehouse adapter
  owns the delete, upsert, and state operations, and incremental materialization
  fails preflight on an adapter lacking the required capabilities.

### Entity-linking resolver registry + vector-similarity resolver (issue #217)

- Introduced a resolver seam behind the `link_entities` `resolver:` option so
  additional resolvers slot in without changing the output schema. The default
  remains `alias_table`; validation is a discriminated union, so an unknown
  resolver is rejected during `dbt-ml compile` with the valid names listed.
- Added the `vector_similarity` resolver, which links a mention's precomputed
  embedding to alias embeddings by `cosine`, `dot`, or `euclidean` similarity at
  or above a required `threshold`, populating the reserved `match_score` column.
  Candidates within `ambiguity_margin` of a namespace's top score are preserved
  as `ambiguous` rows rather than silently resolved to the arg-max; the
  `matched`/`ambiguous`/`unmatched` statuses, privacy defaults, and output
  schema match the alias-table resolver exactly.
- The resolver consumes vectors produced upstream by the `embed` model kind, so
  credentials, provider batching, and versioned embedding identity stay in that
  executor and the linker remains a deterministic, offline transform. It refuses
  to compare vectors from different embedding spaces: when both sides carry an
  `embedding_config_hash`, a mismatch fails the run rather than emitting
  meaningless links. The `alias_set_version` fingerprint now covers the alias
  vector set so alias embedding changes invalidate downstream.
- Added a runnable, credential-free `examples/economic_entity_links_embeddings/`
  pipeline using the built-in `deterministic` embedding provider.

### Atomic parent-scoped child replacement (issue #229)

- The three-step incremental publication sequence (delete old children, insert
  new children, advance state) is now a single atomic warehouse transaction for
  changed parents, eliminating the window where a crash left children and state
  inconsistent. A new `replace_children` adapter method wraps the three steps; a
  new `ATOMIC_PARENT_CHILD_REPLACE` capability advertises support. Both DuckDB
  and BigQuery implement it: DuckDB uses its `_transaction()` context manager;
  BigQuery stages new rows outside the transaction then runs a single
  `BEGIN TRANSACTION … COMMIT TRANSACTION` multi-statement script containing the
  DELETE, MERGE, and state MERGE.
- Removed parents still go through the existing `delete_rows_and_state` path;
  only the changed-parent child replacement is now atomic.
- `chunk`, `llm` (many-cardinality), and incremental Python transform executors
  are migrated to `replace_children`.
- `_require_incremental_capabilities` now checks for
  `ATOMIC_PARENT_CHILD_REPLACE` instead of `ATOMIC_KEYED_UPSERT`.

### Fixed

- BigQuery `embed:` → `search:` pipelines no longer fail with "Search vector
  field must use a numeric list warehouse type" (issue #226). The BigQuery
  adapter now enables Parquet list inference on every load, so `List`-typed
  columns (including embedding vectors) materialize as native `ARRAY<T>` rather
  than a nested `RECORD`, and read back as a numeric Arrow list. Tables that
  were already materialized with the old `RECORD` layout need a
  `--full-refresh` to be rewritten.

## v0.3.0 - 2026-07-28

### Deterministic alias-table entity linking (issue #217)

- Added the `dbt_ml.text.transforms.link_entities` transform, which resolves
  entity mentions to canonical economic identifiers — CIK, ticker, agency,
  ISO 3166, or project-defined keys — through an operator-owned alias table.
  Matching is deterministic and offline: `exact` compares mention text as-is,
  `normalized` applies NFKC + casefold + whitespace collapse, and configured
  methods run in order so the first method producing candidates for a namespace
  wins that namespace.
- Every mention yields an explicit outcome. A namespace resolving to one
  canonical ID is `matched`; several candidates are preserved as one
  `ambiguous` row each rather than silently guessed; nothing found is
  `unmatched`. `on_ambiguity: error` fails the run instead.
- Link rows carry a stable `entity_link_id`, resolver identity and version, and
  an `alias_set_version` fingerprint of the effective alias set, so alias-table
  edits are visible downstream without retaining alias contents. `match_score`
  is reserved (always null) so a future score-producing resolver will not change
  the published schema.
- Matched mention text is withheld unless `include_mention_text: true` is
  explicit and is never folded into `entity_link_id`; `include_fields` keeps the
  allow-list rules used by the NLP transforms.

### Document-level NLP aggregate features (issue #215)

- Added the `dbt_ml.text.transforms.nlp_document_features` transform, rolling the
  token, entity, and entity-link child tables up to one typed row per document so
  downstream dbt models and classic ML no longer each reimplement the same
  aggregation. It requires no optional extra — it reads tables, not text.
- `tokens:` is the required aggregation spine; `entities:`, `links:`, and
  `documents:` are optional. With a `documents:` dependency the parent table
  defines which documents get a row, so a document with no tokens still appears
  with zero counts instead of vanishing.
- Base features (`token_count`, `sentence_count`, `entity_count`,
  `unique_lemma_count`, `lexical_diversity`, `stop_ratio`, `alpha_ratio`) default
  to whatever the configured dependencies support. Every other rollup —
  `pos_counts`, `pos_ratios`, `entity_label_counts`, `link_namespace_counts`,
  `link_status_counts` — is an explicit list, so the output schema is fixed at
  compile time and never depends on warehouse contents.
- Ratios divide by `token_count` and are `null` at a zero denominator, never `0`
  or `NaN`; `sentence_count` is `null` when a pipeline ran without a parser; a
  configured POS or label a document never uses is `0`. Aggregation is vectorized
  polars group-bys rather than per-document Python.
- Identity columns pass through per document, and a document whose child rows
  disagree on model identity — or whose token and entity tables disagree with each
  other — fails rather than claiming one reproducible identity. No document,
  token, or entity text reaches the output.

### Compile-time reconciliation of transform dependencies

- Python transforms may now expose `declared_dependencies(options)` returning
  the complete set of model names their options require. Implementing it asserts
  that options fully determine the transform's inputs, so the compiler enforces
  that `depends_on` matches exactly. A misspelled or stale dependency reference
  in `link_entities` previously passed `compile` and failed partway through
  `build`, after upstream models had already been materialized; it is now
  rejected during preflight with a file, line, and column diagnostic.

### Classic NLP enrichment (issue #43)

- Added the `dbt_ml.text.transforms.nlp_tokens` and
  `dbt_ml.text.transforms.nlp_entities` transforms, which normalize text into
  token and named-entity child tables — stable child IDs, document ID,
  token/sentence index, offsets, lemma, POS/tag, and lexical flags for tokens;
  label, offsets, and nullable confidence for entities — without project-local
  Python.
- spaCy ships behind a new `nlp` extra, imported lazily. Language models are
  never downloaded automatically: a missing model reports the exact
  `python -m spacy download ...` command, and a missing extra reports the
  `dbt-ml[nlp]` install command. Provider, model, version, and language identity
  are published on every output row as `nlp_provider`, `nlp_model`,
  `nlp_model_version`, and `nlp_language`. The `model` option is published
  verbatim, so prefer a package name (`en_core_web_sm`) over a filesystem path
  to a locally trained pipeline if that path is sensitive.
- Transform options are validated during `compile`, before source discovery,
  model loading, credentials, or warehouse mutation.
- Upstream columns are dropped unless named in an explicit `include_fields`
  allow-list. Matched entity text is excluded by default, requires
  `include_text: true`, and is never folded into the stable entity fingerprint.
- Added the runnable `examples/economic_nlp` pipeline.

### MotherDuck deployment mode (issue #186)

- The DuckDB adapter now accepts `path: md:<database>` (or bare `md:` for the
  account default), running MotherDuck as a deployment mode of the existing
  adapter rather than a second adapter. Every materialization, state, snapshot,
  and SQL-model path is shared with local DuckDB unchanged, and local-file
  behavior is byte-identical.
- The `token` field is a credential reference, mirroring BigQuery's handling: it
  must be an exact `{{ env_var('NAME') }}` reference, is valid only on an `md:`
  path,
  and never reaches `manifest.json`, `run_results.json`, logs, or generated dbt
  sources. Omit it and DuckDB reads its own `motherduck_token` environment
  variable.
- Every advertised DuckDB capability was audited against the live service —
  atomic full replace, incremental merge, state upsert/fetch, row+state
  deletion, bounded snapshots, and SQL dry-run all pass with no fork.

### ty as the primary type checker (issue #49)

- ty is now the required primary static type checker for package source, run in
  pull-request and release CI. mypy runs alongside it as a temporary
  migration-parity gate for the bounded observation period in issue #49.
- Ruff `ANN`/`PYI` rules preserve source annotation discipline, and focused
  Pydantic static-compatibility fixtures pin checker behavior.

### Execution, adapter, and CLI boundaries (issue #190)

- Split `runner.py` into per-kind executors (chunk, SQL and Python transforms,
  extraction, embed, llm, search, ML) behind shared execution contracts, split
  `classic_ml` into a package by responsibility, and centralized the
  adapter-neutral schema-change invariant so each adapter no longer restates it.
- Moved the logic behind the CLI into an importable `cli_services/` package —
  shared project/profile bootstrap, the exit-code error contract, watch, and
  serving — keeping Click command declarations, options, help text, and
  user-facing formatting at the edge. Command names, options, exit codes, and
  safe output are unchanged.
- Retired approved compatibility debt (issue #210). **Removed:** the
  `DOCBT_PROFILES_DIR` environment alias (use `DBT_ML_PROFILES_DIR`), legacy
  LLM cache-row pruning, and legacy classic-ML artifact metadata reads — all
  dead paths for current inputs. **Reclassified:** running without a `profile:`
  is now a documented, supported zero-config local DuckDB target rather than
  deprecated, so its `DeprecationWarning` is gone. Declare a `profile:` plus
  `profiles.yml` for warehouse targets, credentials, retrieval, or LLM
  configuration.

### Golden-set retrieval evaluation (issue #137)

- Added `dbt-ml eval` and a `retrieval_tests:` block on `search:` models,
  scoring `recall`, `precision`, `hit_rate`, `mrr`, and `ndcg` at declared `at`
  cutoffs. Threshold keys (`<metric>_at_<k>`) are validated at compile time
  against those cutoffs, and `retrieval_tests` is rejected on non-search models.
- `golden_set` is an ordinary `ref()` to any dbt-ml model — no new "seed"
  concept. The ref adds a real DAG edge so a `build`/`run` materializes the
  golden set ahead of the search index; `dbt-ml eval` itself reads the existing
  golden table rather than building it, so build the project (or at least the
  golden-set model) before evaluating.
- A query with no relevant labels is diagnosed `no_relevant_labels` and excluded
  from aggregate means rather than scored as zero, which would understate
  quality and hide a labeling gap; empty results are a legitimate zero and are
  scored. `required_ids`/`excluded_ids` are hard policy failures independent of
  ranking-metric averaging, so a leaked excluded ID fails even at perfect
  recall.
- Results land in `target-path/retrieval_eval.json` with store and embedding
  identity, golden-set content hash, per-query metrics, aggregates, threshold
  outcomes, and policy violations — no secrets and no raw vectors. `dbt-ml eval`
  exits 1 on any failing test, matching `dbt-ml test`.
- `examples/rag_chunks_pipeline` gains a passing evaluation and a deliberately
  mislabeled one that always fails, so the mechanism is shown catching a real
  mismatch.

### Metric-plus-evidence agent example (issue #147)

- Added `examples/metric_evidence_agent`, combining a deterministic dbt Semantic
  Layer metric fixture with a complete dbt-ml agent-context pipeline: dbt MCP
  metrics orchestrated with all four governed dbt-ml MCP context tools, proving
  entity/time joins, citations, lineage, freshness, and reduced authorization
  results. Includes reviewed answer/tool snapshots, an offline regression test,
  and an opt-in live dbt MCP path.
- dbt MCP `query_metrics` group-by objects now carry the required `type`
  discriminator, matching the live request schema.

### Warehouse-native SQL transform models (issues #141, #143, #142)

- Added `transform.type: sql`, authored as an external `.sql` file and
  materialized **inside the warehouse** via an adapter-owned CTAS — upstream
  rows never enter the dbt-ml process. The accepted design is recorded in
  `docs/architecture/sql-models.md`.
- Templates compile in a sandboxed Jinja environment exposing only
  `ref('literal')`, `is_incremental()`, `this`, and a frozen string-only
  `target`. Ref discovery is AST-based, so dynamic or non-literal refs are
  rejected without executing the template, and a dialect-agnostic guard enforces
  a single `SELECT`.
- SQL transforms derive `depends_on` from their refs before edge validation, so
  lineage, selectors, and `state:` selection work unchanged; missing refs and
  cycles fail before any warehouse access.
- `materialization: incremental` requires `unique_key` (and forbids it
  elsewhere). The runner — not the query text — decides once per run whether
  `is_incremental()` is true, so it can never diverge from what actually
  executed; first runs and `--full-refresh` fall back to the query's own
  non-incremental branch.
- New `SQL_MODEL_MATERIALIZATION` and `SQL_INCREMENTAL_MATERIALIZATION`
  capabilities with typed `materialize_sql_full`, `materialize_sql_incremental`,
  `dry_run_sql`, and `relation_exists`. DuckDB stages into a session-scoped temp
  table then deletes matching keys and inserts transactionally; BigQuery uses an
  atomic `CREATE OR REPLACE TABLE ... AS SELECT` and a single `MERGE`. Core
  never assembles dialect CTAS. Both validate the unique key and apply
  `on_schema_change` before mutating anything.
- `code_version` folds the raw `.sql` content hash, template contract version,
  refs, `unique_key`, and `on_schema_change`, and excludes `warehouse_options`
  and target-compiled SQL so state does not churn across dialects.
- Added `examples/sql_governed_chunks`.

### Google Vertex AI embedding provider (issue #174)

- Added a built-in `vertex` embedding provider using the optional `google-genai`
  SDK behind the `vertex` extra: Vertex `v1`, ADC authentication, requested
  output dimensionality, normalized usage accounting, and sanitized billed
  failures.
- Runner batches split at Vertex model limits — one input for Gemini embedding
  models, at most five for other text embedding models — preserving input IDs
  and reporting logical batches separately from actual provider calls.
- Added operator-owned `embedding:` profile configuration with semantic
  provider-option identity, separate document and query task types for inherited
  retrieval, and preflight rejection of unsupported `api_key_env`.
- `examples/rag_chunks_pipeline` now uses native `embed:` with
  `embedding: inherit` instead of a fixture transform.

### Embedded execution in a dbt-duckdb DAG (issue #177)

- dbt-ml extraction and transform models can now run in-process as native dbt
  nodes, so a single `dbt build` executes them, tests them with native dbt
  tests, and feeds them into downstream dbt SQL models — one DAG, one lineage
  graph, no orchestrator.
- `dbt_ml.dbt_embed.materialize(model, *, project_dir, session, upstreams=...)`
  reuses the standalone runner's single-model path with a capture adapter that
  returns the output frame instead of writing a dbt-ml-owned target, and serves
  injected upstream frames so transform models resolve their inputs without a
  dbt-ml warehouse.
- `dbt-ml codegen` generates one dbt Python-model shim plus a shared
  `schema.yml` (fields and tests, reusing the `emit-dbt-sources` translation)
  for every extraction/transform model whose full dependency closure is
  embeddable. A model that depends — directly or transitively — on a
  non-embeddable chunk, embed, `llm:`, ML, or search model is skipped and
  recorded in `_SKIPPED.txt` so the generated project never references a missing
  dbt node. dbt-ml YAML stays the source of truth; the emitted files are real
  dbt nodes, so `ref()` resolves across dbt and dbt-ml and `dbt docs` shows one
  graph.
- dbt-duckdb only — warehouse-side Python runtimes sandbox network egress and
  are out of scope for embedded extraction. dbt-ml is **not** a dbt package:
  `packages.yml`/`dbt deps` cannot install a Python dependency, so the
  pip-installed `dbt-ml` package and the codegen-generated dbt resources are two
  distinct surfaces. The standalone CLI plus `emit-dbt-sources` remains the path
  for other warehouses.
- Added `examples/dbt_embed_duckdb`, proving a three-level DAG in one
  `dbt build`. The bidirectional `dbt_ref` source remains open work on #177.

### Native llm: transformation models (issue #144)

- Added a first-class `llm:` model kind that maps a prompt over an upstream
  warehouse relation, turning unstructured text into typed rows. The output
  schema is the model's own `fields:` and the prompt is inline, so no new
  file-path surface is introduced and versioning flows through `code_version`.
- `output_cardinality: one` produces one row per input keyed by `id_field`;
  `many` fans out to one row per object keyed by a deterministic `llm_row_id`,
  retaining the parent `id_field` for parent-scoped deletion.
- Every row carries `llm_provider`, `llm_model`, `llm_provider_implementation`,
  `llm_input_hash`, `llm_config_hash`, and `generated_at`. Credentials stay
  operator-owned in `profiles.yml`/environment and never enter the block,
  manifest, or logs.
- Incremental runs key on a content-plus-config fingerprint: unchanged inputs
  skip, content or prompt/schema/provider-identity changes regenerate, and
  removed inputs delete their rows. `code_version` folds the resolved provider
  identity.
- The `llm:` node and `backend: llm` extraction share one execution core, so
  provider resolution, caching, retries, and usage accounting are not
  duplicated. Added a `deterministic` inference provider so examples and tests
  run credential-free.

### Dependencies

- Raised the pypdf floor to 6.14.0 (locked 6.14.2) to clear two new public
  advisories, and bumped transitive pyasn1 to 0.6.4, restoring the
  dependency-audit gate.

## v0.2.10 - 2026-07-21

### Atomic full replacement on BigQuery (issue #171)

- Fixes a v0.2.9 regression: the BigQuery adapter now declares the
  `atomic_full_replace` capability, so full-materialization models (transforms,
  classic ML, extraction) pass the capability preflight instead of being
  rejected as non-atomic. Under the v0.2.9 capability contract (#130), every
  full-materialization model on BigQuery failed to compile.
- `materialize_full` and `materialize_full_chunks` now swap replacements in with
  a single `CREATE OR REPLACE TABLE ... AS SELECT`, which BigQuery executes
  atomically and leaves the target untouched on failure. A staged
  drop-and-rename is used only when the declared layout changes an existing
  table's partitioning spec — the one case BigQuery cannot replace atomically —
  so a bad layout still never destroys the last good table.

## v0.2.9 - 2026-07-20

### Incremental LanceDB search indexes (issue #134)

- Added a distinct `search:` DAG resource and independent `RetrievalStore`
  contract with strict capability preflight, typed predicates, safe target
  descriptors, exact mutation receipts, and manifest v2 serving-resource
  artifacts. Search indexes never masquerade as warehouse relations or appear
  as dbt source tables.
- Added an optional local LanceDB store with owned typed collections, exact or
  approximate vector indexes, full-text and scalar indexes, bounded incremental
  keyed publication, stale-row deletion, and state updates only after durable
  store acknowledgements plus successful index, count, schema, and source-
  generation validation. Physical target identity is alias-independent, so two
  aliases cannot evade collection-collision or state-scope checks.
- Public indexes require operator opt-in in `profiles.yml`. Governed access,
  portable query CLI/API, online replacement, full refresh, and concurrent
  publish/read coordination fail closed or remain follow-up work in #135 and
  #152. The RAG chunks example now provides an offline end-to-end PoC.

### Bounded warehouse snapshot reads (issue #140)

- DuckDB and BigQuery now expose an immutable `table_snapshot()` context with
  projected Arrow batches, validated batch sizes, typed predicate pushdown,
  same-snapshot NULL/uniqueness checks, safe generation fingerprints, and
  deterministic cleanup on completion, failure, or early close.
- DuckDB pins an MVCC transaction and compares a bounded second-scan content
  digest, while BigQuery pages one uncached query result and checks table
  metadata. Both reject generation changes before readiness. Predicate and row
  payloads stay out of errors; BigQuery profile cost and timeout controls still
  apply.
- `streaming_tabular_reads` and `tabular_predicate_pushdown` are explicit
  adapter capabilities with a reusable preflight helper. Eager `read_table()`
  remains for small interactive and existing model-runner paths; this change
  does not claim that every transform now streams.

### Protected credential references (issue #154)

- BigQuery service-account JSON, OAuth tokens, refresh tokens, client secrets,
  and environment-backed keyfile/token endpoints now remain opaque references
  until native Google credential construction. Inline secrets, defaults, mixed
  interpolation, cross-method credential fields, and URL user-info fail safely.
- Protected references and values share one provider/warehouse/future-retrieval
  contract. Their values and environment-variable names are excluded from
  reprs, Pydantic dumps, equality, hashing, artifacts, diagnostics, and debug
  logs; native credential-construction errors are sanitized.
- Existing exact `{{ env_var('NAME') }}` BigQuery references remain valid.
  Literal service-account JSON and OAuth secrets must move to environment
  variables; noncredential profile interpolation remains unchanged.
- The Python provider contract is now v2. `ProviderCredential(env_var, value)`
  becomes the opaque `ProviderCredential(value)` (an alias of
  `ProtectedCredential`), and `.env_var` is removed. `resolve_llm_credential()`
  now returns that protected value or `None`, not an `(env_name, value)` tuple;
  provider implementations reveal it only while constructing the native SDK.

### Metadata-aware incremental state (issue #139)

- Chunk invalidation now fingerprints the effective text and every carried
  upstream value with canonical typed serialization. Metadata-only title,
  source URI, tenant, ACL/filter, date, nested, decimal, timestamp, and binary
  changes can no longer leave stale chunk rows; stable text and positions keep
  their existing `chunk_id` values.
- Incremental state now uses a generic record key scoped by model, stage, and
  a safe serving-target fingerprint. This supports independent `chunk_id`
  publication state without collapsing multiple chunks under one document or
  reusing state after target configuration changes.
- DuckDB and BigQuery automatically migrate the legacy document-specific
  state schema without dropping rows. Unknown schemas fail closed; BigQuery
  also rejects duplicate legacy keys before migration. Legacy chunk models
  perform one metadata-fingerprint rewrite after upgrading; stable chunks keep
  their existing IDs.
- Target-row deletions and state invalidation share one adapter operation, and
  full snapshots replace scoped state atomically. Failed publication cannot
  mark a record current; retries safely replay uncommitted records.

### Clustering and topic modeling (issue #42)

- Two new executable Classic ML tasks over a document-feature matrix:
  `cluster` (providers `builtin.kmeans`, `builtin.dbscan`, `builtin.hdbscan`)
  and `topic_model` (providers `builtin.nmf`, `builtin.lda`). Backed by
  scikit-learn through a new `ml` optional extra
  (`pip install 'dbt-ml[ml]'`).
- The matrix is assembled from an upstream `features` model (long-format
  term/value rows pivoted to documents × terms) or from a dense `embedding`
  column; optional L2 row normalization (`normalize: l2`) makes Euclidean
  distance track cosine similarity for TF-IDF and embeddings.
- Each model emits its primary per-document table (cluster assignments or
  document-topic weights) plus companion tables: `<model>__topics`
  (c-TF-IDF top terms per cluster, or top terms per topic),
  `<model>__representative_docs`, and `<model>__neighbors` when
  `nearest_neighbors > 0`.
- Metrics: `inertia`/`silhouette`/`n_clusters`/`noise_points` (cluster) and
  `reconstruction_error`/`perplexity`/`topic_coherence`/`n_topics`
  (topic_model). Fitting is deterministic under `random_state` and invariant
  to warehouse row order; fitted parameters persist as JSON through the
  existing atomic artifact-publication path.
- `fit_transform` and `fit` for all providers; `predict`/`load_pretrained`
  (assigning new documents to a persisted model) for `builtin.kmeans` and
  `builtin.nmf`. Density-based clustering and `builtin.lda` are fit-only.

### Provider abstraction, plugin discovery, and billed failures (issue #71)

- Added typed, separately registered `InferenceProvider` and `EmbeddingProvider`
  contracts and moved Anthropic synchronous and native-batch execution behind
  the provider boundary. Profile resolution, compilation, caching, state
  identity, manifests, run results, usage, and cost accounting are
  provider-neutral, and semantic identity includes an explicit provider
  `implementation_version` so integration changes invalidate caches and state.
- Added entry-point provider plugin discovery that scans installed
  distributions directly (so same-named entry points cannot shadow each
  other), provider-owned `llm.provider_options:` with per-field secret/identity
  classification, and a `dbt-ml providers list` inspection command.
- Failed provider outcomes are first-class: billed failures carry typed
  `InferenceFailure` provenance and cost accounting instead of being retried
  into silent spend.

### Native warehouse embedding models (issue #138)

- Added executable `embed:` models that materialize canonical,
  provider-identified vectors in the warehouse, with embedding identity
  (provider, model, dimensions, implementation, config hash) recorded per row
  and folded into incremental state so identity changes re-embed.
- A deterministic offline embedding provider supports credential-free local
  development and testing.

### Portable search API and CLI (issue #135)

- Added the provider-neutral `dbt_ml.search()` Python API and `dbt-ml search`
  CLI with portable request/result/filter types, typed scalar predicates,
  score normalization, and core reciprocal-rank-fusion hybrid queries.
- Requests preflight declared store capabilities and fail with explicit
  capability errors instead of silently degrading.

### Bounded, resumable native batches and run budgets (issue #149)

- Native LLM batch execution is bounded and resumable: interrupted batches
  resume across runs by provider job identifier instead of resubmitting paid
  work.
- Enforceable run budgets stop execution before overspend, with ledger-backed
  usage and cost accounting.

### vLLM OpenAI-compatible provider (issue #24)

- Added a vLLM inference provider speaking the OpenAI-compatible API for
  self-hosted open-weight models, registered through the standard provider
  contract.

### Generation-fenced retrieval publication and query readiness (issue #152)

- The warehouse that owns publication state now also owns a per-scope serving
  ledger and shared-lease table: publication acquires an exclusive fenced
  claim, queries pin the active ready generation, and every transition
  re-verifies the fence so a stale publisher cannot advance state or
  readiness after administrative recovery.
- Added `dbt-ml serving status` and `dbt-ml serving recover`; recovery is
  explicit (no timeout-based lease stealing) and requires operator
  confirmation that the previous owner was terminated.
- LanceDB publication adds an OS-level publisher lock, and governed
  (non-public) indexes are now accepted end to end.

### agent_context/v1 contract (issue #145)

- Added the versioned `agent_context/v1` warehouse contract for document
  registries, chunks, and dbt entity links: stable identity, bitemporal
  validity, policy attributes, freshness, provenance, and exact citation
  locators, with cross-relation validation helpers.
- Contract metadata flows through manifests, generated docs, and
  `emit-dbt-sources`; contract-bearing transform output is validated before
  materialization and the declaration participates in model code identity.

### Paged publication-state reconciliation (issue #153)

- Search publication now reconciles its state scope in bounded memory:
  per-batch classification through bounded state key lookups, state
  advancement per batch strictly after exact durable store receipts, and
  complete, deterministic stale discovery streamed in record-key order from
  snapshot-consistent state pages (DuckDB MVCC read transaction; BigQuery
  `FOR SYSTEM_TIME AS OF`).
- Added the `paged_state_reconciliation` and `atomic_state_scope_replace`
  warehouse capabilities, including fence-checked atomic replacement of a
  scope's complete state snapshot against the serving ledger. Compile
  preflight rejects search resources on adapters without paged
  reconciliation; eager `fetch_state()` remains for materialization-scale
  callers.

### Read-only governed context MCP server (issue #146)

- Added `dbt-ml mcp serve` (optional `mcp` extra): a framework-neutral MCP
  stdio server exposing exactly four read-only tools —
  `list_context_models`, `search_context`, `get_document`, and
  `get_context_lineage` — over the `agent_context/v1` contract and portable
  search API.
- Authorization derives from an injectable principal/authorization interface,
  never from tool arguments; policy filters compile server-side, rows are
  independently rechecked on every fetch, and unauthorized or nonexistent
  resources return one indistinguishable `not_found_or_denied` error.
- Responses carry stable IDs, entity links, intervals, freshness, citations,
  and compact lineage, under configurable size, pagination, timeout,
  concurrency, and request-rate limits with structured retryable error codes.

## v0.2.8 - 2026-07-13

### Optional feature dependencies (issue #56)

- PDF, HTML, text-processing, and Presidio dependencies now ship through
  `pdf`, `html`, `text`, and `pii` extras instead of the core install.
- Optional imports are lazy and report the exact `dbt-ml[...]` installation
  command when a requested feature is unavailable; `all` installs every
  optional integration for development and comprehensive deployments.

### Ergonomics and hash hardening (issue #78)

- `--project-dir`, `--profiles-dir`, and `--target` can be placed either before
  or after project-aware subcommands, matching common dbt CLI usage.
- An unmatched `tag:` selector now produces an empty selection instead of a
  configuration error; CLI commands report `No models selected.` and succeed.
- Content, identity, code-version, artifact, and LLM-cache hashes now use
  16-byte BLAKE2b digests. Existing incremental state and LLM cache entries
  miss once after upgrading, and fitted Classic ML artifacts must be rebuilt.

### Classic ML determinism (issue #122)
- Training input is now read in a canonical order (by `document_id`/`id`
  when present, else by row content), so training hashes, vocabularies,
  model payloads, and prediction mappings no longer depend on warehouse
  row-return order.
- Proportional `min_df`/`max_df` now follow vectorizer conventions:
  `min_df: 0.5` keeps terms in at least half the documents
  (`df >= ceil`), `max_df: 0.5` keeps terms in at most half
  (`df <= floor`). An empty corpus selects no terms.
- `builtin.hashing` derives its `alternate_sign` bit from a digest byte
  independent of bucket selection; previously an even `n_features` tied
  sign to bucket parity.
- **Artifact schema bumped to v2**: features and hashes from v1 artifacts
  are not comparable, so `predict`/`load_pretrained` reject them with a
  refit hint instead of silently reusing them. Re-run `fit`/`fit_transform`
  once after upgrading.

### BigQuery model-level parity (issue #91)

- New model-level `warehouse_options:` block, opaque to core and validated by
  the active adapter — BigQuery rejects unknown/malformed keys at run time;
  adapters with no layout knobs (DuckDB) ignore the block, so one project can
  target DuckDB in dev and BigQuery in prod.
- BigQuery honors `partition_by` (time, ingestion-time, and integer-range,
  mirroring dbt-bigquery's config shape) and `cluster_by` (up to 4 columns).
  Layout applies when a table is created or fully rebuilt; changing layout on
  an existing incremental table requires `--full-refresh`. Rebuilds stage and
  validate the replacement before swapping, so a bad layout declaration never
  destroys the last good table.
- Table options: `require_partition_filter`, `partition_expiration_days`,
  `hours_to_expiration` (table TTL), `labels` (applied to the table and as
  job labels on the model's load/query jobs), and `kms_key_name` (set at
  create time via the load job's encryption configuration or CREATE DDL).
- `incremental_strategy: insert_overwrite` — dbt-bigquery's dynamic
  partition-replacement for incremental models (time partitioning with a
  field only). Requires that documents sharing a partition re-extract
  together and that a run's changed documents fit one `flush_every` batch;
  `merge` stays the default and is always correct.
- `warehouse_options` is excluded from `code_version`: declaring or tuning
  layout never reprocesses documents.
- Deferred from #91: `table_format: iceberg` (BigLake) — creating Iceberg
  tables needs explicit column DDL, which lands with #86's table contracts.

### Security

- Local `source.file_pattern` values must be relative and cannot contain `..`.
  Discovery never follows matched file or directory symlinks, hashes through
  verified file descriptors where supported, and gives parsers a hash-checked
  scratch snapshot to close discovery/fetch races. Project, source, and model
  YAML and project Python modules are likewise confined and cannot escape via
  symlinks.
- `llm.api_key_env` is now honored by synchronous extraction, Message Batches,
  and reusable helpers. The profile-owned variable wins deterministically,
  secrets never enter options/artifacts/errors, model YAML cannot select an
  arbitrary environment variable, and secret-value interpolation in
  `api_key_env` is rejected.
- `redact_pii` omits matched substrings from entity evidence by default,
  automatically drops a separately named input text column, and adds explicit
  `keep_fields` / `drop_fields`, `include_raw_text`, and `retain_input_text`
  controls. The customer-facing support-ticket example now uses an allow-list
  projection.
- `clean` no longer invokes adapter database/schema/dataset deletion. It
  removes only named local artifacts, preserves warehouses/caches/unknown
  files, rejects project-root or config-root overlap, and refuses every
  symlink component. The old `--force` option was removed.

### Reliability and correctness

- Incremental inputs now reject missing, NULL, or duplicate keys before
  mutation. DuckDB performs delete+insert and full-table publication in
  transactions; BigQuery stages each write under a unique name and uses one
  atomic `MERGE`. Cleanup errors no longer mask the primary warehouse error.
- Full extraction models publish only after every document succeeds; any
  parser/backend error preserves the prior target and state. Backend and
  zero-match warnings are retained in CLI output and `run_results.json`.
- Extraction `fields[].data_type` defines a typed output contract and enables
  successful zero-document runs to create real zero-row relations on DuckDB
  and BigQuery. Supported logical types are string, integer, float, boolean,
  date, timestamp, and JSON; contract changes invalidate incremental state.
- Selection now limits source discovery and watch paths to the selected graph,
  so unrelated GCS branches do not construct clients or consume credentials.
  Run artifacts record `sources_considered`.

### Compiler and blocker fixes

- All public configuration models reject unknown keys, source/model YAML is
  fixed at schema version 2, configured and dynamic extraction fields cannot
  overwrite lineage columns, and LLM numeric settings are bounded before any
  file, cloud, or API access.
- A shared preflight for compile/run/build/test/watch validates backend names,
  edge kinds, dependency counts, materializations, transform/custom-test
  modules and call signatures, built-in test specifications, and relationship
  targets. Relationship targets are DAG predecessors, so `build` orders them
  before the referencing test.
- All six extraction backends now publish strict typed option contracts and
  capability metadata. Compile and runtime share the same validation, so
  unknown options, wrong types, invalid field schemas, and unsafe bounds fail
  before document reads or API calls.
- Classic-ML preflight now validates executable task/provider pairs,
  provider-specific options, required fields, metrics, and artifact paths.
  Prediction artifacts are validated before the warehouse is queried.
- Project, source, model, profile, and adapter-specific warehouse validation
  errors now report the YAML file, one-based line and column, and full config
  path without including rejected input values.
- Duplicate YAML mapping keys are rejected at the second declaration while
  standard merge defaults with explicit overrides remain supported.
- Incremental and `state:modified` fingerprints now include the effective
  backend implementation and profile-merged LLM model/system settings, so
  operator-level semantic changes reprocess affected documents.
- Classic-ML artifact directories must have one compatible writer contract;
  readers preserve their data relation as `depends_on[0]` and add the writer
  as an ordering dependency. Persisted provider options and payload shapes are
  validated before warehouse reads, and metric selection/include settings now
  control emitted and persisted metrics.
- GCS sources accept an explicit `project:` and missing ADC/project inference
  now exits 2 with actionable guidance instead of a raw traceback (#105).
- Tests against a genuinely missing model relation return a structured
  `relation_exists` failure rather than leaking an adapter-specific 404 (#106).
- `show` safely replaces unsupported characters on narrow Windows console
  encodings such as cp1252 (#107).
- Custom `llm.api_key_env` resolution is consistent at compile and runtime
  (#116), and redacted outputs no longer silently persist raw evidence (#115).

### Upgrade notes

- Unknown configuration keys that were previously ignored now fail. Correct
  misspellings and keep source/model file `version: 2`.
- Extraction backend options no longer coerce quoted booleans or accept
  undocumented/ignored keys. Classic-ML roadmap tasks fail compile until they
  have an executable provider; prediction modes must use the options persisted
  in their artifact rather than declaring `ml.options` again.
- Projects that intentionally share a Classic-ML artifact path must declare a
  single fit writer and list that writer after the reader's first data
  dependency. Generic third-party pretrained files are not yet accepted.
- LLM extraction models require the configured credential variable even when
  their response cache is warm. Put `api_key_env` under profile `llm:`, not
  model options.
- Use `include_raw_text: true` or `retain_input_text: true` only when the
  resulting relation is intentionally sensitive. Use `keep_fields` for
  customer-facing redacted tables.
- `dbt-ml clean` preserves the local DuckDB warehouse; use an explicitly
  scoped administrative workflow when warehouse relations must be dropped.

### Extraction (issue #108)

- html backend: two opt-in heading detectors for corpora that style headings
  instead of using `<h1>`–`<h6>` (SEC inline-XBRL filings):
  `styled_headings: true` heuristically treats short, fully-bold leaf blocks
  as headings with levels ranked by font size, and `heading_selectors:`
  accepts explicit CSS selectors with selector order setting the level.
- `sections` entries now carry a `source` field: `"tag"`, `"selector"`, or
  `"style"`. Both detectors are off by default; existing extractions are
  unaffected.

### Observability

- Backend extraction warnings (missing json fields, empty pdf pages, html
  selectors matching nothing) are no longer dropped: the runner aggregates
  them per model as distinct message → document count, `dbt-ml run`/`build`
  print a WARNING section under each model (capped at 5 distinct messages),
  and `run_results.json` carries the full counts per model plus a run-level
  `counts.warnings` total. Warnings never change the exit code.

## v0.2.7 - 2026-07-10

### Security (issue #65)
- Project-YAML paths are now confined to the project directory: source
  `path:`, `ml.artifact.path`, the layout paths (`source-paths`,
  `model-paths`, `transform-paths`, `target-path`), and model-level llm
  `cache_path` error (exit 2) when they resolve outside it — including via
  `..` and symlinks. Sources and artifact blocks opt out explicitly with
  `external: true`; external llm caches belong in profiles.yml.
- profiles.yml paths stay trusted (operator-local config), but `dbt-ml clean`
  now requires `--force` to delete a warehouse file outside the project
  directory.
- New Trust model & filesystem boundaries section in the README.
- **Upgrade note:** projects whose sources point outside the repo must add
  `external: true` to those sources. `artifact.external` and the boundary
  checks never change `code_version` — incremental state is unaffected.

### Scale (issue #77)
- Extraction models stream rows to the warehouse every `flush_every` documents
  (default 5000) instead of accumulating the whole corpus in memory.
  Incremental models upsert rows and state per flush, so a killed run keeps
  completed chunks and the re-run processes only the remainder; full models
  stream through a `dbt_ml_staging__*` table swapped in atomically at the end.
- New `WarehouseAdapter.materialize_full_chunks` (DuckDB + BigQuery
  implementations); staging tables are hidden from `list_tables`.
- `flush_every` is excluded from `code_version`, so tuning it never
  invalidates incremental state. Empty-corpus full models now drop the target
  table on both adapters (previously DuckDB errored).

### Observability (issue #75, part 2)
- Opt-in `batch: true` on `llm` extraction options routes uncached documents
  through the Anthropic Message Batches API (50% token cost, minutes-latency;
  sync stays the default). Per-document errors stay isolated, responses land
  in the LLM cache, and `estimated_cost_usd` applies the batch discount.
- New `BaseBackend.extract_batch` hook (default: sequential loop with
  per-document error capture).

### Observability (issue #75, part 1)
- LLM extraction records token usage per model: API calls, response-cache hits,
  input/output tokens, and prompt-cache read/write tokens. Totals land on
  `ModelRunResult.metrics`, in `run_results.json`, and as a summary line after
  `dbt-ml run`.
- Optional `pricing:` block in the profile `llm:` config (USD per million
  tokens, user-supplied — no prices ship with dbt-ml) adds `estimated_cost_usd`
  to those metrics.
- New `extract_fields_with_usage` alongside `extract_fields_from_text` for
  transforms that want token accounting; the original keeps its signature.

### Orchestration (issue #87)
- `run`/`build` exit codes now distinguish success (`0`), run failure (`1`), and
  configuration/usage error (`2`) so an orchestrator can branch on the cause.
  Malformed YAML is now reported as a config error instead of an uncaught trace.
- `run`/`build` gain `--json`, printing the `run_results.json` payload to stdout
  (identical to the on-disk artifact) for machine consumption.
- `run_results.json` carries run-level metadata (warehouse target, counts,
  status, elapsed) and per-model `status` + fully-qualified output `relation`;
  `build` records skipped downstream models as `status: "skipped"`.
- `emit-dbt-sources --dagster-meta` stamps `meta.dagster.asset_key` on each
  emitted source table so dbt-ml tables map cleanly onto `dagster-dbt` assets
  (pure dbt ignores the meta).
- New `docs/orchestration-dagster.md`: native `dagster-dbt` wiring — dbt-ml
  materializes the dbt source assets a `@dbt_assets` graph depends on, via
  `get_asset_keys_by_output_name_for_source` and `dbt-ml run --json`.

## v0.1.0 (unreleased)

Initial public preview.

### Backends
- `json` — project keys from JSON objects (deterministic, no API)
- `markdown` — frontmatter + body + word count
- `pdf` — text extraction via pypdf, with empty-text warnings for scanned PDFs
- `html` — body text, CSS selectors, OpenGraph, meta tags via BeautifulSoup
- `llm` — Claude-backed structured extraction with response caching

### Pipeline mechanics
- Declarative YAML: project, sources, extraction models, transform models
- DAG via `graphlib`, `ref()` syntax, cycle detection
- Incremental materialization keyed on content + code version
- `full` / `incremental` materialization
- `target/manifest.json` and `target/run_results.json` artifacts on every run

### CLI
- `init` (with `--template {json,pdf,markdown,html}`)
- `seed`, `compile`, `graph`, `run` (with `--full-refresh`), `test`, `show`, `clean`
- `source freshness` — mtime-vs-threshold check
- `emit-dbt-sources` — write dbt-compatible `sources.yml`

### Selection + filtering
- `--select` / `--exclude` with dbt-shaped syntax: name, `name+`, `+name`, `+name+`
- `tag:` prefix for tag-based selection
- `tags:` on models and sources

### Testing
- Built-in: `not_null`, `unique`, `min_rows`, `not_empty`
- Severity: `severity: warn` downgrades fail → warn (exit 0)
- Custom Python tests: drop `tests/<module>.py` with `run(con, table_ref) -> str | None`

### Profiles
- dbt-shaped `profiles.yml` with per-target warehouse + llm config
- Lookup: `--profiles-dir` → `$DBT_ML_PROFILES_DIR` → `<project>/profiles.yml` → `~/.dbt_ml/profiles.yml`
- `--target` flag selects within active profile
- LLM cache and model id come from profile, with per-model overrides

### Composition
- `dbt-ml emit-dbt-sources` writes dbt-compatible `sources.yml` so a
  `dbt-duckdb` project can `{{ source(...) }}` dbt-ml-materialized tables in the same DuckDB file
- Worked example in `examples/dbt_consumer/` (verified end-to-end with `dbt build`)
