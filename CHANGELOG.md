# Changelog

## Unreleased

## v0.15.3 - 2026-09-01

### pypdf floor raised to 6.16.1 for three CVEs

CVE-2026-84309 (fixed in 6.16.0), CVE-2026-84310 and CVE-2026-84311 (both
fixed in 6.16.1) were published against pypdf 6.15.0, which was this
project's floor. Anyone installing the `pdf` or `all` extra resolved to an
affected version. The floor is now `>=6.16.1`.

### An unfiltered BigQuery snapshot no longer runs a query at all (issue #441)

Two releases in a row (v0.15.1, v0.15.2) still failed on the reported table,
each having changed only how a snapshot read its query's result. The actual
failure is in query *execution*, upstream of both: BigQuery has to
materialize an anonymous destination table for a query's result, and a
projection this wide (a 768-dimensional vector beside full chunk text, over
3.6M rows) exceeds that limit outright — `responseTooLarge`, before either
result-reading path is ever reached. Its own advice, unearthed once the
error's reason code stopped being swallowed by the v0.15.2 fix: "Consider
specifying a destination table in your job configuration."

An unfiltered, explicitly projected snapshot — the shape a search publish
reading its own embeddings table back actually is — now reads straight
through the Storage Read API's own column selection, with no query job and
so no destination table for BigQuery to materialize at all. A predicated read
still runs a query, since translating predicates into the Storage API's own
row-restriction syntax was out of scope here, and streams that query's
destination table exactly as v0.15.2 already did. Either way, the typed
schema and the projection's column positions both come from the read session
itself and never from the request, preserving the ordering guarantee v0.15.2
established. The generation fence, key-domain validation, execution-project
billing, and configured timeout are unchanged on both paths.

## v0.15.2 - 2026-09-01

### BigQuery snapshots leave the REST result endpoint entirely (issue #441)

v0.15.1 moved the snapshot's schema probe from one returned row to zero on the
premise that requesting no rows would keep a wide payload out of the REST
response. That premise was wrong: `jobs.getQueryResults` rejects on the size of
the *underlying* result, not on the number of rows requested, so a relation
projecting a 768-dimensional vector beside full chunk text failed identically
at either value and `search:` models over such a relation still could not
publish.

Both remaining REST result paths are gone. The uncached query is awaited
through the jobs API without fetching a result page, and its finished
destination table is streamed through one BigQuery Storage Read session with
one stream and one queued page. Bypassing the result iterator also bypassed the
exception it raised for a failed query, so a job's terminal error is now
re-raised directly, carrying its BigQuery reason code only — never the native
message, which can quote the SQL and its row values.

The typed Arrow schema and the projection's column positions are both taken
from the read session itself. The Storage Read API does not return the
requested projection's column order, so pairing a schema obtained anywhere else
with these batches would make the positional projection select the wrong
columns. Projection, predicate binding, key-domain validation, generation
checks, typed empty results, and early cancellation are unchanged.

The read session is opened under the query execution project, matching where
the query client runs and bills and where the reference already tells
operators to grant `bigquery.readsessions`; a split-project profile would
otherwise need that role on the data project instead. `job_execution_timeout_seconds`
reaches both the session creation and the row read, and additionally bounds
the whole download — a per-call deadline alone would never trip on a stream
that keeps delivering pages slowly.

## v0.15.1 - 2026-09-01

### BigQuery snapshots keep wide payloads off the REST result path (issue #441)

BigQuery's bounded table snapshot opened its query with a one-row Arrow schema
probe over jobs.getQueryResults. On a multi-million-row relation projecting a
768-dimensional vector and full chunk text, BigQuery rejected that REST result
as too large before search publication could read its first batch. The payload
iterator also passed no Storage Read client, so it retained a second REST
fallback after the probe.

Schema discovery now requests zero rows while retaining the typed Arrow schema.
The payload is streamed through one explicitly owned BigQuery Storage Read
client with one download stream and one queued page, and the client is closed
with the adapter. Snapshot generation checks, projection, predicate binding,
typed empty results, and early cancellation remain unchanged. Generic snapshot
open failures now add only the sanitized native exception type to their public
message; native warehouse text, SQL, and row values remain hidden.

## v0.15.0 - 2026-08-31

### BigQuery user ADC keeps the scopes it was granted

The direct ADC-file path added for Windows MCP startup passed the profile's
dbt-compatible BigQuery, Cloud Platform, and Drive scopes into Google Auth's
file loader. For ordinary gcloud user ADC, that replaced the scopes granted by
`gcloud auth application-default login`; the next token refresh failed with
`invalid_scope`, so the documented default login could not open BigQuery.

ADC files now load without replacing their grant, then receive configured
scopes only when their credential type requires scoping. User ADC keeps its
original grant, service-account and external-account ADC still receive the
profile scopes, and the no-subprocess guarantee from issue #365 remains intact.
The live BigQuery pre-release gate found and now covers this path.

### BigQuery state writes cluster on their MERGE keys (issue #431)

`stel_state` is now clustered by `model_name`, `state_scope`,
`target_identity`, and `record_key`, in the same order used by its MERGE join.
This lets BigQuery prune state writes instead of scanning the full state table
for every flush. Existing v2 state tables receive an in-place metadata patch;
BigQuery reclusters their rows in the background without a copy/swap window or
lost concurrent writes. The schema-changing v1 migration keeps its verified
replacement path.

### Native llm models stream inputs and resume ids (issue #424)

Native `llm:` models no longer load their entire upstream relation, build a
corpus-sized work list, or read generated target columns into memory on resume.
A zero-row probe supplies the schema; projected id/input snapshots first
validate and classify the corpus, then fill one `flush_every` work window at a
time. Existing targets are projected to the typed id column and streamed in
fixed-size batches for deletion reconciliation.

The eager classification pass preserves two cost-safety contracts: invalid ids
still fail before the first provider call, and `max_documents` still rejects
an over-cap run before any partial publication. Streamed and whole-frame input
fingerprints are pinned as equivalent so upgrading cannot silently regenerate
an existing corpus at provider cost. When `id_field` and `input_field` name the
same upstream column, projected reads retain it once, preserving configurations
that were valid before streaming.

### Concurrent provider work no longer multiplies execution-budget caps (issue #435)

Native `llm:` workers used a check-then-act sequence: check the shared ledger,
call the provider, then charge usage. With `max_concurrent: 8`, every worker
could pass the same pre-call check and a `max_api_calls: 1` policy could submit
eight calls.

Budget admission now happens at the real provider-call boundary, after cache
lookup. An uncached call is reserved before submission and its reported API
count is settled without double-charging; cache hits still consume zero calls.
Input-token, output-token, and spend caps cannot reserve an unknown response,
so admission on a response-capped ledger is serialized until the preceding
response is charged, preserving the documented one-response overrun. Admission
ownership now belongs to each ledger: separate model guards coordinate through
their shared run ledger across LLM extraction, native batches, embeddings,
LLM checks, and model-assertion relations. Guards acquire model and run ledgers
in a stable order, avoiding deadlocks without serializing ledgers that have only
statically knowable call caps.

### BigQuery embed resumes no longer collide with their own flushes (issue #436)

An incremental `embed:` run reads reuse candidates from its own target between
flushes. BigQuery's query result is immutable, but the table metadata used by
the generation fence can advance while that result is consumed after the
previous synchronous `MERGE` has already returned. The same race occurs when a
new run overlaps a server-side write left behind by a killed predecessor.

Snapshot generation changes now have a typed adapter error. The mutable-target
reuse reader catches only that error, discards every row from the attempted
lookup, and retries the complete projected read up to three times. Continuous
changes still fail after the bound, and ordinary upstream snapshots continue
to fail immediately rather than accepting a different generation.

### Concurrency: `--threads` stops serializing streaming stages, and embed overlaps its batches (issue #432)

Two independent throttles, both invisible until measured.

- **`_SerializedAdapter` held the shared lock for a snapshot's whole context.**
  A streaming stage keeps that context open for its entire run — provider calls
  and publishes included — so under `--threads N` the first model to open a
  snapshot blocked every other one until it finished. Serialized execution, not
  serialized I/O. It arrived with the bounded-memory work: before #411 and
  #423, embed and chunk read through `read_table`, which takes the generic
  per-call lock and releases it. The lock now guards the open and the close;
  the stream runs unlocked, which is safe for the same reason
  `state_page_reader` always did it — both open a dedicated cursor — and the
  generation fence, not the lock, is what rejects a table that changed
  underneath.
- **`embed:` had no executor-level concurrency.** Provider batches went out one
  at a time and blocked, so the only overlap was whatever a provider arranged
  inside a single call: none at all for one that does not split. New
  `embed.max_concurrent` (default 8) mirrors `llm.max_concurrent` and is
  excluded from `code_version`, so tuning throughput never re-embeds a corpus.

Budget enforcement had to become atomic to allow this. `ensure_headroom` then
charge-on-return is a check-then-act: sequentially fine, but concurrent workers
all pass admission against the same pre-charge total, so a `max_api_calls: 1`
cap would buy one call per worker. `BudgetGuard.reserve_calls()` now charges a
call when it is *admitted* and settles the difference once the provider reports
what it billed. `llm:` has the same pre-existing race and is tracked as #435.

If you are on Vertex, `batch_size` and `max_concurrent` compose with
`provider_options.max_concurrent_requests`: a batch too small to split leaves
the provider's own pool with nothing to overlap.

## v0.14.0 - 2026-08-28

### Upgrading

- **A full refresh of a chunk model is no longer one atomic swap (issue
  #423).** The first window replaces the target and later windows merge into
  it, so an interrupted rebuild leaves a partially rebuilt table that the next
  run completes from state. Exactly the trade v0.13.0 made for embed and llm
  (#401), for the same reason: the previous behavior could not rebuild a large
  corpus at all.
- **A keyed BigQuery table snapshot is now two query jobs (issue #418).** The
  added statement aggregates a single column, so it is cheap next to the
  payload it stopped the warehouse from buffering, but it is billed and it
  shows up in query counts.
- **DuckDB is now bounded by a detected container ceiling (issue #412).**
  Inside a cgroup, stel hands DuckDB 75% of the limit instead of letting it
  size from host RAM. Set `memory_limit:` on the warehouse profile to choose
  the number yourself, or `memory_limit: none` to restore DuckDB's own
  default. Workstation runs are unaffected.
- No incremental state is invalidated by this release. `chunk.flush_every` is
  excluded from `code_version` like every other flush cadence, and the streamed
  reads added for chunk and embed produce byte-identical input fingerprints —
  both pinned by tests, because a silent re-chunk would cascade into a full
  re-embed at provider cost.

### The BigQuery pre-release gate now sees keyed snapshots

`docs/release.md` trusts `test_every_bigquery_operation_is_live_tested_or_listed_as_debt`
to catch BigQuery code shipping without a live test. `table_snapshot` was
invisible to it: the method lives on `WarehouseAdapter` and BigQuery overrides
only the private `_open_table_snapshot`, so neither the class introspection nor
the underscore filter reached it — the same blind spot that let `append_rows`
and `read_relation` ship uncovered in v0.10.0.

That path is what every search publication reads through, and #418 rewrote it.
It is now named in the gate and has a live integration test.

### Partitioned runs: `subset_run` past extraction, and `--read-filter` (issue #417)

`runner.py` computed `subset_run` per invocation and passed it to every
dispatch, but only extraction's signature accepted it — chunk, python
transforms, embed, and search_index all reconciled deletions ungated, so a
partitioned invocation would have had each partition silently deleting every
other partition's rows.

- All four kinds now gate their reconciliation, and search's stale-page pass
  skips the same way `rebuild` already does. Deferred, never lost: the next
  unfiltered run still reconciles.
- New `--read-filter FIELD OP VALUE` pushes a predicate into the read, so a
  partitioned job moves only its slice.
- A read-filtered transform over a multi-row parent aborted, because
  classification saw the filtered rows while the scoped reread did not: the
  read filter now rides along with the key predicate. The watch path applies
  filters too, and `--read-filter` with a `state:` selector no longer claims
  `--state` is missing.

### Embedding drift canary (issue #305)

A hosted embedding alias can re-resolve to a new snapshot with your code,
config, and input text byte-identical. Config hashes are computed from *your*
inputs and structurally cannot observe that, and the existing structural checks
pass on any well-formed vectors, so retrieval degrades silently.

The opt-in `embedding_canary` test re-embeds a handful of frozen sentences and
compares them against a committed baseline model, failing below a
`min_similarity` floor you measure for your provider.

### `stel suggest`: the analysis half (issue #361)

#381 shipped the patching half — rendering candidate rows as a reviewable diff
— but nothing produced a candidate row, so the loop never closed. A new example
project fills the contract relation: the transcript corpus in through a
`warehouse://` source, documentation gaps found by SQL, prose drafted by an
`llm:` model, and the candidate contract assembled for `stel suggest dbt
--from`. No new primitive.

### Incremental transform residency is pinned (issue #383)

The transform incident's binding constraint — `_parent_groups` materializing
every parent's rows as Python dicts — was fixed by #384 and #385, but the last
measurement on #383 predates #385 by four hours, so the shape was never
re-checked. It is now asserted rather than inferred: with *every* parent
changed (the re-key case that caused the OOM), rows arrive one `commit_every`
batch at a time and no whole-table frame is materialized. A full refresh still
reads every parent in one frame, which is the recorded exception — the python
transform contract hands the user one DataFrame — and is now pinned as such so
the bounded case is not mistaken for a general guarantee.

### Chunk models hold the bounded-memory contract (issue #423)

Found by #414's audit. Chunk was unbounded on **both** sides: it read the whole
upstream registry into one frame, and accumulated every chunk row for a single
end-of-run publish. The worst-placed of the remaining gaps — chunk feeds embed,
so a corpus could be OOM-killed here before any of embed's bounded-memory work
was reached — and chunking *amplifies*, so the accumulated output was larger
than the input it came from.

- The upstream schema comes from a zero-row probe, `document_id` is validated
  in one projected streamed pass before anything is written, and documents
  stream in batches.
- Chunks publish every `chunk.flush_every` documents (new, default 5000)
  through the same `FlushPublisher` embed and llm use, so an interrupted run
  keeps its published windows and resumes from state.
- `chunk.flush_every` is excluded from `code_version` like every other flush
  cadence: it changes execution, never chunk content, so no existing corpus
  re-chunks.
- No incremental state is invalidated: `chunk_input_hash` values are unchanged
  across the new read path, pinned by a test rather than assumed.

### The bounded-memory contract, stated and enforced (issue #414)

One root cause reached production five times through five stages (#385, #402,
#407, #411, #420). Each fix was local and correct; the sequence says the
missing piece was an invariant, not another patch.

- **Stated**: for every model kind, peak memory is O(flush window) +
  O(per-parent unit) + O(distinct keys), never O(corpus bytes). The key term is
  quantified rather than waved at — ~108 bytes per id, so ~370MB for a
  3.6M-row corpus, and a resuming embed holds two such sets. See
  `docs/architecture/bounded-memory.md`.
- **Enforced on shape**: every `adapter.read_table()` call site in `src/stel`
  is classified — bounded, an exception with its reason, or a gap with its
  tracking issue — and an AST scan fails when a new one appears unclassified.
- **Enforced on property**: stages run with the adapter instrumented to record
  how many rows they materialize at once, so a regression to a whole-table read
  fails with the call site named.
- **Two gaps found by the audit and filed**: `chunk:` (#423) and `llm:` (#424)
  still read their whole upstream. The audit corrected #414's premise that
  #410 had wired the last unbounded path.

### BigQuery search publish no longer dies on a large table (issue #418)

A keyed table snapshot computed its key-domain validation as two unpartitioned
`OVER()` analytic columns attached to the payload. Without a `PARTITION BY`,
BigQuery buffers the whole result in a single worker to produce two scalars —
including a 768-float embedding column — so publishing a search index over
920k rows failed with `Resources exceeded ... analytic OVER() clauses: 99%`,
while a 23k-row index published fine. Scale-dependent, so it shipped unnoticed;
`search` models always pass a key column, so no large corpus could be indexed
on BigQuery at all.

- The key domain is now validated by one aggregate over the key column alone,
  before the payload query starts. The payload carries no analytic frame.
  This is what the DuckDB adapter has always done; BigQuery was the outlier.
- An invalid key domain no longer costs a scan of the payload: the read is
  never started, rather than started and cancelled.
- Validation applies the read's own predicates, so a filtered read is checked
  against the rows it will actually return, and bypasses the query cache like
  the payload query it guards — a cached aggregate would describe a different
  table state than the read, and the generation fence compares table etags
  rather than query results.
- A keyed BigQuery snapshot is now two query jobs rather than one. The
  aggregate scans a single column, so it is cheap next to the payload it
  replaced buffering for, but it is billed. `docs/reference.md` and
  `docs/architecture/adapter-capabilities.md` describe the new shape.

### DuckDB respects a container ceiling (issue #412)

DuckDB sizes its buffer pool from *host* RAM, so inside a container it grows
past the limit the kernel kills at — a read bounded on stel's side still gets
OOM-killed. Found while measuring #410: streaming a table with DuckDB's default
limit grew 594MB → 1756MB across a 100k → 400k row corpus; with a limit set it
stayed flat at 450MB → 505MB.

- stel now detects a cgroup ceiling and hands DuckDB 75% of it, leaving the
  rest for the Python process. Advisory only: it never raises a limit, does
  nothing outside a container, and an explicit setting always wins.
- New DuckDB warehouse profile fields: `memory_limit` (a DuckDB size string, or
  `none` to opt out of both the setting and detection) and `temp_directory`
  (where a bounded DuckDB spills). A malformed size is rejected at profile
  validation rather than from inside the driver mid-run.
- Workstation runs are unaffected — with no container ceiling to detect,
  DuckDB keeps exactly the sizing it had before.

### A pre-0.13 search index now says how to recover (issue #413)

0.13's serving re-key (#355) left indexes published under 0.11/0.12 with their
ledger row under the old key, where nothing looks for it — and the query path
could not tell that apart from never having been published, so it said
`run 'stel run'`. That sends an operator to re-embed a corpus when a one-time
`stel serving migrate-scope` was all it needed. Reported after a live LanceDB
index went dark on upgrade and was recovered by republishing every collection.

- A query against a scope whose state is still under the pre-0.13 key now names
  `stel serving migrate-scope <model>` and says the rows and embeddings are
  intact. The extra ledger read happens only on the already-failing path.
- The v0.13.0 upgrade notes below now mention the re-key and the command; their
  omission is why this surfaced as a live outage rather than a release-note
  chore.
- The re-key is still not automatic on read: it mutates the ledger, which is
  the publisher's job and needs the publish lease a query does not hold. See
  #413 for the discussion.

### Embed input reads are bounded (issue #410)

The input half of the #401 memory work. #401 streamed the output and #407
bounded the resume lookup, but a fresh embed run still read its whole upstream
into one frame before anything else happened, so peak memory stayed O(corpus)
on precisely the run that matters. A 7.3GB chunk table climbed ~1.2GiB/min for
six minutes with zero flushes committed and zero provider calls made; a 10GiB
container would have been OOM-killed at read time.

- An embed run now takes the upstream schema from a zero-row probe, streams
  the id column once to validate it and count the corpus, and streams the rows
  themselves in batches to fill each flush window. Peak residency is a flush
  window plus a read batch, end to end.
- Upstream id validation (NULL, empty, duplicate) still runs ahead of the
  embedding loop, so a contract violation fails before any provider spend
  rather than after the corpus is paid for.
- `streaming_tabular_reads` is now required of an adapter for embed models and
  checked at preflight, not mid-run.
- No incremental state is invalidated: `input_fingerprint` values are
  unchanged across the new read path, so existing corpora do not re-embed.

## v0.13.0 - 2026-08-27

### Upgrading

- **A search index published before 0.13 needs a one-time re-key (issue #355).**
  The serving scope moved from the physical collection to the logical one, so an
  existing index keeps its publication state under a key nothing reads, and
  queries against it fail until it is moved. Run
  `stel serving migrate-scope <search-index>` once per affected index — it is
  idempotent, touches no rows, and does not republish or re-embed. *(Note added
  after release: its absence here caused a live read outage — issue #413.)*
- **Array-valued attribute filters changed shape (issue #397).** `eq`, `ne`,
  and `in` against an `array[string]` search attribute were previously
  accepted and compiled to SQL the engine could not evaluate as intended;
  they now raise, and the new `array_contains_any` operator (JSON-array
  value) is the one accepted for array fields. Update any `--filter` calls or
  saved queries that touched array attributes.
- **A full refresh of an embed or llm model is no longer one atomic swap
  (issue #401).** The first flush replaces the target and later flushes merge
  into it, so an interrupted rebuild leaves a partially rebuilt table that
  the next run completes from state. The trade is deliberate: the previous
  behavior could not rebuild a large corpus at all.
- No incremental state is invalidated by this release. The new cadence
  options (`embed.flush_every`, `llm.flush_every`) are excluded from
  `code_version`, so existing corpora do not re-embed or re-generate.

### Long-running provider stages fail cheap and resume exactly (issue #401)

The incident behind this: a 3.6M-chunk embed run spent 28 hours of paid
Vertex calls and lost every one to a MemoryError at the final end-of-run
publish.

- **Embed models publish and advance state every `flush_every` rows**
  (default 5000). Peak memory is one flush rather than three corpus-sized
  structures; a failure keeps every published window, and the re-run pays
  only for what was actually lost.
- **llm map models flush the same way** (`flush_every`, default 1000) — one
  provider call per input made them the most expensive all-or-nothing stage.
  Fan-out models publish per window through parent-scoped child replacement.
- **The resume path is bounded.** Resuming used to load the whole existing
  target (vectors included, ~90GB at 3.6M chunks) before the first provider
  call; it now streams the id column once and looks up vector reuse one
  window at a time, keyed and projected.
- **The run budget finally sees embeds.** `max_api_calls`,
  `max_input_tokens`, and `max_documents` gate embed runs; reservations
  account for provider request fan-out (a 128-text gemini-embedding batch is
  128 billed requests, and is reserved as such). A budget stop behaves like a
  crash at the same point and reports `budget_exceeded`; raising the cap
  resumes for the remainder.
- **One checkpoint ordering for every flushing stage** (`FlushPublisher`): a
  rebuild clears state before its first write, state advances only after the
  write lands, and publication failures never carry warehouse error text into
  run artifacts.

### DuckDB-native search store (issue #371)

- `type: duckdb` retrieval stores serve vector (`vss`) and BM25 (`fts`)
  search from a DuckDB file — optionally the warehouse file itself, so a
  DuckDB-warehouse project stands up no second system. Needs no extra.
- Approximate (HNSW) indexing is an explicit opt-in
  (`hnsw_experimental_persistence: true`) because DuckDB's HNSW index is not
  WAL-covered; exact search is the default and returns the same rows.
- Hybrid is core-composed RRF over both legs; per-field BM25 scoring keeps
  multi-field text models correctly fused.
- A shared store conformance suite now runs every registered retrieval store
  (LanceDB and DuckDB) against the same contract, gated on declared
  capabilities.
- `examples/agent_transcripts` gains a `dev_duckdb` target that switches
  stores by configuration alone, with tests asserting identical results
  across both.

### Governed context over the network (issues #392, #397)

- **`stel mcp serve --transport streamable-http|sse --host --port`** serves
  many callers with a per-request principal; the stdio environment resolver
  is refused at startup for network transports. Caller identity comes from
  proxy-set headers behind an authenticating proxy
  (`--trust-proxy-principal-headers`).
- **Operator-owned grants (`--grants-relation`)** move authorization from
  caller-supplied claims to a warehouse relation keyed by subject: a forged
  tenant or access-group header buys nothing. Malformed grant rows surface as
  configuration errors, not denials, and the query log records the tenant the
  policy actually filtered to rather than the tenant the caller claimed.
- **Array-valued policy attributes work** (`access_groups: array[string]`,
  the documented agent-context shape): compiled by set overlap on both the
  prefilter and the row recheck, in both authorization providers.

### CLI

- `stel build` streams a running ledger by default instead of printing one
  final table, and the verbose log channel stays alive alongside the TTY
  progress bar.

### Fixes

- DuckDB table snapshots close their Arrow reader on close, so an
  unexhausted snapshot no longer pins the database file (#408).


## v0.12.0 - 2026-08-26

### Upgrading: one reprocess, paid once (issues #363, #385)

Two changes in this release re-key incremental state, and they are shipped
together deliberately so the cost is paid once rather than twice:

- **`extraction:` models re-extract once.** Backend identity dropped the stel
  release (#363), which changes the identity payload itself. After this run,
  a release that leaves backend-reachable code alone no longer re-extracts
  anything — which is the whole point.
- **`transform: materialization: incremental` models reproject once.** The
  parent fingerprint moved to per-row digests so classification can stream
  (#385).

Cached provider responses survive both: they key on the provider contract
identity, not on backend or transform identity. Plan the first run after
upgrading accordingly, and prefer to take it at a quiet time.

### Atomic generation activation for search indexes (issue #355)

- **A rebuild builds into a private generation and activates by pointer.**
  `materialization: full`, `--full-refresh`, and an incompatible change under
  `on_index_change: rebuild` no longer refuse: the new index is built into a
  collection nothing is querying, validated, and only then does the ledger
  pointer move. The previous generation serves every read until that switch,
  so there is no window where the index is empty, half-built, or mixing two
  configurations.
- **The serving scope keys on the logical collection**, so the ledger stays
  readable while the physical collection behind a name is replaced;
  `active_generation` became a pointer to follow rather than a value to
  compare.
- **Superseded generations are retired.** The collection a rebuild replaced,
  and a half-built one left by a publisher that died before activating, are
  both reclaimed — under the publish lease, where "is anything reading this?"
  is already answered by the existing lease protocol, so no timer or
  generations table was needed.

### `stel suggest dbt` (issue #361)

The feedback loop ended at measurement; this is the half that changes a file.
`stel suggest dbt --from <relation> --dbt-project <dir>` reads candidate
suggestions and renders them as a unified diff against the dbt project.
Nothing is written without `--write`, and nothing is ever committed —
promotion of a suggestion stays a human act, reviewed as a diff where the
context actually lives.

### Incremental transform memory follows the change set (issue #385)

- **The parent table is no longer read whole.** Classification streams it and
  keeps one digest per row (~32 bytes) instead of the row, then rows come back
  only for the parents that actually changed, pushed down as an `IN`
  predicate. Peak residency stops scaling with the corpus and starts scaling
  with what changed — the remaining term after #384.
- **The parent fingerprint now hashes per-row digests** rather than the rows
  themselves (`version=2`). Coverage is deliberately unchanged: every column
  still participates, order still does not matter, and repeated rows still
  count — sorting digests preserves multiplicity where an XOR or sum
  accumulator would have silently absorbed it.
- **One-time cost:** the encoding change re-keys every parent, so the first
  run after adoption reprocesses each incremental transform once, then runs
  incrementally as before. Same shape as #363, and pinned in
  `test_frozen_names.py` so it can never happen unremarked.
- Incremental transforms now require the adapter to advertise
  `STREAMING_TABULAR_READS`; both shipped adapters do, and the alternative is
  the whole-corpus read this change exists to remove.

### Promoting candidates into golden sets (issue #380)

- **`golden_sets/<name>.yml`** is the promotion artifact: a reviewed file in
  the project, not a warehouse write, because a promotion is a human
  judgement that wants git review, blame, and revert.
  `stel.promotion.golden_set` materializes it into the relation
  `retrieval_tests.golden_set` already refs — **the evals need no changes at
  all**, closing #329 phase 3's loop.
- **Every promoted query names its sessions**, and `query_text` is required
  and human-owned: the corpus records only a fingerprint unless text capture
  was opted into, and `retrieval_tests` replays each query through `search()`,
  so the reviewer supplying the text is what makes a row re-runnable.
- **The file's `id_space` is checked against the target index's `id_field`.**
  A set promoted in the wrong id space matches nothing and reports zero
  recall — indistinguishable from broken retrieval — so the mismatch is a hard
  error naming both spaces.

### Candidate retrieval judgments from the transcript corpus (issue #380)

- **`stel.transcripts.transforms.retrieval_judgments`** derives #329 phase 3's
  raw material: one candidate row per id a `search_context` call returned,
  judged `cited`, `returned_not_cited`, or `zero_result`. Failed calls
  contribute nothing — a denied or timed-out search returned nothing because
  it failed, not because the corpus lacked a match.
- **Candidates, never goldens.** Nothing in the relation is read by
  `retrieval_tests:` or `eval:`, and nothing promotes itself at any
  confidence (#329 rule 2). `returned_not_cited` is recorded so a reviewer can
  see what else came back, and is explicitly not evidence of irrelevance.
- Every row carries the provenance a reviewer asks for first — session,
  harness, exchange and call ordinal, timestamp — plus `id_space`, so
  promotion reconciles the candidate against the target index's `id_field`
  instead of assuming which id space it is in.
- `examples/agent_transcripts/` gains the model, and its fixtures now include
  a cited retrieval and a zero-result query.

### Transcripts capture stel's own MCP calls (issue #380)

- **`transcript/v1.1`** adds `context_calls` to each exchange: for a
  `search_context` call against stel's own MCP server, the converter now keeps
  the context model, `query_fingerprint`, returned `context_id`s/`chunk_id`s,
  the zero-result flag, and which returned ids the assistant named in the
  prose that followed. A failed call records its MCP `error_code` instead and
  is never counted as a zero result. This is a deliberate, narrow exception to the #360
  reduction — the result body is still dropped — because that reduction
  otherwise discards exactly the retrieval judgment #329 phase 3 consumes.
- The fingerprint uses the same function and domain as the MCP query log, so a
  transcript row joins to a served-side log row without a second convention.
  Recognition keys on the response's own `mcp_context/v1` marker, not on the
  MCP server name (which the operator chooses), so a similarly named tool from
  another server is not mistaken for stel's.
- Query **text** stays out unless `stel transcripts convert|sync
  --capture-context-queries` is passed, mirroring the query log's separate
  `capture_query_text` opt-in. The field is additive: `transcript/v1`
  documents remain valid and read back with no context calls.

### Agent transcript pipeline: stel can read its own exhaust (issue #360)

- **`stel transcripts convert` / `sync`** turn Claude Code and Codex session
  transcripts into `transcript/v1` landing documents — one reduced JSON per
  session, consumed downstream as an ordinary local `json` source. The
  exchange is the unit: each `## [<ordinal>] <prompt>` heading lets the
  existing chunker attribute every chunk to the request that produced it.
  Reduction is the contract: prompts (truncated beyond 4,000 chars) and
  assistant prose land; thinking blocks, tool result bodies, tool argument
  values, sidechain and meta records never do — each tool call reduces to
  name, argument fingerprint, outcome, and dropped-output byte count, and
  file-bearing arguments become per-exchange `files_touched`. `sync` skips
  live sessions (`--min-idle-seconds`, default 300) and writes atomically per
  session id, so a grown session re-lands exactly one document.
- **`examples/agent_transcripts/`** composes the full pipeline with shipped
  primitives only: landing files → json extraction → heading-attributed
  chunks → `agent_context/v1` wrappers (incremental, registry as a keyed
  reference dep) → deterministic embeddings → a governed search index with
  `harness`, `exchange_heading`, `tools_used`, and `files_touched` attributes.

### Example wrappers construct frames from the contract schema (issue #366)

- The `agent_context_from_builtin_pipeline` transforms build their output with
  an explicit schema from `empty_agent_context_frame(grain).schema` instead of
  polars inference, and `docs/mcp.md` documents the pattern: a batch that is
  all-null in an optional column (e.g. `citation_section_path`) otherwise
  infers a Null-typed column that fails — or quietly drifts the schema — on
  the first mixed batch.

### Keyed reference deps: incremental transforms stop reprojecting the corpus (issue #364)

- **`ReferenceDep(name, join_key=...)`** in `IncrementalContract.reference_deps`
  scopes a reference dependency's invalidation to each parent's joined rows:
  the join-key column holds parent identities, and a changed reference row now
  reprojects only the parents it belongs to instead of flipping every parent
  stale. Plain-string entries keep the table-scoped semantics byte-for-byte,
  so existing contracts see no state re-key. Declaring a join key asserts the
  transform derives a parent's children from that reference only through its
  joined rows; null or empty join-key values are rejected at run time.
- The `agent_context_from_builtin_pipeline` example's chunks wrapper is now
  `materialization: incremental` with the registry as a keyed reference — the
  prescribed wrapper pattern no longer pays a corpus-wide reprojection on
  every registry change. Switching an existing transform to a keyed reference
  changes its module source, so the first run after the switch reprocesses
  every parent once and runs incrementally from then on.

### Extraction state now survives releases (issue #363)

- **Backend identity no longer embeds the stel release.**
  `implementation_identity()` hashes source digests only — the backend class,
  its module, `BaseBackend`, and every stel module the backend module
  transitively imports — so upgrading stel re-keys `extraction:` state only
  when a release actually changes backend-reachable code. Third-party parser
  upgrades still invalidate, via the new `parser_identity()` (beautifulsoup4
  for `html`, pypdf for `pdf`); the release-tagged `version()` string remains
  on extracted rows as provenance but no longer enters `code_version`.
- **One-time migration cost:** adopting this release changes the identity
  payload itself, so the first run re-extracts every corpus once (cached
  provider responses survive — they key on the provider contract identity).
  After that, extraction state is stable across releases that leave backend
  code untouched.

### The concept cloud becomes the star map (issue #345)

- **Concepts are positioned by meaning.** `--embed-model` places each concept
  at the centroid of its mention vectors — which the pipeline already
  computed — projected to 3D at export time. No provider calls, and only
  coordinates enter the bundle: never vectors, never text. In the viewer,
  positioned concepts are pinned; a force simulation would erase the
  projection's semantics in seconds, which is a recorded deviation from the
  design comment's "seed then relax".
- **Categorical dimensions are a feedback loop, not a hardcode.**
  `--with-query-log` joins the MCP query log's `returned_chunk_ids` through
  the linking rows to concepts and buckets the counts into a `retrieval`
  dimension — hot, warm, cold, and `never`, the most actionable value on the
  map. Aggregate-only by construction. `--dimension name=model.column` turns
  any concept-keyed categorical column into a dimension; enum fields (#304)
  fit directly.
- **The artifact grew a color-by picker** over entity type plus every declared
  dimension, with per-value legend toggles; retrieval heat renders as
  temperature rather than arbitrary category colors. The dbt DAG plane became
  **lineage mode**, off by default — the star map is the primary view, and the
  toggle drops the plane in beneath with the beams. The page opens focused on
  the hottest retrieved concept instead of the whole graph.
- **Bundle contract v2** (`schema_version: "2"`): `position` and `dimensions`
  on concepts, `DimensionDef` declarations on the bundle, values validated
  against their declared sets at construction. v1 bundles are rejected by the
  version gate as before.
- **It looks like a star map now.** Concepts render as glowing star sprites
  (tinted live by the active dimension), a faint two-layer starfield drifts
  slowly behind a nebula-tinted sky, depth fog fades distant stars, the
  fourteen brightest concepts carry sky-chart labels, and concept edges read
  as constellation lines — faint silver until a star is selected, when its own
  constellation brightens and the rest recede. Rendering degrades to the
  library's defaults when the THREE namespace is unavailable, and the page
  stays fully self-contained.

### GCS listing pushdown composes across filters (issue #378)

`_static_filter_prefix` narrowed the listing only when every `--source-filter`
glob shared one static leading segment, and returned "list everything" the
moment they disagreed — so one filter got the #348 win and two gave it back
entirely, which is exactly what a batched backfill passes. It now lists the
narrowest covering set of prefixes separately and unions the results, with the
scan ceiling applied to the total.

### Incremental transforms commit in batches (issue #379)

Extraction checkpoints as it goes; incremental transforms did not — they
invoked the module over every changed parent, then published and advanced
state in a single call at the end, so a failure at parent 90-of-100 re-paid
all 100. Changed parents are now processed and committed in batches, each
advancing its own parents' state, so a relaunch reclassifies only what did not
finish.

### Parent groups stream rather than materializing the corpus (issue #383)

`_parent_groups` held every row of the parent table as Python dicts, grouped
by parent, purely to fingerprint them — an overhead larger than the Arrow data
it came from. Groups are now produced one at a time, gathered by recorded row
indices rather than by materializing every partition up front.

### Fixed

- **`stel mcp serve` no longer hangs on Windows with gcloud user credentials**
  (issue #365). `google.auth.default()` resolves gcloud-SDK credentials by
  shelling out to `gcloud config config-helper`; under stdio serving that
  child inherits the MCP protocol pipes and never exits, so every warehouse
  call reported a retryable "timeout" that no retry could clear. The ADC file
  is now loaded directly, with no subprocess. The server also opens the
  warehouse once at startup, so a broken credential setup fails loudly at boot
  instead of mid-session.
- **All text I/O specifies `utf-8`** (issue #369). `read_text()`,
  `write_text()`, and text-mode `open()` otherwise fall back to the platform
  locale, which is not UTF-8 on Windows — so a non-ASCII character in project
  YAML, a source document, or a manifest became a decode failure that
  reproduced on one machine and nowhere else. Three of the affected call sites
  were config and source-document reads. The Python standards this follows are
  now recorded in `AGENTS.md`.

### Documentation

- MotherDuck is documented as the shipped managed deployment of the DuckDB
  adapter rather than a planned one (issue #186), and every `stel init`
  template now carries a commented-out MotherDuck target so the local-dev
  graduation path is visible at init.

## v0.11.0 - 2026-08-23

### Rename follow-up: the residual `dbt_ml` spellings each have a recorded decision (issue #321)

- **Scratch directories are renamed to `stel_fetch_*`** — the one residual
  spelling that is safe to move, because nothing persists across runs, no hash
  consumes it, and no consumer reads it. The startup sweep still recognizes
  the pre-rename prefix: it only ever sees directories a dead process left
  behind, so dropping the old name would not fail anything, it would silently
  leak every such directory forever.
- **The owner-PID marker is deliberately *not* renamed with it.** A missing
  marker reads as "not live", so a newer stel would sweep a staging directory
  an older one is still using. No benefit, real risk — now recorded next to
  the name rather than left to be rediscovered.
- **The migration machinery stays, with a written deletion criterion.**
  `stel migrate`, the `LEGACY_*` name constants, and the
  `LegacyWarehouseNamesError` guards are one removable unit, and the criterion
  is verification rather than a date: no warehouse stel connects to still
  carrying pre-rename names. Local DuckDB files under old `target/`
  directories are warehouses too, and each one still holding `dbt_ml_state` is
  a corpus that gets silently reprocessed at provider cost the moment the
  guard stops looking for it.
- The remaining spellings — the backend identity payload key, the run-results
  version key, the emitted dbt source prefix and meta namespace, and the
  classic-ML artifact runtime key — were already pinned in
  `test_frozen_names.py`, each with the reason it cannot move. They are keys
  inside hashes or inside files already on disk, so renaming them invalidates
  incremental state, artifacts, or a consumer's committed dbt models. "Keep
  permanently" is the decision, and it is now stated rather than implied.

### GCS discovery: the cap counts documents, and `--source-filter` narrows the listing (issue #348)

- **`max_objects` counted the raw listing, before `file_pattern`.** A
  production backfill of 511 per-ticker partitions died at run 25 and every
  run after it: a sibling pipeline had written 9,690 `.json`/`.md` sidecars
  under the same prefix, pushing the raw count past 30,000 while the `.htm`
  count the source actually reads stayed at 19,854. The runs failed at
  discovery, over objects the pattern would have discarded. The cap now counts
  matching documents.
- **A too-broad prefix still fails loudly.** Counting matches would otherwise
  remove the protection the raw count was really providing, so a listing that
  keeps scanning far past the cap without finishing is refused — a typo'd
  prefix crawls loudly rather than quietly.
- **`--source-filter` is pushed into the listing prefix.** A run with
  `--source-filter 'AMAT/*'` listed 30,000 objects and kept about 40. When
  every glob shares a static leading segment, that segment narrows the listing
  itself, which is what makes per-partition runs cheap rather than merely
  possible.
- **Only whole segments qualify.** `AMAT*` does not narrow, because it would
  exclude `AMATX/…` which the glob matches; a glob starting with a wildcard
  narrows nothing; and globs with no shared static segment fall back to the
  old listing. Document identity is unchanged — the source-relative path still
  carries the segment the listing prefix absorbed, so no `document_id` moves.
- The filter reaches backends as a **listing hint**. The authoritative filter
  is still applied by the runner over the returned refs, so a backend that
  narrows incorrectly cannot silently drop documents from a run — it can only
  fail to make the listing cheaper.

### Vertex embedding: pack requests to a token budget and issue them concurrently (issue #350)

- **Every embed batch was chopped into 5-text requests issued one at a time.**
  On a production-shaped run that was 3,796 calls for 18,688 chunks — 4.92
  texts per call, and essentially the entire 56-minute wall time. At full
  corpus scale it projects to ~800k sequential calls, roughly eight days of
  waiting for about $69 of tokens. The per-call latency was never the problem;
  the call *count* was.
- **Requests now pack to a token budget** rather than a hardcoded count.
  Measured on the same corpus shape, that is **8x fewer calls** (3,738 → 467,
  averaging 40 texts per request).
- **Splits are issued concurrently**, bounded by `max_concurrent_requests`
  (default 8). Combined with the packing that turns 3,738 sequential round
  trips into ~58 — the difference between a multi-day backfill and an
  afternoon.
- **Responses are still parsed in request order**, so usage accumulation, the
  billed-request count on a failure, and the positional match between vectors
  and `input_ids` are exactly what the serial loop produced. A concurrent
  failure surfaces the lowest-indexed error, which is the one the serial loop
  would have raised first.
- **The new knobs are execution options, not semantic ones**
  (`max_texts_per_request`, `max_tokens_per_request`,
  `max_concurrent_requests`). Tuning throughput must not change the embedding
  identity — that would invalidate every stored vector to make the pipeline
  faster.
- The token estimate is deliberately pessimistic (3.0 chars/token against a
  ~3.9 English average; ~5.7 measured on real filing prose), because
  overshooting the per-request ceiling fails the call while undershooting only
  costs throughput. Raise `max_tokens_per_request` toward the ~20k ceiling if
  you have measured your own corpus.
- `gemini-embedding-*` models still take one text per request: that is a model
  limit, not a tuning choice, and the budget does not override it.

### `on_index_change: online` widens a live search index in place (issue #344)

- **Adding a filterable attribute no longer costs a rebuild.** With
  `on_index_change: online`, a change classification calls compatible — a new
  attribute, or a wider `display_fields`/`return_text_fields` — is applied to
  the published collection instead of refused. The columns are added in place
  and the rows republished from the warehouse to fill them.
- **No embeddings are recomputed.** Vectors come from the upstream table, so
  the cost is an index rewrite rather than provider spend. On a 20k-document
  corpus that is the difference between adding one filter column and paying to
  embed the whole thing again.
- **Only for compatible changes, and only where the store can do it.** A
  changed vector dimension, metric, id mapping, analyzer, or an existing
  attribute's type or filter role is still refused under `online` — the rows
  already written are genuinely invalid, and a capability flag cannot change
  that. Stores advertise `online_schema_evolution`; the policy is rejected at
  compile time against one that cannot widen a live collection.
- **`on_index_change: rebuild` is still rejected at compile time**, now with an
  error that says why: atomic full replacement needs a store that can prove
  atomic generation activation, and none does yet. The previous message
  implied both remaining modes were merely unimplemented.

### The BigQuery pre-release gate now covers what it claims to (issue #347)

- **`docs/release.md` names `tests/test_bigquery_adapter.py` as the gate for
  warehouse-adapter changes, and it did not test the adapter operations
  v0.10.0 added.** `append_rows`, `read_relation`, and `relation_row_count`
  appeared nowhere in that file — not live, not even against the fake client —
  so the documented gate passed while none of them had ever run against
  BigQuery. A gate that reports success on code it does not execute is worse
  than no gate, because it is trusted. Neither method is abstract on
  `WarehouseAdapter`, so implementing the ABC never forced coverage.
- **Live tests for the three**, including the case with real blast radius: an
  append carrying a new column must widen the table rather than fail, since
  log writes are best-effort and a failure there degrades to a warning that
  could go unnoticed indefinitely.
- **A live test for `rename_table`**, the operation `stel migrate` is built
  on. BigQuery has no `ALTER TABLE ... RENAME`, so the adapter copies under
  the new name and drops the old one only once the copy exists — dialect
  specific, moving real data, and previously proven only against a fake.
- **A ratchet so this cannot recur.** A new BigQuery-specific operation
  without a live test now fails the ordinary suite, and the operations that
  still lack one are listed explicitly instead of being invisible. The list
  may only shrink. Of 27 BigQuery operations, 5 had live coverage before this
  change and 10 do now; the remaining 17 are recorded as debt, including
  `materialize_full_chunks`, `materialize_sql_full`/`_incremental`, and
  `replace_children`.
### Search index configuration changes are classified, not just detected (issue #344)

- **Tuning `batch_size` no longer forces a full re-embed.** A published
  collection's fingerprint hashed the *whole* `search:` block, so fields that
  only change execution cadence invalidated the index: changing `batch_size`
  — how many rows a publish sends per call, never what a row contains —
  demanded a blue/green rebuild of the collection at full embedding cost.
  `index_options` did the same and is not read by any code at all.
- **`on_index_change` was inside the fingerprint that decides how to react to
  a fingerprint change.** Adopting a non-default policy therefore tripped the
  very `fail` gate it was adopted to escape. Latent rather than reported,
  because the compiler still rejects the other modes — it would have surfaced
  the moment `rebuild` shipped.
- **A change is now named.** Collections record a semantic descriptor rather
  than only its digest, and a difference is classified per the change table in
  `docs/architecture/semantic-retrieval.md`: an added attribute or a wider
  projection is *compatible*; a changed vector dimension, metric, id mapping,
  analyzer, or an existing attribute's type or filter role is
  *rebuild-required*. The failure says which field forced it instead of
  "configuration changed".
- `on_index_change: fail` remains the default and the only supported policy —
  `rebuild` and `online` are still rejected at compile time — so a classified
  change still stops the run. What changed is that the operator is told what
  moved, and whether the existing collection could have served it.
- **Existing collections are re-stamped in place, never rebuilt.** A collection
  published before descriptors existed carries only the older stamp; the first
  publish after upgrading recomputes that stamp to prove the configuration is
  unchanged and rewrites it. Rows are untouched and there is nothing to run by
  hand. Had the descriptor simply been narrowed, every live index would have
  failed its next publish and demanded exactly the rebuild this change exists
  to remove.

## v0.10.0 - 2026-08-21

### `chunk:` attributes each chunk to its heading (issue #332)

- **`chunk.headings.pattern`** detects section headings while splitting and
  emits a `section` column: the last heading at or before each chunk's start.
  The column rides the embed step's passthrough onto the search index, turning
  "risk factors mentioning tariffs" into `section = 'Item 1A'` plus similarity
  rather than similarity alone.
- **Attribution is exact, not heuristic.** The splitter sees the full document
  and every boundary offset; a downstream transform sees only chunk texts and
  has to re-derive membership from fragments — missing the cases offsets
  settle outright, such as a chunk carrying the next heading in its tail.
- A capture group in the pattern names the section, so the author decides
  whether trailing punctuation belongs to it rather than stel guessing.
- Text before the first heading has no section. `headings:` requires
  `strategy: recursive` (attribution needs source offsets the token splitter
  does not produce); naming a column the upstream already has, or one the
  chunk model generates itself (`chunk_id`, `chunk_index`, …), fails at config
  load rather than overwriting it.
- The section column is created with an explicit string type, so a first batch
  whose pattern matches no headings cannot fix it as an integer column and
  strand every later batch that does find one.
- Complements in-text metadata (#308) rather than overlapping it: that puts
  document context into the embedded text; this is a structured, filterable
  attribute.

### Fixed: DuckDB sessions are pinned to UTC (issue #339)

- **A genuinely-UTC timestamp read back bearing the developer's local
  offset.** DuckDB defaults `TimeZone` to the host's zone and converts
  `TIMESTAMP WITH TIME ZONE` values into it on read, so the publish-time
  "search timestamp attributes must be UTC" check rejected valid data — making
  `timestamp` search attributes unusable for anyone not sitting in UTC, and
  invisible in CI, whose runners are UTC.
- Every session stel opens is now pinned to UTC, **including cursors**:
  `connection.cursor()` starts a fresh session rather than inheriting the
  parent's, so the Arrow snapshot path stayed host-local until it was pinned
  too. Cursors are created through one helper so the next one cannot
  reintroduce it.
- **Content fingerprints were never affected**, verified before changing
  anything: `hashing.canonical_json` normalizes aware datetimes with
  `astimezone(UTC)` before serializing, so incremental state and content
  hashes are identical either way. A hash that differed by developer timezone
  would have been a far worse bug than the one reported.
- The per-`data_type` filter round-trip test now covers `timestamp` too, which
  it could not while publication rejected the value.

### `chunk_overlap` now snaps to a separator boundary (issue #331)

- **Overlap stepped back an exact number of characters**, landing wherever the
  count fell — mid-word for most chunks, which defeated the recursive
  splitter's separator hierarchy for every chunk after the first. Measured
  downstream on a real 835k-char 10-K: **81.5%** of chunks started mid-word
  with `overlap: 100`, against 4.2% with overlap disabled. Work upstream to
  emit real paragraph structure showed almost no boundary improvement because
  the overlap was reintroducing arbitrary offsets on its own.
- The splitter now steps back *approximately* `chunk_overlap` and snaps to the
  nearest break in the hierarchy it already walks, preferring the strongest
  available. On a structured document of comparable shape this takes broken
  words from **79.9% to 0%**, with 84% of chunks starting at a sentence or
  paragraph boundary.
- Snapping is bounded to between half and twice the requested overlap, and
  falls back to an exact slice when no boundary exists in that band. Like
  `chunk_size`, `chunk_overlap` is a target rather than an exact count.
- **Upgrade note.** This changes chunk text for any model using
  `chunk_overlap > 0`, so `chunk_id` changes and those chunks re-embed once.
  Projects that set `chunk_overlap: 0` to work around the old behavior are
  unaffected until they re-enable it — which is now worth doing.

### Vertex reuses its API client instead of rebuilding it per request (#335)

- **The Vertex provider constructed a fresh `genai.Client` on every embed
  batch and every inference call.** Each construction re-runs
  `google.auth.default()`, which under end-user ADC includes a token refresh
  round trip — measured at a 1:1 construction-to-request ratio, 4,252
  redundant credential resolutions in a single production inference run. Cost
  was unaffected; wall time was the blocker, enough to put a 3.93M-chunk
  backfill at days rather than hours.
- Clients are now cached and reused, keyed on their resolved options so a
  request with a different timeout or retry count still gets its own client
  rather than silently inheriting the first one's.
- The cache is **process-wide, not per provider instance** — which is what
  makes it work at all: `get_embedding_provider`/`get_inference_provider`
  build a fresh provider on every call, and `embed_texts` calls one per batch,
  so a per-instance cache would never see a second hit.
- A shared client is no longer closed after each request; doing so would leave
  the cached entry unusable for every later call.
- Audited the other providers, as the issue suggested: Anthropic keeps
  constructing per request deliberately (no auth round trip to amortize, and
  caching would hold a resolved credential in a long-lived object), and vLLM
  issues stateless `urllib` requests with nothing to cache.


### Fixed: date and timestamp search filters (issue #337)

- **A `data_type: date` or `timestamp` search attribute accepted filters that
  always failed at query time.** The predicate compiler rendered temporal
  values as quoted strings, which the query engine reads as text and refuses
  to compare against a `date32`/timestamp column — so config validation and
  index build both passed and the failure landed on the querying agent as an
  opaque `lancedb_vector_search_failed`. Temporal values are now typed
  literals (`DATE '…'`, `TIMESTAMP '…'`).
- Date scoping ("filings since 2020") is the most natural filter for any dated
  corpus, and the only previous workaround was declaring the column as a
  string and relying on ISO-8601 lexical ordering.
- A regression test now round-trips one filtered query per declared
  `data_type` against a real LanceDB store, plus a compiler-level test that
  executes temporal predicates against real `date32`/timestamp columns —
  asserting on the generated SQL text alone would not have caught this.


### Versioned prompts (issue #303)

- **`prompt: { name, version }`** on an `llm:` model resolves
  `prompts/<name>/<version>.md`. Inline `prompt: "..."` keeps working — this is
  an additional form, not a replacement.
- **Resolution is explicit and compile-time.** No `latest` pointer (a moving
  reference reintroduces the mutable-prompt problem), and a missing or
  misspelled version fails `compile` — before source discovery, credentials, or
  any provider call — listing the versions that do exist.
- **`prompt_name` and `prompt_version` are stamped on every output row**,
  alongside the existing `llm_*` provenance columns, and on the run log. That
  is what makes "which prompt produced this row" and "did v4 cost more per row
  than v3" queries rather than investigations; `llm_config_hash` could only say
  that *something* changed.
- **Prompt text never enters an artifact.** `manifest.json` records the
  resolved name and version. This also fixes a pre-existing leak: a native
  `llm:` model's inline prompt was being written into `manifest.json`
  verbatim, contradicting the documented rule.
- Name and version are charset-validated at config load (they become path
  segments), and no component of the path may be a symlink — a symlinked
  `prompts/` or `prompts/<name>/` would otherwise let a versioned prompt read
  operator-local data and send it to a provider.
- **Upgrade note for existing `materialization: incremental` `llm:` models.**
  The two new output columns change the target's schema the next time the
  model publishes rows. With the default `on_schema_change: fail`, that
  publication is rejected. Set `on_schema_change: append_new_columns`, or run
  once with `--full-refresh`, before it next reprocesses. The upgrade itself
  reprocesses nothing: an inline prompt's config hash is unchanged.
- Append-only logs widen rather than break when a later stel adds a column, so
  a `stel_run_log` created before this release gains `prompt_name`/
  `prompt_version` on the next write instead of failing every write silently.
- **`stel prompts lock` / `stel prompts check`** enforce immutability.
  `prompts/lock.json` records what each released version contained and is
  committed, so its diff surfaces a changed prompt in review; `check` is the CI
  gate and exits non-zero when a released version changed, when a version is
  unlocked, or when a locked version's file is gone.
- **`lock` refuses to re-lock a changed release or launder a deletion** without
  `--force`. Otherwise it would be a one-command bypass that teaches the
  workflow the gate exists to prevent — the fix is a new version, not a new
  hash. The content hash covers the stripped text, so a trailing newline is not
  an edit. A lock whose format version is unrecognized fails closed rather than
  reporting success on a schema it does not understand.


### Append-only logs: run history and MCP query history (issues #306, #329)

- **`run_log:`** on a profile target records one row per model per invocation —
  resolved provider identity, rows processed/skipped/written, token and call
  counts, `estimated_cost_usd`, status, duration. Cross-run questions ("what
  has this project spent over 90 days", "did the prompt change move cost per
  row") become queries instead of a directory of `run_results.json` files. A
  `budget_exceeded` row makes a tripped budget visible after the fact.
- **`mcp_query_log:`** records one row per served `search_context` call —
  principal, model, mode, query fingerprint, result count, `zero_results`,
  returned chunk ids, top score, elapsed. Written after authorization and
  policy filtering, so a row reflects only what the caller could see and a
  denied request logs nothing.
- **`capture_query_text` is a second opt-in**, off even when the log is on.
  The fingerprint answers "which questions repeat, and which return nothing"
  without storing what anyone typed.
- **Both are off by default, append-only, and create their relation on first
  write** — turning one on is the whole setup. Both share one primitive, the
  new `WarehouseAdapter.append_rows`.
- **A log never fails the thing it logs.** Writes are best-effort; a failure is
  one warning naming the exception class, and the run or MCP response is
  unaffected. The query-log write happens outside the MCP request deadline, so
  a stalled warehouse cannot turn a served answer into a timeout.
- Both relations are created with explicit column types rather than types
  inferred from the first batch, so a first run with nulls in optional columns
  cannot fix them as the wrong type and strand every later append.


### Warehouse-table sources: start a pipeline from a table (issue #322)

- **`path: warehouse://<relation>`** on a source treats each row of a
  warehouse relation as a document, so text loaded by Fivetran/Airbyte/dlt or
  built by dbt enters `extraction:` pipelines without an export-to-object-store
  hop. Read through the active adapter — dialect stays behind `adapters/`, and
  per-target `source_paths` overrides point each target at its own copy.
- **`key_column` is the row's identity**: its value ends the document path,
  `document_id` derives from that path exactly as for files, and the content
  hash fingerprints the whole row — so the incremental machinery works
  unchanged (changed row re-extracts, unchanged skips, deleted prunes through
  to chunks). Null or duplicate keys are hard errors, not guesses.
- **`path_columns` compose with `--source-filter`**: the declared columns
  prefix the document path, so `--source-filter 'economics/*'` scopes rows the
  way it scopes object prefixes — the existing orchestrator partition seam,
  no new concept.
- Rows are served to backends as plain JSON (ISO timestamps, decimals as
  strings at declared scale, binary as base64); `backend: json` with declared
  `fields:` is the natural pairing. Discovery snapshots the relation once and
  extraction consumes the snapshot. `max_objects` refuses rather than
  truncates.
- Adapters grew `relation_ref`/`read_relation`/`relation_row_count` for
  validated, quoted cross-schema reads; relation names from project YAML are
  validated per part at config load and again at the adapter.


### Classification eval models (issue #309)

- **A new `eval:` model kind** scores a classifier against labelled ground
  truth and publishes metric rows: accuracy, macro-F1, and per-label
  precision/recall/F1/support. It reads two already-materialized relations, so
  it costs no inference and can run on every change.
- **Long format** — one row per metric, so adding a metric never changes the
  schema and `WHERE metric = 'recall'` works.
- **`min_metric`** thresholds one metric, optionally for one label, so a
  regression on a single label gates a build. `accepted_range` cannot express
  this: it bounds every row of a column, and a long-format eval relation mixes
  rates with counts. An absent metric row fails rather than passing — a label
  that stopped being reported is the regression, not a clean sheet.
- **The label universe comes from the predicted field's `enum`** (issue #304),
  so a label the model stopped predicting reports `recall: 0.0` instead of
  vanishing from the report. `labels:` overrides when there is no enum.
- **`unmatched_rows` is published**, counting expected rows with no prediction
  to join to. An inner join alone would report a model that stopped emitting
  rows as a smaller but equally good one.
- Rows carry `predictions_version`, so an incremental eval keyed on `metric_id`
  replaces a re-run of the same predictions and appends a new version — a
  quality time series rather than a single snapshot.
- The scored relations become ordinary `depends_on` edges — derived when the
  model is validated, so every DAG consumer (`stel ls`, the manifest,
  run_results) sees them, not just the runner.
- **Malformed ground truth is loud.** Duplicate `key` values in either relation
  are a hard error (which duplicate wins would depend on warehouse row order);
  expected rows with a null key or label are counted in a
  `unusable_expected_rows` metric rather than silently dropped.
- **`min_metric` gates on the latest evaluation only**, so a historical dip
  cannot fail the test forever and a stale row cannot vouch for a label the
  current version stopped reporting. An incremental re-run of the same
  predictions version fully replaces that version's metric set — a removed
  label's rows are deleted, not left behind with an old `code_version`.


### Enum fields: declare the label set once (issue #304)

- **`type: enum` with `values:`** on a model field declares a closed set in one
  place. A classification task used to write its labels into the prompt, an
  `accepted_values` test, and the provider schema separately, with nothing to
  catch the three drifting apart.
- **The provider output schema** now carries a real `enum`, constraining the
  field at the API boundary rather than asking the model politely.
- **An `accepted_values` check is derived** from the declaration, so an enum
  field is checked without a `tests:` entry and with no hand-typed list to
  drift. An explicit `accepted_values` on the same column is honoured instead
  of duplicated; when the two lists disagree, compile time says so.
- **Prompt injection is the portability fallback.** Where a provider's
  structured output cannot carry an enum, the constraint is stripped from the
  schema and the labels are rendered into the system prompt, so the taxonomy is
  enforced as far as the provider allows and communicated regardless. Every
  shipped provider carries enums natively; a provider declares this with
  `supports_schema_enum`.
- The column materializes as a string, and `emit-dbt-sources` exports it as
  `string` — `enum` is stel's declaration, not a warehouse column type.
- A no-op for existing models: fields without `values:` derive nothing, and
  no cache keys move.

### `chunk:` can put document context where the embedder can see it (issue #308)

- **`in_text_metadata:` on a `chunk:` model** renders upstream columns into the
  chunk text as a small block ahead of it, so a chunk from the middle of a
  document no longer embeds with no idea which document it came from. Fields
  render in declared order; nulls are skipped; naming a column the upstream
  does not have fails before any document is processed.
- **Additive, never a mode switch.** The columns are still carried onto every
  chunk row. SQL reads columns and the model reads only text, and a rendering
  aimed at one reader must never remove the copy the other depends on.
- **The block counts against `chunk_size`**, in the unit the strategy splits
  by, so it does not push chunks past the size the embedder was configured
  for.
  Adding it on top would have pushed chunks past provider input limits, which
  a provider configured without truncation rejects outright. A block that
  leaves no room for text — or that pushes `chunk_overlap` to or past the
  remaining budget — fails with the numbers named.
- **`chunk_id` tracks the rendered text**, like any other text change. Turning
  the option on or editing its field list re-keys that document's chunks and
  invalidates anything downstream keyed on them. That keeps it consistent with
  the `agent_context` `document_chunks` contract, which recomputes the id from
  the stored text and rejects a mismatch. Worth landing before a corpus is
  embedded at scale rather than after.
- `embed:` needs no change: it reads the chunk model's text field, so the block
  reaches the embedding provider on its own.

### The upgrade path explains itself at the first step (issue #324)

- **A missing `stel_project.yml` now looks for a `dbt_ml_project.yml` beside
  it** and, when it finds one, names it and gives the `git mv` — rather than
  reporting only that a file is absent. This is the first thing a pre-rename
  project hits on upgrade, before any of the warehouse-name guards can run, so
  it was the one #313 hazard whose error explained nothing.
- **The "no profiles.yml was found" error names `~/.dbt_ml/profiles.yml`** when
  that is where the global profile still sits. Moving it to `~/.stel/` is a
  manual upgrade step, so the error now points at it.
- Both are **detection only**. Nothing loads a `dbt_ml_project.yml` or reads a
  profile from `~/.dbt_ml/`: two spellings that both work is how the old one
  never dies, and these need to stay one-time, visible steps.

### Flattened the repository layout

- The Python project moved from `stel/` to the repository root, so `uv sync`,
  `pytest`, `ruff`, `ty`, and `uv build` all run from where you already are.
  The nesting was an artifact of the pre-#313 repo holding a differently named
  package; with the repo and the package finally agreeing, a subdirectory only
  bought an extra `cd`.
- The two READMEs no longer collide, which is what deferred this. The root
  `README.md` stays the landing page; the 2,400-line reference that lived at
  `stel/README.md` is now `docs/reference.md`.
- `.gitignore` is a single file again, and CI drops its `working-directory`.
- Nothing in the package moved: `src/stel/` is still `src/stel/`, so imports,
  the wheel contents, and the frozen `stel/<version>` backend identity string
  are untouched.

### A redirect on the old `dbt-ml` PyPI project

- `compat/dbt-ml/` builds a standalone `dbt-ml` 0.8.1 that depends on `stel` and
  warns on import, so an old pin or a stale link resolves to something that says
  where the project went rather than a version frozen at 0.8.0 with no
  explanation. The old `dbt-ml` console script is kept only to print the same
  message and exit non-zero, so a job still shelling out to it fails visibly
  instead of appearing to succeed at nothing.
- It carries no functionality deliberately. Aliasing stel's submodules so
  `import dbt_ml.adapters` kept working would hide the rename behind a facade
  that then has to be maintained and removed.
- `dbt-ml` 0.8.0 is **not** yanked. It still works, and yanking would break
  exactly the pinned user this redirect exists for. Publishing is a one-time
  manual step documented in `docs/release.md`; it is not part of the `stel`
  release workflow and needs a separately scoped PyPI token.

### `concept-cloud --source-name`, and a loud failure when it does not resolve

- `concept-cloud --dbt-manifest` reconstructed the emitted dbt source name as
  `dbt_ml_<project>` and ignored `--source-name` entirely, so any project that
  had overridden the name got a linking node id matching nothing in the
  manifest. The concept-to-DAG edges are the only reason to pass a manifest, and
  they were built only for a node that resolved — so the export reported success
  and rendered a cloud quietly missing them. `--source-name` now plumbs through,
  and a linking node absent from the manifest raises, naming what it looked for
  and which sources the manifest actually declares.
- The default source name now has one owner, `dbt_export.default_dbt_source_name`.
  It was spelled inline at three call sites; a pin on the value would not have
  caught the drift, because all three copies agreed on the string and disagreed
  only about honoring the override. `test_frozen_names.py` pins the prefix and
  scans `src/` for any module rebuilding it by hand.

### `stel --version`

- The CLI had no version flag at all. It reports the installed distribution
  version through the same `_distribution` lookup used everywhere else.

### Tests

- **Budgets are now observed tripping end to end (issue #310).** Accounting had
  unit coverage; enforcement did not, and those are different claims — "the
  ledger raises past the cap" versus "`stel run` exits non-zero, stops before
  the next provider call, and keeps what it already committed". Eight tests
  drive the real CLI over a real project on the offline `deterministic`
  provider, covering the exit code, no overshoot past the api-call cap, token
  and cost caps, resume-after-stop with published rows and state intact, and
  per-model versus run-wide ledger scoping. Mutation-checked, so a guardrail
  that stopped guarding would fail rather than pass quietly.

### Docs

- `docs/release.md` said to put `PYPI_API_TOKEN` in repository secrets, with the
  `pypi` environment as an optional extra. The publish job declares
  `environment: name: pypi`, so an environment secret is the only place it
  resolves from — a repository-level token is invisible to it, and the failure
  lands after the full build and test matrix has run.

## v0.9.0 - 2026-08-20

### Vertex `thinking_budget`, off by default for structured output (issue #307)

- Gemini 2.5 models run dynamic thinking by default and bill reasoning tokens at
  the output rate. For the one-shot structured extraction stel sends over a
  relation that budget buys little, and it was charged on every row with no
  visibility in the YAML. `provider_options.thinking_budget` now exposes it: `0`
  disables thinking, a positive integer sets a budget, and an explicit value is
  always forwarded as configured.
- Omitted, it defaults to `0` only when the request declares an output schema
  (`fields:`) *and* the model accepts a disabled budget — a declared schema is
  the signal that this is extraction or classification, not open-ended
  reasoning. Every other case sends no `thinking_config` at all and leaves the
  model's own default alone: schema-less requests, Gemini 2.5 Pro (which
  enforces a minimum), pre-2.5 models (which reject `thinking_config`), and
  Gemini 3 (which configures reasoning through `thinking_level`). The capability
  check fails safe — an unrecognized model loses the cost optimization rather
  than gaining a failure.
- `thinking_budget` is a semantic provider option, so it rides the existing
  `profile_options_fingerprint` already folded into both the native `llm:`
  config hash and the `backend: llm` extraction cache key.
- `ProviderUsage` gained `thinking_tokens`, appended after `reported_cost_usd`
  so the positional constructor signature separately installed provider plugins
  rely on keeps working. It is reported only when non-zero, so non-thinking
  providers and cache hits keep byte-identical run-results metrics. Thinking
  tokens stay folded into `output_tokens`, so budget enforcement still sees full
  billable spend.

### `agent_context:` projection helpers and a built-in-pipeline example (issue #300)

- `agent_context.project_document_registry_row` and `project_document_chunk_row`
  build contract-shaped rows from an `extraction:`/`chunk:` pipeline. Wrapping
  those primitives in a `transform:` already worked, but it meant
  hand-assembling ~30 contract fields per row across 75-90 lines of Python with
  no reusable helper. Both compute every id and fingerprint through
  `agent_context`'s own `make_*`/`content_hash` functions rather than
  reimplementing them, and the chunk helper copies the parent registry row's
  bitemporal, policy, and freshness fields verbatim — so the cross-relation
  equality `validate_agent_context_relations` requires holds by construction
  instead of by caller discipline. Policy fields are deny-by-default.
- `examples/agent_context_from_builtin_pipeline/` runs the whole path with no
  credentials: `extraction:` → `chunk:` (the real recursive splitter, 4
  documents into 25 chunks) → two short `transform:` wrappers → `embed:`
  (deterministic) → `search:`. The previous worked example treated each document
  as exactly one chunk, so the multi-chunk shape most real pipelines have was
  never exercised.
- **`agent_context:` stays transform-only, and that is intentional** — now
  documented at the point a reader hits it, in
  `docs/architecture/agent-context-v1.md`, `docs/mcp.md`, and the validation
  error itself. Letting `extraction:`/`chunk:` declare it directly is blocked on
  a real conflict, not an oversight: the contract's `document_id` must equal
  `agent_context.make_document_id(source_system, source_key)`, while
  extraction's `document_id` is a different reserved pipeline-generated field
  used as the incremental-state key throughout. Reconciling them would change
  the id algorithm for every extraction model and invalidate existing
  incremental state on upgrade.

### `dbt_ref()` transforms can also declare `depends_on:` (issue #177)

- A transform reading `dbt_ref('...')` was rejected if it also declared
  `depends_on:`. The restriction lived only in `compiler.py`'s validation —
  `execution/transform.py`, `dag.py`, and `dbt_embed/codegen.py` already
  resolved both into one upstream set with no exclusivity assumption. Such a
  model now falls through to the same existence, duplicate, and kind checks any
  other transform's `depends_on` gets, and `declared_dependencies` unions the
  `dbt_ref` target with `depends_on` so the transform contract check matches
  what `run_transform_model` actually puts in the `deps` dict.
- `examples/dbt_ref_roundtrip` + `examples/dbt_ref_roundtrip_dbt` are a
  self-contained pair proving the full round trip in one `dbt build`: stel
  extraction → dbt SQL → stel transform via `dbt_ref` → dbt SQL. Both READMEs
  document the constraint that made a self-contained pair necessary — every
  generated shim resolves its stel project through one shared
  `STEL_PROJECT_DIR`, so a single `dbt build` embeds models from one stel
  project at a time.

### Renamed the project to Constellations, installed as `stel` (issue #313)

- The package, import name, and CLI are now `stel`; the brand is Constellations.
  `dbt-ml` borrowed another project's mark, read as an official dbt integration
  when it is not one, and described the tool by what it sits next to rather than
  what it does. Everything a user types changes: `pip install stel`, `stel run`,
  `stel_project.yml`, `~/.stel/profiles.yml`, `STEL_*` environment variables,
  and the `stel[...]` extras. There is no compatibility shim — a single known
  user, per `docs/compatibility.md`, so a deprecation window would cost more to
  maintain than it could repay.
- **The package rename moved nothing in a warehouse.** The visible internal
  names moved separately, under `stel migrate` — see the next section. What
  stays put permanently: every fingerprint domain, LanceDB collection metadata,
  the validated classic-ML artifact runtime key, the run_results version key
  that Dagster reads, and the emitted dbt source name (`dbt_ml_<project>`, still
  overridable with `--source-name`). Those are invisible to anyone browsing a
  warehouse, and renaming them would cost a full reprocess to buy nothing.
- Provider implementation identity is pinned across the rename. It deliberately
  excludes the release version so cached provider responses survive an upgrade,
  and it was the one identity that tracked our own module layout — letting the
  rename move it would have re-keyed every cached response and every
  `llm:`/`embed:` model's state at once.
- Extraction `code_version` does change, because backend implementation identity
  mixes in both the release version and the backend module path. Extraction,
  `chunk:`, and `embed:` models therefore recompute once on upgrade, while
  `llm:` and `search:` models skip. Verified against a control: a plain 0.8.0 →
  0.8.1 bump with no other change reprocesses exactly the same models, so the
  rename's effect on incremental state is indistinguishable from any patch
  release. **No provider calls result** — embedding and LLM responses come from
  their caches, which is precisely what pinning provider identity preserves.
- To upgrade: install `stel`, rename `dbt_ml_project.yml` to `stel_project.yml`,
  move `~/.dbt_ml/profiles.yml` to `~/.stel/profiles.yml` if you use the global
  location, rename any `DBT_ML_*` environment variables to `STEL_*`, and re-run
  `stel codegen` for embedded-dbt projects — the generated shims now import
  `stel.dbt_embed` and read `STEL_PROJECT_DIR`, so commit the regenerated files
  before the next `dbt build`. Leave `--source-name` values alone, and see the
  next section for `schema:` and `path:`.

### Migrated the visible warehouse names, behind loud failures (issue #313)

- **`stel migrate`** renames stel's persisted internal tables in place:
  `dbt_ml_state` → `stel_state`, `dbt_ml_serving_ledger` →
  `stel_serving_ledger`, `dbt_ml_serving_leases` → `stel_serving_leases`, and
  any `dbt_ml_test_failures__*` inspection tables. Rows are preserved — DuckDB
  uses a transactional `ALTER TABLE ... RENAME TO`, BigQuery a `CREATE TABLE ...
  COPY` that only drops the original once the copy exists. `--dry-run` prints
  the plan. It touches only tables stel owns, only inside the schema the target
  already points at, and refuses rather than guessing if it finds both spellings
  of the same object.
- **Nothing runs against an unmigrated warehouse.** Connecting to a schema that
  holds the old tables raises and names `stel migrate`, exiting 2 as a setup
  error. Without that guard the run would create empty replacements beside the
  old tables, find no prior state, report every document as new, and reprocess
  the corpus at provider cost — green the whole way, because that is exactly
  what a genuine first run looks like.
- **The default schema is now `stel`** (was `dbt_ml`), and the zero-config
  DuckDB file is `target/stel.duckdb` (was `target/dbt_ml.duckdb`). Neither is
  adopted automatically: a project that never wrote `schema:` and connects to an
  empty `stel` schema while a populated `dbt_ml` one exists is refused, with the
  exact `schema: dbt_ml` line to add. The zero-config path is refused at config
  load for the same reason. A project that names its schema or path explicitly —
  every shipped example does — is unaffected and never second-guessed. Moving a
  whole schema stays the operator's decision, not the migration's.
- **`stel_` is now a reserved model-name prefix, added alongside `dbt_ml_`.**
  The old prefix stays reserved even though no internal table uses it any more:
  dropping it would newly *allow* model names existing projects have always been
  told are stel's.
- `list_tables()` now hides the serving ledger and lease tables. They were never
  filtered, so they appeared in `stel ls` and `stel show` as if a user had
  modeled them. It also still hides the pre-rename `dbt_ml_*` internals, so
  debris a crashed pre-upgrade run left behind does not surface as a model.
- To upgrade an existing target: run `stel migrate` once. If your profile never
  named a schema, add `schema: dbt_ml` first (or move the objects yourself and
  keep the new default).


## v0.8.0 - 2026-08-11

### Coalesce small incremental flushes into one publication (issue #293)

- Extraction models gain `extraction.publish_every` (default `1`), a flush
  multiple that coalesces that many `flush_every`-sized flushes into a single
  incremental upsert. On BigQuery this lets a run of many small flushes share one
  `MERGE` — the combined batch scans the target once instead of once per flush,
  cutting bytes billed roughly in proportion — while `flush_every` keeps its
  memory-bounding role. It is generic: the executor buffers the contracted flush
  frames and issues one combined `materialize_incremental`, so every adapter
  benefits and the default (`1`, one frame per publish) is byte-for-byte the prior
  per-flush path.
- State advances only after a publication succeeds, so the invariant holds at the
  coarser cadence: a crash or budget exhaustion with a partial buffer leaves those
  flushes unpublished and retryable, and already-published batches survive. Only
  same-schema flushes coalesce — a schema-on-read model whose columns drift mid-run
  publishes at the boundary, so `on_schema_change` applies exactly as it did per
  flush and a later flush's new column is never folded into (and dropped by) an
  earlier publication's policy. The tradeoffs — peak memory grows to about
  `publish_every × flush_every`, and crash recovery is coarser — are documented
  alongside the setting. `publish_every` is excluded from `code_version` like
  `flush_every`: it changes execution cadence, never output content.

### Document clustering the BigQuery incremental key (issue #294)

- Documented that clustering the target on the incremental/merge key can let
  BigQuery prune the `MERGE` scan (the read side; `update_when_changed` bounds
  the write side) — a likely optimizer optimization to measure, not a guarantee,
  since dbt-ml emits a column-to-column join rather than a static key predicate.
  Also pins the contract: a `cluster_by` change — like any `warehouse_options`
  layout — is inert on an existing table until `--full-refresh` rebuilds it; a
  layout change is not a storage-format change (so it never trips the #289
  fail-fast); `flush_every` governs MERGE count for extraction models only; and
  serializing external runs remains a project responsibility. Regression tests
  lock in the incremental-path contract.

### Safe BigQuery incremental-publication telemetry (issue #292)

- Each BigQuery incremental publication (DataFrame `MERGE`/`insert_overwrite`
  and SQL-model `MERGE`) now logs safe, structured telemetry at INFO: the output
  relation, the BigQuery job id, bytes processed, and DML-affected row count.
  The job id lets an operator match dbt-ml's own jobs against BigQuery job
  history / `INFORMATION_SCHEMA.JOBS`, so many tiny dbt-ml flushes can be told
  apart from an overlapping external orchestrator run. Only job-level statistics
  and the table name are logged — never SQL text or row values. Surfaced under
  `-v` / `DBT_ML_VERBOSE` like the rest of dbt-ml's progress output.

### `table_format: iceberg` for SQL models (issue #290)

- SQL (`transform.type: sql`) models can now materialize a managed Iceberg
  table, so a project can adopt Iceberg as a uniform storage policy without
  carving out its SQL models with `warehouse_options: {inherit: false}`. Full
  SQL materialization stages the query once, builds an explicit Iceberg
  `CREATE TABLE` from its schema, and `INSERT…SELECT`s the rows across — the same
  non-atomic drop → create → insert shape as the DataFrame Iceberg path, gated by
  `iceberg_table_format`. The former compile-time rejection is removed; SQL
  Iceberg models are now gated on the capability like any other Iceberg model.
- The storage-format fail-fast from #289 now also covers SQL incremental models:
  declaring Iceberg over an existing standard SQL-incremental target (or the
  reverse) raises before staging or merging, naming `--full-refresh`.

### Security

- Bump `pypdf` floor to `>=6.15.0` to clear CVE-2026-71852 and CVE-2026-71870
  (fixed in 6.15.0).

### Fail fast on a BigQuery incremental storage-format mismatch (issue #289)

- An `incremental` model that declares `table_format: iceberg` against a target
  that already exists as a standard BigQuery table (or the reverse) no longer
  writes silently through the standard MERGE path while leaving the stored
  format unchanged. `materialize_incremental` now compares the declared format
  against the target's actual `biglakeConfiguration` and raises before any
  MERGE/load, naming the fix: re-run with `--full-refresh` to rebuild the table
  in the declared format. Matching formats and fresh tables are unaffected.

## v0.7.0 - 2026-08-07

### Incremental change detection: `update_when_changed` (issue #281)

- Incremental models can declare `update_when_changed: [col, ...]`, a
  change-detection fingerprint. A matched row is rewritten only when at least
  one listed column differs (NULL-safe) between the batch and the target, so
  re-publishing an unchanged row no longer rewrites its large payload columns —
  on BigQuery that is far fewer bytes billed for the `MERGE` (BigQuery adds an
  `IS DISTINCT FROM` guard to `WHEN MATCHED`; DuckDB deletes only changed keys
  and inserts new-or-changed rows). New rows still insert and changed rows still
  update.
- The listed columns must exist in both the batch and the target (validated
  before any warehouse mutation); `content_hash` and `code_version` are the
  natural fingerprint for extraction models. The option is only valid with
  `materialization: incremental` and is excluded from `code_version`, so
  enabling it is backward-compatible and never reprocesses documents. Leaving it
  unset preserves the always-overwrite behavior.

### Target-aware BigQuery warehouse defaults (issue #284)

- BigQuery profile targets can declare `warehouse_defaults` once inside their
  `warehouse:` block. Model-level top-level `warehouse_options` override those
  defaults, while `inherit: false` opts a model out for a plain or separately
  configured table.
- Iceberg defaults use `external_volume` to derive a unique
  `{target}/{dataset}/{model}` Cloud Storage prefix. A literal default
  `storage_uri` is rejected so separate models or environments cannot silently
  share one physical location.
- Effective defaults and overrides are adapter-validated before source
  discovery, credential construction, or warehouse mutation, and target-level
  defaults also participate in capability and filtered-run safety checks.

### Extraction-time field derivation (issue #282)

- Extraction models can declare a project-local `post_extract` hook that
  replaces each backend result's fields before staging or warehouse
  publication. This lets envelope sources derive text or structured fields in
  memory and omit large raw payloads instead of materializing a raw table for a
  second transform pass.
- Hook modules and options are validated during compilation, apply to ordinary
  and native-batch extraction while verified snapshots still exist, participate
  in incremental `code_version`, preserve backend warnings/metrics, and sanitize
  failures that might otherwise reveal raw document content.
- Generated manifests omit `post_extract.options`, and compiler diagnostics for
  module/option validation failures use a stable message with the unsafe
  exception chain severed.

### Tooling (issue #283)

- Added dbt-ml project skills that package repository workflows for agent-driven
  development. No runtime or public config/behavior contract change.

## v0.6.0 - 2026-08-07

### Cloud object-store retrieval stores (issue #271)

- The `lancedb` retrieval store's `path` now accepts a cloud object-store URI
  (`s3://`, `gs://`, `az://`, plus `s3a`/`gcs`/`abfs`/`abfss` aliases) as well as
  a local path, so a `search:` index can be shared between the machine that runs
  `dbt-ml build` and the machines that serve it (MCP server, search API) instead
  of being pinned to one local disk. A cloud path bypasses project-relative
  resolution and the local `mkdir`, and flows straight to `lancedb.connect()`.
- Cloud credentials follow the repository's reference contract. `storage_options`
  carries non-secret routing only (region, endpoint, …) and is part of the
  store's physical identity, so a changed endpoint yields a distinct safe target
  and state scope; secret-looking keys are rejected. Secrets are supplied through
  `storage_options_env`, a map of option key to environment-variable *name* that
  is redacted from every dump and resolved to its value only at
  `lancedb.connect()`, never stored or fingerprinted.
- Equivalent URI spellings (`s3://`≡`s3a://`, `gs://`≡`gcs://`, trailing slashes)
  canonicalize to one physical target for both the store identity and the
  publisher lock. Local-store identities are unchanged (byte-identical
  fingerprint), so existing state scopes keep resolving.
- The single-host publisher lock, which has no local disk to live on for a cloud
  store, defaults to a fixed per-machine directory (not a `TMPDIR`-derived path,
  which varies per process/container). Deployments whose publishers run in
  separate mount namespaces set `publisher_lock_dir` to a shared volume. The
  single-host boundary is unchanged: cross-host publication still requires the
  reserved provider-enforced fencing capability.

### Progress output for long-running builds (issue #268)

- `dbt-ml run` and `dbt-ml build` accept `-v` (also `DBT_ML_VERBOSE=1` for
  orchestrated runs) to emit per-source discovery counts, per-model start/finish
  lines, and a live per-model progress bar on a TTY. Non-TTY stderr (e.g. a
  Dagster capture) gets the same events as plain INFO log lines. Default output
  is unchanged, and all progress goes to stderr so `--json` on stdout stays a
  single parseable payload.
- Exactly one verbose channel is active at a time (bar *or* log lines), so the
  bar is never corrupted by overlapping log records and events are not
  double-printed. `run --threads N` uses the log-line channel even on a TTY,
  since parallel per-model bars on one terminal would interleave. With
  `--source-filter`, the reported per-source count is the post-filter selected
  count, so it reflects what is actually processed.
- Verbose is deliberately capped at INFO; DEBUG-level log sites carry
  unsanitized exception text and are not exposed through the flag.

### Runtime partitioned extraction: `--source-filter` (issue #266)

- `dbt-ml run`/`build` accept a repeatable `--source-filter GLOB` to process only
  source documents whose relative path matches a glob (e.g. `--source-filter
  'AAPL/*'`), for partitioned or backfill runs over a large corpus. A filtered
  run is additive/upsert-only: it never deletes and cannot be combined with
  `--full-refresh`.

### Provider retry classification (issues #258, #263)

- Inference provider errors now preserve the underlying HTTP status so the
  runner can distinguish retryable (429/5xx) from permanent (4xx) failures
  instead of treating every provider error alike.
- Anthropic native-batch failures now carry the same retry-vs-permanent
  classification rather than surfacing unclassified.

### BigQuery incremental state at scale (issues #256, #260)

- The BigQuery adapter's state and child-row methods no longer pass the full
  key/record set as unbounded array query parameters, which failed at scale (a
  ~75k-record corpus). `_merge_state`, `replace_children`,
  `delete_rows_and_state`, `delete_rows`, `delete_state`, and
  `_fetch_state_subset` now either batch by a bounded request size or stage the
  keys/records into a temp table referenced inside the single transaction, so
  the atomic multi-statement guarantees (#229) are preserved.

### 3D concept cloud (issue #255)

- Added an end-to-end proof-of-concept that renders a 3D concept cloud over the
  dbt DAG (milestones 1–3).

### Reliability and safety fixes

- Harden GCS source error handling and resource cleanup, mapping SDK errors to
  concise, response-body-free messages that surface the HTTP status (#265).
- Keep raw warehouse text out of SQL-model error messages (#262).
- Preserve the sanitized exception cause on generic snapshot/credential adapter
  failures instead of dropping the chain (#261).
- Bound and self-heal fetch-staging disk usage during extraction so a large or
  interrupted run cannot leave unbounded scratch behind (#273).

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
