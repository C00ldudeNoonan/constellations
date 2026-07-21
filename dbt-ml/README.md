# dbt-ml

**dbt for unstructured data.** Declarative YAML pipelines that turn folders of
documents — PDFs, markdown, HTML, JSON, email, free-form text — into warehouse
tables. Incremental processing, schema tests, dbt-style selectors, profiles,
and a manifest artifact you can wire into other tools.

The current v0.2 preview is pure Python and supports DuckDB and BigQuery
warehouses, local and GCS sources, document chunk models, executable classic
text-ML providers, native warehouse-materialized embedding models with a
deterministic offline provider, and an incremental local LanceDB search-index
proof of concept. Additional warehouse and hosted retrieval adapters and
production embedding providers remain roadmap work; Rust and PyO3 are
explicitly out of scope through v0.2.

## Where dbt-ml fits

The 2026 landscape for unstructured document pipelines has two stable poles:

- **Managed RAG-as-a-Service** (Vectara, Bedrock Knowledge Bases, Vertex AI
  Search, Snowflake Cortex Search, Glean) — best when time-to-value matters
  and the team can't dedicate ML engineers.
- **Compose best-of-breed Python components** (LlamaParse → contextual
  chunking → Voyage embeddings → Qdrant → Cohere Rerank → Ragas) — best when
  retrieval quality, multi-tenant isolation, or unusual document types
  matter and you have ≥2 ML engineers.

dbt-ml is the **opinionated, declarative path through the second lane**.
Where LlamaIndex is imperative Python, dbt-ml is YAML + a manifest + tests +
lineage. Where Snowflake Cortex Search hides everything, dbt-ml makes every
stage inspectable and reproducible. It's *dbt-shaped*: the same DAG +
selectors + tests + artifacts pattern, applied to unstructured data.

---

## You have a folder of files. Get them into your warehouse.

```bash
# Install from PyPI with the PDF parser used below
uv add 'dbt-ml[pdf]'

# 1. Scaffold a project for whatever shape your data is
uv run dbt-ml init my_project --template pdf      # or json, markdown, html

# 2. Drop your files into ./my_project/data/pdfs/  (or wherever the source points)

# 3. Run it
cd my_project
uv run dbt-ml run

# 4. Query the result
duckdb target/dbt_ml.duckdb -c "SELECT * FROM my_project.raw_pdf_text LIMIT 5"
```

That's the whole loop. Everything else (selectors, profiles, tests, LLM
extraction, dbt handoff) is opt-in on top.

### Optional dependencies

The core install stays lean. Add only the feature groups a project uses:

| Extra | Features |
|-------|----------|
| `pdf` | PDF extraction and synthetic PDF generation (`pypdf`, `fpdf2`) |
| `html` | HTML extraction (`beautifulsoup4`) |
| `text` | Token counting, encoding cleanup, language detection, and near-duplicate detection |
| `pii` | Presidio PII detection and redaction; a spaCy language model is still installed separately |
| `bigquery` | BigQuery warehouse adapter |
| `gcs` | Google Cloud Storage document sources |
| `lancedb` | Local LanceDB search-index publication and queries |
| [`mcp`](docs/mcp.md) | Read-only governed context server over MCP stdio |
| `all` | Every optional feature above |

For example, `uv add 'dbt-ml[pdf,text]'` installs PDF and text processing,
while `uv add 'dbt-ml[all]'` provides the complete development/runtime feature
set. Invoking a feature whose extra is absent raises an error with the exact
installation command.

## What dbt-ml actually does

| Concept            | What it means                                                                  |
|--------------------|--------------------------------------------------------------------------------|
| **Source**         | A glob over a folder. `*.pdf`, `*.json`, `*.html`, `*.md` — your choice.        |
| **Extraction model** | One row per source file, produced by a backend (JSON, Markdown, PDF, HTML, email, or LLM). |
| **Transform model**  | A Python module returning a Polars DataFrame, depends on other models via `ref()`. |
| **Chunk model**      | An executable `chunk:` model producing stable, lineage-carrying retrieval units. |
| **Embed model**      | An executable `embed:` model producing canonical, provider-identified vectors in the warehouse. |
| **Classic ML model** | An executable `ml:` model for deterministic features and classifiers, with persisted artifacts. |
| **Materialization**  | `full` (always replace) or `incremental` (skip unchanged input on re-runs).      |
| **Tests**          | `not_null`, `unique`, `min_rows`, custom Python — with `severity: warn` if you want.|
| **Profile**        | Warehouse + LLM config, swappable per `--target dev|prod`. No credentials in models. |
| **Artifacts**      | `target/manifest.json`, `target/run_results.json`, `target/sources.yml` (for dbt). |

## Backends

| Backend    | Reads             | Notes                                                                                     |
|------------|-------------------|-------------------------------------------------------------------------------------------|
| `json`     | `*.json`          | Projects keys per `options.fields`. Deterministic, no API.                                |
| `markdown` | `*.md`            | YAML frontmatter + `body` + optional `word_count`. Deterministic, no API.                 |
| `pdf`      | `*.pdf`           | Per-page text via pypdf. Warns on empty extracts (likely scanned). Deterministic, no API. |
| `html`     | `*.html`/`*.htm`  | Body text + CSS selectors + OpenGraph/meta via BeautifulSoup. Deterministic, no API.      |
| `email`    | `*.eml`           | from/to/subject/date/body via stdlib `email`. Deterministic, no API.                      |
| `llm`      | `*.txt`/`*.md`    | Registered inference provider → structured fields. Provider, model, and protected credential reference come from the active profile. |

Add a new backend by inheriting from `BaseBackend`, defining a strict Pydantic
option model, and decorating it with `@register(options_model=...)`. Bare
`@register` remains a pass-through compatibility path for existing third-party
backends, but new backends should publish a typed option contract so compile
and runtime enforce the same configuration.

## Security Notes

dbt-ml projects are local code-and-data projects. Only run projects you trust:
Python transforms and custom Python tests execute in your Python process, and
project configuration controls source globs, generated paths, and executable
modules. The discovered profile controls warehouse, cache, and protected
credential references. Reference names and values are omitted from artifacts
and user-facing diagnostics.

Document parsers process local files with third-party libraries. Keep
dependencies current before running dbt-ml over untrusted PDFs, HTML, email, or
other documents, since malformed files can trigger parser CPU or memory bugs.

The `llm` backend sends document text to the configured model provider and stores
cached structured responses in plaintext in the configured cache database. New
POSIX cache databases and transient write-ahead logs are forced to owner-only
mode (`0600`), but the files still contain extracted document data and must be
handled as sensitive. Use
deterministic local backends for sensitive documents unless remote processing is
intended.

Local LanceDB collections contain the projected chunk text, embeddings, and
returned/filter attributes in plaintext beneath the operator-configured profile
path. The first slice supports only `access: public`, and only when the active
profile explicitly sets `retrieval.allow_public_indexes: true`. Do not use it
for tenant- or ACL-governed data; governed publication is rejected until the
trusted policy and publish/read coordination work in #152 lands.

### Trust model & filesystem boundaries

Paths declared in **project YAML** ship with a repo, so they are confined to
the project directory — a path that resolves outside it (via `..`, an absolute
path, or a symlink) is a configuration error (exit 2):

| Path | Confined | Opt-out |
|---|---|---|
| `source.path` | yes | `external: true` on the source |
| `source.file_pattern` | relative only; absolute paths and `..` are rejected | none |
| matched local source files | must stay below the resolved source root; symlinks are not followed | none |
| `ml.artifact.path` | yes | `external: true` on the artifact block |
| `source-paths` / `model-paths` / `transform-paths` / `target-path` | always | none |
| model-level llm `cache_path` | always | put it in profiles.yml instead |
| legacy inline `duckdb.path` | always | move external paths into profiles.yml |

```yaml
sources:
  - name: filings
    path: "D:/corpora/filings/"   # outside the repo — reviewable opt-in:
    external: true
```

`external: true` permits the declared source root outside the project. It does
not permit pattern traversal or symlinked source files. Local discovery hashes
through no-follow file descriptors where the platform supports them; fetches
are verified snapshots in per-run scratch space, so a path swap after discovery
does not change the bytes sent to a parser or remote model.

Project, source, and model YAML must be regular files under their configured
roots. Configuration discovery does not follow symlinked files or directories.

**profiles.yml paths** (warehouse `path:`, llm `cache_path:`) are operator
configuration, like dbt's, and are trusted as-is. An implicit project-local
profiles file must be a regular file; pass `--profiles-dir` when intentionally
using an operator-managed symlink.

`dbt-ml clean` removes only known local artifacts under `target-path`
(`manifest.json`, `run_results.json`, generated `sources.yml`, `docs/`, and
classic-ML `artifacts/`). It preserves configured warehouse/cache files and
unknown files, never calls an adapter-level database/schema/dataset reset,
rejects project-root or source/model/transform overlap, and refuses symlinked
paths. There is no `--force` option.

Running a third-party project still executes its Python transforms and custom
tests, and remote sources (`gs://…`) reach whatever your ambient credentials
allow — review projects you didn't write before running them.

For scheduled/orchestrated runs, the `llm` backend can route uncached documents
through a provider's native batch API. The built-in Anthropic provider applies
its 50% batch multiplier, at the price of minutes-scale latency (the run blocks
until the batch completes). Cache hits still resolve locally, and the cost
estimate in run results applies the selected provider's batch multiplier.
Keep it off for dev loops:

```yaml
extraction:
  backend: llm
  options:
    batch: true                  # provider-native batch; higher latency, often cheaper
    batch_size: 1000             # deterministic partition size (capped by the provider)
    batch_poll_seconds: 30       # initial poll interval; backs off toward the max
    batch_poll_max_seconds: 300  # poll backoff ceiling
    batch_timeout_seconds: 86400 # cancel the provider job past this deadline
    on_partial_batch: fail       # or publish_successful (per-doc errors, successes kept)
```

Uncached documents stream through deterministic partitions of at most
`batch_size` requests (never above the provider's own limit), so memory stays
bounded regardless of corpus size. Each partition's provider job identifier is
persisted in the response cache database before polling: a crashed or
interrupted run resumes the submitted job on the next invocation instead of
resubmitting it, so the work is billed exactly once. Batch mode without
`cache_path` still runs, but cannot resume — `compile` warns about it. By
default a partition containing a failed document publishes nothing further
(`on_partial_batch: fail`); opt into `publish_successful` to record
per-document failures and keep the successes, advancing state only for
published documents.

Execution budgets cap what a run may consume before the next provider call is
made. Per-model caps live in the model's extraction options; run-wide caps are
operator policy in profiles.yml:

```yaml
# model YAML
extraction:
  backend: llm
  options:
    budget:
      max_documents: 5000
      max_api_calls: 5000
      max_cost_usd: 25.0

# profiles.yml
llm:
  budget:               # shared by every model in one invocation
    max_total_bytes: 500000000
    max_cost_usd: 100.0
```

Available caps: `max_documents`, `max_file_bytes`, `max_total_bytes`,
`max_input_tokens`, `max_output_tokens`, `max_api_calls`, and `max_cost_usd`
(provider-reported spend wins over the pricing-table estimate). A tripped
budget stops the model with the distinct `budget_exceeded` status: `full`
materializations publish nothing, and incremental runs keep only chunks that
already committed with their state. Token and spend caps are measured from
responses, so the stopping call may overshoot the cap by at most one response.

The built-in `vllm` provider supports local, Docker, Kubernetes, and remote
OpenAI-compatible endpoints. See the [vLLM provider guide](docs/vllm.md) for
server startup, profile configuration, authentication, timeout, model-name,
and concurrency recommendations.

### LLM credentials

`api_key_env` selects an environment-variable reference, never a secret.
Runtime resolves the exact profile-owned reference and passes an opaque value
to the selected provider. It never substitutes a different provider's default.
Model YAML cannot choose a credential reference. Missing credentials fail with
the provider and field policy—not the private reference name—before a provider
request is submitted, and `compile` applies the same redacted warning policy.

Provider integration authors upgrading to provider contract v2 should accept
`ProviderCredential(value)` and call `reveal()` only at SDK construction. The
old two-argument constructor and `.env_var` attribute are removed, and
`resolve_llm_credential()` returns a protected value (or `None`) instead of a
tuple.

Reusable transform helpers are not profile-ambient. A transform that calls one
must declare the dependency so profile changes invalidate state and provider
provenance appears in artifacts:

```yaml
transform:
  type: python
  module: transforms.enrich
  uses_llm: true
```

Pass the effective `ctx.llm.provider`, `model`, `api_key_env`, `base_url`, and
`timeout_seconds` to `extract_fields_from_text()`; when
`ctx.llm.system_prompt` is set, pass it as the helper's `system=` argument.
This keeps provider selection, routing, and credentials operator-governed. LLM
extraction models preflight credentials even if their response cache is warm.

## The CLI

```
dbt-ml init <name> [--template {json,pdf,markdown,html}]   # scaffold a fresh project
dbt-ml seed [--count N] [--type {invoices,posts,...,tickets,emails}]
dbt-ml compile                                             # parse YAML, validate DAG, write manifest.json
dbt-ml graph                                               # Mermaid DAG to stdout
dbt-ml run [--select EXPR] [--exclude EXPR] [--full-refresh] [--threads N] [--watch] [--state DIR]
dbt-ml test [--select EXPR] [--exclude EXPR] [--store-failures] [--state DIR]
dbt-ml build [--select EXPR] [--exclude EXPR] [--full-refresh] [--threads N] [--store-failures] [--state DIR]
dbt-ml ls [--select EXPR] [--resource-type {model,source,search_index,all}] [--output {name,json}]
dbt-ml show <model> [--limit N]                            # peek at a materialized table
dbt-ml search --model NAME --query TEXT [--mode {vector,text,hybrid}] [--filter FIELD OP VALUE] [--output {table,json}]
dbt-ml serving status <search-index>                       # publication ledger: status, fence, counts, leases
dbt-ml serving recover <search-index> --owner-terminated   # explicit authority reassignment after a crash
dbt-ml providers list [--output {table,json}]              # built-in + entry-point providers, incompatible plugins flagged
dbt-ml source freshness                                    # mtime vs warn_after/error_after
dbt-ml docs generate [--output DIR]                        # static HTML site from manifest.json
dbt-ml docs serve [--port N]                               # local http.server over target/docs/
dbt-ml emit-dbt-sources [--output PATH]                    # write dbt-compatible sources.yml
dbt-ml clean                                               # remove known target artifacts; preserve warehouses

# Global flags (work on every command):
dbt-ml --project-dir <dir> --profiles-dir <dir> --target <name> <command>
```

Project, source, model, and profile models reject unknown keys; source/model
YAML accepts schema `version: 2`. Before profile resolution, source discovery,
or warehouse mutation, `compile`, `run`, and `build` validate registered
backend names, source/model edge kinds, supported materializations, transform
and custom-test modules/call signatures, built-in test option shapes, and
relationship targets. Relationship tests add a DAG predecessor so their target
relation is built first. Every shipped extraction backend has a strict,
backend-specific option schema; unknown options, wrong types, invalid LLM field
schemas, and out-of-range execution settings fail before source discovery.
Executable classic-ML tasks, providers, provider options, metrics, and artifact
paths are checked by the same preflight. YAML schema diagnostics include the
file, one-based line and column, and full configuration path without echoing
the rejected input value; duplicate mapping keys are rejected at their second
declaration. Configuration failures exit 2.

### Useful flags

- `--watch` on `run` listens to source paths and re-runs on file changes
  (debounced 500ms). Ctrl-C to stop.
- `--threads N` parallelizes per-document extraction within an extraction
  model. Most useful for PDF / LLM / HTML (I/O- or API-bound). The LLM cache
  is lock-serialized so threading is safe.
- `--select` / `--exclude` limit source discovery as well as model execution;
  an unrelated GCS branch is never listed or authenticated.

## Selectors

dbt-shaped. Whitespace-separated tokens, optional `+` modifiers, `tag:` prefix.

```bash
dbt-ml run --select raw_pdf_text       # one model
dbt-ml run --select 'raw_pdf_text+'    # plus all downstream
dbt-ml run --select '+invoice_summary' # plus all upstream
dbt-ml run --select 'tag:raw+'         # all models tagged "raw" + their downstream
dbt-ml run --exclude tag:expensive
dbt-ml run --select 'state:modified+' --state ./main-manifest/
                                       # only models whose config or transform
                                       # code changed vs a previous manifest,
                                       # plus their downstream
```

`state:modified` compares each model's `code_version` (a hash of its
extraction/transform/ml config and transform module source) against a
manifest written by a previous `compile` or `run`. The CI recipe: store
`target/manifest.json` from main, then on PRs run
`dbt-ml build --select 'state:modified+' --state path/to/main-manifest/`.

## Profiles

Warehouse and LLM config live in `profiles.yml`, *not* in `dbt_ml_project.yml`.
Project YAML says `profile: my_project`; profile says where to write and which
LLM to call. Swap `--target prod` to switch environments.

```yaml
# profiles.yml — sits next to dbt_ml_project.yml, or in ~/.dbt_ml/profiles.yml
my_project:
  target: dev
  outputs:
    dev:
      warehouse:
        type: duckdb
        path: ./target/dbt_ml.duckdb
        schema: my_project
      source_paths:
        filings: ./data/dev/filings
      llm:
        provider: anthropic
        model: claude-haiku-4-5
        api_key_env: ANTHROPIC_API_KEY
        cache_path: ./target/llm_cache.duckdb
        pricing:                       # optional — enables estimated_cost_usd
          input_usd_per_mtok: 1.00     # in run summaries + run_results.json.
          output_usd_per_mtok: 5.00    # USD per million tokens; you own these
          cache_read_usd_per_mtok: 0.10   # numbers, dbt-ml ships no price table.
    prod:
      warehouse:
        type: duckdb
        path: "{{ env_var('DBT_ML_PROD_DB', '/data/prod/dbt_ml.duckdb') }}"
        schema: my_project_prod
      source_paths:
        filings: "{{ env_var('DBT_ML_FILINGS_ROOT', '/data/prod/filings') }}"
      llm:
        model: claude-sonnet-4-6
        cache_path: /data/prod/llm_cache.duckdb
```

Lookup order: `--profiles-dir` flag → `$DBT_ML_PROFILES_DIR` →
`<project>/profiles.yml` → `~/.dbt_ml/profiles.yml`.

Set `api_key_env` to the name of the credential variable itself, as above; do
not wrap it in `env_var()`. dbt-ml deliberately rejects secret-value
interpolation in this field so validation errors and resolved configuration
cannot contain the key.

### Provider plugins and provider options

Separately packaged inference/embedding providers install as normal Python
distributions and are discovered through versioned entry-point groups
(`dbt_ml.inference_providers.v3` / `dbt_ml.embedding_providers.v3`) — no
wrapper import needed. Discovery is deterministic and fails closed before any
source or provider I/O: duplicate or built-in-shadowing names, broken plugins,
and name mismatches are configuration errors, and a plugin built against a
different provider contract version is reported as incompatible rather than
"not found". `dbt-ml providers list` shows every provider with its
distribution and implementation identity.

A provider may publish a strict options model; operators configure it under
`llm.provider_options:` in the profile (opaque to core, validated by the
selected provider, rejected in model YAML). Every provider option field is
classified: `credential` fields are protected references that never enter
artifacts or fingerprints, `semantic` fields join the response-cache key and
model identity, `execution` fields never invalidate state, and
`artifact-safe` fields may appear in manifest descriptors. See
[docs/architecture/provider-abstraction.md](docs/architecture/provider-abstraction.md).

### BigQuery

Install the extra, then point a target at a GCP project. Non-secret profile
fields mirror dbt-bigquery. Authentication supports ADC (`method: oauth`, the
default), a literal or environment-backed `keyfile:` (service account), an
environment-backed `keyfile_json:`, or environment-backed `token` /
`refresh_token` / `client_secret` fields (`oauth-secrets`), plus
`impersonate_service_account`, `scopes`, `execution_project`,
`quota_project`, `priority`, `maximum_bytes_billed`, and the
`job_retries` / `job_retry_deadline_seconds` /
`job_creation_timeout_seconds` / `job_execution_timeout_seconds` knobs.
`method:` may be omitted — it's inferred from which credential fields are
set. (dbt's `dataproc_*` fields don't apply: dbt-ml transforms run
in-process, not on Dataproc.)

```
pip install 'dbt-ml[bigquery]'
```

```yaml
my_project:
  target: prod
  outputs:
    prod:
      warehouse:
        type: bigquery
        project: my-gcp-project
        dataset: dbt_ml                # `schema:` works too
        location: US                   # optional
        # Omit auth fields for ADC, or choose exactly one auth family:
        # keyfile: ./secrets/service-account.json
        # keyfile: "{{ env_var('DBT_ML_BQ_KEYFILE') }}"
        # keyfile_json: "{{ env_var('DBT_ML_BQ_SERVICE_ACCOUNT_JSON') }}"
        # token: "{{ env_var('DBT_ML_BQ_ACCESS_TOKEN') }}"
```

Secret-bearing BigQuery fields accept only an exact, quoted
`{{ env_var('NAME') }}` reference with no default or surrounding text. The
reference is preserved without reading the environment—even on an inactive
target—and is resolved only while constructing Google credentials. The value
of `keyfile_json` may still be JSON or base64-encoded JSON; the serialized value
belongs in the environment variable, never inline in YAML. Refresh-token auth
requires all of `refresh_token`, `client_id`, `client_secret`, and `token_uri`;
an access token may be supplied alone or with that complete refresh set.
Credential fields from different auth methods cannot be combined, and
`token_uri` must be an absolute URL without URL user-info.

Migration is intentionally narrow: existing exact `env_var()` references keep
working. Move inline `keyfile_json` mappings/JSON/base64 and literal OAuth
secrets into environment variables, then replace the YAML value with one exact
reference. Replace mixed interpolation such as
`Bearer {{ env_var('TOKEN') }}` and credential defaults with an environment
variable containing the complete value. Literal `keyfile` paths remain valid.

Materialized tables, `--store-failures` tables, and incremental state all
live in the configured dataset — no DuckDB involved. `dbt-ml clean` does not
drop or mutate the BigQuery dataset; it only removes known local target
artifacts. `emit-dbt-sources` emits `database: <project>` / `schema: <dataset>`
so a dbt-bigquery project can consume the tables directly.

#### Partitioning & clustering (`warehouse_options`)

Models may declare adapter-specific physical layout under
`warehouse_options:` (issue #91), mirroring dbt-bigquery's `partition_by` /
`cluster_by` resource configs:

```yaml
- name: filings_chunks
  materialization: incremental
  warehouse_options:
    partition_by:
      field: filing_date        # omit for ingestion-time partitioning
      data_type: date           # timestamp | date (default) | datetime | int64
      granularity: day          # hour | day (default) | month | year
      # int64 instead takes: range: {start: 0, end: 100, interval: 10}
    cluster_by: [cik, form_type] # up to 4 columns; a single string works too
    require_partition_filter: true
    partition_expiration_days: 365
    hours_to_expiration: 72      # whole-table TTL
    labels: {team: econ, env: prod}   # table labels + job labels
    kms_key_name: projects/p/locations/us/keyRings/r/cryptoKeys/k
    incremental_strategy: merge  # or insert_overwrite (see below)
```

The block is validated by the *active* adapter: BigQuery rejects unknown or
malformed keys at run time, while adapters with no layout knobs (DuckDB
today) ignore it entirely — so one project can run DuckDB in dev and
BigQuery in prod. Layout applies when the table is created or fully
rebuilt (`full` models rebuild every run); an existing incremental table
keeps its layout, so adding or changing `partition_by` on an incremental
model needs one `--full-refresh`. Rebuilds are staged and swapped: the
replacement table is built and validated first, so a bad layout
declaration fails the run without touching the last good table.
`warehouse_options` never changes `code_version` — declaring it does not
reprocess documents. `labels` are applied to the table and to the load /
query jobs the run issues for that model (for cost attribution).

**`incremental_strategy: insert_overwrite`** replaces every partition
present in the incoming batch instead of merging by `document_id` —
dbt-bigquery semantics, with partition pruning instead of a full-table
key scan. Two contracts come with it: documents sharing a partition must
always re-extract together (unchanged documents in a touched partition
are dropped, because incremental batches contain only changed documents),
and one run's changed documents must fit in a single flush
(`flush_every`, default 5000) so a partition is never split across
flushes. Time partitioning with a `field` is required. When in doubt,
stay on `merge` — it is always correct.

String values support `{{ env_var('NAME') }}` and
`{{ env_var('NAME', 'default') }}` — the one piece of dbt's Jinja grammar
profiles need for non-secret routing and per-environment paths. Protected
BigQuery credential fields are the deliberate exception: they require one
exact reference with no default and resolve only at native SDK construction.
`api_key_env` remains a literal variable name rather than an `env_var()` call.
An unset ordinary interpolated variable with no default is a load-time error. Each
`warehouse:` block is validated against the config schema of the adapter named
by `type:`; unknown types and typo'd fields fail at resolve time with the
adapter named.
Use target-level `source_paths:` when the same source should read from
different local roots or `gs://` prefixes in dev/staging/prod. Keys are source
names from project YAML; values replace only `source.path`, leaving
`document_id` and incremental identity based on the source-relative object path
and content/generation hash.

## GCS sources

Sources can point at Google Cloud Storage instead of local directories —
raw documents stay in the bucket, dbt-ml materializes into the warehouse:

```
pip install 'dbt-ml[gcs]'
```

```yaml
# sources/documents.yml
version: 2
sources:
  - name: report_html
    path: gs://my-raw-bucket/reports   # bucket + prefix
    project: my-gcp-project             # optional when ADC cannot infer it
    file_pattern: "*.html"             # basename match; "2026/*.html" matches paths
    max_objects: 20000                 # listing bound (default 5000)

  - name: meeting_transcripts
    path: gs://my-raw-bucket/transcripts
    file_pattern: "*.pdf"
    freshness:
      warn_after: { count: 45, period: day }
```

Incremental identity comes from the object listing (md5 → crc32c →
generation), so unchanged objects are skipped **without downloading
anything**; changed objects are fetched generation-pinned into a per-run
scratch directory. Extraction rows gain `source_uri`
(`gs://bucket/name#generation` — exact lineage to the raw object version)
and a `source_metadata` JSON column (size, updated, content type, hashes).
`source freshness` uses object `updated` timestamps.

Auth is Application Default Credentials: `gcloud auth application-default
login` locally, or `GOOGLE_APPLICATION_CREDENTIALS` pointing at a
service-account JSON in CI. User ADC may not carry a default Google Cloud
project; set `GOOGLE_CLOUD_PROJECT` or add `project:` to the GCS source when
project inference is unavailable.

## Document extraction contract

Every extraction row carries identity, lineage, and parser provenance:
`document_id`, `source_path`, `source_uri` (local `file://` URI, or
`gs://bucket/name#generation` for GCS), `content_hash`, `code_version`,
`backend_name`, `backend_version` (the parsing library's version, e.g.
`pypdf/6.1`), and `extracted_at` (one UTC timestamp per run). Remote
sources populate the nullable `source_metadata` JSON column.

> Upgrading note: these columns are new — existing *incremental*
> extraction models will report a schema change on their next reprocess;
> run once with `--full-refresh` (or set
> `on_schema_change: append_new_columns`).

### Declared extraction schema

Top-level model `fields:` is the warehouse output contract for extraction
payload columns. Lineage columns above are automatic; when `fields:` is
non-empty, undeclared backend payload fields are dropped before materialization.

```yaml
fields:
  - name: invoice_id
    data_type: string
  - name: total
    data_type: float
  - name: paid
    data_type: boolean
```

Supported types are `string`, `integer`, `float`, `boolean`, `date`,
`timestamp`, and `json` (`type:` and `dtype:` are accepted input aliases for
`data_type:`). A successful zero-document run materializes a typed, zero-row
relation from this contract, so downstream tests and models see a real table.
Type changes participate in `code_version`; invalid casts fail without
publishing a full-model staging table. A declared field without `data_type`
defaults to string. Omitting `fields:` retains legacy dynamic backend output,
but cannot type payload columns for an initially empty corpus.

Structure-preserving options for document parsing:

```yaml
# Sectioned HTML (reports, filings): headings/tables as JSON with char
# offsets into `text`, so a downstream parser slices sections without
# touching HTML.
- name: raw_reports
  source: ref('report_html')
  extraction:
    backend: html
    options:
      include_structure: true   # emits `sections` and `tables`
  materialization: incremental

# Multi-page PDF (transcripts, reports): per-page char offsets into
# `text`, so e.g. speaker-turn parsing can attribute any match to a page.
- name: raw_transcripts
  source: ref('meeting_transcripts')
  extraction:
    backend: pdf
    options:
      include_pages: true       # emits `pages` [{page, char_start, char_end}]
  materialization: incremental
```

`sections` entries are `{level, heading, char_start, source, anchor?}`;
`tables` are `{index, char_start, n_rows, n_cols, cells}`. Domain-specific
logic (section taxonomy, speaker parsing) belongs in a transform layered
after extraction — the backends stay generic.

By default `sections` only sees semantic `<h1>`–`<h6>` tags
(`source: "tag"`). Corpora that style their headings instead — SEC
inline-XBRL filings render headings as bold `<div>`/`<span>` blocks — need
one of the opt-in detectors:

```yaml
- name: raw_filings
  source: ref('filing_html')
  extraction:
    backend: html
    options:
      include_structure: true
      styled_headings: true      # heuristic: short, fully-bold leaf blocks
      heading_selectors:         # and/or explicit CSS selectors
        - "div.doc-title"        # matches become level 1
        - "div[id^='item']"      # matches become level 2, and so on
  materialization: incremental
```

`styled_headings` treats a leaf block element whose text is short and
entirely bold as a heading, ranking levels by font size (largest = level 1);
entries carry `source: "style"`. `heading_selectors` names headings
explicitly (`source: "selector"`), with selector order setting the level;
its matches win over the heuristic, and semantic heading tags always work.
A selector that matches nothing logs a warning on the run.

### Streaming large corpora

Extraction streams rows to the warehouse every `flush_every` documents
(default 5000), so corpus size is bounded by the flush size, not memory:

```yaml
- name: raw_filings
  source: ref('filing_html')
  extraction:
    backend: html
    flush_every: 1000   # smaller = lower memory, finer crash recovery
  materialization: incremental
```

Incremental writes are atomic per flush: DuckDB uses a transaction and
BigQuery loads a unique staging table then executes one `MERGE`. Missing,
NULL, or duplicate incremental keys are rejected before mutation. A killed
run keeps successful earlier flushes and their state, and the re-run picks up
the remainder. With BigQuery `append_new_columns`, schema addition happens
before the `MERGE`; a failed merge preserves all rows but can leave the new,
nullable column in place.

Full models publish a unique staging table only after every document
succeeds. A parser/backend error preserves the previous target and state.
Backend warnings and zero-source-match warnings appear in the CLI and
`run_results.json`. Changing `flush_every` never invalidates incremental
state. One edge: with `on_schema_change: fail` and more than one flush, the
first flush is compared against the existing table — heterogeneous corpora
whose early documents lack a column can fail where a whole-run union carried
it; use `append_new_columns` there.

### Bounded warehouse snapshots

Warehouse consumers that publish to a serving sink can use the adapter's
`table_snapshot()` context instead of eager `read_table()`. The context exposes
one immutable Arrow schema, opaque safe snapshot and generation fingerprints,
and one-shot record batches whose size is validated between 1 and 100,000 rows. Projection,
AND-combined typed predicates, and an optional same-snapshot NULL/uniqueness
check for a stable key all execute inside the adapter:

```python
from dbt_ml.adapters import ReadPredicate, ReadPredicateOperator

with adapter.table_snapshot(
    "document_chunks",
    columns=("chunk_id", "text", "embedding", "tenant_id"),
    batch_size=2_000,
    predicate=ReadPredicate(
        "tenant_id", ReadPredicateOperator.EQUAL, trusted_tenant_id
    ),
    key_column="chunk_id",
) as snapshot:
    for batch in snapshot:
        publish(batch)
```

DuckDB holds one MVCC read transaction through the context, derives a content
generation fingerprint while consuming it, and performs a second bounded scan
before successful close to reject a newer table version. BigQuery pages one
uncached query result and rejects the read if the table generation changes
while the snapshot is opened or consumed; normal query billing and the
profile's `maximum_bytes_billed` limit still apply. Both adapters push
projection and predicates into the warehouse. Predicate values are bound
parameters and redacted from diagnostics.

Batch ordering is deliberately unspecified. Consumers must use stable row keys
and must keep the context open through their final snapshot validation. The
DuckDB `generation_fingerprint` becomes available only after full iteration;
an early close has no publishable generation. The
existing transform, chunk, and classic-ML runners still use eager
`read_table()`; this contract bounds serving-sink input reads rather than every
dbt-ml execution path.

Incremental state is keyed by a stable record identity within a model, stage,
and target scope. Extraction and chunk generation use `document_id` because a
whole document is their retry unit; downstream publication can independently
track every `chunk_id`. Serving-target descriptors are stored only as a
canonical fingerprint, so changing non-secret semantic target configuration
forces publication without persisting the descriptor itself. Target rows and
their scoped state are deleted together, and new state is recorded only after
the corresponding materialization succeeds.

Existing state upgrades automatically on the first adapter connection. The
legacy `(model_name, document_id)` rows are preserved under the
`materialization` / `warehouse-v1` scope. DuckDB migrates in one transaction;
BigQuery rejects duplicate legacy keys, builds and verifies a v2 staging copy,
then atomically replaces the state table. An unrecognized state-table shape
fails closed with a recovery message instead of being guessed or discarded.
The first incremental chunk run after this migration performs one deliberate
rewrite because the metadata-aware fingerprint replaces the legacy text-only
hash. Chunk IDs remain stable wherever document ID, position, and text are
unchanged; budget for the one-time warehouse write on large corpora.

## Chunking (RAG)

A `chunk:` model splits an upstream document's text into one row per chunk —
the grain RAG and agent retrieval need. Chunk IDs are deterministic and
content-addressed, so an unchanged document re-runs to identical IDs (safe
for incremental MERGE into a warehouse or keyed publish to a retrieval store).

```yaml
- name: document_chunks
  depends_on: [ref('document_registry')]   # an extraction model
  chunk:
    strategy: recursive        # recursive (char splitter) | tokens (tiktoken)
    text_field: text           # upstream column to split
    chunk_size: 800            # chars (recursive) or tokens (tokens)
    chunk_overlap: 100
  materialization: incremental
```

Each chunk row carries `chunk_id`, `document_id`, `chunk_index`,
`chunk_count`, `text`, `chunk_strategy`, `chunked_at`, plus every upstream
column except the split text field — so document lineage (`source_uri`,
`content_hash`, parser provenance) flows onto every chunk for free.
Incremental chunk models skip unchanged documents, re-chunk changed ones
without leaving orphan chunks, and prune chunks of deleted documents.

Chunk identity and row invalidation are deliberately separate:

| upstream change | changes `chunk_id` | invalidates materialized chunk rows |
|---|---:|---:|
| `document_id`, chunk position, or chunk text | yes | yes |
| title, source URI, tenant, ACL/access groups, dates, or other carried metadata | no | yes |
| native nested mapping key order only | no | no |
| native nested/list value or list order | no | yes |
| splitter/code configuration | only if position or text changes | yes |

The invalidation fingerprint uses canonical typed serialization for mappings,
lists, nulls, timestamps, decimals, and binary values. It includes the split
text plus every upstream value that survives on the chunk row. Only fields
replaced by the chunk model (`chunk_id`, `chunk_index`, `chunk_count`, output
`text`, `chunk_strategy`, `code_version`, and `chunked_at`) are excluded;
`document_id` is included explicitly. As a result, an ACL-only change rewrites
the affected rows while preserving stable chunk IDs when their text and
positions are unchanged.

The recommended document-layer shape (GCS raw files → BigQuery tables):

| model | grain | kind |
|-------|-------|------|
| `document_registry` | one row per document/version | `extraction` (`include_structure`) |
| `document_chunks`   | one row per chunk            | `chunk` |
| `document_extractions` | one row per structured field set | `extraction` (llm) or `transform` |

See `examples/rag_chunks_pipeline/` for a runnable registry → chunks project.
Domain keys (symbol, filing date, …) belong in transforms or downstream dbt
models layered on top — the chunk grain stays generic.

## Embedding models

An `embed:` model batches one upstream text field through an
`EmbeddingProvider` and materializes one canonical row per stable upstream ID.
It preserves upstream text, document/chunk lineage, and filter metadata while
adding the vector and its safe provider identity.

```yaml
- name: document_embeddings
  depends_on: [ref('document_chunks')]
  embed:
    provider: deterministic
    model: contract-v1
    text_field: text
    id_field: chunk_id
    vector_field: embedding
    dimensions: 8
    batch_size: 128
  materialization: incremental
```

The built-in `deterministic` provider is offline and reproducible. It exists for
tests, examples, and pipeline integration—not semantic similarity quality. A
production provider implements the same `EmbeddingProvider` contract.

Canonical output adds `embedding_provider`, `embedding_model`,
`embedding_dimensions`, `embedding_provider_implementation`,
`embedding_input_hash`, `embedding_config_hash`, and `embedded_at`. Vectors are
portable numeric list values. Manifest and run-results artifacts expose only
safe identity and aggregate usage metadata; input text and credentials are not
copied into artifacts.

Incremental runs distinguish three cases:

- unchanged rows are skipped;
- metadata-only changes reuse the existing vector and refresh the warehouse row;
- text, model, provider, dimensions, or implementation changes recompute it.

Removed upstream IDs are deleted downstream. Provider results are validated for
cardinality, dimensions, and finite numbers before any rows or state are
published. `dbt_ml.embedding.embed_query()` accepts the identity recorded in
the manifest so query-time vectors cannot silently use a different provider
implementation or configuration.

## Search indexes (local proof of concept)

A `search:` resource publishes exactly one upstream warehouse model to an
independently configured retrieval store. It is a leaf serving resource, not a
warehouse relation. Install `dbt-ml[lancedb]`, configure the operator-owned
store in `profiles.yml`, and explicitly opt in to public indexes:

```yaml
my_project:
  target: dev
  outputs:
    dev:
      warehouse:
        type: duckdb
        path: ./target/dbt_ml.duckdb
        schema: my_project
      retrieval:
        default: local
        allow_public_indexes: true
        stores:
          local:
            type: lancedb
            path: ./target/lancedb
```

The project model declares the portable serving contract:

```yaml
- name: chunk_search
  depends_on: [ref('chunk_embeddings')]
  materialization: incremental
  search:
    access: public
    store: local
    collection: document_chunks
    id_field: chunk_id
    text_fields: [text]
    return_text_fields: [text]
    vector:
      field: embedding
      dimensions: 768
      metric: cosine
      search: exact
      embedding: inherit
    full_text:
      fields: [text]
    attributes:
      - name: source_uri
        data_type: string
        filter_role: user
        returned: true
    query:
      modes: [vector, text, hybrid]
      consistency: strong
```

`run` and `build` stream projected Arrow batches from the warehouse, validate
the declared row contract before each mutation, upsert changed rows, delete
stale rows, and advance warehouse state only after exact durable receipts,
index validation, and the snapshot generation check all succeed.
`ls --resource-type search_index` lists serving resources; `show` rejects them
because they have no warehouse table. Manifest v2 exposes a non-secret
`serving_resource` descriptor with the resolved embedding identity.

Query the index from the CLI:

```bash
dbt-ml search --model chunk_search --query "latest inflation release" --mode hybrid
dbt-ml search --model chunk_search --query "inflation" \
  --filter source_uri eq reports/cpi.md --output json
dbt-ml search --model chunk_search --query "labor market" \
  --filter category in '["employment", "wages"]'
```

Filters are repeatable `FIELD OP VALUE` triples. Operators are `eq`, `ne`,
`lt`, `le`, `gt`, `ge`, and `in`; `in` takes a JSON array. Values are parsed
against the attribute's declared type, and only attributes with
`filter_role: user` can be supplied by a caller. Multiple filters are combined
with AND.

The same request is available as a provider-neutral Python API:

```python
from dbt_ml.search import SearchMode, SearchRequest, search

results = search(
    ".",
    SearchRequest(
        model="chunk_search",
        query="latest inflation release",
        mode=SearchMode.HYBRID,
        limit=10,
    ),
)
```

Vector queries can provide a precomputed `vector=` instead of query text. When
`embedding: inherit` points directly to a native `embed:` model, dbt-ml reuses
that model's exact provider identity for query-time embedding and rejects stale
or dimension-incompatible indexes. Externally generated vectors still declare
a complete embedding identity and require a precomputed query vector.

### Serving readiness and coordination

Publication is generation-fenced (issue #152). The active warehouse owns a
per-index serving ledger plus publish/query leases: a publisher acquires an
exclusive fenced claim (and an OS-enforced per-collection lock on the LanceDB
store) before any store mutation, and marks the scope `ready` only after
receipts, index validation, the snapshot generation check, and state
advancement all succeed. A failed or interrupted publish leaves the scope
unavailable to queries until a later publish succeeds. Queries take a shared
lease that pins the ready physical generation through query embedding, store
search, and result validation; they are rejected while a publisher is active,
and publication is rejected while query leases are held.

There is no timeout-based lease stealing. If a publisher crashes, terminate
it, then explicitly reassign authority:

```bash
dbt-ml serving status chunk_search     # ledger status, fence, counts, leases
dbt-ml serving recover chunk_search --owner-terminated
```

Recovery advances the fencing token (so a surviving zombie fails its next
check), clears leases, and leaves the scope failed until the next `dbt-ml run`
republishes it. After upgrading to this contract, run `dbt-ml run` once per
search index to establish its ledger before querying.

Governed indexes (`access: governed`) are supported on stores that declare
strong read-after-write consistency and metadata filtering. Changed governed
records are deleted before their replacement is upserted, so a failed policy
revocation leaves the old row absent rather than queryable. Governed queries
fail closed unless the calling service supplies trusted `policy_filters=` that
constrain every policy-role attribute; they are composed with user filters as
mandatory in-store prefilters and are rejected on public indexes. The
`dbt-ml search` CLI serves public indexes only — an interactive flag is not a
trusted authorization context.

This slice still deliberately rejects search-resource tests, full refresh,
online/rebuild schema changes, arbitrary predicate strings, and
adapter-specific index options. Bounded state paging and atomic full
replacement remain #153; distributed-store fencing (provider-enforced fencing
or immutable-generation activation) remains declared-but-unclaimed until a
hosted adapter (#136) implements it. These are unsupported guarantees, not
silent best-effort behavior.

## Built-in text preprocessing

Reference any of these as a Python transform module — no project-local code
needed. Users can override by writing their own `transforms/<name>.py`
(project-local files win over installed packages).

```yaml
- name: post_text_stats
  depends_on: [ref('raw_posts')]
  transform:
    type: python
    module: dbt_ml.text.transforms.text_stats   # built-in, ships with dbt-ml
    options:
      text_field: body
      emit: [word_count, sentence_count]
```

| Module                                    | What it does                                                                   |
|-------------------------------------------|--------------------------------------------------------------------------------|
| `dbt_ml.text.transforms.text_stats`        | Adds `word_count` / `char_count` / `sentence_count` / `paragraph_count`         |
| `dbt_ml.text.transforms.clean_encoding`    | Fixes mojibake (UTF-8-as-Latin-1 confusion) via ftfy                            |
| `dbt_ml.text.transforms.detect_language`   | Adds a 2-letter ISO language code per row via langdetect                        |
| `dbt_ml.text.transforms.count_tokens`      | Adds `token_count` for an OpenAI / Claude-style tokenizer (tiktoken)            |
| `dbt_ml.text.transforms.find_duplicates`   | Flags near-duplicate rows via MinHash + LSH (Jaccard threshold configurable)    |
| `dbt_ml.text.transforms.redact_pii`        | Detects + redacts PII via Microsoft Presidio (requires `en_core_web_sm` spaCy model) |

All are pure functions importable via `from dbt_ml.text import …` if you'd
rather wire them into your own transforms.

**PII setup** — `redact_pii` uses spaCy under the hood. First-time install:

```bash
python -m spacy download en_core_web_sm
```

Without the model, calls into `redact_pii` raise a clear `PIIError` pointing
at this command.

For a customer-facing relation, use an allow-list projection:

```yaml
- name: redacted_tickets
  depends_on: [ref('raw_tickets')]
  transform:
    type: python
    module: dbt_ml.text.transforms.redact_pii
    options:
      text_field: summary
      output_field: summary_redacted
      entities_field: pii_entities
      keep_fields: [ticket_id, summary_redacted, pii_entities]
```

`entities_field` stores type, offsets, and confidence by default; it does not
store the matched substring. `include_raw_text: true` opts back into raw PII
evidence and makes that output sensitive. When `output_field` differs from
`text_field`, the original text is dropped unless `retain_input_text: true` is
set. `keep_fields` and `drop_fields` are mutually exclusive, and unknown
projection fields fail loudly. Other upstream columns are otherwise retained,
so use `keep_fields` for a relation that must exclude names, email addresses,
or other sensitive source columns.

## Classic text and document ML

Classic ML is a first-class dbt-ml lane alongside LLM/RAG work. The `ml:`
model block executes deterministic text/document workflows and persists their
artifacts; shipped providers cover Count/TF-IDF/hashing features and Naive
Bayes classification. Additional regression, clustering, topic-model, and NLP
providers remain roadmap work.

```yaml
- name: ticket_tfidf
  depends_on: [ref('raw_tickets')]
  ml:
    task: features
    mode: fit_transform
    provider: builtin.tfidf
    text_field: body
    artifact:
      path: target/artifacts/ticket_tfidf
    metrics: [vocabulary_size]
    options:
      ngram_range: [1, 2]
      max_features: 50000
```

Executable feature providers are `builtin.count`, `builtin.tfidf`, and
`builtin.hashing`. They write long-form sparse feature tables with stable
`row_id`, `term`, `term_index`, `count`, `tf`, `idf`, `tfidf`, and `value`
columns where applicable. Fitted vocabulary providers persist
`target/artifacts/<model>/metadata.json` plus `vocabulary.json`; hashing is
stateless and persists metadata only.

Common options include `analyzer: word | char | char_wb`, `ngram_range`,
`min_df`, `max_df`, `max_features`, `stop_words`, `binary`, `n_features`, and
`alternate_sign`. See `docs/classic-ml.md` for the full design contract.

The first supervised provider is `builtin.naive_bayes`, which trains a
deterministic text classifier from `text_field` and `label_field`, persists a
model artifact, and materializes prediction rows with scores/probabilities.

## Tests

**Structural:**

```yaml
tests:
  - not_null: [vendor, total]            # column-level, fails the run
  - unique: invoice_id                   # single-column
  - unique: [a, b]                       # composite (compiled to dbt_utils on emit)
  - min_rows: 100
  - not_empty                            # bare-string form of min_rows: 1
  - not_null: total                      # warn doesn't fail the run
    severity: warn
  - relationships: { column: vendor_id, to: ref('vendors'), field: id }  # referential integrity
  - python: tests.my_check               # custom: tests/my_check.py defines run(con, table_ref) -> str | None
```

**Traditional ML / statistical data-quality checks** (deterministic, no LLM, no
sampling — see [issue #10](https://github.com/C00ldudeNoonan/dbt-ml/issues/10)
for the full design including the optional LLM-judge tier):

```yaml
tests:
  - matches_regex: { column: arxiv_id, pattern: '^\d{4}\.\d{4,5}$' }
  - accepted_values: { column: primary_category, values: [cs.LG, cs.CL, stat.ML] }
  - accepted_range: { column: n_authors, min: 1, max: 30 }
  - null_rate: { column: title, max: 0.0 }       # silent-extraction-failure guard
  # deterministic faithfulness — extracted value must appear in the source text,
  # catching hallucinated values with zero LLM calls:
  - grounded_in: { value: title, source: abstract, method: exact }
```

`grounded_in` also supports `method: fuzzy` with a `min_score`. These run as
full-table aggregates, so they stay cheap and reproducible.

**Inspecting failures.** Pass `--store-failures` to `dbt-ml test` or `dbt-ml
build` to persist the offending rows of each failing test to a
`dbt_ml_test_failures__<model>__<test>[__<column>]` table (replaced each run).
The test output reports the table name and row count. These tables are
inspection artifacts and are kept out of the model namespace (they don't show up
in `dbt-ml ls` or `emit-dbt-sources`).

**`dbt-ml build`** runs and tests each model in dependency order, skipping a
model's descendants when it errors or fails a test — so a bad upstream extraction
stops before it pollutes everything downstream.

## Examples in this repo

| Path                                | What it shows                                                          |
|-------------------------------------|------------------------------------------------------------------------|
| `examples/invoice_pipeline/`        | JSON extraction → per-vendor + monthly aggregations                    |
| `examples/blog_pipeline/`           | Markdown frontmatter → per-author word counts                          |
| `examples/pdf_invoice_pipeline/`    | PDFs → text via pypdf → LLM-extracted structured fields                |
| `examples/llm_invoice_pipeline/`    | Free-form invoice text → LLM extraction (no PDF stage)                 |
| `examples/support_tickets_pipeline/`| JSON tickets → open queue + SLA breaches + per-team workload (no LLM)  |
| `examples/arxiv_papers/`            | arXiv metadata → deterministic data-quality checks (incl. `grounded_in`) |
| `examples/dbt_consumer/`            | dbt-duckdb project consuming dbt-ml-materialized tables                 |
| `examples/classic_text_ml/`         | deterministic sparse text features + Naive Bayes classification        |
| `examples/rag_chunks_pipeline/`     | document registry → deterministic RAG chunks                           |

Each example is runnable end-to-end with `uv run dbt-ml --project-dir examples/<name> ...`.

## Composing with dbt

dbt-ml does the unstructured→structured "E" and dbt does the SQL "T".
`emit-dbt-sources` targets the matching adapter: dbt-duckdb can share the
DuckDB file, and dbt-bigquery can read the configured BigQuery dataset. The
DuckDB bridge:

```bash
uv run dbt-ml --project-dir examples/invoice_pipeline run
uv run dbt-ml --project-dir examples/invoice_pipeline emit-dbt-sources \
  --output examples/dbt_consumer/models/sources/_dbt_ml_sources.yml

cd examples/dbt_consumer && uv sync && uv run dbt build --profiles-dir .
```

`emit-dbt-sources` translates dbt-ml tables into a dbt-compatible `sources.yml`.
Column tests carry over (`not_null`, single-column `unique`); composite unique
becomes a `dbt_utils.unique_combination_of_columns` macro test.

## Artifacts

`dbt-ml compile` writes the manifest; `run` and `build` write the manifest and
run results under `target-path`:

- **`manifest.json`** — project, sources, models, refs, tags, `code_version` per
  model, DAG nodes+edges+execution order. Re-generated each run.
- **`run_results.json`** — run-level metadata (warehouse target, status, counts,
  elapsed, and `sources_considered`) plus per-model documents
  processed/skipped, rows written, duration, warnings, errors, `status`, and
  the fully-qualified output `relation`. LLM extraction models also carry
  token accounting in `metrics` (API calls, cache hits, input/output/cache
  tokens, and `estimated_cost_usd` when the profile sets `pricing:`).
  `run`/`build` also accept `--json` to print this payload to stdout.
- **`sources.yml`** — only when you call `emit-dbt-sources`. dbt-shaped.
- **`docs/`** — static HTML site (`dbt-ml docs generate`) with project overview,
  Mermaid DAG, per-model pages. Serve locally with `dbt-ml docs serve`.

External tools (lineage viewers, CI dashboards, the dbt-consumer above)
consume these. `run`/`build` exit `0` on success, `1` on run failure, and `2` on
a configuration error, so an orchestrator can branch on the cause. Because
dbt-ml tables are dbt sources, they wire natively into the `dagster-dbt`
integration — see
[`docs/orchestration-dagster.md`](docs/orchestration-dagster.md) (use
`emit-dbt-sources --dagster-meta` to pin the Dagster asset keys).

## Benchmarks

```bash
uv run python scripts/benchmark.py --count 5000
```

5000-doc benchmark on the JSON backend:

```
seed 5000 invoices                          0.8s    →   6.3k docs/sec
first run (cold)                            4.8s    →   1.0k docs/sec
second run (all skipped)                    0.3s    →  19.9k docs/sec
third run (1 changed)                       0.3s    →  18.2k docs/sec
full-refresh                                4.3s    →   1.2k docs/sec
```

These historical v0.1 numbers are a local baseline, not a service-level
guarantee. Current runs support `--threads` for per-document extraction and
parallel independent model batches; benchmark your own parser, warehouse, and
source mix.

## Layout

```
src/dbt_ml/
├── cli.py                 # click: init/seed/compile/graph/run/test/show/clean/source freshness/emit-dbt-sources
├── config/                # pydantic models for project/source/model/profile + loader
├── profile.py             # profile discovery + resolution (warehouse + llm)
├── dag.py                 # graphlib-based DAG, selectors (+ name +, tag:foo), Mermaid render
├── adapters/              # warehouse adapters + adapter-owned incremental state
├── runner.py              # extract → materialize orchestration
├── manifest.py            # target/manifest.json + run_results.json
├── dbt_export.py          # target/sources.yml (dbt-shaped)
├── freshness.py           # source mtime check
├── backends/              # json, markdown, pdf, html, email, llm
├── transforms/runner.py   # loads user Python transform modules + TransformContext
├── checks/                # schema tests + custom Python tests + severity
├── synth/                 # synthetic data generators per shape
└── templates/             # init scaffolds for {json,pdf,markdown,html}
```

## Roadmap

The live plan is maintained in GitHub issues tagged
[`roadmap`](https://github.com/C00ldudeNoonan/dbt-ml/issues?q=is%3Aissue+label%3Aroadmap).
Already shipped in the v0.2 preview: the warehouse adapter seam, DuckDB and
BigQuery, GCS sources, recursive/token chunk models, layout-preserving HTML/PDF
metadata, PII redaction, the first classic-ML providers, and the local LanceDB
search-index proof of concept.

Next adapter work follows dbt-core's warehouse set over time: Postgres first,
then Snowflake, Databricks, and Redshift. Production embeddings, hosted
retrieval stores, more parser providers, and evaluation/reranking remain
roadmap items. Incremental state stays adapter-owned. Rust, PyO3, and Metaxy
remain explicitly deferred.

The accepted [semantic retrieval architecture](docs/architecture/semantic-retrieval.md)
defines the `search:` DAG resource, `RetrievalStore` boundary, typed filters,
incremental publication state, and serving-resource artifacts. The local
LanceDB publication and portable Python/`dbt-ml search` query surfaces ship
with generation-fenced readiness, publish/query leases, explicit recovery, and
governed policy-prefilter queries (issue #152) inside their documented
single-host boundary; bounded state paging (#153) and distributed-store
fencing (#136) remain roadmap work and fail closed.

The versioned [agent context contract](docs/architecture/agent-context-v1.md)
defines the document registry, chunk, and dbt-entity link grains used to carry
bitemporal validity, policy, freshness, provenance, and exact citations from
warehouse models into governed retrieval projections.
