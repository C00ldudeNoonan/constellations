# Constellations (`stel`)

**dbt for unstructured data.** Declarative YAML pipelines that turn folders of
documents — PDFs, markdown, HTML, JSON, email, free-form text — into warehouse
tables. Incremental processing, schema tests, dbt-style selectors, profiles,
and a manifest artifact you can wire into other tools.

The current v0.2 preview is pure Python and supports DuckDB and BigQuery
warehouses, local and GCS sources, document chunk models, executable classic
text-ML providers, native warehouse-materialized embedding models with a
deterministic offline provider plus Google Vertex AI, and an incremental local
LanceDB search-index proof of concept. Additional warehouse, embedding, and
retrieval work follows the focused platform scope below; Rust and PyO3 are
explicitly out of scope through v0.2.

### Supported and planned platforms

| Role | Shipped | Active roadmap |
|------|---------|----------------|
| Warehouse | DuckDB, MotherDuck, BigQuery | [Snowflake](https://github.com/C00ldudeNoonan/dbt-ml/issues/187) |
| Document source | local files, GCS | improvements within the same local/GCP scope |
| Retrieval store | local LanceDB | production hardening of the portable LanceDB contract |
| Embedded dbt execution | dbt-duckdb preview | remaining dbt-duckdb integration work |

Additional warehouse, cloud, and hosted retrieval integrations are not on the
current roadmap. Shipped inference providers and extraction backends remain
supported; this table narrows new platform work rather than removing existing
features. BigQuery and future Snowflake projects compose with dbt through the
standalone CLI and `emit-dbt-sources`, not warehouse-hosted Python execution.

## Where stel fits

The 2026 landscape for unstructured document pipelines has two stable poles:

- **Managed RAG-as-a-Service** (Vectara, Bedrock Knowledge Bases, Vertex AI
  Search, Snowflake Cortex Search, Glean) — best when time-to-value matters
  and the team can't dedicate ML engineers.
- **Compose best-of-breed Python components** (LlamaParse → contextual
  chunking → Voyage embeddings → Qdrant → Cohere Rerank → Ragas) — best when
  retrieval quality, multi-tenant isolation, or unusual document types
  matter and you have ≥2 ML engineers.

stel is the **opinionated, declarative path through the second lane**.
Where LlamaIndex is imperative Python, stel is YAML + a manifest + tests +
lineage. Where Snowflake Cortex Search hides everything, stel makes every
stage inspectable and reproducible. It's *dbt-shaped*: the same DAG +
selectors + tests + artifacts pattern, applied to unstructured data.

---

## You have a folder of files. Get them into your warehouse.

```bash
# Install from PyPI with the PDF parser used below
uv add 'stel[pdf]'

# 1. Scaffold a project for whatever shape your data is
uv run stel init my_project --template pdf      # or json, markdown, html

# 2. Drop your files into ./my_project/data/pdfs/  (or wherever the source points)

# 3. Run it
cd my_project
uv run stel run

# 4. Query the result
duckdb target/stel.duckdb -c "SELECT * FROM my_project.raw_pdf_text LIMIT 5"
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
| `vertex` | Google Vertex AI text embeddings (`google-genai`) |
| `lancedb` | Local LanceDB search-index publication and queries |
| [`mcp`](mcp.md) | Read-only governed context server over MCP stdio |
| `all` | Every optional feature above |

For example, `uv add 'stel[pdf,text]'` installs PDF and text processing,
while `uv add 'stel[all]'` provides the complete development/runtime feature
set. Invoking a feature whose extra is absent raises an error with the exact
installation command.

## What stel actually does

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
| **Profile**        | Warehouse + LLM + embedding-provider config, swappable per `--target dev|prod`. No credentials in models. |
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

stel projects are local code-and-data projects. Only run projects you trust:
Python transforms and custom Python tests execute in your Python process, and
project configuration controls source globs, generated paths, and executable
modules. The discovered profile controls warehouse, cache, and protected
credential references. Reference names and values are omitted from artifacts
and user-facing diagnostics.

Document parsers process local files with third-party libraries. Keep
dependencies current before running stel over untrusted PDFs, HTML, email, or
other documents, since malformed files can trigger parser CPU or memory bugs.

The `llm` backend and hosted embedding providers send document text to the
configured model service. The LLM backend stores cached structured responses
in plaintext in the configured cache database. New
POSIX cache databases and transient write-ahead logs are forced to owner-only
mode (`0600`), but the files still contain extracted document data and must be
handled as sensitive. Use
deterministic local backends for sensitive documents unless remote processing is
intended.

Local LanceDB collections contain the projected chunk text, embeddings, and
returned/filter attributes in plaintext beneath the operator-configured profile
path. Public indexes require the active profile to set
`retrieval.allow_public_indexes: true`. Governed indexes require a trusted
calling service to supply complete mandatory `policy_filters`; the interactive
CLI cannot manufacture that authorization context and serves public indexes
only. Profiles select resources but are not an identity or authorization
service.

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

`stel clean` removes only known local artifacts under `target-path`
(`manifest.json`, `run_results.json`, generated `sources.yml`, `docs/`, and
classic-ML `artifacts/`). It preserves configured warehouse/cache files and
unknown files, never calls an adapter-level database/schema/dataset reset,
rejects project-root or source/model/transform overlap, and refuses symlinked
paths. There is no `--force` option.

Running a third-party project still executes its Python transforms, custom
tests, and post-extract hooks, and remote sources (`gs://…`) reach whatever
your ambient credentials allow — review projects you didn't write before
running them.

### Derive fields before warehouse publication

When a source object is an envelope around a large payload, use a project-local
`post_extract` hook to derive the useful representation before stel builds a
warehouse row. The hook replaces the backend's field mapping; fields it omits
never enter the staging frame or target table. This avoids a raw-payload table
and a second warehouse transform pass:

```yaml
extraction:
  backend: json
  options:
    fields: [accession_number, content]
  post_extract:
    module: post_extract.sec_text
    options:
      html_field: content
      output_field: text
```

The dotted module is a `.py` file inside the project. For the configuration
above, create `post_extract/sec_text.py`:

```python
from collections.abc import Mapping
from typing import Any


def validate_options(options: Mapping[str, Any]) -> None:
    required = {"html_field", "output_field"}
    if set(options) != required:
        raise ValueError(f"options must be exactly {sorted(required)}")


def run(fields: dict[str, Any], ctx: Any) -> dict[str, Any]:
    from bs4 import BeautifulSoup  # stel[html]

    html = fields[ctx.options["html_field"]]
    return {
        "accession_number": fields["accession_number"],
        ctx.options["output_field"]: BeautifulSoup(
            html, "html.parser"
        ).get_text("\n", strip=True),
    }
```

`run` may accept `(fields)` or `(fields, ctx)` and must return a mapping with
string field names. `fields` is a copy of the backend output. `ctx` exposes the
document/source identity, source metadata, configured hook options, and the
verified local snapshot path. Backend warnings and numeric usage metrics are
preserved automatically. A shorthand without options is also valid:
`post_extract: post_extract.sec_text`.

stel imports the module and calls its optional `validate_options(options)`
during compilation, before source discovery, credentials, or warehouse access.
The hook runs once per successful backend result, including native-batch
results, while the verified source snapshot still exists. Its module source and
options participate in `code_version`, so an incremental model reprocesses
documents when derivation logic changes. Hook failure details are sanitized
because the hook may be holding raw document content or sensitive options.
Hook option values are omitted from generated manifests; the artifact records
the module and resulting `code_version`, not arbitrary project configuration.

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
OpenAI-compatible endpoints. See the [vLLM provider guide](vllm.md) for
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
stel init <name> [--template {json,pdf,markdown,html}]   # scaffold a fresh project
stel seed [--count N] [--type {invoices,posts,...,tickets,emails}]
stel compile                                             # parse YAML, validate DAG, write manifest.json
stel graph                                               # Mermaid DAG to stdout
stel run [--select EXPR] [--exclude EXPR] [--full-refresh] [--threads N] [--watch] [--state DIR] [--source-filter GLOB] [-v]
stel test [--select EXPR] [--exclude EXPR] [--store-failures] [--state DIR]
stel eval [--select EXPR] [--exclude EXPR] [--json]      # golden-set retrieval evaluation (recall/precision/MRR/NDCG@k)
stel build [--select EXPR] [--exclude EXPR] [--full-refresh] [--threads N] [--store-failures] [--state DIR] [--source-filter GLOB] [-v]
stel ls [--select EXPR] [--resource-type {model,source,search_index,all}] [--output {name,json}]
stel show <model> [--limit N]                            # peek at a materialized table
stel search --model NAME --query TEXT [--mode {vector,text,hybrid}] [--filter FIELD OP VALUE] [--output {table,json}]
stel serving status <search-index>                       # publication ledger: status, fence, counts, leases
stel serving recover <search-index> --owner-terminated   # explicit authority reassignment after a crash
stel providers list [--output {table,json}]              # built-in + entry-point providers, incompatible plugins flagged
stel source freshness                                    # mtime vs warn_after/error_after
stel docs generate [--output DIR]                        # static HTML site from manifest.json
stel docs serve [--port N]                               # local http.server over target/docs/
stel emit-dbt-sources [--output PATH]                    # write dbt-compatible sources.yml
stel codegen --output DIR                                # generate dbt Python-model shims + schema.yml (embedded path)
stel clean                                               # remove known target artifacts; preserve warehouses
stel migrate [--dry-run]                                 # one-time: rename pre-#313 internal warehouse tables

# Global flags (work on every command):
stel --project-dir <dir> --profiles-dir <dir> --target <name> <command>
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
- `--source-filter GLOB` (repeatable) on `run`/`build` scopes a run to the
  *documents* whose source-relative path matches the glob (`*` spans `/`, so
  `--source-filter 'AAPL/*'` selects a whole prefix) — distinct from `--select`,
  which chooses models. It's the seam for orchestrator-driven **partitioned**
  processing (one ticker/partition per run, parallelized, backfillable). A
  filtered run is **additive/upsert-only**: it never deletes, requires
  incremental extraction models, and is rejected with `--full-refresh`. Deletion
  of removed documents is reconciled by a periodic unfiltered full run.

### Upgrading a warehouse built before the rename

stel's internal warehouse objects are named after the tool, so renaming the
tool renamed them: `dbt_ml_state` became `stel_state`, the serving ledger and
leases followed, the default schema went from `dbt_ml` to `stel`, and the
zero-config DuckDB file from `target/dbt_ml.duckdb` to `target/stel.duckdb`.

Those tables are data. `stel_state` holds every incremental fingerprint, so a
run that cannot find it concludes that every document is new and reprocesses
the whole corpus at provider cost — with no error, because that is exactly
what a genuine first run looks like. Nothing is allowed to reach that state
silently, so each way of arriving at it is a hard stop with the fix named.

The rename also moved two files you own, and both come *before* the warehouse
in the order things fail.

**The project file.** `dbt_ml_project.yml` is now `stel_project.yml`. This is
the first thing an upgrading project hits, so the missing-file error looks for
the old name beside it and gives you the rename:

```
git mv dbt_ml_project.yml stel_project.yml
```

**A global profile.** If yours lives in your home directory rather than the
project, it moved from `~/.dbt_ml/profiles.yml` to `~/.stel/profiles.yml`. The
"no profiles.yml was found" error names the old path when it is still there.

Neither old name is ever loaded — only reported. Two spellings that both work
is how the old one never dies, so these stay one-time, visible steps.

**Internal tables under their old names.** Run the migration once per target:

```
stel migrate --dry-run   # what would be renamed
stel migrate             # rename in place, rows preserved
```

It renames only the tables stel owns, only inside the schema the target
already points at. If it finds both spellings of the same object it refuses
rather than choosing which one holds the live rows.

**A schema you never named.** A project with no `schema:` in its profile used
to get `dbt_ml` and now gets `stel`. If the old schema still holds your data,
say so explicitly — moving a whole schema is your call, not the migration's:

```yaml
warehouse:
  type: duckdb
  path: ./target/stel.duckdb
  schema: dbt_ml
```

An explicit `schema:` is never second-guessed, so a project that already had
one is unaffected either way.

**A zero-config project.** Same shape: a project with no `profile:` that never
set `duckdb.path` now defaults to `target/stel.duckdb`. Config load refuses to
open it while `target/dbt_ml.duckdb` exists; point `path:` at the existing file
(then `stel migrate`), or delete it if you meant to start over.

Model names beginning with `stel_` are now reserved alongside `dbt_ml_`, which
stays reserved even though no internal table uses it any more.

## Selectors

dbt-shaped. Whitespace-separated tokens, optional `+` modifiers, `tag:` prefix.

```bash
stel run --select raw_pdf_text       # one model
stel run --select 'raw_pdf_text+'    # plus all downstream
stel run --select '+invoice_summary' # plus all upstream
stel run --select 'tag:raw+'         # all models tagged "raw" + their downstream
stel run --exclude tag:expensive
stel run --select 'state:modified+' --state ./main-manifest/
                                       # only models whose config or transform
                                       # code changed vs a previous manifest,
                                       # plus their downstream
```

`state:modified` compares each model's `code_version` (a hash of its
extraction/transform/ml config and transform module source) against a
manifest written by a previous `compile` or `run`. The CI recipe: store
`target/manifest.json` from main, then on PRs run
`stel build --select 'state:modified+' --state path/to/main-manifest/`.

## Progress output

`stel run` and `stel build` are silent by default beyond their final
summary table. Pass `-v` for per-source discovery lines, per-model
start/finish lines, and a live per-model progress bar on a TTY; on
non-TTY stderr (e.g. a Dagster capture) the same events land as plain
INFO log lines instead. Exactly one channel is active at a time so the
bar isn't corrupted by log lines writing over its redraws. `run --threads N`
runs independent models concurrently, so it uses the log-line channel even on
a TTY (parallel bars on one terminal would interleave). With `--source-filter`,
the reported per-source count is the post-filter selected count, so it always
reflects what is actually processed. For runs launched by an orchestrator,
`STEL_VERBOSE=1` enables verbose output without changing the CLI invocation.
All progress output goes to stderr, so `--json` on stdout stays a single
parseable payload.

The verbose flag is deliberately capped at INFO. DEBUG-level log sites
(transform failures, provider errors) carry unsanitized exception text
and traceback frames that the user-facing error path scrubs but a raw
log stream would not — attach your own DEBUG handler if you need it for
troubleshooting.

Under verbose, each BigQuery incremental publication also emits safe
telemetry (issue #292) on whichever channel is active — the progress reporter
on a TTY, the INFO log on a captured/orchestrator run: the output relation, the
BigQuery **job id**, bytes processed, and DML-affected row count. The job id
lets you match stel's own jobs against BigQuery job history /
`INFORMATION_SCHEMA.JOBS`, so many tiny stel flushes can be told apart from an
overlapping orchestrator run. Only job-level statistics and the table name are
surfaced — never SQL text or row values.

## Matrix model expansion (`for_each`)

Declare `for_each` on any model to turn it into a template. stel expands
it into one concrete model per cartesian-product combination of the axis
values before the DAG is built, so selectors, lineage, incremental state,
and the manifest all see ordinary models.

```yaml
models:
  - name: ticket_tfidf
    depends_on: [ref('raw_tickets')]
    for_each:
      min_df:    [1, 2, 5]
      ngram_range: [[1, 1], [1, 2]]
    ml:
      task: features
      mode: fit_transform
      provider: builtin.tfidf
      text_field: body
      artifact:
        path: target/artifacts/ticket_tfidf
      options:
        min_df:      ${matrix.min_df}
        ngram_range: ${matrix.ngram_range}
```

This produces six models named
`ticket_tfidf__min_df_1__ngram_range_1_1`,
`ticket_tfidf__min_df_1__ngram_range_1_2`, …,
`ticket_tfidf__min_df_5__ngram_range_1_2`.

**Placeholder syntax** — write `${matrix.<axis>}` anywhere in a string value
in the model config:

- An **exact-match** placeholder (`"${matrix.min_df}"`) substitutes the axis
  value type-preservingly: an integer axis value produces an integer, a list
  produces a list. Typed config fields such as `chunk_size` or `dimensions`
  work correctly.
- A placeholder **embedded** in a longer string
  (`"artifacts/${matrix.label}"`) is interpolated as a string.

**Naming** — variant names are `<base>__<axis>_<slug>__…`. Slugs are
identifier-safe (letters, digits, underscores; `.` and spaces become `_`).
Long values are truncated with an 8-character SHA-256 suffix.

**Selecting variants** — every variant is automatically tagged with the base
model name, so `--select tag:ticket_tfidf` (or `--select tag:ticket_tfidf+`)
runs all six variants and their downstream:

```bash
stel run --select 'tag:ticket_tfidf+'
stel run --select ticket_tfidf__min_df_1__ngram_range_1_2
```

**Limits and errors** — a template may expand to at most 256 variants.
Axis names must be valid identifiers. Empty axis lists and slug collisions
(two combinations that produce the same name) are rejected at project load
time with a clear error.

## Profiles

Warehouse and LLM config live in `profiles.yml`, *not* in `stel_project.yml`.
Project YAML says `profile: my_project`; profile says where to write and which
LLM to call. Swap `--target prod` to switch environments.

```yaml
# profiles.yml — sits next to stel_project.yml, or in ~/.stel/profiles.yml
my_project:
  target: dev
  outputs:
    dev:
      warehouse:
        type: duckdb
        path: ./target/stel.duckdb
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
          cache_read_usd_per_mtok: 0.10   # numbers, stel ships no price table.
    prod:
      warehouse:
        type: duckdb
        path: "{{ env_var('STEL_PROD_DB', '/data/prod/stel.duckdb') }}"
        schema: my_project_prod
      source_paths:
        filings: "{{ env_var('STEL_FILINGS_ROOT', '/data/prod/filings') }}"
      llm:
        model: claude-sonnet-4-6
        cache_path: /data/prod/llm_cache.duckdb
```

Lookup order: `--profiles-dir` flag → `$STEL_PROFILES_DIR` →
`<project>/profiles.yml` → `~/.stel/profiles.yml`.

Set `api_key_env` to the name of the credential variable itself, as above; do
not wrap it in `env_var()`. stel deliberately rejects secret-value
interpolation in this field so validation errors and resolved configuration
cannot contain the key.

### MotherDuck

MotherDuck is the managed deployment of DuckDB — the same `type: duckdb`
adapter and the same capability contract, reached over the network instead of a
local file. Point `path:` at a `md:` database and supply the service token:

```yaml
      warehouse:
        type: duckdb
        path: md:economic_data            # or "md:" (quoted) for the account default
        token: "{{ env_var('MOTHERDUCK_TOKEN') }}"
        schema: analytics
```

- `path` forms: `"md:"` (account-default database — quote it, or YAML reads the
  bare trailing colon as mapping syntax) or `md:<database>`. Credential-bearing
  query parameters (`?motherduck_token=…`) are rejected; the token belongs only
  in the protected `token:` field.
- `token` must be an exact `{{ env_var('NAME') }}` reference — literal tokens
  are rejected. It is never written to `manifest.json`, `run_results.json`,
  logs, or generated dbt sources, and is revealed only at connection. If you
  omit it, DuckDB reads its own `motherduck_token` environment variable.
- `token` is only valid on a `md:` path; a local DuckDB file needs none.

Because MotherDuck runs the DuckDB engine, the full capability set
(transactions, atomic replace, incremental merge, paged state, SQL models,
bounded snapshots) is advertised unchanged. Behavior against the live service is
exercised by a credential-gated integration test (`MOTHERDUCK_TOKEN`); the
default suite covers it with deterministic unit tests.

### Provider plugins and provider options

Separately packaged inference/embedding providers install as normal Python
distributions and are discovered through versioned entry-point groups
(`stel.inference_providers.v3` / `stel.embedding_providers.v3`) — no
wrapper import needed. Discovery is deterministic and fails closed before any
source or provider I/O: duplicate or built-in-shadowing names, broken plugins,
and name mismatches are configuration errors, and a plugin built against a
different provider contract version is reported as incompatible rather than
"not found". `stel providers list` shows every provider with its
distribution and implementation identity.

A provider may publish a strict options model; operators configure it under
`llm.provider_options:` or `embedding.provider_options:` in the profile
(opaque to core, validated by the selected provider, rejected in model YAML).
Every provider option field is
classified: `credential` fields are protected references that never enter
artifacts or fingerprints, `semantic` fields join the response-cache key and
model identity, `execution` fields never invalidate state, and
`artifact-safe` fields may appear in manifest descriptors. See
[docs/architecture/provider-abstraction.md](architecture/provider-abstraction.md).

### Vertex AI embeddings

Install the extra and authenticate with
[Application Default Credentials](https://cloud.google.com/docs/authentication/provide-credentials-adc).
User ADC and service-account ADC follow the same path; the provider deliberately
rejects `api_key_env` during profile resolution. stel splits runner batches at
Vertex model limits—one input for Gemini embedding models and five for other
text embedding models—while preserving input order and reporting the actual
API-call count.

```bash
pip install 'stel[vertex]'
gcloud auth application-default login
```

Bind the model's provider to operator-owned project and location settings in
the active target:

```yaml
my_project:
  target: prod
  outputs:
    prod:
      warehouse:
        type: bigquery
        project: my-gcp-project
        dataset: stel
      embedding:
        provider: vertex
        timeout_seconds: 60
        provider_options:
          project: my-gcp-project       # optional if ADC can infer it
          location: global             # use a model-supported Vertex location
          task_type: RETRIEVAL_DOCUMENT
          query_task_type: RETRIEVAL_QUERY
          auto_truncate: false
```

The model ID and output dimensionality remain reviewable model semantics:

```yaml
- name: document_embeddings
  depends_on: [ref('document_chunks')]
  embed:
    provider: vertex
    model: gemini-embedding-001         # or text-embedding-005 /
                                        # text-multilingual-embedding-002
    text_field: text
    id_field: chunk_id
    dimensions: 768
    batch_size: 128
    max_retries: 4
  materialization: incremental
```

stel passes `dimensions` as Vertex `output_dimensionality`, sends each runner
batch as one SDK request, and configures the SDK with the model's retry count
and the profile timeout. Document and query task types are separate so an
inherited search identity uses `RETRIEVAL_QUERY` at query time. Vertex model
availability, input limits, supported dimensions, and locations can change;
check the current
[Vertex text embeddings documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings)
before choosing production settings.

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
set. (dbt's `dataproc_*` fields don't apply: stel transforms run
in-process, not on Dataproc.)

```
pip install 'stel[bigquery]'
```

```yaml
my_project:
  target: prod
  outputs:
    prod:
      warehouse:
        type: bigquery
        project: my-gcp-project
        dataset: stel                  # `schema:` works too
        location: US                   # optional
        # Omit auth fields for ADC, or choose exactly one auth family:
        # keyfile: ./secrets/service-account.json
        # keyfile: "{{ env_var('STEL_BQ_KEYFILE') }}"
        # keyfile_json: "{{ env_var('STEL_BQ_SERVICE_ACCOUNT_JSON') }}"
        # token: "{{ env_var('STEL_BQ_ACCESS_TOKEN') }}"
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
live in the configured dataset — no DuckDB involved. `stel clean` does not
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

#### Incremental change detection (`update_when_changed`)

By default an incremental merge rewrites every column of a matched row,
including large payload columns, even when the row is byte-identical to
what is already stored. Declare a change-detection fingerprint to skip
those no-op rewrites (issue #281):

```yaml
  materialization: incremental
  update_when_changed: [content_hash, code_version]
```

A matched row is updated only when at least one listed column differs
(NULL-safe) between the batch and the target; unchanged rows are left in
place, so re-publishing them does not rewrite payload columns — on
BigQuery that is far fewer bytes billed for the `MERGE`. New rows still
insert and changed rows still update. The listed columns must exist in
both the batch and the target; `content_hash` and `code_version` are the
natural fingerprint for extraction models. Leaving it unset keeps the
always-overwrite behavior. It is a publication optimization, so it does
not change `code_version` — enabling it never reprocesses documents.

#### Reducing the incremental MERGE scan (clustering)

`update_when_changed` bounds how much a MERGE *writes*; clustering the target on
the incremental/merge key can reduce how much it *reads*. Each incremental
publication issues a `MERGE` joining the target to a staging table on the key;
against a large target that join can scan a lot of data. Clustering the target
on the merge key gives BigQuery's optimizer the option to prune that scan to the
blocks holding the batch's keys (issue #294). Declare it in `warehouse_options`,
listing the model's key first:

```yaml
  materialization: incremental          # keyed on document_id (chunk_id, …)
  warehouse_options:
    cluster_by: [document_id]           # merge key first
  update_when_changed: [content_hash, code_version]
```

Notes and boundaries:

- **Clustering helps the read; it is not a guaranteed bound.** stel emits a
  column-to-column join (`ON target.key = source.key`), not a static
  `WHERE key IN (…)` predicate, so pruning is an optimizer decision, not a
  guarantee — a small batch can still scan more than its keys. Treat clustering
  as a likely optimization and confirm the win with the bytes-processed
  telemetry below rather than assuming it. `update_when_changed` (the write
  side) composes with it.
- **A layout change needs a rebuild.** Like all `warehouse_options`, `cluster_by`
  applies when the table is created; an existing table keeps its physical layout
  until `--full-refresh` rebuilds it. Adding or changing `cluster_by` on a table
  that already exists is silently inert on the incremental path until you
  `--full-refresh` (which routes through the full-materialization path and
  recreates the table with the new clustering).
- **A layout change is not a format change.** Re-clustering never trips the #289
  storage-format fail-fast — that guard fires only on an Iceberg-vs-standard
  mismatch, not on a partition/cluster change.
- **`cluster_by` is a general layout knob.** stel does not force it to include
  the merge key; cluster for your query patterns as well. Pruning of the MERGE
  scan is only possible when the key is among the clustering columns (BigQuery
  prunes left-to-right, so list the key first).
- **Run count and overlap stay project responsibilities.** For **extraction**
  models, `flush_every` (default 5000) splits a run into flush-sized batches and
  `publish_every` (default 1, issue #293) coalesces that many flushes per `MERGE`,
  so together they govern how many MERGEs an extraction run issues; other
  incremental kinds (`embed`, `llm`, SQL transforms) publish their whole output in
  a single `MERGE` per run, and `chunk` models fully replace, so neither applies
  to them. Neither clustering nor
  `update_when_changed` coordinates overlapping orchestrator runs against the
  same target. The publication telemetry from issue #292 (`-v` /
  `STEL_VERBOSE`) surfaces each MERGE's job id and bytes processed so you can
  measure all of this against real cost.

**BigLake managed Apache Iceberg tables** (issue #163) — set
`table_format: iceberg` to store a model as Iceberg in Cloud Storage, queryable
through BigQuery and external Iceberg readers:

```yaml
  warehouse_options:
    table_format: iceberg
    connection: my-project.us.my-biglake-conn   # Cloud Resource connection, or DEFAULT
    storage_uri: gs://my-bucket/filings_chunks   # gs:// location for the table data
    partition_by: {field: filing_date}           # time partitioning only
    cluster_by: [cik]
```

Set a BigQuery storage policy once per profile target with
`warehouse_defaults` inside its `warehouse:` block (issue #284). Defaults can
carry any BigQuery warehouse option except `storage_uri`; model-level top-level
keys override them:

```yaml
# profiles.yml
economic_data:
  target: prod
  outputs:
    prod:
      warehouse:
        type: bigquery
        project: my-project
        dataset: economics_marts
        warehouse_defaults:
          table_format: iceberg
          connection: "{{ env_var('BQ_CONNECTION') }}"
          external_volume: "gs://{{ env_var('ICEBERG_BUCKET') }}/stel"
          labels: {managed_by: stel}
```

For Iceberg defaults, stel derives each model's location as
`{external_volume}/{target}/{dataset}/{model}`. That keeps dev/staging/prod and
every model on distinct prefixes without templating model YAML. A literal
`storage_uri` is rejected in `warehouse_defaults` because it would send every
model to one location; a model may still declare its own `storage_uri`, which
overrides the derived path. To opt one model out completely (for example, a
plain native scratch table), start its options from an empty policy:

```yaml
warehouse_options:
  inherit: false
```

An opted-out model can add its own options below `inherit`. Merging is shallow:
each top-level model option replaces the corresponding target default. Effective
options are validated before source discovery, credential resolution, or any
warehouse mutation. Targets without `warehouse_defaults` retain the existing
behavior.

Iceberg targets are created with explicit column DDL derived from the model's
output schema (`List` columns — including embedding vectors — become
`ARRAY<T>`); `connection` and `storage_uri` are required. Because BigQuery
Iceberg tables support neither `CREATE OR REPLACE` nor a truncating load, a
`full` model is replaced by drop → create → append and is therefore **not
atomic** (a failed run leaves the table empty and the next run repopulates it);
this path is gated by the adapter's `iceberg_table_format` capability rather than
`atomic_full_replace`. `incremental` models `MERGE`/`insert_overwrite` in place.
SQL (`transform.type: sql`) models materialize Iceberg too (issue #290): the
query is staged once, an explicit Iceberg `CREATE TABLE` is built from its
schema, and the rows are `INSERT…SELECT`ed across — the same non-atomic
drop → create → insert shape — so a project can adopt Iceberg as a uniform
storage policy without carving out its SQL models. Because an incremental merge
cannot change a table's storage format, declaring `table_format: iceberg`
against a target that already exists as a standard table (or the reverse) fails
fast rather than silently leaving the format unchanged; `--full-refresh`
rebuilds the table in the declared format (issue #289).
Current limits: time partitioning only (no `int64` range), no `kms_key_name`, and
BigQuery's unsupported Iceberg column types (`JSON`, `GEOGRAPHY`, `BIGNUMERIC`,
`INTERVAL`) are rejected before any warehouse call.

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
raw documents stay in the bucket, stel materializes into the warehouse:

```
pip install 'stel[gcs]'
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
    publish_every: 20    # coalesce 20 flushes into one MERGE (issue #293)
  materialization: incremental
```

Incremental writes are atomic per publication: DuckDB uses a transaction and
BigQuery loads a unique staging table then executes one `MERGE`. Missing,
NULL, or duplicate incremental keys are rejected before mutation. A killed
run keeps successful earlier publications and their state, and the re-run picks
up the remainder. With BigQuery `append_new_columns`, schema addition happens
before the `MERGE`; a failed merge preserves all rows but can leave the new,
nullable column in place.

By default (`publish_every: 1`) every flush publishes on its own, so on
BigQuery a run of many small flushes issues one billed `MERGE` per flush.
`publish_every` coalesces that many flushes into a single upsert: chunks
accumulate and one combined `MERGE` scans the target once instead of per
flush, cutting BigQuery bytes billed roughly in proportion. It is a distinct
lever from `flush_every` — `flush_every` bounds memory, `publish_every` bounds
publication cost — so keep flushes small and raise `publish_every` to trade
memory for fewer merges. Only flushes that share one schema coalesce: a
schema-on-read model whose columns drift mid-run publishes at the boundary, so
`on_schema_change` still applies exactly as it did per flush and a later flush's
new column is never dropped — coalescing changes cadence, never output. The
costs: peak resident rows grow to about `publish_every × flush_every`, and crash
recovery is coarser — a crash or budget exhaustion with a partial buffer discards
those unpublished flushes and re-extracts them next run (already-published
batches always survive, and state never advances before publication). Like
`flush_every`, changing `publish_every` never invalidates incremental state.
Coordinating overlapping orchestrator runs against one target remains a project
responsibility.

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
from stel.adapters import ReadPredicate, ReadPredicateOperator

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
stel execution path.

Incremental state is keyed by a stable record identity within a model, stage,
and target scope. Extraction and chunk generation use `document_id` because a
whole document is their retry unit; downstream publication can independently
track every `chunk_id`. Serving-target descriptors are stored only as a
canonical fingerprint, so changing non-secret semantic target configuration
forces publication without persisting the descriptor itself. Target rows and
their scoped state are deleted together, and new state is recorded only after
the corresponding materialization succeeds.

Search publication reconciles its state scope in bounded memory (issue #153),
complementing the bounded upstream reads above: these are two separate memory
ceilings. Each upstream batch is classified new/changed/unchanged through
bounded state key lookups, state advances per batch only behind exact durable
store receipts, and stale IDs stream back in strict record-key order through
snapshot-consistent state pages filtered to keys absent upstream — so delete
discovery is complete even for an empty upstream, and publication memory does
not grow with total state size. DuckDB pins pages to one MVCC read
transaction; BigQuery pins them to one `FOR SYSTEM_TIME AS OF` timestamp.
Adapters advertise `paged_state_reconciliation` for this contract and
`atomic_state_scope_replace` for fenced, atomic replacement of a scope's
complete state snapshot; the eager `fetch_state()` remains for
materialization-scale callers. Warehouses without these capabilities are
rejected at compile preflight for search resources.

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
tests, examples, and pipeline integration—not semantic similarity quality. The
`vertex` provider implements the same contract for
`gemini-embedding-001`, `text-embedding-005`, and
`text-multilingual-embedding-002`; its project, location, task types, timeout,
and ADC behavior live under profile `embedding:` configuration.

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
published. `stel.embedding.embed_query()` accepts the identity recorded in
the manifest so query-time vectors cannot silently use a different provider
implementation or configuration.

## LLM transformation models

An `llm:` model maps a prompt over one upstream warehouse relation, turning
unstructured text into typed, agent-ready rows — the first-class path for
transformations that need semantic interpretation, while SQL and Python stay
the deterministic surfaces. The model's `fields:` are the structured output
schema; the prompt is inline. It shares one execution core with `backend: llm`
extraction, so provider resolution, caching, retries, and usage accounting live
in one place. Credentials stay operator-owned in `profiles.yml`/environment —
the `llm:` block never carries an api key.

```yaml
- name: chunk_facts
  depends_on: [ref('document_chunks')]
  llm:
    mode: map
    input_field: text          # upstream column holding the content
    id_field: chunk_id         # stable upstream key, carried to the output
    output_cardinality: one    # one | many
    prompt: "Extract the key factual claim and its topic."
    provider: deterministic    # default -> the profile's LLM provider
    model: deterministic-v1
  fields:                      # the structured output schema (required)
    - {name: claim, type: string}
    - {name: topic, type: string}
  materialization: incremental
```

`output_cardinality: one` produces one row per input, keyed by `id_field`.
`output_cardinality: many` fans out a list of objects into one row each, keyed
by a deterministic `llm_row_id` (`f"{id}__{ordinal}"`) with the parent
`id_field` retained. Every output row gets provenance columns: `llm_provider`,
`llm_model`, `llm_provider_implementation`, `llm_input_hash`, `llm_config_hash`,
and `generated_at`.

Incremental runs skip inputs whose content and configuration are unchanged,
regenerate rows when the content or the prompt/schema/provider identity change,
and delete an input's rows when it is removed upstream (parent-scoped for
fan-out). The built-in `deterministic` provider runs offline for tests and
examples; production providers implement the same `InferenceProvider`
contract — `anthropic`, `vllm`, and `vertex` (Gemini models on Vertex AI via
`google-genai`, ADC-only, selecting the GCP project and location under profile
`llm:` configuration; install `stel[vertex]`). Gemini 2.5 models think by
default and bill reasoning tokens as output on every row, so `vertex` defaults
`thinking_budget` to `0` when the model declares `fields:` — a declared output
schema is extraction, not open-ended reasoning. That automatic default applies
only to models that accept a disabled budget (the Gemini 2.5 Flash family);
every other model keeps its own default and is sent no thinking configuration.
Set `provider_options: {thinking_budget: N}` to choose a budget explicitly,
which is always forwarded as configured. Reasoning tokens are reported as
`thinking_tokens` in run metrics when a provider bills them, and still count
toward `output_tokens` for budget enforcement.
Manifest and run-results
artifacts expose only the safe resolved
identity and aggregate usage — prompts, input text, and credentials are never
copied into artifacts. Native provider batch execution for `llm:` models is
deferred (issue #149 covers the batch machinery `backend: llm` already uses).

## Search indexes (local proof of concept)

A `search:` resource publishes exactly one upstream warehouse model to an
independently configured retrieval store. It is a leaf serving resource, not a
warehouse relation. Install `stel[lancedb]`, configure the operator-owned
store in `profiles.yml`, and explicitly opt in to public indexes:

```yaml
my_project:
  target: dev
  outputs:
    dev:
      warehouse:
        type: duckdb
        path: ./target/stel.duckdb
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
stel search --model chunk_search --query "latest inflation release" --mode hybrid
stel search --model chunk_search --query "inflation" \
  --filter source_uri eq reports/cpi.md --output json
stel search --model chunk_search --query "labor market" \
  --filter category in '["employment", "wages"]'
```

Filters are repeatable `FIELD OP VALUE` triples. Operators are `eq`, `ne`,
`lt`, `le`, `gt`, `ge`, and `in`; `in` takes a JSON array. Values are parsed
against the attribute's declared type, and only attributes with
`filter_role: user` can be supplied by a caller. Multiple filters are combined
with AND.

The same request is available as a provider-neutral Python API:

```python
from stel.search import SearchMode, SearchRequest, search

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
`embedding: inherit` points directly to a native `embed:` model, stel reuses
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
stel serving status chunk_search     # ledger status, fence, counts, leases
stel serving recover chunk_search --owner-terminated
```

Recovery advances the fencing token (so a surviving zombie fails its next
check), clears leases, and leaves the scope failed until the next `stel run`
republishes it. After upgrading to this contract, run `stel run` once per
search index to establish its ledger before querying.

Governed indexes (`access: governed`) are supported on stores that declare
strong read-after-write consistency and metadata filtering. Changed governed
records are deleted before their replacement is upserted, so a failed policy
revocation leaves the old row absent rather than queryable. Governed queries
fail closed unless the calling service supplies trusted `policy_filters=` that
constrain every policy-role attribute; they are composed with user filters as
mandatory in-store prefilters and are rejected on public indexes. The
`stel search` CLI serves public indexes only — an interactive flag is not a
trusted authorization context.

This slice still deliberately rejects search-resource tests, full refresh,
online/rebuild schema changes, arbitrary predicate strings, and
adapter-specific index options. Bounded state paging is implemented. Atomic
full replacement and distributed-store fencing remain declared-but-unclaimed
capabilities for any future store that can prove them. These are unsupported
guarantees, not silent best-effort behavior.

## Built-in text preprocessing

Install optional NLP support and a spaCy language model before running the NLP
child-table transforms:

```bash
pip install 'stel[nlp]'
python -m spacy download en_core_web_sm
```

Reference any of these as a Python transform module — no project-local code
needed. Users can override by writing their own `transforms/<name>.py`
(project-local files win over installed packages).

```yaml
- name: post_text_stats
  depends_on: [ref('raw_posts')]
  transform:
    type: python
    module: stel.text.transforms.text_stats   # built-in, ships with stel
    options:
      text_field: body
      emit: [word_count, sentence_count]
```

| Module                                    | What it does                                                                   |
|-------------------------------------------|--------------------------------------------------------------------------------|
| `stel.text.transforms.text_stats`        | Adds `word_count` / `char_count` / `sentence_count` / `paragraph_count`         |
| `stel.text.transforms.clean_encoding`    | Fixes mojibake (UTF-8-as-Latin-1 confusion) via ftfy                            |
| `stel.text.transforms.detect_language`   | Adds a 2-letter ISO language code per row via langdetect                        |
| `stel.text.transforms.count_tokens`      | Adds `token_count` for an OpenAI / Claude-style tokenizer (tiktoken)            |
| `stel.text.transforms.find_duplicates`   | Flags near-duplicate rows via MinHash + LSH (Jaccard threshold configurable)    |
| `stel.text.transforms.redact_pii`        | Detects + redacts PII via Microsoft Presidio (requires `en_core_web_sm` spaCy model) |
| `stel.text.transforms.nlp_tokens`         | Emits one normalized child row per spaCy token                                 |
| `stel.text.transforms.nlp_entities`       | Emits one normalized child row per spaCy named entity                          |
| `stel.text.transforms.link_entities`      | Links entity mentions to canonical IDs by alias table, vector, or fuzzy match  |
| `stel.text.transforms.extract_relations`  | Emits typed relations between entity mentions (deterministic co-occurrence)     |
| `stel.text.transforms.nlp_document_features` | Rolls the NLP child tables up to one aggregate feature row per document     |
| `stel.text.transforms.document_tone`      | Scores per-document tone from the token table + an operator-owned lexicon      |
| `stel.text.transforms.extract_keyphrases` | Ranks keyphrases per document by n-gram frequency; child table with stable IDs |

All are pure functions importable via `from stel.text import …` if you'd
rather wire them into your own transforms.

The NLP transforms require one upstream table with unique, nonempty document
IDs and a string text column. They use spaCy's batched `nlp.pipe` API and
record provider, package, model, model-version, and language identity on every
output row. Configuration is validated during `stel compile`, before spaCy
or the configured model is loaded.

```yaml
- name: document_entities
  depends_on: [ref('raw_documents')]
  transform:
    type: python
    module: stel.text.transforms.nlp_entities
    options:
      document_id_field: document_id
      text_field: text
      model: en_core_web_sm
      language: en
      batch_size: 32
      include_fields: [publisher, published_at]
      include_text: false
```

`nlp_tokens` emits stable `token_id`, document/token/sentence indexes, character
offsets, token text, lemma, POS/tag values, and stop/alpha flags.
`nlp_entities` emits stable `entity_id`, document/entity/sentence indexes,
character offsets, label, and nullable confidence. Matched `entity_text` is
excluded unless `include_text: true` is explicit. Source columns are also
excluded unless named in `include_fields`, and the raw text and document-ID
source fields cannot be repeated through that option. See
`examples/economic_nlp/` for a complete economic-document pipeline.

### Entity linking to canonical identifiers

`link_entities` resolves entity mentions (for example `nlp_entities` output
with `include_text: true`) to canonical identifiers — CIK numbers, tickers,
agency IDs, country codes, or project-defined keys — through an operator-owned
alias table. It needs no optional extra, no network access, and no credentials.

```yaml
- name: entity_links
  depends_on: [ref('document_entities'), ref('entity_aliases')]
  transform:
    type: python
    module: stel.text.transforms.link_entities
    options:
      mentions: document_entities
      aliases: entity_aliases
      match_methods: [exact, normalized]
      on_ambiguity: keep
```

The alias model supplies `alias`, `entity_namespace`, and `canonical_id`
columns (names configurable). Matching is deterministic: `exact` compares the
mention text as-is; `normalized` applies NFKC + casefold + whitespace collapse.
Methods run in configured order and the first method that produces candidates
for a namespace wins that namespace. Every mention yields explicit rows —
`matched` (one canonical ID in a namespace), `ambiguous` (one row per
candidate, never a silent guess), or `unmatched` — and `on_ambiguity: error`
fails the run instead. Each row records a stable `entity_link_id`, the
resolver identity and version, a `match_score` reserved for future
score-producing resolvers, and an `alias_set_version` fingerprint of the whole
alias table so alias edits are visible downstream. Mention text is not
retained unless `include_mention_text: true` is explicit, and `include_fields`
follows the same allow-list rules as the NLP transforms. The `mentions:` and
`aliases:` values must name exactly the models in `depends_on`; a misspelled or
stale reference is rejected during `stel compile`, before any model is
materialized. See `examples/economic_entity_links/` for a runnable pipeline.

The resolver is selected by the `resolver:` option, defaulting to `alias_table`
(above). Set `resolver: vector_similarity` to link by embedding similarity
instead: `mention_vector_field` and `alias_vector_field` name precomputed vector
columns — produced upstream by the ordinary `embed` model kind — and each
mention resolves to alias candidates whose `metric` similarity (`cosine`,
`dot`, or `euclidean`) is at or above `threshold`, with the score written to
`match_score`. Candidates within `ambiguity_margin` of a namespace's top score
are `ambiguous` rather than silently arg-maxed. Because the vectors are computed
by the `embed` kind, credentials and provider batching stay in that executor and
this resolver remains an offline transform; the mention/status/privacy contract
and output schema are identical to the alias-table resolver. Vectors from
different embedding models occupy unrelated spaces, so when both sides carry the
`embed` kind's `embedding_config_hash` a mismatch fails the run rather than
emitting meaningless links (`embedding_config_hash_field: null` bypasses the
check). See `examples/economic_entity_links_embeddings/` for a runnable,
credential-free pipeline using the built-in `deterministic` embedding provider.

Set `resolver: fuzzy` to match mention text against alias text by deterministic
string similarity — the option to reach for when surface forms vary (spelling
variants, legal suffixes, reordered words) but you have no embeddings. `metric`
selects `trigram_dice` (character-trigram Dice, default; robust to typos and
suffixes) or `jaccard_token` (whitespace-token Jaccard; suits reordered
multi-word names); both are in `[0, 1]`. A mention resolves to alias candidates
whose similarity is at or above the required `threshold`, with the score written
to `match_score`; `ambiguity_margin` and the `matched`/`ambiguous`/`unmatched`
statuses behave exactly as for `vector_similarity`. Matching is case- and
width-insensitive by default (`normalize: false` scores the raw surface forms).
Like `alias_table` it needs no optional extra, no network access, and no
credentials — the similarity math is pure and deterministic, so identical
inputs always produce identical links.

All three resolvers support `materialization: incremental`: parents are the
documents in the `mentions` model and the `aliases` model is a whole-table
reference input, so an unchanged corpus re-links nothing while any alias-table
edit re-links every document (child rows are keyed by `entity_link_id`). The
example projects materialize incrementally.

To join documentary evidence to governed structured metrics, project matched
links into the agent-context `context_entity_links` grain (see
[agent-context](architecture/agent-context-v1.md)) with
`stel.agent_context.project_entity_link`: the `canonical_id` becomes the row's
`entity_key`, so a governed metric keyed on the same namespace/name/canonical id
resolves to the identical `entity_id` — the cross-plane join key. Record the
resolver identity with `entity_link_method(resolver, resolver_version)`.

### Relation extraction

`extract_relations` emits a child table of relations between the entity mentions
in a document (for example `nlp_entities` output), one row per related pair. It
keeps three kinds of relationship strictly distinguishable via the `method`
column so a consumer never mistakes proximity for a semantic assertion:
`co_occurrence` (proximity), `rule` (deterministic typed rules), and
`model_assertion` (a learned/LLM extractor). All three ship: `co_occurrence` and
`rule` are deterministic and offline, and `model_assertion` calls a governed
inference provider through the same registry seam.

```yaml
- name: document_relations
  depends_on: [ref('document_entities')]
  transform:
    type: python
    module: stel.text.transforms.extract_relations
    options:
      mentions: document_entities
      scope: sentence          # or `window` with `max_char_gap`
      relation_type: co_occurs_with
      labels: [ORG, GPE]       # optional: only these labels participate
  materialization: incremental
```

Two mentions co-occur when they share a sentence (`scope: sentence`, requires a
non-null `sentence_index`) or fall within `max_char_gap` characters
(`scope: window`). Co-occurrence is symmetric, so each unordered pair yields one
row with `directed: false`; the subject is the earlier-positioned mention and
the object the later one, giving every pair a stable orientation and
`relation_id`. Every row records the `relation_type`, the `method`, a `status`
(`asserted` for the deterministic extractors; `ambiguous`/`no_relation` are
reserved for learned extractors), a `confidence` (null for the deterministic
extractors), the subject/object mention IDs and offsets, the participating
labels, and the extractor identity and version. Evidence text is withheld unless
`include_mention_text: true` (and then the mentions model must carry it via the
NLP transform's `include_text: true`).

Set `extractor: rule` for **directed, typed** relations instead of symmetric
proximity. Each rule asserts a `relation_type` from a subject mention of one
label to an object mention of another when the two co-occur in scope; the
distinct `relation_type` values across the rules are exactly the relations the
model can emit (schema-controlled), and the subject/object orientation follows
the rule rather than text position. The rules are deterministic and offline —
no provider, no network.

```yaml
- name: document_typed_relations
  depends_on: [ref('document_entities')]
  transform:
    type: python
    module: stel.text.transforms.extract_relations
    options:
      mentions: document_entities
      extractor: rule
      scope: sentence
      rules:
        - {subject_label: ORG, object_label: GPE, relation_type: references_geography}
        - {subject_label: ORG, object_label: MONEY, relation_type: references_amount}
  materialization: incremental
```

Set `extractor: model_assertion` for a **learned/LLM** extractor. Per document it
asks a governed inference provider whether one of the schema-controlled
`relation_types` holds between each in-scope candidate pair, with a `confidence`;
assertions at/above `threshold` are `asserted`, below are `no_relation`, and a
pair the model maps to conflicting types is `ambiguous`. The `relation_types`
list is the allow-list — the model may only assert those, enforced in the prompt
and re-validated (out-of-list or hallucinated mention pairs are dropped). It runs
through the same shared inference core as the `llm:` kind (caching, retries), so
the model requires `transform.uses_llm: true` and resolves provider, model, and
credentials from the profile's `llm:` block only — never from project YAML.
Switching the provider or model reprocesses the table (the model identity folds
into the code version). Evidence text and provider responses never enter
artifacts.

```yaml
- name: document_relations_llm
  depends_on: [ref('document_entities')]   # entities must carry text (include_text: true)
  transform:
    type: python
    module: stel.text.transforms.extract_relations
    uses_llm: true
    options:
      mentions: document_entities
      extractor: model_assertion
      relation_types: [acquired, subsidiary_of, references_geography]
      threshold: 0.6
  materialization: incremental
```

Relations materialize incrementally on the same one-to-many path as the other
child tables: a changed document re-derives exactly its relation rows. See
`examples/economic_nlp/` for runnable co-occurrence and rule pipelines.

### Document-level aggregate features

`nlp_document_features` rolls the token, entity, and entity-link child tables
back up to one row per document, so downstream dbt models and classic ML do not
each reimplement the same aggregation. It needs no optional extra — it reads
tables, not text.

```yaml
- name: document_features
  depends_on:
    [ref('document_tokens'), ref('document_entities'), ref('entity_links'),
     ref('raw_documents')]
  transform:
    type: python
    module: stel.text.transforms.nlp_document_features
    options:
      tokens: document_tokens        # required — the aggregation spine
      entities: document_entities    # optional
      links: entity_links            # optional
      documents: raw_documents       # optional — row universe + metadata
      documents_id_field: economic_id
      pos_counts: [NOUN, PROPN]
      entity_label_counts: [ORG, GPE]
      link_namespace_counts: [agency, iso3166]
      include_fields: [publisher, published_at]
```

Base features come from `emit:`, which defaults to every feature the configured
dependencies support: `token_count`, `sentence_count`, `entity_count`,
`unique_lemma_count`, `lexical_diversity`, `stop_ratio`, and `alpha_ratio`.
Naming a feature that its dependency is missing — `entity_count` with no
`entities:` — is a compile-time error rather than a silent null.

Every other rollup is an explicit list, so the output schema is fixed at compile
time and never depends on what happens to be in the warehouse:

| Option | Column pattern | Meaning |
|---|---|---|
| `pos_counts` | `pos_noun_count` | tokens with that POS |
| `pos_ratios` | `pos_noun_ratio` | that count over `token_count` |
| `entity_label_counts` | `entity_org_count` | entities with that label |
| `link_namespace_counts` | `linked_agency_count` | distinct canonical IDs in that namespace |
| `link_status_counts` | `link_ambiguous_count` | mentions with that link status |

Conventions worth knowing:

- Ratios divide by `token_count`, which counts the rows in the token table —
  space tokens are excluded unless that table was built with `include_space`.
- A document with no tokens has counts of `0` and ratios of `null`; ratios are
  undefined at a zero denominator, never `0` or `NaN`.
- `sentence_count` is `null`, not `0`, when the pipeline had no parser and every
  `sentence_index` is null. A configured POS or label a document never uses is
  `0`, not null.
- With a `documents:` dependency the parent table defines which documents get a
  row, so empty documents still appear and a stale child row for a document that
  no longer exists is excluded rather than resurrecting it. Without that
  dependency, only documents present in the token table get a row.
- Identity columns pass through per document. If one document's child rows
  disagree on `nlp_model`/`nlp_model_version` — or tokens and entities disagree
  with each other — the run fails rather than claim a single reproducible
  identity.

No document, token, or entity text reaches the output. Counting distinct lemmas
is not retaining them, and `include_fields` allow-lists parent metadata only.

### Document tone / sentiment

`document_tone` scores per-document tone by matching the token table against an
operator-owned tone lexicon. It is deterministic and reads tables, not text, so
it needs no optional extra and no LLM — a general sentiment score is never
presented as an economic fact. The lexicon is a normal upstream model (rows of
`term`, `category`, optional `weight`), exactly like the entity-linking alias
table.

```yaml
- name: document_tone
  depends_on: [ref('document_tokens'), ref('tone_lexicon')]
  transform:
    type: python
    module: stel.text.transforms.document_tone
    options:
      tokens: document_tokens      # the token child table
      lexicon: tone_lexicon        # operator-owned term/category/weight table
      match_field: lemma           # token column matched (case-insensitively)
      language: en
      emit: [positive, negative, uncertainty, hawkish, dovish]
      include_fields: [publisher, published_at]
```

`emit` is an explicit list of lexicon categories, so the output schema is fixed
at compile time regardless of the lexicon rows in the warehouse. Each emitted
category `c` produces `c_score` and `c_hits`; general polarity
(`positive`/`negative`) and domain signals (`uncertainty`, `hawkish`/`dovish`, …)
are just different categories in the lexicon, so they stay separate by
construction.

Conventions worth knowing:

- A category score is the sum of matched term weights normalized by
  `token_count`; `coverage` is `matched_token_count / token_count`. Both are
  `null` when there is too little text (below `min_tokens`, `status` is
  `insufficient_text`), never a misleading `0`.
- With `negation` on (the default), a matched term preceded by a negator within
  a bounded same-sentence window flips its contribution; `*_hits` still counts
  the raw match. Negators are configurable for non-English lexicons.
- The lexicon's content is fingerprinted as `lexicon_version`, so an edit is
  visible to downstream invalidation without retaining the lexicon. `scorer` and
  `scorer_version` identify the deterministic path so a future learned scorer can
  be added without a schema change.
- Tokens whose `nlp_language` disagrees with the configured `language` fail the
  run rather than being scored against the wrong lexicon.
- No document text or matched phrases reach the output; `include_fields`
  allow-lists parent metadata (publisher, release date) so tone joins to them on
  the same row.

### Keyphrase extraction

`extract_keyphrases` ranks per-document keyphrases by normalized n-gram
frequency from the NLP token child table. No IDF, no learned model, no optional
extra — the same token table and the same options always produce the same ranked
list.

```yaml
- name: document_keyphrases
  depends_on: [ref('document_tokens')]
  transform:
    type: python
    module: stel.text.transforms.extract_keyphrases
    options:
      tokens: document_tokens     # the token child table
      language: en
      min_phrase_length: 1        # minimum tokens per candidate phrase
      max_phrase_length: 3        # maximum tokens per candidate phrase
      top_k: 15                   # phrases to keep per document
      # include_phrase_text: true # opt-in: phrase text is a verbatim excerpt
```

The output is a child table with one row per `(document_id, phrase_lemma)`:

| column | notes |
|--------|-------|
| `phrase_id` | stable hash of `(document_id, phrase_lemma)` |
| `rank` | 1-indexed position within the document |
| `score` | occurrence count / total candidate n-grams in the document |
| `phrase_lemma` | space-joined lemmas |
| `phrase_length` | token count |
| `token_start` / `token_end` | first occurrence offsets (token indexes) |
| `sentence_index` | sentence of first occurrence |
| `phrase_text` | surface form — present only when `include_phrase_text: true` |
| `extractor` / `extractor_version` | `ngram_freq` / `1` |
| `nlp_provider` … `nlp_language` | 5 NLP identity columns from the token table |

Conventions worth knowing:

- Candidates are contiguous lemma n-grams within sentence boundaries. Boundary
  tokens (first and last) must not be stop words and must not carry a POS tag in
  the configurable `stop_pos` set (default: `PUNCT`, `SPACE`, `NUM`, `SYM`, `X`);
  interior tokens are unrestricted, so "rate of return" is a valid 3-gram.
- Score is normalized term frequency: occurrence count / total candidate n-gram
  count in the document. Rank tie-breaking is alphabetic on `phrase_lemma` for
  deterministic output regardless of corpus order.
- Multi-token extraction (`max_phrase_length > 1`) requires sentence boundaries
  (`sentence_index` non-null). Rebuild the token table with a spaCy pipeline that
  includes the sentencizer or dependency parser, or set `max_phrase_length: 1` to
  restrict to unigrams.
- **Phrase text is opt-in.** `include_phrase_text: true` emits the `phrase_text`
  column using the `token_text` values already in the token table. Phrase text is
  a verbatim excerpt of the source document and may contain sensitive content —
  the default keeps it out of the output.
- `extract_keyphrases` supports `declared_incremental_contract` with
  `parent_key="document_id"` and `child_key="phrase_id"`, consistent with
  `nlp_tokens` and `nlp_entities`.

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
    module: stel.text.transforms.redact_pii
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

Classic ML is a first-class stel lane alongside LLM/RAG work. The `ml:`
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

**Distribution checks** (deterministic statistics over a single column):

```yaml
tests:
  # a summary statistic within bounds (stat: mean|min|max|sum|stddev|median|quantile)
  - column_stat: { column: n_authors, stat: mean, min: 1, max: 10 }
  - column_stat: { column: score, stat: quantile, quantile: 0.95, max: 1.0 }
  # distinct-value count and/or distinct ratio (distinct / total rows)
  - cardinality: { column: primary_category, min: 2 }
  - cardinality: { column: id, min_ratio: 1.0 }        # every row distinct
  # fraction of numeric outliers, by IQR (default, k·IQR) or z-score
  - outlier_rate: { column: n_authors, method: iqr, max_rate: 0.02 }
```

`column_stat` and `outlier_rate` operate on a numeric column (nulls and
non-finite values are skipped; a non-numeric column fails with an actionable
message). `outlier_rate` reads only the target column and supports
`--store-failures`.

**Drift checks** (run-over-run distribution change against a baseline model):

```yaml
tests:
  # distribution of `n_authors` vs the same field in a snapshot you maintain
  - drift: { column: n_authors, to: ref('papers_baseline'), metric: psi, max: 0.2 }
  - drift: { column: score, to: ref('scores_baseline'), metric: ks, max: 0.1 }
  # categorical proportion drift; `field` maps to a differently-named baseline col
  - drift:
      column: primary_category
      to: ref('papers_baseline')
      field: category
      metric: jensen_shannon
      max: 0.05
```

The baseline is an **ordinary model you snapshot and `ref()`** — an explicit,
git-reviewable run-over-run comparison, not an implicit last-run store — and
stel builds it before the check (same dependency path as `relationships`).
`metric` is `psi` (default), `ks` (numeric only), `jensen_shannon`, or
`chi_squared`; numeric columns are compared over baseline-quantile bins
(`bins`, default 10) and categoricals over their value proportions. The check
fails when the divergence exceeds `max`. (Note `chi_squared` is a raw statistic
that scales with sample size, so calibrate its `max` per corpus, unlike the
bounded PSI/KS/JS.)

**Golden-set checks** (compare a model to checked-in expected rows):

```yaml
tests:
  - golden:
      to: ref('extractions_golden')   # a model holding the expected output rows
      key: invoice_id                 # join key present in both
      columns: [vendor, total]        # default: all shared non-key columns
      tolerance: { total: 0.01 }      # per-column absolute numeric tolerance
      exhaustive: false               # true also fails on unexpected extra rows
```

The golden model is an ordinary model you `ref()` (a seed or a snapshot),
reviewable in git and built first as a dependency. Every golden key must appear
in the model and match each compared column exactly (or within `tolerance`);
`--store-failures` persists the offending keys and which columns diverged.

**LLM-judge check** (optional, sampled — the subjective escape hatch):

```yaml
tests:
  - llm_judge:
      column: summary
      criterion: "is a faithful, single-sentence summary of the source"
      sample_size: 20            # rows sampled per run (deterministic by `seed`)
      seed: 0
      min_pass_rate: 0.95        # fail if fewer than 95% of sampled rows pass
```

`llm_judge` samples rows deterministically (a stable sort before seeded
sampling, so the same `seed` selects the same rows regardless of warehouse row
order), asks the profile's `llm:` provider whether each `column` value meets
`criterion` (structured boolean verdict via the shared #144 inference path), and
fails when the pass rate drops below `min_pass_rate`. It honors the same
`llm.provider_options` and `llm.budget` caps as `llm:` models — each judge call
is charged to the run budget and stops at the run-wide `max_api_calls` /
`max_cost_usd`, and a project that declares `llm_judge` without an `llm:` profile
fails preflight before any model is built. It is a sampled, cost-bounded escape
hatch for subjective qualities — not a deterministic CI gate — so keep it off the
critical path and prefer the deterministic checks above. Tests run against the
offline `deterministic` provider.

**Embedding-quality checks** (deterministic, over the vector column of an
`embed` model — no provider call):

```yaml
tests:
  # dimensionality + finiteness + L2-norm bounds + zero-vector rate
  - embedding_valid: { column: embedding, dimensions: 1536, max_zero_rate: 0.0 }
  # collapse guard: mean per-dimension variance must stay above a floor
  - embedding_variance: { column: embedding, min_variance: 0.0001 }
  # exact-duplicate-vector rate (redundant copies / total) — usually a cache/join bug
  - embedding_duplicates: { column: embedding, max_rate: 0.0 }
  # fraction of vectors beyond `z` std-devs of the centroid distance
  - embedding_outliers: { column: embedding, z: 3.0, max_rate: 0.01 }
```

These read only the vector column (memory proportional to the embeddings, not
the whole relation) and compute norms, per-dimension variance, exact-duplicate
rates, and centroid-distance outliers in process. A zero or NaN embedding is a
common silent provider failure, and near-zero variance catches representation
collapse — both invisible to `not_null`.

**Inspecting failures.** Pass `--store-failures` to `stel test` or `stel
build` to persist the offending rows of each failing test to a
`stel_test_failures__<model>__<test>[__<column>]` table (replaced each run).
The test output reports the table name and row count. These tables are
inspection artifacts and are kept out of the model namespace (they don't show up
in `stel ls` or `emit-dbt-sources`).

**`stel build`** runs and tests each model in dependency order, skipping a
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
| `examples/dbt_consumer/`            | dbt-duckdb project consuming stel-materialized tables                 |
| `examples/dbt_embed_duckdb/`        | stel embedded in one `dbt build` via generated Python models (#177)   |
| `examples/classic_text_ml/`         | deterministic sparse text features + Naive Bayes classification        |
| `examples/document_clustering/`     | deterministic TF-IDF, K-means clustering, and NMF topics                |
| `examples/economic_nlp/`            | economic documents → normalized spaCy token and entity child tables    |
| `examples/economic_entity_links/`   | entity mentions → canonical CIK/ticker/agency IDs via an alias table   |
| `examples/rag_chunks_pipeline/`     | document registry → deterministic RAG chunks                           |
| `examples/sql_governed_chunks/`     | warehouse-native SQL model applying document permissions               |
| [`examples/metric_evidence_agent/`](../examples/metric_evidence_agent/) | dbt metric + governed, cited stel evidence over two MCP servers |

The stel-native examples run with
`uv run stel --project-dir examples/<name> ...`. The two dbt composition
examples and the agent example include their own commands in local READMEs.

## Composing with dbt

stel does the unstructured→structured "E" and dbt does the SQL "T". There are
two ways to compose them, depending on whether you want a staged handoff or one
`dbt build`.

### Staged handoff — `emit-dbt-sources` (any adapter)

`emit-dbt-sources` targets the matching adapter: dbt-duckdb can share the
DuckDB file, and dbt-bigquery can read the configured BigQuery dataset. The
DuckDB bridge:

```bash
uv run stel --project-dir examples/invoice_pipeline run
uv run stel --project-dir examples/invoice_pipeline emit-dbt-sources \
  --output examples/dbt_consumer/models/sources/_stel_sources.yml

cd examples/dbt_consumer && uv sync && uv run dbt build --profiles-dir .
```

`emit-dbt-sources` translates stel tables into a dbt-compatible `sources.yml`.
Column tests carry over (`not_null`, single-column `unique`); composite unique
becomes a `dbt_utils.unique_combination_of_columns` macro test.

### Embedded in one `dbt build` — `codegen` (dbt-duckdb)

> Status: preview (issue #177). dbt-duckdb only; extraction and transform models.

`stel codegen` turns each stel model into a native dbt **Python model** so a
single `dbt build` runs stel and dbt in one DAG — dbt and stel models
`ref()` each other and share one `dbt docs` lineage graph, no orchestrator.

stel is **not** a dbt package (`packages.yml` / `dbt deps` can't install a
Python dependency). Instead it works through two surfaces:

1. **Python package** — `pip install stel` into the same environment as your
   dbt-duckdb project. This provides the `stel` CLI and the
   `stel.dbt_embed.materialize` engine the generated models call in-process.
   (`materialize` has no dbt dependency and needs no extra — it ships in the
   core install.)
2. **Generated dbt resources** — `stel codegen` writes one Python-model shim
   per model plus a `schema.yml` (fields + tests) **into** your dbt project's
   `models/` tree. You commit these like any other model; the stel YAML stays
   the source of truth and you regenerate when it changes.

```bash
# In your dbt project's environment (dbt-duckdb + stel installed):
stel --project-dir path/to/stel_project codegen --output models/stel
STEL_PROJECT_DIR=path/to/stel_project dbt build
```

Each generated Python model imports `stel.dbt_embed.materialize` lazily at run
time; dbt-duckdb executes it in-process, so extraction, LLM calls, and the LLM
response cache all run locally. See
[`examples/dbt_embed_duckdb`](../examples/dbt_embed_duckdb) for a runnable
three-level DAG (extraction → transforms → SQL mart).

The direction can also reverse: a stel transform can declare
`source: dbt_ref('<dbt_model>')` to read a **dbt-built** table back into stel
— optionally alongside ordinary `depends_on:` stel models in the same
transform — closing the loop in one `dbt build`. See
[`examples/dbt_ref_roundtrip_dbt`](../examples/dbt_ref_roundtrip_dbt) for a
runnable stel → dbt → stel → dbt round trip.

## Concept cloud (visualization, proof of concept)

`stel concept-cloud` renders extracted entities as an explorable 3D **concept
cloud** floating over a 2D plane of the dbt DAG, with lines tying each concept
down to the model that produced it — a way to *see* how the fuzzy unstructured
layer maps onto the deterministic structured one (issue #255).

The command writes a single **self-contained HTML file** (the rendering library
is inlined, so it opens offline in any browser — the inline preview panels in
some tools do not run WebGL, so open the file directly):

```bash
# Try it with built-in bundles — no project needed:
stel concept-cloud --demo -o cloud.html         # ~45-entity economic-data sample
stel concept-cloud --placeholder -o cloud.html  # minimal example

# Export a real project's concepts:
stel --project-dir path/to/stel_project concept-cloud \
  --linking-model link_entities \
  --relation-model extract_relations \
  --dbt-manifest path/to/dbt/target/manifest.json \
  -o cloud.html
```

The export job is a three-way join over artifacts stel already produces: the
entity-linking output supplies canonical concepts (sized by mention frequency,
colored by entity type) and the mention→canonical map; the relation grain
supplies typed concept-to-concept edges; and the DAG plane comes from the
downstream dbt `manifest.json` (or stel's own if `--dbt-manifest` is omitted).
The viewer has orbit controls, click-to-trace cross-layer beams, a text search,
entity-type toggles, an orphan highlight (nodes with no edges), and a
min-frequency filter.

**Prerequisites.** The cloud is keyed on `canonical_id`, so an entity-linking
(`link_entities`) model must have run; and human-readable node labels require the
NLP/linking models to set `include_text: true` (otherwise nodes show ids).

**Boundaries.** The artifact reads only exported/queried output tables — no
warehouse credentials ever enter it, and raw document text appears only when the
operator opted into it upstream. The bundle is a versioned contract
(`ConceptCloudExport`, `schema_version`), so the static viewer and the export job
evolve independently.

## Artifacts

`stel compile` writes the manifest; `run` and `build` write the manifest and
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
- **`docs/`** — static HTML site (`stel docs generate`) with project overview,
  Mermaid DAG, per-model pages. Serve locally with `stel docs serve`.

External tools (lineage viewers, CI dashboards, the dbt-consumer above)
consume these. `run`/`build` exit `0` on success, `1` on run failure, and `2` on
a configuration error, so an orchestrator can branch on the cause. Because
stel tables are dbt sources, they wire natively into the `dagster-dbt`
integration — see
[`docs/orchestration-dagster.md`](orchestration-dagster.md) (use
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
src/stel/
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

Active platform work is limited to DuckDB/MotherDuck, BigQuery/GCP, and
Snowflake. LanceDB remains the reference retrieval store; additional hosted
retrieval adapters and unrelated warehouse/provider integrations are not on
the current roadmap. Retrieval evaluation remains active work. Incremental
state stays adapter-owned. Rust, PyO3, and Metaxy remain explicitly deferred.

The accepted [semantic retrieval architecture](architecture/semantic-retrieval.md)
defines the `search:` DAG resource, `RetrievalStore` boundary, typed filters,
incremental publication state, and serving-resource artifacts. The local
LanceDB publication and portable Python/`stel search` query surfaces ship
with generation-fenced readiness, publish/query leases, explicit recovery,
governed policy-prefilter queries (issue #152), and bounded paged
publication-state reconciliation (issue #153) inside their documented
single-host boundary. Atomic full replacement and distributed-store fencing
remain unsupported and fail closed; no hosted retrieval adapter is currently
planned.

The versioned [agent context contract](architecture/agent-context-v1.md)
defines the document registry, chunk, and dbt-entity link grains used to carry
bitemporal validity, policy, freshness, provenance, and exact citations from
warehouse models into governed retrieval projections.
