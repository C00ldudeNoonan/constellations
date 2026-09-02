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
| `lancedb` | Local LanceDB search-index publication and queries (the `duckdb` search store needs no extra) |
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
(or `validate_options(options, project_dir)`, when the options name a file)
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
Native `llm:` workers reserve uncached calls before submitting them, so
`max_concurrent` cannot multiply an API-call cap. When an input-token,
output-token, or spend cap is active, provider admission on that ledger is
serialized until the preceding response is charged; ledgers without those caps
keep their configured concurrency. Admission belongs to the ledger, so the
run-scope cap coordinates every model and provider stage sharing it, including
LLM extraction, native batches, embeddings, LLM checks, and model-assertion
relations. A run-wide API cap therefore cannot admit one extra call per model,
and a response-measured cap overshoots by at most one response across the whole
run. Cache hits consume no call budget.

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
stel serving migrate-scope <search-index>                # one-time move onto the logical-collection serving key
stel suggest dbt --from RELATION --dbt-project DIR       # propose `description:` for under-documented dbt models
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
  On object-store sources each glob's leading static segment narrows the
  listing itself, and **several filters list one narrowed prefix each rather
  than falling back to the whole bucket prefix** — so K filters cost K cheap
  listings and batch size stays an orchestration knob. A glob with no static
  segment (`'*/2024*'`) has nothing to narrow on and lists everything.
- `--read-filter FIELD OP VALUE` (repeatable) extends the same partitioned-run
  seam past extraction to `ref()`-based models (issue #417). It builds a typed
  predicate pushed down to the warehouse and narrows what **python transform
  parent reads** and **embed source reads** see — ops are `eq`, `ne`, `lt`,
  `le`, `gt`, `ge`, and `in` (a JSON array of strings):

  ```bash
  stel build --select document_embeddings --read-filter cik eq 0000320193
  stel build --select sec_document_text --read-filter filing_date ge 2024-01-01
  ```

  Either filter flag makes the **whole invocation** a subset run: chunk,
  transform, embed, and search-index models all skip their removed-row
  reconciliation, because absence from a deliberately narrowed run is not
  removal — a partitioned invocation must never delete every other
  partition's rows. Reconciliation is deferred, not lost: the next unfiltered
  run performs it. Like `--source-filter`, a read filter is rejected with
  `--full-refresh`, and every selected transform/embed model must be
  incremental — a full materialization would replace its whole table with
  the slice. Values are passed as strings; a filter on a typed column relies
  on the warehouse's comparison semantics for string literals against that
  type, so prefer string partition keys.

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

`stel run` and `stel build` narrate as they go. By default they stream a
**ledger** to stderr — a header, one line per model as it completes, and a
footer — before the summary table lands on stdout:

```
Running 3 models (target: dev, duckdb)
[1/3] raw_invoices            extraction         40 rows     1.0s  OK
[2/3] invoice_summary         transform          38 rows     0.0s  OK
[3/3] monthly_totals          transform          13 rows     0.0s  OK
Completed in 1.1s: 3 ok
```

The counter tracks completions, not launch order, so `--threads N` finishing
out of order still counts up. `stel build` also prints a `SKIPPED (upstream
failed)` line for each model it blocks. The ledger is not TTY-gated: a CI log
wants it as much as a terminal does.

Pass `-v` for detail on top of that — per-source discovery lines, the phases
(profile resolution, warehouse connect, the incremental to-process/unchanged
split, publication telemetry, test execution), and a live per-model progress
bar on a TTY. Bars cover the model kinds with a countable unit of work:
extraction, `llm:`, `embed:` and `chunk:`. `search:` reports indexed batches as
lines instead, because its read is a one-shot bounded stream with no row count
to render a bar against.

On a TTY the log channel and the bar run together: records are routed through
the progress reporter, which holds them while a bar is live and prints them
once it finishes, so the bar is never written over. Events the reporter renders
itself — source discovery, model completion, publication telemetry — are
emitted once, not once per channel. A bar that stays live for a long time caps
how many lines it holds; if any are dropped, the count is printed before the
rest rather than silently swallowed. `run --threads N` runs independent models
concurrently, so it falls back to the plain log-line channel even on a TTY
(parallel bars on one terminal would interleave).

`--json` is the machine path and goes quiet: the payload on stdout is the whole
output, with no ledger on stderr. Add `-v` alongside it to get the log channel
back while keeping stdout a single parseable payload.

With `--source-filter`, the reported per-source count is the post-filter
selected count, so it always reflects what is actually processed. For runs
launched by an orchestrator, `STEL_VERBOSE=1` enables verbose output without
changing the CLI invocation. `-v` is also available on `stel eval` and
`stel concept-cloud`.

The verbose flag is deliberately capped at INFO. DEBUG-level log sites
(transform failures, provider errors) carry unsanitized exception text
and traceback frames that the user-facing error path scrubs but a raw
log stream would not — attach your own DEBUG handler if you need it for
troubleshooting.

Under verbose, each incremental publication also emits safe telemetry
(issue #292) — the progress reporter renders it on a TTY, the INFO log carries
it on a captured/orchestrator run. DuckDB reports the relation, row count and
key; BigQuery adds the job-level statistics below: the output relation, the
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

### Bounding DuckDB's memory

DuckDB sizes its buffer pool from **host** RAM — about 80% of it. Inside a
container that is the wrong number in the dangerous direction: the cgroup
ceiling is invisible to DuckDB, so it will grow past the limit the kernel
actually kills the process at, and a read that is bounded on stel's side still
gets OOM-killed.

stel therefore detects a container ceiling and hands DuckDB 75% of it, leaving
the remainder for the Python process — the frames, provider buffers, and flush
window, all of which are bounded by design. Detection is advisory: it only
supplies a limit where DuckDB would otherwise size itself from the host, never
raises one, and does nothing at all outside a container, so a workstation run
is unaffected.

Override it per target when you know better:

```yaml
warehouse:
  type: duckdb
  path: ./target/stel.duckdb
  memory_limit: 4GB           # a DuckDB size string; wins over detection
  temp_directory: /var/tmp/stel   # where a bounded DuckDB spills
```

`memory_limit: none` opts out of both the setting and the detection, restoring
DuckDB's own host-sized default. A malformed size is rejected when the profile
is validated rather than from inside the driver partway through a run.

Set `temp_directory` when the default — a directory beside the database file —
is on a volume too small to spill into. A bounded DuckDB spills rather than
failing, so it needs somewhere to go.

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

For gcloud user ADC, stel preserves the scopes granted by
`gcloud auth application-default login`; it does not replace them with the
profile's dbt-compatible default list. Credential types that require scoping,
including service-account and external-account ADC, still receive the
configured `scopes`. This matches Google Auth's normal discovery behavior
while retaining stel's no-subprocess ADC loading on Windows.

```
pip install 'stel[bigquery]'
```

The extra includes both the BigQuery query client and the BigQuery Storage
Read client. The Storage API is enabled with the BigQuery API, but the runtime
principal needs roles/bigquery.readSessionUser (or equivalent
bigquery.readsessions permissions) on the query execution project
(execution_project when set, otherwise project), plus its existing table
access on the data project. Bounded snapshot payloads use that API so wide
vectors and document text do not pass through the REST result endpoint.

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
  so together they govern how many MERGEs an extraction run issues. Other model
  kinds use their own checkpoint cadence and do not read `publish_every`. Neither
  clustering nor
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

## Warehouse-table sources

A `warehouse://` source treats each **row** of a relation as a document, so
text that arrived in the warehouse — Fivetran/Airbyte/dlt loads, upstream dbt
models — enters an `extraction:` pipeline the same way files do:

```yaml
sources:
  - name: reddit_rows
    path: warehouse://economics_raw.reddit_comments_raw
    key_column: comment_id          # the row's identity across runs
    path_columns: [subreddit]       # optional: prefix the document path
```

The relation is read through the **active adapter**, so warehouse dialect
stays behind `adapters/`, and per-target `source_paths` overrides point each
target at its own copy — dev at a sampled table, prod at the real one. The
name may be `table`, `schema.table`, or `project.dataset.table`; each part is
validated at config load and quoted per dialect.

**Identity.** `key_column` is the row-grain analogue of an object path: its
value becomes the final segment of the document's source-relative path
(prefixed by `path_columns` values), and `document_id` derives from that path
exactly as it does for files. The content hash fingerprints the whole row, so
the incremental machinery works unchanged: a changed row re-extracts, an
unchanged row skips, a deleted row prunes its documents — and its chunks.
Null keys and duplicate keys are hard errors, not guesses: a row without a
key has no identity, and which duplicate became the document would depend on
warehouse row order. Filter with a view when the raw table is imperfect.

**`--source-filter` composes unchanged.** The globs address the document
path, so with `path_columns: [subreddit]`, `--source-filter 'economics/*'`
scopes a run to those rows the way `'AAPL/*'` scopes an object prefix — the
same partition seam orchestrators already use.

**Fetch and extraction.** Each discovered row is served to the backend as a
plain JSON object (timestamps as ISO strings, decimals as strings at their
declared scale, binary as base64), so `backend: json` with declared `fields:`
is the natural pairing. Discovery snapshots the relation once and extraction
consumes that snapshot, never a re-query of a table that may have moved —
the row-grain analogue of the object sources' verified-snapshot rule.
`max_objects` (default 5000) bounds the read and refuses rather than
truncates; narrow with a view or raise it deliberately.

**Not `dbt_ref()`.** The two answer different questions: `dbt_ref('model')`
lets a *transform* consume a dbt-built table, resolved by dbt in embedded
(dbt-duckdb) mode. A `warehouse://` source starts a *document pipeline* from
rows, resolved by the active adapter in standalone mode. Source freshness
reports row counts but no modification time — rows carry none a listing could
read; a declared watermark column is future work.

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

**`max_objects` counts matching documents.** The cap is about the documents
the source reads, so objects the `file_pattern` discards do not consume it — a
sibling pipeline writing `.json` sidecars under the same prefix cannot fail a
`*.htm` source. A prefix broad enough to keep scanning far past the cap without
finishing still fails, so a typo'd prefix crawls loudly rather than quietly.

**`--source-filter` narrows the listing.** When every glob shares a static
leading path segment, that segment is pushed into the listing prefix, so
`--source-filter 'AMAT/*'` lists one ticker's objects instead of the whole
prefix and then discarding the rest. Only whole segments qualify: `AMAT*` does
not narrow (it would exclude `AMATX/…`, which the glob matches), and a glob
beginning with a wildcard narrows nothing. Document identity is unaffected —
the source-relative path still carries the segment the listing prefix absorbed.

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
`timestamp`, `json`, and `enum` (`type:` and `dtype:` are accepted input
aliases for `data_type:`). A successful zero-document run materializes a typed, zero-row
relation from this contract, so downstream tests and models see a real table.
Type changes participate in `code_version`; invalid casts fail without
publishing a full-model staging table. A declared field without `data_type`
defaults to string. Omitting `fields:` retains legacy dynamic backend output,
but cannot type payload columns for an initially empty corpus.

#### Enum fields: declare the label set once

A classification field is a closed set of values, and writing that set out in
several places is how a taxonomy rots — add a label to the prompt and forget
the test and invented labels pass silently; add it to the test and forget the
prompt and the model never produces it.

`type: enum` declares the set once:

```yaml
fields:
  - name: signal
    type: enum
    values: [churn_risk, expansion, pricing, support, none]
  - name: evidence
    type: string
```

Three things derive from that one declaration:

1. **The provider's output schema** carries a real `enum` constraint, so the
   model is constrained at the API boundary rather than asked politely for one
   of the labels.
2. **An `accepted_values` check** runs on the column, with no hand-typed list
   to drift. It needs no `tests:` entry — the field is the declaration. An
   explicit `accepted_values` on the same column is honoured instead (yours
   runs, not two), and a disagreement between the two lists is reported at
   compile time: the declared set is what the schema and prompt use, so a
   test allowing something else is checking a different taxonomy.
3. **The prompt, as a portability fallback.** Where a provider's structured
   output cannot carry an enum, the constraint is stripped from the schema and
   the labels are rendered into the system prompt instead, so the taxonomy is
   enforced as far as the provider allows and communicated regardless. Every
   shipped provider carries enums natively, so this is for third-party
   providers; a provider declares its ability with `supports_schema_enum`.

`values:` requires `type: enum`, and an `enum` with no `values:` is rejected —
it would constrain nothing. The column materializes as a string; `enum` is
stel's declaration, not a warehouse column type, and `emit-dbt-sources`
exports it as `string`.

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

Incremental **transforms** get the same treatment through `commit_every`,
which sets how many changed parents are invoked and published per batch:

```yaml
- name: chunk_entities
  depends_on: [ref('document_chunks')]
  transform:
    type: python
    module: transforms.chunk_entities
    commit_every: 250   # smaller = finer crash recovery, more MERGEs
  materialization: incremental
```

Each batch reconciles `on_schema_change`, publishes its children, and advances
those parents' state before the next begins, so a run that fails partway keeps what committed and a relaunch
reprocesses only the parents whose state never advanced. Without it, a failure
at the last parent — or at the publish — re-pays the whole corpus, which on a
multi-million-row wrapper is tens of minutes and a multi-gigabyte republish.

The default (1000) is high enough that a run with fewer changed parents than
that is a single batch: identical behavior, one MERGE. Lower it when a run is
long enough that losing it hurts; raise it to cut MERGE count, and BigQuery
bytes billed, on runs that rarely fail.

**Embed models** flush the same way, and for them the argument is sharper. A
failed extraction or transform costs CPU to redo; a failed embed costs the
provider bill again. Embedded rows publish and advance state every
`flush_every` rows (default 5000):

```yaml
- name: chunk_embeddings
  depends_on: [ref('document_chunks')]
  embed:
    provider: vertex
    model: text-embedding-005
    dimensions: 768
    flush_every: 2000   # publish and bank 2000 embedded rows at a time
  materialization: incremental
```

`embed.max_concurrent` (default 8) is how many provider batches are in flight
at once. It is the throughput knob: raising it overlaps provider latency, and
`batch_size` decides how much work each request carries. On Vertex the two
compose — a batch is split again by `max_texts_per_request` and
`max_tokens_per_request`, and those splits are issued concurrently, so a
`batch_size` too small to split leaves `provider_options.max_concurrent_requests`
with nothing to overlap. Neither setting is part of `code_version`, so tuning
throughput never re-embeds a corpus.

Peak memory is one flush rather than the corpus, and a run that dies at hour
28 keeps every row it already paid for — the next run resumes at the last
flush instead of re-embedding everything (issue #401).

The input is bounded the same way. An embed run reads its upstream's schema
from a zero-row probe, streams the id column once to validate it and count the
corpus, then streams the rows themselves in batches to fill each flush window
— it never materializes the upstream as a single frame (issue #410). Before
that, a *fresh* run's peak was O(corpus) no matter how small `flush_every`
was, because the whole upstream was read before the first provider call.

The resume is bounded too: a resumed run reads the existing target's
id column once (streamed and projected — no vectors), then looks up reuse
candidates one window at a time by key, so resuming a large corpus never
costs more memory than running it.

On BigQuery the reuse target is also the table each window just updated. If
table metadata advances while one of those immutable query results is being
consumed, stel discards that entire advisory lookup and retries a bounded
number of complete projected reads. This also lets a new run settle after a
server-side write left behind by an interrupted predecessor. Upstream snapshot
reads do not retry generation changes, and a continuously changing target still
fails the run.

Both paths need `streaming_tabular_reads` from the adapter. It is checked at
preflight, so a warehouse without it fails before credentials are resolved
rather than partway through a corpus.

Embed runs also charge the run-scope budget in `profiles.yml`. The
enforceable dimensions are `max_api_calls`, `max_input_tokens`, and
`max_documents` — embedding *cost* is not modeled, so `max_cost_usd` does not
gate embeds. A budget stop behaves exactly like a crash at the same point:
published windows stay, state covers exactly them, and the run reports
`budget_exceeded` so descendants are skipped rather than fed a partial
corpus. Raising the cap resumes for the remainder:

```yaml
      llm:
        budget:
          max_api_calls: 40000   # shared across every model in the invocation
```

**llm map models** flush the same way, on `flush_every` inputs (default
1000). This is the stage where an all-or-nothing write costs the most: one
provider call per input, so a failure — or a budget ceiling — near the end of
a long run used to discard every completion already paid for. Windows already
published now stay, with their state advanced, and only the partial window in
flight is re-called:

```yaml
- name: filing_facts
  depends_on: [ref('filing_registry')]
  llm:
    provider: vertex
    model: gemini-2.5-flash
    input_field: body
    prompt: extract_facts
    flush_every: 500
  materialization: incremental
```

Input memory is bounded independently of corpus size. Native llm models take
their schema from a zero-row probe, stream only the id and input columns to
validate ids and classify changed records before provider spend, then stream
those columns again to fill one `flush_every` work window at a time. On an
incremental resume, the existing generated target is projected to its id
column and streamed in fixed-size batches; generated text is never loaded just
to reconcile deletions (issue #424).

The preliminary classification scan is intentional. It preserves the
`max_documents` contract that an over-cap run stops before its first provider
call or publication, while still avoiding a corpus-sized list of input text.

For `output_cardinality: many` each window publishes through
`replace_children` scoped to that window's parents, so a later window never
disturbs children an earlier one published.

One consequence worth stating plainly: a **full refresh** of an embed or llm
model is
no longer a single atomic swap. The first flush replaces the target and later
flushes merge into it, so an interrupted rebuild leaves a partially rebuilt
table that the next run completes from state. That is a deliberate trade — the
alternative is the previous behavior, where a corpus large enough to matter
could not be rebuilt at all.

### Which stages checkpoint

| Stage | Cadence knob | Default | Why |
| --- | --- | --- | --- |
| `extraction` | `flush_every`, `publish_every` | 5000, 1 | Parser and backend cost |
| `transform` | `commit_every` | 1000 | CPU over changed parents |
| `chunk` | `flush_every` | 5000 | Chunk rows amplify the input |
| `embed` | `flush_every`, `max_concurrent` | 5000, 8 | **Metered provider spend** |
| `llm` | `flush_every` | 1000 | **One provider call per row** |
| `search` | per-batch upsert | — | Publishes and advances state per batch |
| `chunk`, `ml`, `eval` | none | — | CPU-only; a re-run costs time, not money |

The rule behind the table: a stage checkpoints when losing its work costs more
than the bookkeeping. `chunk`, `ml`, and `eval` publish once because redoing
them is cheap and deterministic. Everything that spends money, or runs long
enough that losing the run hurts, publishes as it goes and advances state only
for what it actually wrote.

Like `flush_every`, `commit_every` is excluded from `code_version` — it changes
execution cadence, never output content, so tuning it does not invalidate
existing state. The same holds for an embed model's `flush_every`: it must
not move `code_version`, because that would silently re-embed every existing
corpus at provider prices.

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
before successful close to reject a newer table version. BigQuery reads an
unfiltered, explicitly projected snapshot directly through the Storage Read
API's own column selection — no query job at all, and so no anonymous
destination table for BigQuery to materialize before the read can begin.
That materialization step is what a wide-enough projection actually exceeds
(a `responseTooLarge` job failure), upstream of any question about how a
result is fetched back; skipping it is what makes a snapshot over a
768-dimensional-vector-plus-document-text relation possible at all. A
predicated read still runs a query — translating predicates into the Storage
API's own row-restriction syntax remains unsupported — and streams that
finished job's destination table through the same Storage Read path instead.
Either way, the typed Arrow schema and the projection's column positions both
come from the read session itself, never from the request or a query's own
schema: the Storage Read API does not return the requested projection's
column order, so pairing a schema from anywhere else with its batches would
select the wrong columns by position. It rejects the read if the table
generation changes while the snapshot is opened or consumed. A read that
names a key column adds one further uncached aggregate over that column
alone, ahead of the payload read, so the key check never touches the
projection; it is cheap next to the payload, but it is its own query and
normal query billing and the profile's `maximum_bytes_billed` limit apply to
it. Both adapters push projection and predicates into the warehouse when a
query runs. Predicate values are bound parameters and
redacted from diagnostics.

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

`chunk_overlap` carries the tail of each chunk into the next, so a concept
straddling a boundary is findable from either side. The carried text **starts
on a separator boundary**: the splitter steps back approximately
`chunk_overlap` characters and snaps to the nearest break in the same
hierarchy it splits on (`

` → `
` → sentence → word), preferring the
strongest available. A fixed step back would land wherever the count fell —
mid-word for most chunks — which defeats the separator hierarchy for every
chunk after the first. Like `chunk_size`, `chunk_overlap` is therefore a
target rather than an exact count; snapping is bounded to between half and
twice the requested overlap, and falls back to an exact slice when no
boundary exists in that band (one very long token, say).

### Heading attribution

For a long structured document — a filing, a contract, a report — "which
section" is the most useful retrieval filter there is. The splitter is the
only place that sees both the full document and every boundary position, so
it can answer that exactly:

```yaml
  chunk:
    strategy: recursive
    chunk_size: 1600
    headings:
      pattern: '^(Item\s+\d{1,2}[A-C]?)[.:]'   # line-anchored on the source
      column: section                           # default
```

Each chunk gets a `section` column: **the last heading at or before the
chunk's start**. That column rides the embed step's passthrough onto the
search index like any other, turning "risk factors mentioning tariffs" into
`section = 'Item 1A'` plus similarity, rather than similarity alone.

The pattern is matched with `re.MULTILINE` against the source text. **A
capture group names the section**; without one the whole match is used — so
`^(Item\s+\d+[A-C]?)[.:]` yields `Item 1A` while `^Item\s+\d+[A-C]?[.:]`
yields `Item 1A.`. That keeps the choice about trailing punctuation with the
author instead of stel guessing.

Two behaviors worth knowing:

- **A chunk that straddles a boundary belongs to where it starts.** A chunk
  can carry the next section's heading in its tail and still be mostly the
  previous section's content; it is attributed to the previous one. (This is
  the case a downstream transform re-deriving sections from chunk text gets
  wrong — it sees the heading and claims the whole chunk.)
- **Text before the first heading has no section** (`NULL`), rather than being
  attributed to a heading that comes after it.

`headings:` requires `strategy: recursive`: attribution works from source
character offsets, which the token splitter does not produce. Declaring it
with `strategy: tokens` fails at config load rather than silently emitting
nulls. Naming a `column:` the upstream already has — or one the chunk model
generates itself, such as `chunk_id` — fails at config load rather than
overwriting it.

This complements [in-text metadata](#metadata-the-embedder-can-actually-see)
rather than overlapping it: that puts document context *into the embedded
text* so the vector carries identity; this is a structured, filterable
*attribute*. A corpus generally wants both — one helps unfiltered semantic
queries, the other makes scoped queries exact.

### Metadata the embedder can actually see

Carried columns serve SQL perfectly — filtering, joining, building a citation.
But the embedding model and the LLM read only what is inside the text field.
For them, metadata in a sibling column does not exist, and a chunk from the
middle of a document embeds with no idea which document it came from. That is
exactly the ambiguity that hurts retrieval on short chunks: "the rate rose 40
basis points" is far more findable when the vector also encodes which report
and which quarter it came from.

`in_text_metadata` renders upstream columns into the chunk text itself:

```yaml
  chunk:
    strategy: tokens
    text_field: text
    chunk_size: 800
    in_text_metadata: [title, published_date, source_uri]
```

producing

```text
title: Q3 Monetary Policy Report
published_date: 2026-03-14
source_uri: gs://raw/reports/q3.pdf
---
<original chunk text follows>
```

**Additive, never a mode switch.** The columns are still emitted on every chunk
row; the block is a second copy inside the text. The same metadata serves two
readers with different eyes, and a rendering aimed at one must never remove the
structured copy the other depends on — otherwise downstream SQL has to regex
values back out of prose.

Fields render in declared order, and null values are skipped rather than
written as `None`. Naming a column the upstream model does not have fails
before any document is processed.

**The block counts against `chunk_size`**, in whichever unit the strategy uses,
so it does not push chunks past the size the embedder was configured for. (The
recursive splitter's own overlap merging can still exceed `chunk_size` on a
hard cut, as it always could; the block does not add to that.) Providers
configured without truncation reject an oversized request rather than quietly
shortening it, so the alternative — adding the block on top — would turn a
retrieval-quality feature into a runtime failure. If the block leaves no room
for text, or pushes `chunk_overlap` to or past the remaining budget, the model
fails with the numbers named.

Because the block is part of the emitted `text`, `chunk_id` tracks it like any
other text change: turning `in_text_metadata` on, off, or editing its field
list re-keys that document's chunks and invalidates anything downstream keyed
on them (embeddings, retrieval-store rows). This is the documented rule, not an
exception to it — and it is why `chunk_id` stays consistent with the
`agent_context` `document_chunks` contract, which recomputes the id from the
stored text and rejects a mismatch.

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
schema; the prompt is inline or a
[versioned reference](#versioned-prompts). It shares one execution core with `backend: llm`
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

### Versioned prompts

A prompt is a program input that changes the output, exactly like the SQL in a
transform, so it can have what code has: a name, a version, and a diff in
review. Reference one instead of inlining it:

```yaml
  llm:
    prompt: { name: signal_classify, version: v3 }   # prompts/signal_classify/v3.md
```

The useful analogy is a database migration. Each version is a file, referenced
explicitly, and improving one means writing the next version rather than
editing a released one.

**Inline `prompt: "..."` keeps working** — it is right for quick projects and
examples. This is an additional form, not a replacement.

**Version resolution is explicit and required.** There is deliberately no
`latest` pointer: a moving reference would make two runs of the same committed
project resolve to different text, which is the mutable-prompt problem
versions exist to solve.

**Resolved at compile time.** A missing or misspelled version fails `compile`,
before source discovery, credentials, or any provider call — and the error
lists the versions that do exist:

```
Model 'classified' references prompt signal_classify/v9, but
prompts/signal_classify/v9.md does not exist. Available versions of
'signal_classify': v1, v2.
```

Name and version are path segments, so they are charset-validated at config
load; a traversal never parses. Prompt files must be regular, non-symlink
files inside the project, the same rule project configuration follows.

**Every row records which prompt produced it.** `prompt_name` and
`prompt_version` join the existing `llm_*` provenance columns, so a query can
group by them:

```sql
select prompt_version, count(*), avg(...) from classified group by 1
```

`llm_config_hash` records that *something* changed — prompt, schema, provider,
and model identity mixed into one opaque value. These columns say *what ran*.
An inline prompt leaves both null: there is no stable identity to record,
which is precisely the gap versioned prompts close, and the config hash omits
prompt identity entirely in that case so existing models are not re-keyed.

> **Upgrading an existing incremental `llm:` model.** The two columns change
> the target's schema the next time it publishes rows, which the default
> `on_schema_change: fail` rejects. Set `on_schema_change: append_new_columns`
> or run once with `--full-refresh` before it next reprocesses. The upgrade
> itself reprocesses nothing.

The [run log](#append-only-logs) carries the same two columns, which is what
makes "did v4 cost more per row than v3" a query rather than an investigation.

**Prompt text never enters an artifact.** `manifest.json` records the resolved
name and version only. Changing a version file still invalidates incremental
state, because the resolved text is part of the model's config hash.

#### The immutability gate

Versions are only immutable if something enforces it. `prompts/lock.json`
records what each released version contained, and is committed — its diff is
what makes a changed prompt visible in review:

```bash
stel prompts lock     # record every version; commit the result
stel prompts check    # CI gate: fails if a released version changed
```

`check` exits non-zero and names the offender:

```
Error: Prompt lock check failed:
  signal_classify/v1 was released and has since changed. Add the next version
  instead of editing this one.
```

It also reports a version present in the tree but not the lock (run `lock`),
and one in the lock whose file is gone — rows already produced under a version
record it, so deleting it strands their provenance.

**`lock` refuses to launder an edit.** Re-locking a released version whose
contents changed requires `--force`, and says what it re-locked. Without that
refusal, `lock` would be a one-command bypass and would teach exactly the
workflow the gate exists to prevent: the fix for a prompt that needs changing
is a new version, not a new hash.

The hash covers the stripped text, so an editor adding a trailing newline is
not a released-prompt edit.

Add it to CI beside the other checks:

```bash
uv run stel prompts check
```

## Search indexes (local proof of concept)

A `search:` resource publishes exactly one upstream warehouse model to an
independently configured retrieval store. It is a leaf serving resource, not a
warehouse relation. Two stores ship: `duckdb`, which needs no extra and can
live in the warehouse file itself, and `lancedb`, which needs
`stel[lancedb]`. Configure the operator-owned store in `profiles.yml` and
explicitly opt in to public indexes:

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

### DuckDB-native search

When the warehouse is already DuckDB, a separate retrieval system is an extra
moving part for no reason. The `duckdb` store serves vector and full-text
search from a DuckDB file — optionally the same file as the warehouse — using
the `vss` and `fts` extensions:

```yaml
      retrieval:
        default: local
        stores:
          local:
            type: duckdb
            path: ./target/stel.duckdb
```

`duckdb` is a core dependency, so this needs no extra; the `vss` and `fts`
extensions are installed on first connect, which requires network access once
(an air-gapped host must have them pre-installed). The store opens its own
connection and closes only that connection, so pointing it at the warehouse
file does not close the database out from under the warehouse adapter.

Three behaviors are worth knowing before choosing it:

**Approximate vector search is opt-in.** DuckDB will not build a persistent
HNSW index unless `hnsw_experimental_persistence` is set, because that index
is not covered by the write-ahead log and a crash can leave it inconsistent
with the table. stel will not set that flag for you: declaring
`vector: {search: approximate}` without it fails at publish with an
explanation rather than silently accepting the risk. Exact search needs no
index and returns the same rows — the cost of declining is latency, not
correctness:

```yaml
          local:
            type: duckdb
            path: ./target/stel.duckdb
            hnsw_experimental_persistence: true
            hnsw_ef_construction: 128
            hnsw_m: 16
```

**Indexes are rebuilt at publish, not maintained.** DuckDB's HNSW index does
not compact on delete, so an incrementally churned index would grow without
bound; the BM25 index is a snapshot of the table at build time rather than a
live view. Both are rebuilt when a publish completes, which keeps publish cost
predictable and query cost flat.

**Hybrid search is composed, not native.** DuckDB has no operator that blends
`vss` and `fts` ranking, so hybrid runs both legs and stel fuses them with
RRF. This is a supported shape, not a limitation of the store's honesty about
itself.

Ownership is stamped in the table comment and read back through DuckDB's
catalog, so a table stel did not create is refused rather than published into.

#### What is and is not atomic

Stating this explicitly because the guarantees differ from a remote store's,
and the difference is invisible until it matters:

- **One upsert batch is atomic.** It runs as a single statement inside a real
  transaction, so a partial batch is not a state any reader can observe.
- **A delete batch is atomic**, for the same reason.
- **A publish as a whole is not.** Creating the collection, writing rows, and
  rebuilding indexes are separate transactions. A crash between them leaves a
  collection that exists and is stamped but whose indexes lag its rows;
  re-running the publish converges it.
- **Index rebuilds are not atomic with the writes they follow.** Between the
  bulk mutation and the rebuild, full-text queries reflect the previous BM25
  snapshot. Vector queries are unaffected, since exact search reads the table
  directly.
- **Concurrent readers during a publish see the collection mid-update.** stel
  fences concurrent *publishers* on one host, but it does not snapshot the
  collection for readers, and DuckDB offers no rename-based swap that would
  give one without a second copy of the data.

The single-host publisher lock is the boundary from issue #152 and is unchanged
here: it excludes another publisher on the same machine and cannot fence one on
another machine sharing the file. DuckDB's own single-writer rule does catch
that case, but as a lock error rather than as coordination — stel reports it
as `duckdb_database_locked` so it reads as a concurrent publisher rather than
a misconfiguration.

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
`lt`, `le`, `gt`, `ge`, `in`, and `array_contains_any`; the last two take a
JSON array. Values are parsed against the attribute's declared type, and only
attributes with `filter_role: user` can be supplied by a caller. Multiple
filters are combined with AND.

An `array[string]` attribute holds a list per row, so it is filtered by
overlap and `array_contains_any` is the only operator accepted for it — the
scalar operators would compare a whole list against one value:

```bash
stel search --model chunk_search --query "inflation"   --filter access_groups array_contains_any '["analysts", "ops"]'
```

The row matches when its list shares at least one value with the array given.
The retrieval store must advertise the `array_containment_filters` capability;
one that cannot express overlap refuses the query rather than dropping the
filter.

Every declared `data_type` is filterable, temporal types included:

```bash
stel search --model chunk_search --query "tariffs" --mode vector   --filter filing_date_dt ge 2020-01-01
```

Date and timestamp values are rendered as typed SQL literals (`DATE '…'`,
`TIMESTAMP '…'`) rather than quoted strings, because a quoted string is text
to the query engine and will not compare against a `date32` or timestamp
column. A round-trip test executes one filter per declared type against a real
store, so a filter that validates at authoring time cannot fail at query time
on the caller.

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

### Changing a published index's configuration

A published collection records a **semantic descriptor** of the configuration
it was built from. On the next publish stel compares the two and classifies the
difference, so a change is named rather than merely detected.

Fields that only affect execution cadence are not part of the descriptor and
never invalidate an index: **`batch_size`**, **`index_options`**, and
`on_index_change` itself. Tuning publish pacing is free.

Everything that defines a row's shape or meaning is:

| Change | Classification |
| --- | --- |
| new attribute, wider `display_fields`/`return_text_fields` | compatible |
| `vector` field, dimensions, metric, or embedding identity | rebuild required |
| `id_field`, `document_id_field`, `chunk_id_field` | rebuild required |
| `text_fields`, `full_text`, `access`, `query` modes | rebuild required |
| an existing attribute's `data_type` or `filter_role` | rebuild required |
| removing an attribute or a projected field | rebuild required |

`on_index_change: fail` is the default. A classified change stops the run, and
the error names the field that forced it and says whether the existing
collection could have served the change:

```
Search index configuration changed and requires a rebuild: vector: changed
from {...} to {...}. Rows already written were indexed under the previous
definition, so publish under a new collection name, validate it, and cut
consumers over.
```

### `on_index_change: online`

Set `on_index_change: online` to apply a change the table above calls
*compatible* — a new attribute, or a wider `display_fields` /
`return_text_fields` — to the live collection instead of refusing it:

```yaml
search:
  on_index_change: online
```

The new columns are added to the published collection in place, and the rows
are then republished from the warehouse to fill them. **No embeddings are
recomputed**: vectors come from the upstream table, so the cost is an index
rewrite, not provider spend. That is the difference between adding a filter
attribute to a 20k-document index for the price of a republish and paying to
embed the corpus again.

Two limits are deliberate:

- **It applies only to compatible changes.** A changed vector dimension,
  metric, id mapping, analyzer, or an existing attribute's type or filter role
  is still refused under `online`, because the rows already written really are
  invalid. A capability flag cannot make an incompatible change safe.
- **The store must advertise `online_schema_evolution`.** The policy is
  rejected at compile time against a store that cannot widen a live
  collection, rather than failing mid-publish.

### `on_index_change: rebuild`

Set `on_index_change: rebuild` to absorb a change the table above calls
rebuild-required, instead of stopping the run. stel builds a **new generation**
— a physical collection nothing is querying — validates it, and only then
points the logical collection at it. The previous generation keeps serving
every read until that switch, so there is no window in which the index is
empty, half-built, or serving rows from two configurations.

```yaml
search:
  on_index_change: rebuild
```

The same machinery backs a full replacement asked for directly:

```bash
stel run --select chunk_search --full-refresh
```

or, permanently, `materialization: full` on the search resource.

Three things follow from how activation works, and are worth knowing before
turning any of them on:

- **It re-embeds everything.** A rebuild is a full republication at provider
  prices. It is never inferred from a change — you ask for it, by policy or by
  flag. An unannounced full re-embed is the behavior this design rejects.
- **The store must advertise `private_generation_build`.** A store that cannot
  build a collection under a private name cannot replace a live one
  atomically, and the run fails before touching anything.
- **The superseded generation is retired after activation**, once nothing can
  be reading it. A generation left behind by a publisher that died mid-build
  is reclaimed by the next successful publish.

Recovery is safe across all of this: `stel serving recover` preserves the
record of which generation is live, so recovering authority does not force a
re-embed.

Changing `store` or `collection` is not an evolution at all — it selects a
different physical collection, published independently under its own state.

Collections published before descriptors existed carry only the older stamp.
The first publish after upgrading recomputes that older stamp to prove the
configuration is unchanged and rewrites it in place. Rows are untouched: there
is no rebuild, no re-embed, and nothing to run by hand.

#
## Suggesting context improvements

`stel suggest dbt` turns the agent-transcript corpus into a **reviewable
patch** against a dbt project. dbt keeps its context as files in git, so the
acceptance mechanism is a diff a human reads and merges — not a table of
recommendations nobody opens.

```bash
stel suggest dbt --from analytics.doc_suggestions --dbt-project ../analytics
```

It prints a unified diff and changes nothing. Re-run with `--write` to apply;
stel never commits.

### Where the analysis lives

Not in the command. Deciding *which* models are under-documented, and what
their descriptions should say, is an ordinary stel project over the transcript
corpus — the same provider, prompt-provenance, and incremental machinery every
other model uses. `stel suggest dbt` reads the relation those models produce.

That relation is the contract:

| column | meaning |
|---|---|
| `dbt_model` | the dbt model to document |
| `dbt_column` | column to document; null for a model-level description |
| `suggested_description` | the proposed prose |
| `evidence_count` | distinct sessions supporting the suggestion |
| `evidence_sessions` | which sessions — the provenance a reviewer asks for first |

`--min-evidence` (default 3) is the bar. One session is an anecdote: an agent
opens a model's SQL for all sorts of reasons. Repetition across sessions is
what separates "someone looked at this once" from "this keeps costing people
time".

### What it will not do

Each of these is a way a well-meaning suggestion could destroy work:

- **Never overwrites an existing description.** Only absent ones are filled —
  and a `description:` key that is present but empty or null counts as
  existing, because inserting a second one beside it produces a duplicate
  mapping key that loaders resolve back to the empty value.
- **Never touches any key but `description:`.** Tests, columns, and config are
  out of reach.
- **Never documents the wrong object.** The entry is located inside the
  `models:` block and at that sequence's own depth, so a source table or a
  column sharing the model's name cannot be edited in its place.
- **Never leaves `models/**/*.yml`.** `dbt_project.yml`, seeds, and snapshots
  are not searched, so they cannot be edited. Symlinks are refused rather than
  followed — including a symlinked `models/` or any symlinked directory
  beneath it, which would otherwise put files outside the project in reach.
- **Never applies without `--write`,** and never commits.

Descriptions are ordinary prose and may contain colons, quotes, or newlines.
They are emitted as quoted YAML scalars, so a description cannot introduce
keys into the file it lands in.

## Serving readiness and coordination

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
check) and clears leases. It grants nobody publication authority — a new
publisher still claims the scope in the ordinary way. After upgrading to this
contract, run `stel run` once per search index to establish its ledger before
querying.

A failed or recovered publication does not necessarily stop queries. A
generation build writes to a collection nothing is reading, so its failure
leaves the previously-active generation correct, and the scope is left
`degraded`: it keeps answering queries from that generation, and the recorded
`safe_error_code` stays visible in `stel serving status` so a pipeline that
has been broken for days does not hide behind a working endpoint. The next
successful publish returns it to `ready` with no operator action.

An in-place publish is the exception. It writes into the collection the
activation pointer names, so a failure there may have corrupted what was
live; both pointers are cleared, the scope becomes `failed`, and queries are
refused until a successful republish. A publisher that is *killed* rather
than failing cleanly leaves no record of its intent, so the claim records it
up front: an in-place claim clears the activation pointer when it is taken,
which is what lets `stel serving recover` fail closed on a crashed in-place
publish while still serving through a crashed rebuild.

A generation retains the configuration fingerprint it was published under, so
a rebuild forced by a *configuration change* leaves the old generation
serving only the queries it can still answer correctly: a query issued under
the new configuration is refused (`published under a different
configuration`), because the retained index was never built for it. A
rebuild forced by `--full-refresh`, where the configuration is unchanged,
keeps serving normally.

#### Migrating the serving scope

The serving scope is keyed on the *logical* collection. Earlier versions keyed
it on the physical collection, which cannot survive a logical collection having
more than one physical generation behind it: resolving the ledger row would
require already knowing the active generation that row names.

An index published before this change keeps its ledger row and publication
state under the old key, where nothing looks for it. stel treats an index with
unreachable publication state as unpublished, which has two consequences and
the first one bites immediately: **queries fail** ("still has its publication
state under the pre-0.13 serving key") until the index is moved, and the next
run would re-embed it in full. The rows and embeddings are intact throughout.
Move it — once per affected index:

```bash
stel serving migrate-scope chunk_search
```

The command is idempotent; a second run reports nothing to migrate. It is
refused while query leases are outstanding (let readers finish, or run
`stel serving recover`), and refused if the destination scope already holds a
published ledger row rather than picking a winner between two publications.

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

That path classifies parents from a **streamed** read of the parent table,
keeping one digest per row rather than the row, and then reads back only the
parents that changed (issue #385). Peak memory therefore follows the change
set rather than the corpus, and an incremental transform requires its adapter
to advertise `STREAMING_TABULAR_READS`. What invalidates a parent is
unchanged — every column still participates, order still does not matter, and
repeated rows still count.

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

## Classification eval models

An `eval:` model scores a classifier against labelled ground truth and
publishes metric rows. It reads two relations that already exist, so it costs
no inference and is cheap enough to run on every change.

It answers a different question from the surfaces beside it. `golden` compares
to expected rows and fails on any mismatch — "is this exactly right", a
pass/fail gate that cannot tell you *how* wrong. `stel eval` scores ranked
document lists for **retrieval**. This scores predicted labels, which is what a
prompt or model change actually raises: did per-label recall move, and which
labels regressed.

```yaml
- name: signal_eval
  eval:
    kind: classification
    predictions: ref('signals')          # the model under test
    predicted_field: signal
    expected: ref('signals_labeled')     # ground truth
    expected_field: expected_signal
    key: chunk_id                        # joins the two
  tests:
    - min_metric: { metric: recall, label: churn_risk, min: 0.70 }
```

`min_metric` reads the **latest evaluation only** (the rows sharing the newest
`evaluated_at`). An incremental eval keeps one metric set per predictions
version — that history is the point — but a gate must read the classifier's
current state, not the worst it has ever been: aggregating history would fail
forever after one bad version, and a stale row could satisfy the existence
check for a label the current version no longer reports. An absent metric row
in the latest evaluation fails the test.

Output is **long format** — one row per metric, so adding a metric never
changes the schema and `WHERE metric = 'recall'` works:

| metric | label | value |
|---|---|---|
| accuracy | | 0.87 |
| macro_f1 | | 0.71 |
| evaluated_rows | | 1430 |
| unmatched_rows | | 12 |
| unusable_expected_rows | | 3 |
| precision | churn_risk | 0.91 |
| recall | churn_risk | 0.72 |
| f1 | churn_risk | 0.80 |
| support | churn_risk | 143 |

`macro_f1` is the unweighted mean over labels: a collapsed rare label moves it
where accuracy hides that behind the majority class, so it is usually the
number worth gating on. `unmatched_rows` counts expected rows with no
prediction to join to — an inner join alone would report a model that stopped
emitting rows as a smaller but equally good one. `unusable_expected_rows`
counts ground-truth rows that could not be scored at all (a null key or a null
label): defects in the labelled set itself, reported rather than silently
inflating quality over the rows that survived. A duplicate `key` value in
either relation is a hard error — which duplicate would be scored depends on
warehouse row order, which silently corrupts an experiment.

Every row also carries `metric_id`, `predictions_version`, `code_version`, and
`evaluated_at`. With `materialization: incremental`, `metric_id` keys on the
metric, the label, and the version of the predictions scored — so re-running
the same predictions replaces the row and a new predictions version appends
one. That is a quality time series, and it makes a prompt change a measured
decision rather than a guess. (`predictions_version` reads the upstream
relation's `code_version`; it is null when the relation carries none, or
carries more than one.)

### Two conventions worth knowing

**Zero denominators score 0.0; they do not vanish.** A label nothing predicted
has no precision in the mathematical sense. Dropping those rows would make a
collapsed label disappear from the report exactly when it most needs reading,
so they are emitted as 0.0 and `support` tells you which kind of zero it is.
This matches scikit-learn's `zero_division=0`.

**The label universe is declared, not observed.** It comes from the predicted
field's `enum` values (see [enum fields](#enum-fields-declare-the-label-set-once)),
so a label the model stopped predicting reports `recall: 0.0` rather than
leaving the report. Set `labels:` explicitly when the predictions model does
not declare an enum.

The two relations become ordinary `depends_on` edges, so selectors, lineage,
and run ordering treat an eval like any other model. Declaring `depends_on:`
yourself is rejected — the edges have one source of truth.

Single-label classification only. Multi-label and regression are different
metric families and would get their own `kind:`.

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
  - min_metric: { metric: recall, label: churn_risk, min: 0.70 }   # classification eval only
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

**`embedding_canary`** covers the one failure every check above is blind to: a
hosted model alias re-resolving to a new snapshot under a pinned name, with
your code, config, and input text byte-identical. Config hashes are computed
from your own inputs and cannot observe the provider; the structural checks
pass on any well-formed vectors. The canary re-embeds a handful of frozen
probe strings and compares against a blessed, committed baseline by cosine
similarity — the measure retrieval already ranks by, so the threshold means
"would this difference change a search result":

```yaml
- name: chunk_embeddings
  embed:
    provider: vertex
    model: gemini-embedding-001
    ...
  tests:
    - embedding_canary:
        enabled: true                       # off by default; see below
        to: ref('embedding_canary_baseline')  # a committed model: text + vector
        min_similarity: 0.999                 # your provider's measured floor
```

The baseline is an ordinary model you `ref()` — probe text in a `text` column
and the blessed vector in an `embedding` column (`text_column` /
`vector_column` override the names; a JSON-encoded array string works, which
is how extraction stores one). Every probe re-embeds through the *tested
model's own* provider identity, and probes are capped at 64 rows: a canary is
a handful of frozen sentences, not a corpus.

Three deliberate behaviors:

- **`enabled: false` is the default, and the skip is visible.** `stel build`
  runs model tests automatically, every probe is a billed provider call, and
  drift happens on the provider's schedule rather than yours — so the canary
  belongs to a scheduled `stel test --select <model>` invocation, not to every
  ad-hoc build. A disabled canary reports `skipped`, never `pass`.
- **`min_similarity` has no default.** The right threshold is your provider's
  measured replica-noise floor, and any shipped number would be a guess
  wearing a default's authority. Measure it: embed the same probe N times
  across separate calls and take the worst pairwise cosine —

  ```python
  from stel.embedding import EmbeddingIdentity, embed_texts
  from stel.config.model import EmbedConfig

  identity = EmbeddingIdentity.from_config(EmbedConfig(
      provider="vertex", model="gemini-embedding-001", dimensions=768))
  runs = [embed_texts(["frozen probe sentence"], identity).vectors[0]
          for _ in range(20)]
  ```

  Set `min_similarity` with real margin *below* the observed floor.
- **Degenerate baselines fail rather than pass.** An empty baseline, a
  zero vector, or a dimension mismatch all fail the check — a monitor that
  can only pass is worse than none.

When the canary trips, the response is a human decision, not an automatic
re-embed: bless a new baseline (accepting the provider's new behavior), or
re-embed the corpus so index and queries agree again.

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
| `examples/agent_transcripts/`       | Claude Code / Codex sessions → exchange-attributed governed search (#360) |

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
response cache all run locally.

Embedded mode works against MotherDuck with no stel-side configuration: dbt
owns the target in this mode, so pointing the **dbt-duckdb profile** at
`md:<database>` (with dbt-duckdb's own `motherduck_token` handling) materializes
the generated models there. stel's internal capture database is a throwaway
local temp file either way and never touches the dbt target.

See
[`examples/dbt_embed_duckdb`](../examples/dbt_embed_duckdb) for a runnable
three-level DAG (extraction → transforms → SQL mart).

The direction can also reverse: a stel transform can declare
`source: dbt_ref('<dbt_model>')` to read a **dbt-built** table back into stel
— optionally alongside ordinary `depends_on:` stel models in the same
transform — closing the loop in one `dbt build`. See
[`examples/dbt_ref_roundtrip_dbt`](../examples/dbt_ref_roundtrip_dbt) for a
runnable stel → dbt → stel → dbt round trip.

## Star map (visualization)

`stel concept-cloud` renders extracted entities as an explorable 3D **star
map**: concepts positioned by meaning, colored by declared or usage-derived
dimensions, with a toggleable *lineage mode* that drops the dbt DAG plane in
beneath and ties each concept down to the model that produced it (issues
#255, #345).

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
The viewer has orbit controls, a color-by picker over every dimension, a
text search, per-value legend toggles, an orphan highlight, a min-frequency
filter, and lineage mode (off by default) with click-to-trace beams. It opens
focused on the hottest retrieved concept — or the most frequent one — rather
than the whole graph.

**Semantic positions** (`--embed-model <model>`). Point the export at an embed
model over the linking mentions and each concept is placed at the centroid of
its mention vectors, projected to 3D at export time (PCA; deterministic).
Proximity then means how the corpus uses a concept. Only coordinates enter the
bundle — never vectors or text — and concepts stay pinned to their positions
in the viewer, because position *is* the meaning. Without the flag, layout
falls back to the force simulation.

**Categorical dimensions** (issue #345). Beyond the built-in entity-type
coloring:

- `--with-query-log` derives a **retrieval heat** dimension
  (hot/warm/cold/never) from the MCP query log: how often agents' queries
  actually returned each concept's chunks. Aggregate-only — query text and
  principal ids never leave the warehouse. `never` is the value worth
  looking at first: well-covered concepts agents cannot reach.
- `--dimension name=model.column` (repeatable) turns any concept-keyed
  categorical column into a dimension. `llm:` enum fields fit directly: the
  label set is already declared and validated upstream.

**Prerequisites.** The cloud is keyed on `canonical_id`, so an entity-linking
(`link_entities`) model must have run; and human-readable node labels require the
NLP/linking models to set `include_text: true` (otherwise nodes show ids).

**Boundaries.** The artifact reads only exported/queried output tables — no
warehouse credentials ever enter it, and raw document text appears only when the
operator opted into it upstream. The bundle is a versioned contract
(`ConceptCloudExport`, `schema_version`), so the static viewer and the export job
evolve independently.

## Append-only logs

Artifacts are per-run files, replaced each run, so any question spanning more
than one run means scraping a directory of JSON. Two opt-in warehouse tables
make those questions queries instead. Both are **append-only** (the history is
the artifact), both create their relation on first write, and both are **off by
default**.

```yaml
my_project:
  outputs:
    dev:
      warehouse: { type: duckdb, path: ./target/stel.duckdb }
      run_log:
        enabled: true
        relation: stel_run_log        # default
      mcp_query_log:
        enabled: true
        relation: stel_mcp_query_log  # default
        capture_query_text: false     # default — see below
```

**`run_log`** (issue #306) — one row per model per invocation: `invocation_id`,
`model_name`, `kind`, `status`, resolved `provider`/`provider_model`/
`provider_implementation`, `rows_processed`/`rows_skipped`/`rows_written`,
`api_calls`, `cache_hits`, `input_tokens`, `output_tokens`,
`estimated_cost_usd` (when the profile sets `pricing:`), `duration_seconds`,
`started_at`, `completed_at`. This is a durable sink for numbers stel already
meters, not a second meter. A `status: budget_exceeded` row makes a tripped
budget visible after the fact rather than only in the terminal output of the
run that hit it.

**`mcp_query_log`** (issue #329) — one row per served `search_context` call:
`logged_at`, `principal_id`, `tenant_id`, `model_name`, `mode`,
`query_fingerprint`, `requested_limit`, `result_count`, `zero_results`,
`returned_chunk_ids`, `top_score`, `elapsed_ms`. Written **after**
authorization and policy filtering, so a row reflects what the caller was
allowed to see — a log of pre-filter hits would leak the existence of
documents the principal cannot read — and a denied request logs nothing.

`zero_results` is the cheapest retrieval-quality signal there is: a question
the index cannot answer is what a chunking or metadata gap looks like from
outside, so it is a column rather than something to reconstruct.

### Two rules worth knowing

**A log never fails the thing it logs.** Writes are best-effort: a warehouse
that rejects one, a permission an operator forgot, a relation someone renamed
— none of that turns a successful run into a failed one, or a served MCP
answer into an error. Failures are a single warning naming the exception
class. The query-log write also happens *outside* the MCP request deadline,
so a stalled warehouse cannot spend a caller's timeout budget.

Both relations are created with **explicit column types** rather than types
inferred from the first batch — otherwise a first run with no LLM model (or a
first query returning nothing) would fix a column as the wrong type and every
later row would fail to append, silently, given the best-effort rule above.

**`capture_query_text` is a second opt-in.** It stays off even when the query
log is on. The fingerprint is always recorded and answers "which questions
repeat, and which return nothing" without storing what anyone typed; turning
the text on records user-authored content in your warehouse, which is a
separate decision. Everything else written to either log follows the artifact
rules — resolved identity and aggregates only, no prompt text, no document
text, no credential values or environment-variable names.

Retention and pruning are yours: the tables are primitives, and anything
reading them is a downstream concern.

## Agent transcripts

`stel transcripts` converts Claude Code and Codex session transcripts into
`transcript/v1` landing documents — one reduced JSON file per session — that
an ordinary local `json` source then consumes (issue #360). The session is
the document and the *exchange* (one user prompt plus every assistant and
tool turn it caused) is the unit chunks attribute to: each exchange renders
under a `## [<ordinal>] <prompt>` markdown heading, so a `chunk:` model with
`headings.pattern: '^## (\[\d+\] .+)$'` names every chunk's exchange in its
`section` column.

```bash
stel transcripts convert <transcript.jsonl> --out <landing-dir>   # e.g. from a SessionEnd hook
stel transcripts sync --out <landing-dir>                          # scan ~/.claude/projects and ~/.codex/sessions
```

Reduction is the contract, not a tuning knob: user prompts (truncated beyond
4,000 characters — pasted bulk content does not land whole) and assistant
prose are kept; thinking blocks, tool result bodies, and tool argument values
are dropped. Each tool call contributes one line — name, argument
fingerprint, ok/error, and the byte count of the output that was dropped —
and the file paths named by file-bearing arguments become per-exchange
`files_touched`, the corpus's best search filter. Sidechain (subagent) and
meta records, and Codex instruction/environment messages, never land.

**stel's own MCP calls are the one exception** (issue #380). A
`search_context` call against stel's server carries the retrieval judgment
the feedback loop is built on, so each one lands in the exchange's
`context_calls` as structured fields — the context model queried, the
`query_fingerprint`, the returned `context_id`s and `chunk_id`s, whether the
call returned nothing, and which of the returned ids the assistant went on to
name in the prose that followed. A call that *failed* records its MCP
`error_code` and is never marked `zero_results`: a denied or timed-out search
returned nothing because it failed, and counting it as an empty retrieval
would corrupt the zero-result rate this corpus exists to measure. The result body itself is still dropped, and
the fingerprint uses the same function and domain as the MCP query log, so a
transcript row joins directly to a served-side log row.

Recognition keys on the response's own `mcp_context/v1` marker rather than on
the MCP server's name, which is operator-chosen in client configuration: a
tool named like stel's that answers something else is not treated as one.
Query *text* stays out unless `--capture-context-queries` is passed, the same
separate opt-in `capture_query_text` is for the query log. Documents carrying
`context_calls` declare `transcript/v1.1`; the field is additive and `v1`
documents remain valid.

`sync` skips any transcript modified within `--min-idle-seconds`
(default 300): that file is a live session, and its sealed exchanges land on
a later pass. Landing writes are atomic and named `{harness}-{session_id}.json`,
so a grown session rewrites exactly one document and content-hash-based
extraction reprocesses only it. Both parsers are tolerant by contract —
unknown record types and torn tail lines skip rather than fail, since neither
harness versions its transcript format.

`examples/agent_transcripts/` composes the full pipeline: landing files →
`json` extraction → heading-attributed chunks → `agent_context/v1` wrapper
transforms (incremental, with the registry as a keyed reference dep) →
deterministic embeddings → a governed search index with `harness`,
`exchange_heading`, `tools_used`, and `files_touched` attributes.

### Candidate retrieval judgments

`stel.transcripts.transforms.retrieval_judgments` turns the captured context
calls into one candidate row per returned id (#329 phase 3, issue #380). Point
it at the model holding `transcript/v1.1` rows:

```yaml
transform:
  type: python
  module: stel.transcripts.transforms.retrieval_judgments
  options:
    transcripts: raw_transcripts
```

Each row carries its `judgment` — `cited`, `returned_not_cited`, or
`zero_result` — plus the `query_fingerprint` that joins it to the MCP query
log, the session and exchange it came from, and `id_space`, which names the id
space the candidate is expressed in (`search_context` results carry both a
`context_id` and a `chunk_id`, and an index keys on one or the other, so
promotion reconciles it against the target's `id_field` rather than assuming).

`returned_not_cited` is **not** a negative: an agent may use a chunk without
naming its id. `zero_result` is the one honest negative. Failed calls
contribute nothing at all.

These are candidates, never goldens: nothing here is read by
`retrieval_tests:` or `eval:`, and nothing promotes itself at any confidence
(#329 rule 2). Promotion is a separate human step.

### Promoting candidates into a golden set

A promotion is a human judgement, so the artifact is **a reviewed file in the
project**, not a warehouse write — it wants git review, blame, and revert, and
a table nobody opens gives you none of those. `golden_sets/<name>.yml`:

```yaml
version: 1
# Must match the target index's `id_field`.
id_space: context_id
queries:
  - query_id: refund_rounding
    query_text: "rounding policy for refunds"
    relevant_ids: ["11111111111111111111111111111111"]
    promoted_by: alex
    promoted_at: 2026-08-25
    evidence:
      sessions: ["0f5a2c1e-1111-4aaa-8bbb-000000000001"]
      harness: claude-code
      query_fingerprint: ed3b7566a129f02e9b61b2a32da0b58d
```

`stel.promotion.golden_set` materializes it into the relation
`retrieval_tests.golden_set` already refs, so **the evals need no changes**:

```yaml
  - name: promoted_goldens
    depends_on: [ref('retrieval_judgment_candidates')]
    transform:
      type: python
      module: stel.promotion.golden_set
      options:
        path: golden_sets/context_search.yml
        search_model: context_search
    materialization: full
```

Two rules the file enforces, each guarding a way a promotion could produce a
worthless test:

- **Every promoted query names its sessions.** The first question a reviewer
  asks is where a row came from; one that cannot answer is indistinguishable
  from an invented golden. A set written from scratch is a fine model — it is
  just not a promotion, and does not belong in this file.
- **`id_space` is checked against `search_model`'s `id_field`.** A set
  promoted in the wrong space matches nothing and reports zero recall, which
  reads as broken retrieval rather than a mislabelled golden set. The mismatch
  is a hard error naming both spaces.

`query_text` is required and human-owned. The corpus records only a query
fingerprint unless text capture was opted into, and `retrieval_tests` replays
each query through `search()` — so the reviewer supplying or confirming the
text is the step that turns an observation into a re-runnable test.

`depends_on` names the candidates the set was promoted from. Those rows are
not read: the file is the source of truth, and a promoted golden must survive
the sessions it came from being rotated away.

### Drafting a golden set from candidates

Writing that file by hand against a relation of candidate rows is tedious and
error-prone, so `stel promote` drafts it. It drafts; it does not promote —
the output is a file a human reads, edits and merges, and nothing becomes a
golden until that happens:

```bash
stel promote \
  --from-candidates transcripts.retrieval_judgment_candidates \
  --output golden_sets/context_search.yml \
  --promoted-by alex          # prints the draft; nothing is written
stel promote ... --write      # writes it, for review
```

Candidates are grouped by `query_fingerprint`, the identity the corpus and the
MCP query log agree on. That fingerprint hashes the query string alone, so the
same question asked of two context models shares one — candidates spanning
more than one index are **refused**, and `--context-model` narrows them to the
index the set is for. Merging them would put ids from one index into a set run
against another, which the `id_space` check cannot catch when both key on the
same space. The drafted file names its context model in the header.

What it does and does not do:

- **Only `cited` ids become `relevant_ids`.** An id that was returned and not
  cited is left out, because an agent may use a chunk without naming it: that
  is absence of evidence, not evidence of irrelevance. It never becomes an
  `excluded_id` either.
- **A query with no citation is skipped and reported**, not dropped. A
  zero-result query is a real signal, but only a human can say what should
  have matched it.
- **`query_text` is filled in from the corpus when it was captured**, and
  shown for confirmation — a transcribed query is more faithful than a
  remembered one. Where the corpus is fingerprint-only, which is the
  sensitivity default, the row carries a placeholder that `load_golden_set`
  **refuses**: an unreviewed draft fails loudly rather than running as a test
  that asks the wrong question.
- **An existing file is never overwritten** without `--force`. It is
  human-owned, and re-drafting over it would discard the review that is the
  point of the artifact. `--output` is confined to the project and refuses a
  path passing through a symlink, matching the loader, which will not read a
  golden set through one.

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

## Memory and corpus size

stel targets a bounded-memory contract: for every model kind, peak memory
follows the flush window, the per-parent unit, and the number of distinct keys
— never the corpus in bytes. The intent is that corpus size is bounded by
warehouse capacity rather than process memory.

Size the key term when you size a container: stages that reconcile deletions
hold every id at once, at roughly 108 bytes each, so a 3.6M-row corpus costs
~370MB per key set and a resuming embed holds two. That scales with row count
rather than row width, which is what makes a large corpus finish at all, but it
is not constant.

It holds today for extraction, transform (incremental), chunk, embed, and
search publication. It does **not** yet hold for `llm:`
([#424](https://github.com/C00ldudeNoonan/constellations/issues/424)), which
still reads its whole upstream and its whole existing target — so size that one
against your corpus until it lands. Some stages read whole tables by contract and always will:
classic ML training fits a single matrix, a transform full refresh needs every
parent, and `concept-cloud` builds one artifact from several models at once.

The invariant, what it costs a stage to hold it, and what enforces it are in
[`docs/architecture/bounded-memory.md`](architecture/bounded-memory.md).

On DuckDB, remember that the engine sizes its own buffer pool separately — see
[Bounding DuckDB's memory](#bounding-duckdbs-memory).

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
