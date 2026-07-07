# dbt-ml

**dbt for unstructured data.** Declarative YAML pipelines that turn folders of
documents — PDFs, markdown, HTML, JSON, email, free-form text — into warehouse
tables. Incremental processing, schema tests, dbt-style selectors, profiles,
and a manifest artifact you can wire into other tools.

This is the v0.1 PoC: pure Python, DuckDB warehouse. **v0.2 is in scope** —
adding RAG support (chunking, embeddings, vector storage via LanceDB) and a
warehouse adapter pattern aimed at the dbt-core set (Postgres, Snowflake,
BigQuery, Databricks, …). A full Rust+Python rebuild is sketched as a
longer-term v2 direction.

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
# Install (once it's published; today: clone and `uv sync`)
uv add git+https://github.com/<your-org>/dbt-ml    # or local: uv pip install -e .

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

## What dbt-ml actually does

| Concept            | What it means                                                                  |
|--------------------|--------------------------------------------------------------------------------|
| **Source**         | A glob over a folder. `*.pdf`, `*.json`, `*.html`, `*.md` — your choice.        |
| **Extraction model** | One row per source file, produced by a backend (pdf, json, markdown, html, llm). |
| **Transform model**  | A Python module returning a Polars DataFrame, depends on other models via `ref()`. |
| **Classic ML model** | A planned `ml:` model for deterministic text/document ML: features, classifiers, clustering, topic models, NLP enrichment. |
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
| `llm`      | `*.txt`/`*.md`    | Claude tool-use → structured fields. Responses cached. Requires `ANTHROPIC_API_KEY`.      |

Add a new backend = drop a file under `src/dbt_ml/backends/`, inherit from
`BaseBackend`, decorate with `@register`. No plugin system needed for v1.

## Security Notes

dbt-ml projects are local code-and-data projects. Only run projects you trust:
Python transforms and custom Python tests execute in your Python process, and
project YAML controls source globs, warehouse paths, cache paths, and artifact
paths.

Document parsers process local files with third-party libraries. Keep
dependencies current before running dbt-ml over untrusted PDFs, HTML, email, or
other documents, since malformed files can trigger parser CPU or memory bugs.

The `llm` backend sends document text to the configured model provider and stores
cached structured responses in the configured cache database. Use deterministic
local backends for sensitive documents unless remote processing is intended.

## The CLI

```
dbt-ml init <name> [--template {json,pdf,markdown,html}]   # scaffold a fresh project
dbt-ml seed [--count N] [--type {invoices,posts,...,tickets,emails}]
dbt-ml compile                                             # parse YAML, validate DAG, write manifest.json
dbt-ml graph                                               # Mermaid DAG to stdout
dbt-ml run [--select EXPR] [--exclude EXPR] [--full-refresh] [--threads N] [--watch] [--state DIR]
dbt-ml test [--select EXPR] [--exclude EXPR] [--store-failures] [--state DIR]
dbt-ml build [--select EXPR] [--exclude EXPR] [--full-refresh] [--threads N] [--store-failures] [--state DIR]
dbt-ml ls [--select EXPR] [--resource-type {model,source,all}] [--output {name,json}]
dbt-ml show <model> [--limit N]                            # peek at a materialized table
dbt-ml source freshness                                    # mtime vs warn_after/error_after
dbt-ml docs generate [--output DIR]                        # static HTML site from manifest.json
dbt-ml docs serve [--port N]                               # local http.server over target/docs/
dbt-ml emit-dbt-sources [--output PATH]                    # write dbt-compatible sources.yml
dbt-ml clean                                               # delete the project's DuckDB

# Global flags (work on every command):
dbt-ml --project-dir <dir> --profiles-dir <dir> --target <name> <command>
```

### Useful flags

- `--watch` on `run` listens to source paths and re-runs on file changes
  (debounced 500ms). Ctrl-C to stop.
- `--threads N` parallelizes per-document extraction within an extraction
  model. Most useful for PDF / LLM / HTML (I/O- or API-bound). The LLM cache
  is lock-serialized so threading is safe.

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
      llm:
        provider: anthropic
        model: claude-haiku-4-5
        api_key_env: ANTHROPIC_API_KEY
        cache_path: ./target/llm_cache.duckdb
    prod:
      warehouse:
        type: duckdb
        path: "{{ env_var('DBT_ML_PROD_DB', '/data/prod/dbt_ml.duckdb') }}"
        schema: my_project_prod
      llm:
        model: claude-sonnet-4-6
        cache_path: /data/prod/llm_cache.duckdb
```

Lookup order: `--profiles-dir` flag → `$DBT_ML_PROFILES_DIR` →
`<project>/profiles.yml` → `~/.dbt_ml/profiles.yml`.

### BigQuery

Install the extra, then point a target at a GCP project. Profile fields
mirror dbt-bigquery, so an existing dbt profile ports over: auth via ADC
(`method: oauth`, the default), `keyfile:` (service-account),
`keyfile_json:` (inline/base64 JSON — CI-friendly with `env_var()`), or
`token`/`refresh_token` + client secrets (`oauth-secrets`), plus
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
        # keyfile: "{{ env_var('DBT_ML_BQ_KEYFILE') }}"   # optional; omit for ADC
```

Materialized tables, `--store-failures` tables, and incremental state all
live in the configured dataset — no DuckDB involved. `dbt-ml clean` drops
the whole dataset. `emit-dbt-sources` emits `database: <project>` /
`schema: <dataset>` so a dbt-bigquery project can consume the tables
directly.

String values support `{{ env_var('NAME') }}` and
`{{ env_var('NAME', 'default') }}` — the one piece of dbt's Jinja grammar
profiles need, so credentials and per-environment paths stay out of the file.
An unset variable with no default is a load-time error. Each `warehouse:`
block is validated against the config schema of the adapter named by `type:`;
unknown types and typo'd fields fail at resolve time with the adapter named.

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
service-account JSON in CI.

## Document extraction contract

Every extraction row carries identity, lineage, and parser provenance:
`document_id`, `source_path`, `source_uri` (local `file://` URI, or
`gs://bucket/name#generation` for GCS), `content_hash`, `code_version`,
`backend_name`, `backend_version` (the parsing library's version, e.g.
`pypdf/6.1`), and `extracted_at` (one UTC timestamp per run). Remote
sources add `source_metadata` JSON.

> Upgrading note: these columns are new — existing *incremental*
> extraction models will report a schema change on their next reprocess;
> run once with `--full-refresh` (or set
> `on_schema_change: append_new_columns`).

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

`sections` entries are `{level, heading, char_start, anchor?}`; `tables`
are `{index, char_start, n_rows, n_cols, cells}`. Domain-specific logic
(section taxonomy, speaker parsing) belongs in a transform layered after
extraction — the backends stay generic.

## Chunking (RAG)

A `chunk:` model splits an upstream document's text into one row per chunk —
the grain RAG and agent retrieval need. Chunk IDs are deterministic and
content-addressed, so an unchanged document re-runs to identical IDs (safe
for incremental MERGE into a warehouse/vector store).

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

The recommended document-layer shape (GCS raw files → BigQuery tables):

| model | grain | kind |
|-------|-------|------|
| `document_registry` | one row per document/version | `extraction` (`include_structure`) |
| `document_chunks`   | one row per chunk            | `chunk` |
| `document_extractions` | one row per structured field set | `extraction` (llm) or `transform` |

See `examples/rag_chunks_pipeline/` for a runnable registry → chunks project.
Domain keys (symbol, filing date, …) and embeddings belong in transforms /
downstream dbt models layered on top — the chunk grain stays generic.

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

## Classic text and document ML

Classic ML is a first-class dbt-ml lane alongside LLM/RAG work. The v0.2 design
adds an `ml:` model block for deterministic text and document workflows such as
Count/TF-IDF/hashing features, supervised classification/regression,
clustering, topic models, and NLP enrichment.

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
  - not_null: total, severity: warn      # warn doesn't fail the run
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

Each example is runnable end-to-end with `uv run dbt-ml --project-dir examples/<name> ...`.

## Composing with dbt (dbt-duckdb)

dbt-ml and dbt can share a DuckDB file: dbt-ml does the unstructured→structured
"E", dbt does the SQL "T". The bridge:

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

Every `dbt-ml compile` / `dbt-ml run` writes to `target/`:

- **`manifest.json`** — project, sources, models, refs, tags, `code_version` per
  model, DAG nodes+edges+execution order. Re-generated each run.
- **`run_results.json`** — per-model documents processed/skipped, rows written,
  duration, errors.
- **`sources.yml`** — only when you call `emit-dbt-sources`. dbt-shaped.
- **`docs/`** — static HTML site (`dbt-ml docs generate`) with project overview,
  Mermaid DAG, per-model pages. Serve locally with `dbt-ml docs serve`.

External tools (lineage viewers, CI dashboards, the dbt-consumer above)
consume these.

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

Linear through 5k. Bottleneck is single-threaded extraction; parallelism is a v2 item.

## Layout

```
src/dbt_ml/
├── cli.py                 # click: init/seed/compile/graph/run/test/show/clean/source freshness/emit-dbt-sources
├── config/                # pydantic models for project/source/model/profile + loader
├── profile.py             # profile discovery + resolution (warehouse + llm)
├── dag.py                 # graphlib-based DAG, selectors (+ name +, tag:foo), Mermaid render
├── state.py               # DuckDB-backed incremental state
├── runner.py              # extract → materialize orchestration
├── manifest.py            # target/manifest.json + run_results.json
├── dbt_export.py          # target/sources.yml (dbt-shaped)
├── freshness.py           # source mtime check
├── backends/              # json, markdown, pdf, html, llm
├── transforms/runner.py   # loads user Python transform modules + TransformContext
├── checks/                # schema tests + custom Python tests + severity
├── synth/                 # synthetic data generators per shape
└── templates/             # init scaffolds for {json,pdf,markdown,html}
```

## Roadmap

**v0.2 — RAG + warehouse adapter pattern.** Tracked in GitHub issues
tagged `roadmap`. The four headline pieces:

1. **Warehouse adapter pattern** matching dbt-core's set. v0.2 starts with
   DuckDB (current) + LanceDB (lakehouse-style vector store); subsequent
   versions add Postgres, then Snowflake / BigQuery / Databricks / Redshift.
2. **Chunking primitives** as a first-class model kind: recursive (default),
   token-aware, layout-aware, optional Anthropic Contextual Retrieval
   (49–67% retrieval failure reduction per published numbers).
3. **Embedding primitives** as a first-class model kind: Voyage, Cohere,
   OpenAI, and local sentence-transformers providers. Same cache mechanic
   as today's LLM backend so re-runs are free.
4. **Layout-aware OSS parsers** as additional backends: Docling (privacy +
   table quality), Marker (best OSS layout fidelity).

**Deferred beyond v0.2:**

- Rust CLI + PyO3 bridge.
- Metaxy integration (replace `state.py` with `MetadataStore`).
- Field-level lineage (`version_from: [ref('x').field_a]`).
- Parallel *model* execution (today's `--threads` parallelizes within a model).
- Managed parser backends (Reducto, Mistral OCR 3, LlamaParse) — generic
  remote-parser adapter pattern when there's a real ask.
- Reranker hooks (Cohere Rerank, Voyage Rerank).
- Multi-LLM-provider adapters (Bedrock, Vertex, OpenAI structured output).
- PII detection / redaction (Microsoft Presidio).
- Ragas integration (`dbt-ml eval`).
