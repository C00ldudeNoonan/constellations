# dbt_embed_duckdb — dbt-ml embedded inside a dbt-duckdb DAG (#177 prototype)

This is the round-trip proof for the "dbt drives dbt-ml" integration (design
issue [#177]): a **single `dbt build`** runs a dbt-ml extraction model as a dbt
Python model, tests it with native dbt tests, and feeds it into a downstream dbt
SQL model — one DAG, one lineage graph, **no orchestrator**.

Unlike [`../dbt_consumer`](../dbt_consumer) (which reads dbt-ml tables as external
`sources` after a separate `dbt-ml run`), here dbt-ml runs *inside* dbt:

```
models/dbt_ml/raw_invoices.py   →  def model(dbt, session):
                                       from dbt_ml.dbt_embed import materialize
                                       return materialize("raw_invoices", ...).to_arrow()
models/marts/invoice_facts.sql  →  select ... from {{ ref('raw_invoices') }}
```

`raw_invoices` is a real dbt node, so `ref()` works across the two worlds and
`dbt docs` shows one graph.

## How it works

`dbt_ml.dbt_embed.materialize()` reuses the standalone runner's single-model path
but swaps a `CaptureAdapter` in for the warehouse: dbt-ml does document discovery,
extraction, and (for `llm:` models) its response cache in-process, then hands dbt
the resulting frame. **dbt owns materialization, tests, docs, and lineage; dbt-ml
owns the unstructured→structured extraction.** No dbt-ml table or state is written
to the dbt database.

Under the chosen authoring surface (A1), the `.py` shim and `schema.yml` here are
what `dbt-ml codegen` will generate from the dbt-ml YAML; they're checked in so the
example runs without the (not-yet-built) codegen step.

## Run it

From `examples/dbt_embed_duckdb/`:

```bash
# Seed sample documents into the colocated dbt-ml project (once)
uv run --project ../.. dbt-ml --project-dir ../invoice_pipeline seed --count 8

# Install this dbt project's env (dbt-duckdb + editable dbt-ml) and build
uv sync
DBT_ML_PROJECT_DIR="$(cd ../invoice_pipeline && pwd)" uv run dbt build --profiles-dir .
```

Expected: `PASS=6` — the Python extraction model, four schema tests, and the SQL
mart all succeed, materializing `raw_invoices` and `invoice_facts` into one
`target/embedded.duckdb`.

## Prototype scope / caveats

- **dbt-duckdb only.** Python models on warehouse-side runtimes (Snowpark,
  BigQuery/Dataproc) sandbox network egress, so extraction backends that call
  Anthropic or read files can't run there. Other warehouses keep using the
  standalone CLI + `emit-dbt-sources` handoff.
- **Extraction/transform models.** The bidirectional `dbt_ref` source (a dbt-ml
  model consuming a dbt table) and `dbt-ml codegen` are the next slices in #177.
- **No dbt-ml-side incremental** in embedded mode; the LLM response cache still
  makes re-runs cheap under dbt full-refresh.
- `DBT_ML_PROJECT_DIR` locates the colocated dbt-ml project because dbt-duckdb
  copies Python models to a temp file (so `__file__` is unreliable); codegen will
  bake in a project-relative path.

[#177]: https://github.com/C00ldudeNoonan/dbt-ml/issues/177
