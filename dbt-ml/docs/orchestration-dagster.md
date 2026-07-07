# Orchestrating dbt-ml with Dagster (dagster-dbt)

dbt-ml is a dbt-shaped package: its materialized tables are declared as dbt
**sources** (`dbt-ml emit-dbt-sources`). That means it slots straight into the
[`dagster-dbt`](https://docs.dagster.io/integrations/libraries/dbt) integration —
when your dbt project is loaded with `@dbt_assets`, any model that does
`{{ source('dbt_ml_<project>', '<table>') }}` gets an upstream asset dependency,
and dbt-ml is the asset that produces it. No custom lineage glue.

This page shows that native wiring. dbt-ml itself stays pure Python and gains no
Dagster dependency; the code below lives in your Dagster project (e.g.
`economic-data-project`).

## How the keys line up

`dagster-dbt`'s default asset key for a dbt source table is
`[source_name, table_name]`. So for the source dbt-ml emits — `dbt_ml_<project>` —
table `raw_invoices` becomes the asset key `dbt_ml_<project>/raw_invoices`.

`dbt-ml emit-dbt-sources --dagster-meta` pins that exact key onto each table via
`meta.dagster.asset_key`, so the producer (dbt-ml) and the consumers (your dbt
models) agree on one key without you hand-copying anything. Pure dbt ignores the
`meta`, so the same file still works in a plain dbt project.

```yaml
# _dbt_ml_sources.yml (emitted)
sources:
  - name: dbt_ml_invoice_pipeline
    tables:
      - name: raw_invoices
        meta:
          dagster:
            asset_key: [dbt_ml_invoice_pipeline, raw_invoices]
```

## Step 1 — emit sources into the dbt project

Run this **before** Dagster loads the dbt manifest (the `@dbt_assets` decorator
reads the manifest at definition time, so the source must exist when `dbt parse`
runs). A natural home is the `DbtProject` prepare step, a Makefile, or CI:

```bash
uv run dbt-ml --project-dir /opt/pipelines/document_extraction \
  emit-dbt-sources --dagster-meta \
  --output /opt/pipelines/dbt/models/sources/_dbt_ml_sources.yml
```

## Step 2 — load the dbt project

Standard `dagster-dbt` setup:

```python
from pathlib import Path

from dagster import AssetExecutionContext
from dagster_dbt import DbtCliResource, DbtProject, dbt_assets

dbt_project = DbtProject(project_dir=Path("/opt/pipelines/dbt"))
dbt_project.prepare_if_dev()


@dbt_assets(manifest=dbt_project.manifest_path)
def dbt_models(context: AssetExecutionContext, dbt: DbtCliResource):
    yield from dbt.cli(["build"], context=context).stream()
```

## Step 3 — make dbt-ml the producer of the source assets

Model the dbt-ml run as one `@multi_asset` whose outputs are exactly the source
asset keys dbt-ml feeds. `get_asset_keys_by_output_name_for_source` reads those
keys back out of the dbt manifest, so the two sides can never drift. The op runs
`dbt-ml run --json` once and attaches per-table metadata from `run_results.json`.

```python
import json
import subprocess

from dagster import AssetOut, MaterializeResult, multi_asset
from dagster_dbt import get_asset_keys_by_output_name_for_source

DBT_ML_PROJECT = "/opt/pipelines/document_extraction"
SOURCE_NAME = "dbt_ml_invoice_pipeline"

_keys_by_output = get_asset_keys_by_output_name_for_source([dbt_models], SOURCE_NAME)


@multi_asset(
    outs={name: AssetOut(key=key) for name, key in _keys_by_output.items()},
    can_subset=True,
)
def dbt_ml_documents(context: AssetExecutionContext):
    proc = subprocess.run(
        ["uv", "run", "dbt-ml", "--project-dir", DBT_ML_PROJECT, "run", "--json"],
        capture_output=True,
        text=True,
    )
    # 2 = misconfigured project, 1 = a model/document/test failed, 0 = ok.
    if proc.returncode == 2:
        raise RuntimeError(f"dbt-ml project is misconfigured:\n{proc.stderr}")
    if proc.returncode == 1:
        context.log.error(proc.stdout or proc.stderr)
        raise RuntimeError("dbt-ml run failed")

    payload = json.loads(proc.stdout)
    by_model = {r["model_name"]: r for r in payload["results"]}
    target = payload["metadata"]["target"]

    for output_name, key in _keys_by_output.items():
        # output_name mirrors the source table name, which is the dbt-ml model.
        r = by_model.get(key.path[-1], {})
        yield MaterializeResult(
            asset_key=key,
            metadata={
                "relation": r.get("relation", {}).get("fully_qualified"),
                "rows_written": r.get("rows_written", 0),
                "documents_processed": r.get("documents_processed", 0),
                "documents_skipped": r.get("documents_skipped", 0),
                "failed_documents": len(r.get("errors", [])),
                "failed_tests": len(r.get("test_failures", [])),
                "warehouse": target["adapter_type"],
            },
        )
```

Wire it all together:

```python
from dagster import Definitions

defs = Definitions(
    assets=[dbt_ml_documents, dbt_models],
    resources={"dbt": DbtCliResource(project_dir=dbt_project)},
)
```

Dagster now shows one asset graph: `dbt_ml_documents` (per table) → the dbt models
that `{{ source(...) }}` them → downstream marts.

### Single-table sources

If the dbt-ml source has one table, skip the multi-asset and use the single-key
helper:

```python
from dagster import asset
from dagster_dbt import get_asset_key_for_source

@asset(key=get_asset_key_for_source([dbt_models], "dbt_ml_invoice_pipeline"))
def dbt_ml_documents(context): ...
```

## Why the exit codes matter here

The subprocess branches on dbt-ml's status codes:

| Code | Meaning | Dagster reaction |
| ---- | ------- | ---------------- |
| `0`  | Success | materialize with metadata |
| `1`  | A model/document/test failed | fail the asset loudly |
| `2`  | Misconfigured project (bad YAML, DAG cycle, bad selector, profile) | fail with a distinct message |

## Related

- `examples/dbt_consumer/` — the round-trip proof that dbt-ml's emitted
  `sources.yml` is dbt-parseable and that dbt models read dbt-ml tables directly.
- `emit-dbt-sources` reference in the main `README.md`.
