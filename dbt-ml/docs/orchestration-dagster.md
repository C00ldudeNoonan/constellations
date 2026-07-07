# Orchestrating dbt-ml with Dagster

dbt-ml doesn't need a deep Dagster integration to be operable inside a Dagster
project. It exposes a stable **CLI + artifact contract** so a Dagster asset can
launch a run as a subprocess, fail loudly on error, read run metadata, and hand
its output tables to downstream dbt models. This page sketches that pattern; it
is intentionally a code sketch, not a runnable example, and adds no Dagster
dependency to dbt-ml itself.

## The contract

### Exit codes

`dbt-ml run` (and `build`) return a status code an orchestrator can branch on:

| Code | Meaning |
| ---- | ------- |
| `0`  | Success — every selected model ran clean (and, for `build`, tests passed). |
| `1`  | Run failure — the run started but a model errored, a document failed to extract, or a test hard-failed. |
| `2`  | Configuration/usage error — the project couldn't be set up: malformed YAML, DAG cycle, bad selector, unresolved profile, or bad `--state`. |

Distinguishing `2` from `1` lets you alert differently: a `2` is "someone broke
the project definition," a `1` is "a run had bad data."

### Artifacts

Every `run`/`build` writes two files under `target/`:

- `manifest.json` — project structure: sources, models, kinds, DAG, per-model
  `code_version`.
- `run_results.json` — what the run did. With `--json`, the identical payload is
  printed to stdout so you can capture it without reading the file.

`run_results.json` shape (abridged):

```json
{
  "metadata": {
    "dbt_ml_version": "0.2.7",
    "invocation": "run",
    "status": "success",
    "elapsed_seconds": 0.94,
    "target": {
      "profile": "invoice_pipeline", "name": "dev",
      "adapter_type": "duckdb", "schema": "dbt_ml",
      "catalog": "dbt_ml", "location": "/abs/path/dbt_ml.duckdb"
    },
    "counts": {"total": 3, "success": 3, "error": 0, "skipped": 0}
  },
  "results": [
    {
      "model_name": "raw_invoices", "kind": "extraction",
      "status": "success",
      "documents_processed": 8, "documents_skipped": 0,
      "documents_deleted": 0, "rows_written": 8, "errors": [],
      "relation": {
        "catalog": "dbt_ml", "schema": "dbt_ml", "name": "raw_invoices",
        "fully_qualified": "dbt_ml.dbt_ml.raw_invoices"
      }
    }
  ]
}
```

`skipped` downstream models (a `build` where an upstream failed) appear in
`results` with `"status": "skipped"`.

## A Dagster asset wrapper

Invoke dbt-ml as a subprocess with `--json`, branch on the exit code, and attach
the run metadata to the materialization.

```python
import json
import subprocess
from pathlib import Path

from dagster import AssetExecutionContext, MaterializeResult, asset

DBT_ML_PROJECT = Path("/opt/pipelines/document_extraction")


@asset
def dbt_ml_documents(context: AssetExecutionContext) -> MaterializeResult:
    proc = subprocess.run(
        ["uv", "run", "dbt-ml", "--project-dir", str(DBT_ML_PROJECT), "run", "--json"],
        capture_output=True,
        text=True,
    )

    if proc.returncode == 2:
        raise RuntimeError(f"dbt-ml project is misconfigured:\n{proc.stderr}")
    if proc.returncode == 1:
        # Surface the run so the failure is visible, then fail the asset.
        context.log.error(proc.stdout or proc.stderr)
        raise RuntimeError("dbt-ml run failed (a model errored or a test failed)")

    payload = json.loads(proc.stdout)
    meta = payload["metadata"]
    rows = sum(r["rows_written"] for r in payload["results"])
    errored = sum(len(r["errors"]) for r in payload["results"])

    return MaterializeResult(
        metadata={
            "target": meta["target"]["location"],
            "adapter": meta["target"]["adapter_type"],
            "models": meta["counts"]["total"],
            "rows_written": rows,
            "documents_skipped": sum(r["documents_skipped"] for r in payload["results"]),
            "failed_documents": errored,
            "relations": [r["relation"]["fully_qualified"] for r in payload["results"]],
            "elapsed_seconds": meta["elapsed_seconds"],
        }
    )
```

For finer-grained lineage, emit one asset per dbt-ml model by iterating
`manifest.json`'s `models` and keying each `MaterializeResult` off the matching
entry in `run_results.json`.

## Handing off to downstream dbt

`emit-dbt-sources` writes a dbt-parseable `sources.yml` declaring the dbt-ml
tables. Point it at a dbt project's sources directory and downstream dbt models
`{{ source(...) }}` straight into dbt-ml output — no glue:

```bash
uv run dbt-ml --project-dir /opt/pipelines/document_extraction \
  emit-dbt-sources \
  --output /opt/pipelines/dbt/models/sources/_dbt_ml_sources.yml
```

Run this as its own asset (or the tail of the run asset) so it lands before the
dbt build. A Dagster `@dbt_assets` graph then depends on the generated source,
giving you dbt-ml → dbt lineage in one asset graph. See
`examples/dbt_consumer/` for the round-trip proof that the emitted `sources.yml`
is dbt-parseable and that dbt models read dbt-ml tables directly.

## Sketch of the asset graph

```
dbt_ml_documents ──> emit_dbt_ml_sources ──> @dbt_assets (staging → marts)
     (run --json)        (sources.yml)              ({{ source('dbt_ml_*', ...) }})
```
