"""Classic ML model executor (issue #190).

Thin orchestration seam over classic_ml: runs training, materializes the
primary and secondary relations, and publishes (or discards) the artifact
atomically. runner.py re-exports run_ml_model for compatibility.
"""

from __future__ import annotations

from pathlib import Path

from ..adapters import WarehouseAdapter
from ..classic_ml import run_classic_ml_model
from ..config.model import ModelConfig
from ..config.project import ProjectConfig
from .contracts import ModelRunResult, RunError
from .warehouse import warehouse_options


def run_ml_model(
    *,
    model: ModelConfig,
    project: ProjectConfig,
    project_dir: Path,
    adapter: WarehouseAdapter,
) -> ModelRunResult:
    assert model.ml is not None
    if model.materialization == "incremental":
        raise RunError(
            f"ML model '{model.name}' declares `materialization: incremental`, "
            "but ML models only support `full` today. Set `materialization: full` "
            "(or omit it) — see issue #53."
        )
    output = None
    try:
        output = run_classic_ml_model(
            model=model,
            project=project,
            project_dir=project_dir,
            adapter=adapter,
        )
        rows_written = adapter.materialize_full(
            model.name, output.df, options=warehouse_options(adapter, model)
        )
        for suffix, table_df in output.secondary_tables.items():
            adapter.materialize_full(
                f"{model.name}__{suffix}",
                table_df,
                options=warehouse_options(adapter, model),
            )
        output.publish_artifact()
    except BaseException as e:
        if output is not None:
            output.discard_staged_artifact()
        if not isinstance(e, Exception):
            raise
        raise RunError(f"ML model '{model.name}' failed: {e}") from e

    return ModelRunResult(
        model_name=model.name,
        materialization=model.materialization,
        kind="ml",
        rows_written=rows_written,
        artifact_path=str(output.artifact_path),
        artifact_version=output.artifact_version,
        training_input=output.training_input,
        metrics=output.metrics,
        artifact_metadata=output.artifact_metadata,
    )
