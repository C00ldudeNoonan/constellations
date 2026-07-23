"""Library entry point for running one dbt-ml model from a dbt Python model.

`materialize(model, project_dir=..., session=...)` runs a single dbt-ml model's
extraction/transform in-process and returns its output frame for dbt to
materialize. It reuses the standalone runner's single-model path (source
discovery + `_run_model`) but swaps a `CaptureAdapter` in for the real warehouse,
so no dbt-ml-owned table or state is written — dbt owns the target.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from ..compiler import validate_project_contract
from ..config import load_project
from ..dag import ProjectDAG
from ..profile import apply_source_path_overrides, resolve_profile
from ..runner import _discover_sources, _run_model
from .adapter import CaptureAdapter


def materialize(
    model: str,
    *,
    project_dir: str | Path,
    session: Any | None = None,
    upstreams: dict[str, pl.DataFrame] | None = None,
    target: str | None = None,
    profiles_dir: str | Path | None = None,
) -> pl.DataFrame:
    """Run dbt-ml model `model` and return its output as a polars DataFrame.

    Parameters mirror what a dbt Python model can supply: `project_dir` points at
    the colocated dbt-ml project; `session` is the dbt-duckdb connection (reserved
    for the bidirectional `dbt_ref` path). `upstreams` maps each dbt-ml dependency
    name to the frame the dbt Python model read from `dbt.ref(...)` — transform
    models read their inputs from here instead of a dbt-ml warehouse.
    `target`/`profiles_dir` select the dbt-ml profile that carries backend/LLM
    configuration.
    """
    project_path = Path(project_dir)
    project, sources, models = load_project(project_path)
    dag: ProjectDAG = validate_project_contract(project, sources, models, project_path)

    models_by_name = {m.name: m for m in models}
    if model not in models_by_name:
        available = ", ".join(sorted(models_by_name)) or "(none)"
        raise ValueError(
            f"Model '{model}' is not defined in dbt-ml project "
            f"'{project.name}'. Available models: {available}"
        )

    resolved = resolve_profile(
        project,
        project_path,
        target=target,
        profiles_dir=Path(profiles_dir) if profiles_dir is not None else None,
    )

    # Apply the selected target's source_paths overrides before discovery, so an
    # embedded build scans the same dev/prod roots the standalone runner would.
    sources = apply_source_path_overrides(sources, resolved)

    required_sources = set(dag.required_sources([model]))
    source_docs = _discover_sources(
        [source for source in sources if source.name in required_sources],
        project_path,
    )

    with CaptureAdapter(
        schema=resolved.warehouse.schema_name, upstreams=upstreams
    ) as adapter:
        _run_model(
            model=models_by_name[model],
            models_by_name=models_by_name,
            project=project,
            project_dir=project_path,
            source_docs=source_docs,
            adapter=adapter,
            resolved=resolved,
            full_refresh=True,
        )
        return adapter.captured
