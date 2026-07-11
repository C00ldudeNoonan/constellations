from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import Any

from .config import load_project
from .config.model import ModelConfig
from .dag import ProjectDAG, parse_ref
from .profile import ProfileError, resolve_profile
from .runner import ModelRunResult
from .versioning import compute_code_version

MANIFEST_VERSION = 1
MANIFEST_FILENAME = "manifest.json"
RUN_RESULTS_FILENAME = "run_results.json"


def build_manifest(
    project_dir: Path,
    *,
    target: str | None = None,
    profiles_dir: Path | None = None,
) -> dict[str, Any]:
    project, sources, models = load_project(project_dir)
    resolved = resolve_profile(
        project, project_dir, target=target, profiles_dir=profiles_dir
    )
    dag = ProjectDAG(sources, models)

    return {
        "manifest_version": MANIFEST_VERSION,
        "generated_at": _now(),
        "project": {
            "name": project.name,
            "version": project.version,
            "profile": resolved.profile_name,
            "target": resolved.target_name,
            "duckdb_path": resolved.warehouse.storage_location(),
            "duckdb_schema": resolved.warehouse.schema_name,
        },
        "sources": [
            {
                "name": s.name,
                "description": s.description,
                "path": s.path,
                "file_pattern": s.file_pattern,
                "recursive": s.recursive,
                "external": s.external,
                "tags": s.tags,
            }
            for s in sources
        ],
        "models": [_model_dict(m, project_dir) for m in models],
        "dag": {
            "execution_order": dag.execution_order(),
            "nodes": [
                {"name": n, "kind": dag.nodes[n].kind.value}
                for n in dag.nodes
            ],
            "edges": [
                [pred, succ]
                for succ, preds in dag.predecessors.items()
                for pred in preds
            ],
        },
    }


def write_manifest(
    project_dir: Path,
    *,
    target: str | None = None,
    profiles_dir: Path | None = None,
) -> Path:
    project, _, _ = load_project(project_dir)
    target_dir = (project_dir / project.target_path).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    out = target_dir / MANIFEST_FILENAME
    out.write_text(
        json.dumps(
            build_manifest(project_dir, target=target, profiles_dir=profiles_dir),
            indent=2,
        )
    )
    return out


def build_run_results(
    project_dir: Path,
    results: list[ModelRunResult],
    *,
    target: str | None = None,
    profiles_dir: Path | None = None,
    invocation: str = "run",
    skipped: list[str] | None = None,
    elapsed_seconds: float | None = None,
    test_failures: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Assemble the run_results.json payload: run-level metadata (target,
    counts, status) plus per-model status and the fully-qualified relation each
    model materialized into. This is the contract Dagster reads to attach
    materialization metadata (issue #87).

    `test_failures` maps model_name → hard-failed test labels (from `build`). A
    model that materialized fine but failed a test has empty `errors`, so its
    status must be derived here too — otherwise a failing test on a leaf model
    (no skipped descendants) would report `success` while the command exits 1."""
    skipped = skipped or []
    test_failures = test_failures or {}
    target_block = _target_block(project_dir, target=target, profiles_dir=profiles_dir)
    catalog = target_block["catalog"] if target_block else None
    schema = target_block["schema"] if target_block else None
    _, sources, models = load_project(project_dir)
    dag = ProjectDAG(sources, models)
    model_names = {model.name for model in models}
    considered_models = [
        name
        for name in [*(r.model_name for r in results), *skipped]
        if name in model_names
    ]
    sources_considered = dag.required_sources(considered_models)

    result_rows: list[dict[str, Any]] = []
    n_error = 0
    for r in results:
        failed = test_failures.get(r.model_name, [])
        status = "error" if (r.errors or failed) else "success"
        n_error += status == "error"
        row = asdict(r)
        row["status"] = status
        row["test_failures"] = failed
        row["relation"] = _relation(catalog, schema, r.model_name)
        result_rows.append(row)

    for name in skipped:
        row = asdict(ModelRunResult(model_name=name, materialization="", kind="skipped"))
        row["status"] = "skipped"
        row["test_failures"] = []
        row["relation"] = _relation(catalog, schema, name)
        result_rows.append(row)

    overall = "error" if (n_error or skipped) else "success"
    metadata: dict[str, Any] = {
        "dbt_ml_version": _dbt_ml_version(),
        "generated_at": _now(),
        "invocation": invocation,
        "status": overall,
        "elapsed_seconds": elapsed_seconds,
        "target": target_block,
        "sources_considered": sources_considered,
        "counts": {
            "total": len(results) + len(skipped),
            "success": len(results) - n_error,
            "error": n_error,
            "skipped": len(skipped),
        },
    }
    return {"metadata": metadata, "results": result_rows}


def write_run_results(
    project_dir: Path,
    results: list[ModelRunResult],
    *,
    target: str | None = None,
    profiles_dir: Path | None = None,
    invocation: str = "run",
    skipped: list[str] | None = None,
    elapsed_seconds: float | None = None,
    test_failures: dict[str, list[str]] | None = None,
) -> Path:
    project, _, _ = load_project(project_dir)
    target_dir = (project_dir / project.target_path).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    payload = build_run_results(
        project_dir,
        results,
        target=target,
        profiles_dir=profiles_dir,
        invocation=invocation,
        skipped=skipped,
        elapsed_seconds=elapsed_seconds,
        test_failures=test_failures,
    )
    out = target_dir / RUN_RESULTS_FILENAME
    out.write_text(json.dumps(payload, indent=2))
    return out


def _target_block(
    project_dir: Path,
    *,
    target: str | None,
    profiles_dir: Path | None,
) -> dict[str, Any] | None:
    """Warehouse target descriptor for run_results. Returns None if the profile
    can't be resolved (never fail a completed run just to describe its target)."""
    project, _, _ = load_project(project_dir)
    try:
        resolved = resolve_profile(
            project, project_dir, target=target, profiles_dir=profiles_dir
        )
    except ProfileError:
        return None
    w = resolved.warehouse
    return {
        "profile": resolved.profile_name,
        "name": resolved.target_name,
        "adapter_type": w.type,
        "schema": w.schema_name,
        "catalog": w.catalog_name(),
        "location": w.storage_location(),
    }


def _relation(
    catalog: str | None, schema: str | None, name: str
) -> dict[str, Any]:
    parts = [p for p in (catalog, schema, name) if p]
    return {
        "catalog": catalog,
        "schema": schema,
        "name": name,
        "fully_qualified": ".".join(parts),
    }


def _dbt_ml_version() -> str:
    try:
        return _pkg_version("dbt-ml")
    except PackageNotFoundError:
        return "unknown"


def _model_dict(model: ModelConfig, project_dir: Path) -> dict[str, Any]:
    if model.extraction is not None:
        kind = "extraction"
    elif model.ml is not None:
        kind = "ml"
    elif model.transform is not None:
        kind = "transform"
    elif model.chunk is not None:
        kind = "chunk"
    else:
        kind = "unknown"

    return {
        "name": model.name,
        "description": model.description,
        "kind": kind,
        "materialization": model.materialization,
        "tags": model.tags,
        "source": model.source,
        "depends_on": model.depends_on or [],
        "extraction": model.extraction.model_dump() if model.extraction else None,
        "transform": model.transform.model_dump() if model.transform else None,
        "ml": model.ml.model_dump(mode="json") if model.ml else None,
        "chunk": model.chunk.model_dump() if model.chunk else None,
        "fields": [f.model_dump() for f in model.fields],
        "tests": model.tests,
        "code_version": compute_code_version(
            extraction=model.extraction,
            transform=model.transform,
            ml=model.ml,
            chunk=model.chunk,
            depends_on=_code_version_depends_on(model),
            fields=model.fields,
            project_dir=project_dir,
        ),
    }


def _code_version_depends_on(model: ModelConfig) -> list[str] | None:
    if model.chunk is None or not model.depends_on:
        return None
    return [parse_ref(dep) for dep in model.depends_on]


class StateError(Exception):
    pass


def read_state_code_versions(state_path: Path) -> dict[str, str]:
    """Read {model_name: code_version} from a previous manifest.

    `state_path` may be the manifest.json itself or a directory containing
    one (dbt's --state convention).
    """
    path = state_path / MANIFEST_FILENAME if state_path.is_dir() else state_path
    if not path.exists():
        raise StateError(
            f"No manifest found at {path}. Point --state at a manifest.json "
            "(or its directory) written by a previous `compile` or `run`."
        )
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise StateError(f"{path} is not valid JSON: {e}") from e
    models = data.get("models")
    if not isinstance(models, list):
        raise StateError(f"{path} has no `models` list; is it a dbt-ml manifest?")
    return {
        m["name"]: m.get("code_version", "")
        for m in models
        if isinstance(m, dict) and "name" in m
    }


def compute_modified_models(
    models: list[ModelConfig], project_dir: Path, state_path: Path
) -> set[str]:
    """Models whose code_version differs from the state manifest, or that
    the state manifest has never seen."""
    previous = read_state_code_versions(state_path)
    modified: set[str] = set()
    for model in models:
        current = compute_code_version(
            extraction=model.extraction,
            transform=model.transform,
            ml=model.ml,
            chunk=model.chunk,
            fields=model.fields,
            project_dir=project_dir,
        )
        if previous.get(model.name) != current:
            modified.add(model.name)
    return modified


def _now() -> str:
    return datetime.now(UTC).isoformat()
