from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import load_project
from .config.model import ModelConfig
from .dag import ProjectDAG
from .profile import resolve_profile
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
            "duckdb_path": str(resolved.warehouse.path),
            "duckdb_schema": resolved.warehouse.schema_name,
        },
        "sources": [
            {
                "name": s.name,
                "description": s.description,
                "path": s.path,
                "file_pattern": s.file_pattern,
                "recursive": s.recursive,
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


def write_run_results(project_dir: Path, results: list[ModelRunResult]) -> Path:
    project, _, _ = load_project(project_dir)
    target_dir = (project_dir / project.target_path).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": _now(),
        "results": [asdict(r) for r in results],
    }
    out = target_dir / RUN_RESULTS_FILENAME
    out.write_text(json.dumps(payload, indent=2))
    return out


def _model_dict(model: ModelConfig, project_dir: Path) -> dict[str, Any]:
    if model.extraction is not None:
        kind = "extraction"
    elif model.ml is not None:
        kind = "ml"
    elif model.transform is not None:
        kind = "transform"
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
        "fields": [f.model_dump() for f in model.fields],
        "tests": model.tests,
        "code_version": compute_code_version(
            extraction=model.extraction,
            transform=model.transform,
            ml=model.ml,
            project_dir=project_dir,
        ),
    }


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
            project_dir=project_dir,
        )
        if previous.get(model.name) != current:
            modified.add(model.name)
    return modified


def _now() -> str:
    return datetime.now(UTC).isoformat()
