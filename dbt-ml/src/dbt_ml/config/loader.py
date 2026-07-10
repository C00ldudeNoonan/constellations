from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from .model import ModelConfig, ModelFile
from .project import ProjectConfig
from .source import SourceConfig, SourceFile


class ConfigError(Exception):
    pass


def load_project(
    project_dir: Path,
) -> tuple[ProjectConfig, list[SourceConfig], list[ModelConfig]]:
    project_path = project_dir / "dbt_ml_project.yml"
    if not project_path.exists():
        raise ConfigError(f"No dbt_ml_project.yml found at {project_path}")

    project = _parse_yaml(project_path, ProjectConfig)

    # Local import: paths.py imports ConfigError from this module.
    from ..paths import resolve_within_project

    # Layout paths always stay inside the project — there is no sane reason
    # for models/ or target/ to live outside the repo, and no opt-in (#65).
    layout: list[tuple[str, Path]] = [
        *(("source-paths", p) for p in project.source_paths),
        *(("model-paths", p) for p in project.model_paths),
        *(("transform-paths", p) for p in project.transform_paths),
        ("target-path", project.target_path),
    ]
    for label, layout_path in layout:
        resolve_within_project(layout_path, project_dir, surface=f"`{label}`")

    sources: list[SourceConfig] = []
    for source_dir in project.source_paths:
        sources.extend(_load_yaml_dir(project_dir / source_dir, SourceFile, lambda f: f.sources))

    models: list[ModelConfig] = []
    for model_dir in project.model_paths:
        models.extend(_load_yaml_dir(project_dir / model_dir, ModelFile, lambda f: f.models))

    return project, sources, models


def _parse_yaml[T](path: Path, model: type[T]) -> T:
    try:
        with path.open() as f:
            data: Any = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise ConfigError(f"Malformed YAML at {path}:\n{e}") from e
    try:
        return model.model_validate(data)  # type: ignore[attr-defined,no-any-return]
    except ValidationError as e:
        raise ConfigError(f"Invalid YAML at {path}:\n{e}") from e


def _load_yaml_dir[F, I](directory: Path, file_model: type[F], extract: Any) -> list[I]:
    if not directory.exists():
        return []
    out: list[I] = []
    for path in sorted(directory.glob("**/*.yml")):
        parsed = _parse_yaml(path, file_model)
        out.extend(extract(parsed))
    return out
