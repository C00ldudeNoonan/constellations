from __future__ import annotations

import os
import stat
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
    if not project_path.exists() and not project_path.is_symlink():
        raise ConfigError(f"No dbt_ml_project.yml found at {project_path}")
    _validate_config_file(
        project_path,
        allowed_root=project_dir,
        description="Project configuration",
    )

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
    if project.profile is None:
        resolve_within_project(
            project.duckdb.path,
            project_dir,
            surface="Legacy inline `duckdb.path`",
            hint="Declare a profile and put external warehouse paths in profiles.yml.",
        )

    sources: list[SourceConfig] = []
    for source_dir in project.source_paths:
        configured = project_dir / source_dir
        resolved = resolve_within_project(
            source_dir, project_dir, surface="`source-paths`"
        )
        _reject_symlink_directory_components(
            configured, project_dir, description="Source configuration root"
        )
        sources.extend(
            _load_yaml_dir(
                resolved,
                SourceFile,
                lambda f: f.sources,
                description="Source configuration",
            )
        )

    models: list[ModelConfig] = []
    for model_dir in project.model_paths:
        configured = project_dir / model_dir
        resolved = resolve_within_project(
            model_dir, project_dir, surface="`model-paths`"
        )
        _reject_symlink_directory_components(
            configured, project_dir, description="Model configuration root"
        )
        models.extend(
            _load_yaml_dir(
                resolved,
                ModelFile,
                lambda f: f.models,
                description="Model configuration",
            )
        )

    return project, sources, models


def _parse_yaml[T](path: Path, model: type[T]) -> T:
    try:
        with path.open() as f:
            data: Any = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise ConfigError(f"Malformed YAML at {path}:\n{e}") from e
    except OSError as e:
        raise ConfigError(f"Could not read YAML configuration file {path}: {e}") from e
    try:
        return model.model_validate(data)  # type: ignore[attr-defined,no-any-return]
    except ValidationError as e:
        raise ConfigError(f"Invalid YAML at {path}:\n{e}") from e


def _load_yaml_dir[F, I](
    directory: Path,
    file_model: type[F],
    extract: Any,
    *,
    description: str,
) -> list[I]:
    if not directory.exists():
        return []
    if directory.is_symlink():
        raise ConfigError(
            f"Refusing to traverse symlinked {description.lower()} directory: "
            f"{directory}"
        )
    if not directory.is_dir():
        raise ConfigError(f"{description} root is not a directory: {directory}")

    def _walk_error(error: OSError) -> None:
        path = error.filename or str(directory)
        raise ConfigError(
            f"Could not traverse {description.lower()} directory {path}: {error}"
        ) from error

    out: list[I] = []
    for root, dirnames, filenames in os.walk(
        directory,
        topdown=True,
        onerror=_walk_error,
        followlinks=False,
    ):
        root_path = Path(root)
        dirnames.sort()
        filenames.sort()
        for dirname in dirnames:
            candidate = root_path / dirname
            if candidate.is_symlink():
                raise ConfigError(
                    f"Refusing to traverse symlinked {description.lower()} "
                    f"directory: {candidate}"
                )
        for filename in filenames:
            if not filename.endswith(".yml"):
                continue
            path = root_path / filename
            _validate_config_file(
                path,
                allowed_root=directory,
                description=description,
            )
            parsed = _parse_yaml(path, file_model)
            out.extend(extract(parsed))
    return out


def _validate_config_file(
    path: Path,
    *,
    allowed_root: Path,
    description: str,
) -> None:
    if path.is_symlink():
        raise ConfigError(
            f"Refusing to load {description.lower()} symlink {path}. "
            f"Use a regular YAML file beneath {allowed_root.resolve()}."
        )
    try:
        file_stat = path.lstat()
    except OSError as e:
        raise ConfigError(f"Could not inspect {description.lower()} {path}: {e}") from e
    if not stat.S_ISREG(file_stat.st_mode):
        raise ConfigError(
            f"Refusing to load {description.lower()} {path}: expected a regular "
            "non-symlink file."
        )

    root = allowed_root.resolve()
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise ConfigError(
            f"Refusing to load {description.lower()} {path}: it resolves outside "
            f"the allowed configuration root {root}."
        )


def _reject_symlink_directory_components(
    configured: Path,
    project_dir: Path,
    *,
    description: str,
) -> None:
    project_root = Path(os.path.abspath(project_dir))
    lexical = Path(os.path.abspath(configured))
    try:
        parts = lexical.relative_to(project_root).parts
    except ValueError:
        return
    current = project_root
    for part in parts:
        current /= part
        if current.is_symlink():
            raise ConfigError(
                f"Refusing to traverse {description.lower()} with symlink "
                f"component {current}. Configure a real directory beneath "
                f"{project_root}."
            )
