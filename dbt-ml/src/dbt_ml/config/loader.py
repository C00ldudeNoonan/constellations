from __future__ import annotations

import os
import stat
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ValidationError

from .model import ModelConfig, ModelFile, protect_model_llm_credential_option
from .project import ProjectConfig
from .source import SourceConfig, SourceFile
from .yaml_diagnostics import (
    YamlDocument,
    YamlProvenance,
    format_yaml_parse_error,
    parse_yaml_document,
)


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

    project, project_document = _parse_yaml_with_document(
        project_path,
        ProjectConfig,
    )
    project._yaml_provenance = YamlProvenance(
        file_path=project_path,
        config_path=(),
        _document=project_document,
    )

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
            surface="Inline `duckdb.path`",
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
    default_backend = project.extraction.default_backend

    def protect_loaded_model(model: ModelConfig) -> None:
        extraction = model.extraction
        if extraction is not None and (extraction.backend or default_backend) == "llm":
            protect_model_llm_credential_option(model)

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
                attach_model_provenance=True,
                postprocess=protect_loaded_model,
            )
        )

    _populate_sql_depends_on(models, project_dir)

    return project, sources, models


def _populate_sql_depends_on(
    models: list[ModelConfig], project_dir: Path
) -> None:
    """Derive `depends_on` for SQL transforms from their `.sql` ref()s at load
    time, so every DAG consumer — the compiler, manifest, run_results, and the
    runner — sees the same lineage edges. Only fills an unset `depends_on`;
    explicit declarations are preserved for the compiler's agreement check.
    Unreadable/invalid SQL is left for the compiler to report with YAML context.
    """
    # Local imports avoid a cycle (paths imports ConfigError from here; sql_models
    # is dependency-free beyond jinja2).
    from ..paths import resolve_within_project
    from ..sql_models import SqlModelError, discover_refs, read_sql_source

    for model in models:
        transform = model.transform
        if (
            transform is None
            or transform.type != "sql"
            or not transform.path
            or model.depends_on
        ):
            continue
        try:
            resolved = resolve_within_project(
                transform.path, project_dir, surface="transform.path"
            )
            sql_text = read_sql_source(resolved, model_name=model.name)
            refs = discover_refs(sql_text, model_name=model.name)
        except (ConfigError, SqlModelError):
            continue
        if refs:
            model.depends_on = [f"ref('{name}')" for name in refs]


def _parse_yaml_with_document[T: BaseModel](
    path: Path,
    model: type[T],
) -> tuple[T, YamlDocument]:
    document: YamlDocument | None = None
    load_error: ConfigError | None = None
    try:
        with path.open() as f:
            document = parse_yaml_document(f.read())
    except yaml.YAMLError as e:
        load_error = ConfigError(format_yaml_parse_error(path, e))
    except OSError as e:
        load_error = ConfigError(
            f"Could not read YAML configuration file {path}: {e}"
        )
    if load_error is not None:
        raise load_error
    assert document is not None
    data: Any = document.data if document.data is not None else {}
    document = document.without_data()
    validation_failure: ConfigError | None = None
    try:
        parsed = model.model_validate(data)
    except ValidationError as e:
        diagnostics = document.format_validation_errors(path, e)
        validation_failure = ConfigError(
            f"Invalid YAML at {path}:\n{diagnostics}"
        )
    if validation_failure is not None:
        data = None
        raise validation_failure
    return parsed, document


def _load_yaml_dir[F: BaseModel, I](
    directory: Path,
    file_model: type[F],
    extract: Callable[[F], Iterable[I]],
    *,
    description: str,
    attach_model_provenance: bool = False,
    postprocess: Callable[[I], None] | None = None,
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
            parsed, document = _parse_yaml_with_document(path, file_model)
            items: list[I] = list(extract(parsed))
            if postprocess is not None:
                for item in items:
                    postprocess(item)
            if attach_model_provenance:
                for index, item in enumerate(items):
                    if not isinstance(item, ModelConfig):
                        raise TypeError(
                            "Model YAML provenance can only be attached to ModelConfig"
                        )
                    item._yaml_provenance = YamlProvenance(
                        file_path=path,
                        config_path=("models", index),
                        _document=document,
                    )
            out.extend(items)
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
