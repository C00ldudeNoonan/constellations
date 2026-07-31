from __future__ import annotations

import hashlib
import itertools
import os
import re
import stat
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .model import ModelConfig, protect_model_llm_credential_option
from .project import ProjectConfig
from .source import SourceConfig, SourceFile
from .yaml_diagnostics import (
    ConfigPath,
    YamlDocument,
    YamlProvenance,
    format_yaml_parse_error,
    parse_yaml_document,
)


class ConfigError(Exception):
    pass


_MAX_FOR_EACH_VARIANTS = 256
_MAX_SLUG_LEN = 32
_MATRIX_RE = re.compile(r"\$\{matrix\.([A-Za-z_][A-Za-z0-9_]*)\}")


def _value_slug(value: Any) -> str:
    if isinstance(value, list):
        return "_".join(_value_slug(v) for v in value) or "empty"
    raw = str(value).lower()
    slug = re.sub(r"[^a-z0-9]+", "_", raw).strip("_")
    if not slug:
        slug = "x"
    if len(slug) > _MAX_SLUG_LEN:
        h = hashlib.sha256(slug.encode()).hexdigest()[:8]
        slug = slug[: _MAX_SLUG_LEN - 9] + "_" + h
    return slug


def _variant_name(base: str, axis_names: list[str], combo: tuple[Any, ...]) -> str:
    parts = [base]
    for key, value in zip(axis_names, combo, strict=True):
        parts.append(f"{key}_{_value_slug(value)}")
    return "__".join(parts)


def _substitute(value: Any, matrix: dict[str, Any]) -> Any:
    if isinstance(value, str):
        m = _MATRIX_RE.fullmatch(value)
        if m:
            key = m.group(1)
            if key not in matrix:
                raise ConfigError(
                    f"Matrix placeholder '${{matrix.{key}}}' references unknown axis '{key}'"
                )
            return matrix[key]

        def _repl(mo: re.Match[str]) -> str:
            k = mo.group(1)
            if k not in matrix:
                raise ConfigError(
                    f"Matrix placeholder '${{matrix.{k}}}' references unknown axis '{k}'"
                )
            return str(matrix[k])

        return _MATRIX_RE.sub(_repl, value)
    if isinstance(value, dict):
        return {k: _substitute(v, matrix) for k, v in value.items()}
    if isinstance(value, list):
        return [_substitute(v, matrix) for v in value]
    return value


def _expand_for_each(models: list[ModelConfig]) -> list[ModelConfig]:
    """Expand for_each template ModelConfigs into concrete variants.

    Used for programmatic/test construction where models are already validated.
    The load-time path uses _expand_single_raw_model instead, which expands
    before strict Pydantic validation so typed fields can hold placeholders.
    """
    result: list[ModelConfig] = []
    for model in models:
        if model.for_each is None:
            result.append(model)
            continue

        axes = model.for_each
        axis_names = list(axes.keys())

        # P2: check size before materialising all combinations.
        total = 1
        for vals in axes.values():
            total *= len(vals)
        if total > _MAX_FOR_EACH_VARIANTS:
            raise ConfigError(
                f"Model '{model.name}': for_each expands to {total} variants "
                f"(max {_MAX_FOR_EACH_VARIANTS})"
            )

        base_name = model.name
        base_dict = model.model_dump(mode="python", exclude={"for_each"})
        seen_names: set[str] = set()

        for combo in itertools.product(*[axes[k] for k in axis_names]):
            matrix = dict(zip(axis_names, combo, strict=True))
            variant_name = _variant_name(base_name, axis_names, combo)

            if variant_name in seen_names:
                raise ConfigError(
                    f"Model '{base_name}': variant name collision '{variant_name}'; "
                    "two axis value combinations produce the same slug"
                )
            seen_names.add(variant_name)

            variant_dict: dict[str, Any] = _substitute(base_dict, matrix)
            variant_dict["name"] = variant_name

            tags: list[str] = list(variant_dict.get("tags") or [])
            if base_name not in tags:
                tags.append(base_name)
            variant_dict["tags"] = tags

            try:
                variant = ModelConfig.model_validate(variant_dict)
            except ValidationError as e:
                raise ConfigError(
                    f"Model '{base_name}' for_each variant '{variant_name}' failed validation:\n"
                    f"{e}"
                ) from e

            variant._yaml_provenance = model._yaml_provenance
            result.append(variant)

    return result


# ---------------------------------------------------------------------------
# Raw-dict model loading — for_each expansion before strict validation
# ---------------------------------------------------------------------------


class _RawModelFile(BaseModel):
    """Parses the file-level YAML structure without validating individual models.
    Models are kept as raw dicts so for_each expansion can happen before
    strict per-field Pydantic validation."""

    model_config = ConfigDict(extra="forbid")

    version: Literal[2] = 2
    models: list[dict[str, Any]] = Field(default_factory=list)


_AXIS_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _expand_single_raw_model(
    raw_dict: dict[str, Any],
    index: int,
    path: Path,
    document: YamlDocument,
) -> list[ModelConfig]:
    """Validate one raw model dict from a YAML file, expanding for_each first.

    For templates the raw dict is never validated as ModelConfig directly —
    substitution runs first, then each expanded variant is validated.  This
    lets typed fields (chunk_size, dimensions, materialization, …) carry
    ${matrix.KEY} placeholder strings in the template YAML.
    """
    for_each = raw_dict.get("for_each")
    model_prefix: ConfigPath = ("models", index)

    if for_each is None:
        config_error: ConfigError | None = None
        try:
            model = ModelConfig.model_validate(raw_dict)
        except ValidationError as e:
            diag = document.format_validation_errors(path, e, prefix=model_prefix)
            config_error = ConfigError(f"Invalid YAML at {path}:\n{diag}")
        if config_error is not None:
            raw_dict = {}  # drop sensitive input before raising
            raise config_error
        _assert_has_kind(model, path, document, model_prefix)
        model._yaml_provenance = YamlProvenance(
            file_path=path,
            config_path=model_prefix,
            _document=document,
        )
        return [model]

    # --- Template with for_each ---
    name: str = raw_dict.get("name") or "<unknown>"

    if not isinstance(for_each, dict) or not for_each:
        raise ConfigError(
            document.format_message(
                path, (*model_prefix, "for_each"), "for_each must be a non-empty mapping"
            )
        )

    for axis_name, values in for_each.items():
        if not _AXIS_IDENT_RE.match(str(axis_name)):
            raise ConfigError(
                document.format_message(
                    path,
                    (*model_prefix, "for_each"),
                    f"axis name '{axis_name}' must be a valid identifier "
                    "(letters, digits, underscores; start with letter or _)",
                )
            )
        if not isinstance(values, list) or not values:
            raise ConfigError(
                document.format_message(
                    path,
                    (*model_prefix, "for_each", axis_name),
                    f"axis '{axis_name}' must be a non-empty list",
                )
            )

    axis_names: list[str] = [str(k) for k in for_each]
    total = 1
    for vals in for_each.values():
        total *= len(vals)
    if total > _MAX_FOR_EACH_VARIANTS:
        raise ConfigError(
            document.format_message(
                path,
                (*model_prefix, "for_each"),
                f"model '{name}' for_each expands to {total} variants "
                f"(max {_MAX_FOR_EACH_VARIANTS})",
            )
        )

    base_dict = {k: v for k, v in raw_dict.items() if k != "for_each"}
    seen_names: set[str] = set()
    result: list[ModelConfig] = []

    for combo in itertools.product(*[for_each[k] for k in axis_names]):
        matrix: dict[str, Any] = dict(zip(axis_names, combo, strict=True))
        variant_name = _variant_name(name, axis_names, combo)

        if variant_name in seen_names:
            raise ConfigError(
                f"Model '{name}': variant name collision '{variant_name}'; "
                "two axis value combinations produce the same slug"
            )
        seen_names.add(variant_name)

        variant_dict: dict[str, Any] = _substitute(base_dict, matrix)
        variant_dict["name"] = variant_name

        tags: list[str] = list(variant_dict.get("tags") or [])
        if name not in tags:
            tags.append(name)
        variant_dict["tags"] = tags

        try:
            variant = ModelConfig.model_validate(variant_dict)
        except ValidationError as e:
            raise ConfigError(
                f"Model '{name}' for_each variant '{variant_name}' failed validation:\n{e}"
            ) from e

        _assert_has_kind(variant, path, document, model_prefix)
        variant._yaml_provenance = YamlProvenance(
            file_path=path,
            config_path=model_prefix,
            _document=document,
        )
        result.append(variant)

    return result


def _assert_has_kind(
    model: ModelConfig,
    path: Path,
    document: YamlDocument,
    config_path: ConfigPath,
) -> None:
    if model.kind_block_count == 0:
        raise ConfigError(
            document.format_message(
                path,
                config_path,
                f"Model '{model.name}' is missing an "
                "extraction/transform/ml/chunk/embed/llm or search block",
            )
        )


def _load_model_yaml_dir(
    directory: Path,
    *,
    postprocess: Callable[[ModelConfig], None] | None = None,
) -> list[ModelConfig]:
    """Walk a model directory, expanding for_each templates before validation.

    Mirrors the walk logic of _load_yaml_dir but uses _RawModelFile + raw-dict
    expansion so typed fields in for_each templates can hold placeholder strings.
    """
    description = "Model configuration"

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
        err_path = error.filename or str(directory)
        raise ConfigError(
            f"Could not traverse {description.lower()} directory {err_path}: {error}"
        ) from error

    out: list[ModelConfig] = []
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
            file_path = root_path / filename
            _validate_config_file(
                file_path,
                allowed_root=directory,
                description=description,
            )
            raw_file, document = _parse_yaml_with_document(file_path, _RawModelFile)
            for index, raw_model in enumerate(raw_file.models):
                for model in _expand_single_raw_model(raw_model, index, file_path, document):
                    if postprocess is not None:
                        postprocess(model)
                    out.append(model)

    return out


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
            _load_model_yaml_dir(
                resolved,
                postprocess=protect_loaded_model,
            )
        )

    # for_each expansion is inline in _load_model_yaml_dir, so models here
    # are already fully expanded.  _populate_sql_depends_on therefore runs on
    # the concrete variants, which is correct when transform.path itself
    # contains a ${matrix.KEY} placeholder.
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
