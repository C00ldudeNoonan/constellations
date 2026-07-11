from __future__ import annotations

import re
from pathlib import Path

from .backends import list_backends
from .config.loader import ConfigError
from .config.model import ModelConfig
from .config.project import ProjectConfig
from .config.source import SourceConfig
from .dag import DAGError, ProjectDAG, parse_ref
from .test_specs import TestSpecError, parse_test_spec
from .transforms import load_transform, transform_call_arity

_MODULE_PATTERN = re.compile(r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*$")


def validate_project_contract(
    project: ProjectConfig,
    sources: list[SourceConfig],
    models: list[ModelConfig],
    project_dir: Path,
) -> ProjectDAG:
    source_names = {source.name for source in sources}
    model_names = {model.name for model in models}
    duplicates = source_names & model_names
    if duplicates:
        raise ConfigError(
            f"Source and model names must be unique; duplicated: {sorted(duplicates)}"
        )

    available_backends = set(list_backends())
    default_backend = project.extraction.default_backend
    if default_backend not in available_backends:
        raise ConfigError(
            f"Default extraction backend '{default_backend}' is not registered. "
            f"Available: {sorted(available_backends)}"
        )

    for model in models:
        _validate_tests(model, source_names, model_names, project_dir)
        _validate_model_edges(model, source_names, model_names)
        _validate_materialization(model)
        if model.extraction is not None:
            backend = model.extraction.backend or default_backend
            if backend not in available_backends:
                raise ConfigError(
                    f"Extraction model '{model.name}' uses unregistered backend "
                    f"'{backend}'. Available: {sorted(available_backends)}"
                )
        if model.transform is not None:
            _validate_transform(model, project_dir)

    try:
        return ProjectDAG(sources, models)
    except DAGError as e:
        raise ConfigError(f"Invalid project DAG: {e}") from e


def _validate_tests(
    model: ModelConfig,
    source_names: set[str],
    model_names: set[str],
    project_dir: Path,
) -> None:
    for index, spec in enumerate(model.tests):
        try:
            parsed = parse_test_spec(spec)
        except TestSpecError as e:
            raise ConfigError(
                f"Model '{model.name}' test[{index}] is invalid: {e}"
            ) from e
        if parsed.name == "python":
            module_path = parsed.argument
            assert isinstance(module_path, str)
            _validate_python_test(model, index, module_path, project_dir)
        target = parsed.relationship_target
        if target is None:
            continue
        if target in source_names:
            raise ConfigError(
                f"Model '{model.name}' relationships test target '{target}' is a "
                "source; relationship targets must be models"
            )
        if target not in model_names:
            raise ConfigError(
                f"Model '{model.name}' relationships test references unknown model "
                f"'{target}'"
            )


def _validate_python_test(
    model: ModelConfig, index: int, module_path: str, project_dir: Path
) -> None:
    if not _MODULE_PATTERN.fullmatch(module_path):
        raise ConfigError(
            f"Model '{model.name}' test[{index}] python module '{module_path}' is not "
            "a valid dotted Python module path"
        )
    # Local import avoids a compiler <-> checks package import cycle.
    from .checks.python import CustomTestError, load_python_test

    try:
        load_python_test(module_path, project_dir)
    except CustomTestError as e:
        raise ConfigError(
            f"Model '{model.name}' test[{index}] python module '{module_path}' is "
            f"invalid: {e}"
        ) from e


def _validate_model_edges(
    model: ModelConfig, source_names: set[str], model_names: set[str]
) -> None:
    if model.kind_block_count != 1:
        raise ConfigError(
            f"Model '{model.name}' must declare exactly one of "
            "extraction/transform/ml/chunk"
        )

    if model.extraction is not None:
        if not model.source:
            raise ConfigError(
                f"Extraction model '{model.name}' must declare exactly one `source:`"
            )
        if model.depends_on is not None:
            raise ConfigError(
                f"Extraction model '{model.name}' must use `source:`, not `depends_on:`"
            )
        target = parse_ref(model.source)
        if target in model_names:
            raise ConfigError(
                f"Extraction model '{model.name}' source '{target}' is a model; "
                "extraction sources must reference source nodes"
            )
        if target not in source_names:
            raise ConfigError(
                f"Extraction model '{model.name}' references unknown source '{target}'"
            )
        return

    if model.source is not None:
        raise ConfigError(
            f"{_kind_label(model)} model '{model.name}' must use `depends_on:`, "
            "not `source:`"
        )

    dependencies = model.depends_on or []
    if model.transform is not None and not dependencies:
        raise ConfigError(
            f"Transform model '{model.name}' must declare at least one `depends_on:` model"
        )
    if model.ml is not None and not dependencies:
        raise ConfigError(
            f"ML model '{model.name}' must declare at least one `depends_on:` model"
        )
    if model.chunk is not None and len(dependencies) != 1:
        raise ConfigError(
            f"Chunk model '{model.name}' must declare exactly one `depends_on:` model"
        )

    dependency_targets = [parse_ref(dependency) for dependency in dependencies]
    duplicate_targets = sorted(
        target for target in set(dependency_targets) if dependency_targets.count(target) > 1
    )
    if duplicate_targets:
        raise ConfigError(
            f"{_kind_label(model)} model '{model.name}' declares duplicate "
            f"dependencies: {duplicate_targets}"
        )

    for target in dependency_targets:
        if target in source_names:
            raise ConfigError(
                f"{_kind_label(model)} model '{model.name}' dependency '{target}' is "
                "a source; non-extraction models must depend on models"
            )
        if target not in model_names:
            raise ConfigError(
                f"{_kind_label(model)} model '{model.name}' references unknown model "
                f"'{target}'"
            )


def _validate_materialization(model: ModelConfig) -> None:
    if model.transform is not None and model.materialization != "full":
        raise ConfigError(
            f"Transform model '{model.name}' only supports `materialization: full`"
        )
    if model.ml is not None and model.materialization != "full":
        raise ConfigError(f"ML model '{model.name}' only supports `materialization: full`")


def _validate_transform(model: ModelConfig, project_dir: Path) -> None:
    assert model.transform is not None
    transform = model.transform
    if transform.type != "python":
        raise ConfigError(
            f"Transform model '{model.name}' has unsupported type '{transform.type}'; "
            "supported: python"
        )
    if not transform.module:
        raise ConfigError(f"Transform model '{model.name}' requires a `module:`")
    if not _MODULE_PATTERN.fullmatch(transform.module):
        raise ConfigError(
            f"Transform model '{model.name}' module '{transform.module}' is not a "
            "valid dotted Python module path"
        )
    try:
        transform_fn = load_transform(transform.module, project_dir)
        transform_call_arity(transform_fn)
    except (Exception, SystemExit) as e:
        raise ConfigError(
            f"Transform model '{model.name}' module '{transform.module}' is invalid: {e}"
        ) from e


def _kind_label(model: ModelConfig) -> str:
    if model.transform is not None:
        return "Transform"
    if model.ml is not None:
        return "ML"
    if model.chunk is not None:
        return "Chunk"
    return "Unknown"
