from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

from .adapters import WarehouseCapability, adapter_capabilities
from .backends import BackendOptionsError, list_backends, validate_backend_options
from .config.loader import ConfigError
from .config.model import ModelConfig, protect_model_llm_credential_option
from .config.project import ProjectConfig
from .config.source import SourceConfig
from .config.yaml_diagnostics import ConfigPath
from .dag import DAGError, ProjectDAG, parse_ref
from .ml_contracts import MLContractError, validate_ml_project_contracts
from .paths import resolve_within_project
from .providers import (
    ProviderConfigurationError,
    ProviderNotFoundError,
    get_inference_provider,
)
from .test_specs import TestSpecError, parse_test_spec
from .transforms import load_transform, transform_call_arity

_MODULE_PATTERN = re.compile(r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*$")


def validate_project_contract(
    project: ProjectConfig,
    sources: list[SourceConfig],
    models: list[ModelConfig],
    project_dir: Path,
) -> ProjectDAG:
    default_backend = project.extraction.default_backend
    for model in models:
        extraction = model.extraction
        if extraction is not None and (extraction.backend or default_backend) == "llm":
            protect_model_llm_credential_option(model)

    source_names = {source.name for source in sources}
    model_names = {model.name for model in models}
    duplicates = source_names & model_names
    if duplicates:
        duplicate_model = next(model for model in models if model.name in duplicates)
        raise _model_error(
            duplicate_model,
            f"Source and model names must be unique; duplicated: {sorted(duplicates)}",
            ("name",),
        )

    available_backends = set(list_backends())
    if default_backend not in available_backends:
        raise ConfigError(
            project.format_yaml_diagnostic(
                f"Default extraction backend '{default_backend}' is not registered. "
                f"Available: {sorted(available_backends)}",
                relative_path=("extraction", "default_backend"),
            )
        )

    for model in models:
        _validate_tests(model, source_names, model_names, project_dir)
        _validate_model_edges(model, source_names, model_names)
        _validate_materialization(model)
        if model.extraction is not None:
            backend = model.extraction.backend or default_backend
            if backend not in available_backends:
                raise _model_error(
                    model,
                    f"Extraction model '{model.name}' uses unregistered backend "
                    f"'{backend}'. Available: {sorted(available_backends)}",
                    ("extraction", "backend"),
                )
            if backend == "llm" and "api_key_env" in model.extraction.options:
                raise _model_error(
                    model,
                    "llm option 'api_key_env' is operator-owned configuration; "
                    "set it under `llm:` in profiles.yml, not in model "
                    "extraction options",
                    ("extraction", "options", "api_key_env"),
                )
            try:
                canonical_options = validate_backend_options(
                    backend, model.extraction.options
                )
            except BackendOptionsError as e:
                error_path = getattr(e, "path", ("options",))
                raise _model_error(
                    model,
                    f"Extraction model '{model.name}' has {e}",
                    ("extraction", *error_path),
                ) from e
            # Provider checks here only apply when the model pins one — the
            # canonical default may not be the effective provider, which the
            # profile selects. resolve_llm_options re-validates registration
            # and batch capability against the resolved profile.
            if backend == "llm" and "provider" in model.extraction.options:
                provider_name = str(canonical_options["provider"])
                try:
                    provider = get_inference_provider(provider_name)
                except (ProviderNotFoundError, ProviderConfigurationError) as e:
                    raise _model_error(
                        model,
                        str(e),
                        ("extraction", "options", "provider"),
                    ) from e
                if canonical_options.get("batch") and not provider.supports_native_batch:
                    raise _model_error(
                        model,
                        f"Inference provider '{provider_name}' does not support "
                        "native batch execution",
                        ("extraction", "options", "batch"),
                    )
            if backend == "llm" and "cache_path" in model.extraction.options:
                try:
                    resolve_within_project(
                        model.extraction.options["cache_path"],
                        project_dir,
                        surface=f"Model '{model.name}' llm cache_path",
                        hint="Set llm.cache_path in profiles.yml for locations "
                        "outside the project.",
                    )
                except ConfigError as e:
                    raise _model_error(
                        model,
                        str(e),
                        ("extraction", "options", "cache_path"),
                    ) from e
        if model.transform is not None:
            _validate_transform(model, project_dir)
    try:
        validate_ml_project_contracts(models, project, project_dir)
    except MLContractError as e:
        implicated = next(
            (model for model in models if model.name == e.model_name),
            None,
        )
        if implicated is None:
            raise ConfigError(str(e)) from e
        raise _model_error(implicated, str(e), e.path) from e

    try:
        return ProjectDAG(sources, models)
    except DAGError as e:
        raise ConfigError(f"Invalid project DAG: {e}") from e


def validate_warehouse_capabilities(
    models: list[ModelConfig], adapter_type: str
) -> None:
    available = adapter_capabilities(adapter_type)
    for model in models:
        required: dict[WarehouseCapability, str] = {}
        if model.materialization == "full":
            required[WarehouseCapability.ATOMIC_FULL_REPLACE] = (
                "full materialization"
            )
        else:
            required[WarehouseCapability.ATOMIC_KEYED_UPSERT] = (
                "incremental materialization"
            )
        if model.extraction is not None:
            required[WarehouseCapability.TYPED_EMPTY_RELATIONS] = (
                "empty extraction results"
            )
            required[WarehouseCapability.CHUNKED_WRITES] = (
                "bounded extraction writes"
            )
        if model.transform is not None or model.ml is not None or model.chunk is not None:
            required[WarehouseCapability.TABULAR_READS] = (
                f"{_kind_label(model).lower()} input reads"
            )
        if model.tests:
            required[WarehouseCapability.SQL_SCHEMA_TESTS] = "model tests"
        if (
            model.materialization == "incremental"
            and model.on_schema_change == "append_new_columns"
        ):
            required[WarehouseCapability.SCHEMA_EVOLUTION] = (
                "on_schema_change=append_new_columns"
            )

        missing = sorted(set(required) - available, key=lambda item: item.value)
        if not missing:
            continue
        details = ", ".join(
            f"{capability.value} ({required[capability]})"
            for capability in missing
        )
        raise _model_error(
            model,
            f"Warehouse adapter '{adapter_type}' cannot execute model "
            f"'{model.name}'; missing capabilities: {details}",
        )


def validate_warehouse_operation_capabilities(
    adapter_type: str,
    required: Mapping[WarehouseCapability, str],
    *,
    operation: str,
) -> None:
    """Preflight a non-model warehouse operation before adapter construction."""
    available = adapter_capabilities(adapter_type)
    missing = sorted(set(required) - available, key=lambda item: item.value)
    if not missing:
        return
    details = ", ".join(
        f"{capability.value} ({required[capability]})"
        for capability in missing
    )
    raise ConfigError(
        f"Warehouse adapter '{adapter_type}' cannot execute {operation}; "
        f"missing capabilities: {details}"
    )


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
            raise _model_error(
                model,
                f"Model '{model.name}' test[{index}] is invalid: {e}",
                ("tests", index),
            ) from e
        if parsed.name == "python":
            module_path = parsed.argument
            assert isinstance(module_path, str)
            _validate_python_test(model, index, module_path, project_dir)
        target = parsed.relationship_target
        if target is None:
            continue
        if target in source_names:
            raise _model_error(
                model,
                f"Model '{model.name}' relationships test target '{target}' is a "
                "source; relationship targets must be models",
                ("tests", index),
            )
        if target not in model_names:
            raise _model_error(
                model,
                f"Model '{model.name}' relationships test references unknown model "
                f"'{target}'",
                ("tests", index),
            )


def _validate_python_test(
    model: ModelConfig, index: int, module_path: str, project_dir: Path
) -> None:
    if not _MODULE_PATTERN.fullmatch(module_path):
        raise _model_error(
            model,
            f"Model '{model.name}' test[{index}] python module '{module_path}' is not "
            "a valid dotted Python module path",
            ("tests", index),
        )
    # Local import avoids a compiler <-> checks package import cycle.
    from .checks.python import CustomTestError, load_python_test

    try:
        load_python_test(module_path, project_dir)
    except CustomTestError as e:
        raise _model_error(
            model,
            f"Model '{model.name}' test[{index}] python module '{module_path}' is "
            f"invalid: {e}",
            ("tests", index),
        ) from e


def _validate_model_edges(
    model: ModelConfig, source_names: set[str], model_names: set[str]
) -> None:
    if model.kind_block_count != 1:
        raise _model_error(
            model,
            f"Model '{model.name}' must declare exactly one of "
            "extraction/transform/ml/chunk",
        )

    if model.extraction is not None:
        if not model.source:
            raise _model_error(
                model,
                f"Extraction model '{model.name}' must declare exactly one `source:`",
                ("source",),
            )
        if model.depends_on is not None:
            raise _model_error(
                model,
                f"Extraction model '{model.name}' must use `source:`, not `depends_on:`",
                ("depends_on",),
            )
        target = parse_ref(model.source)
        if target in model_names:
            raise _model_error(
                model,
                f"Extraction model '{model.name}' source '{target}' is a model; "
                "extraction sources must reference source nodes",
                ("source",),
            )
        if target not in source_names:
            raise _model_error(
                model,
                f"Extraction model '{model.name}' references unknown source '{target}'",
                ("source",),
            )
        return

    if model.source is not None:
        raise _model_error(
            model,
            f"{_kind_label(model)} model '{model.name}' must use `depends_on:`, "
            "not `source:`",
            ("source",),
        )

    dependencies = model.depends_on or []
    if model.transform is not None and not dependencies:
        raise _model_error(
            model,
            f"Transform model '{model.name}' must declare at least one `depends_on:` model",
            ("depends_on",),
        )
    if model.ml is not None and not dependencies:
        raise _model_error(
            model,
            f"ML model '{model.name}' must declare at least one `depends_on:` model",
            ("depends_on",),
        )
    if model.chunk is not None and len(dependencies) != 1:
        raise _model_error(
            model,
            f"Chunk model '{model.name}' must declare exactly one `depends_on:` model",
            ("depends_on",),
        )

    dependency_targets = [parse_ref(dependency) for dependency in dependencies]
    duplicate_targets = sorted(
        target for target in set(dependency_targets) if dependency_targets.count(target) > 1
    )
    if duplicate_targets:
        raise _model_error(
            model,
            f"{_kind_label(model)} model '{model.name}' declares duplicate "
            f"dependencies: {duplicate_targets}",
            ("depends_on",),
        )

    for index, target in enumerate(dependency_targets):
        if target in source_names:
            raise _model_error(
                model,
                f"{_kind_label(model)} model '{model.name}' dependency '{target}' is "
                "a source; non-extraction models must depend on models",
                ("depends_on", index),
            )
        if target not in model_names:
            raise _model_error(
                model,
                f"{_kind_label(model)} model '{model.name}' references unknown model "
                f"'{target}'",
                ("depends_on", index),
            )


def _validate_materialization(model: ModelConfig) -> None:
    if model.transform is not None and model.materialization != "full":
        raise _model_error(
            model,
            f"Transform model '{model.name}' only supports `materialization: full`",
            ("materialization",),
        )
    if model.ml is not None and model.materialization != "full":
        raise _model_error(
            model,
            f"ML model '{model.name}' only supports `materialization: full`",
            ("materialization",),
        )


def _validate_transform(model: ModelConfig, project_dir: Path) -> None:
    assert model.transform is not None
    transform = model.transform
    if transform.type != "python":
        raise _model_error(
            model,
            f"Transform model '{model.name}' has unsupported type '{transform.type}'; "
            "supported: python",
            ("transform", "type"),
        )
    if not transform.module:
        raise _model_error(
            model,
            f"Transform model '{model.name}' requires a `module:`",
            ("transform", "module"),
        )
    if not _MODULE_PATTERN.fullmatch(transform.module):
        raise _model_error(
            model,
            f"Transform model '{model.name}' module '{transform.module}' is not a "
            "valid dotted Python module path",
            ("transform", "module"),
        )
    try:
        transform_fn = load_transform(transform.module, project_dir)
        transform_call_arity(transform_fn)
    except (Exception, SystemExit) as e:
        raise _model_error(
            model,
            f"Transform model '{model.name}' module '{transform.module}' is invalid: {e}",
            ("transform", "module"),
        ) from e


def _model_error(
    model: ModelConfig,
    message: str,
    relative_path: ConfigPath = (),
) -> ConfigError:
    return ConfigError(
        model.format_yaml_diagnostic(message, relative_path=relative_path)
    )


def _kind_label(model: ModelConfig) -> str:
    if model.transform is not None:
        return "Transform"
    if model.ml is not None:
        return "ML"
    if model.chunk is not None:
        return "Chunk"
    return "Unknown"
