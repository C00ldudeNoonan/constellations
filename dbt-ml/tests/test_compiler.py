from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from dbt_ml.adapters import WarehouseCapability
from dbt_ml.cli import cli
from dbt_ml.compiler import (
    validate_project_contract,
    validate_warehouse_capabilities,
)
from dbt_ml.config import ConfigError, load_project
from dbt_ml.config.model import ModelConfig
from dbt_ml.config.project import ProjectConfig
from dbt_ml.config.source import SourceConfig
from dbt_ml.credentials import CredentialReference
from dbt_ml.dag import ProjectDAG
from dbt_ml.runner import build_project
from dbt_ml.test_specs import TestSpecError as SpecError
from dbt_ml.test_specs import parse_test_spec


def _source(name: str = "docs") -> SourceConfig:
    return SourceConfig(name=name, path=f"data/{name}")


def _extraction(
    name: str, source: str = "docs", *, backend: str = "json"
) -> ModelConfig:
    return ModelConfig(
        name=name,
        source=f"ref('{source}')",
        extraction={"backend": backend},
    )


def test_capability_preflight_rejects_missing_typed_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = ModelConfig(
        name="chunks",
        depends_on=["ref('raw')"],
        chunk={"text_field": "text"},
    )
    available = frozenset(WarehouseCapability) - {
        WarehouseCapability.TABULAR_READS,
    }
    monkeypatch.setattr(
        "dbt_ml.compiler.adapter_capabilities",
        lambda _adapter_type: available,
    )

    with pytest.raises(ConfigError, match=r"tabular_reads.*chunk input reads"):
        validate_warehouse_capabilities([model], "vector_only")


def test_capability_preflight_rejects_sql_tests_without_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _extraction("raw")
    model.tests = ["not_empty"]
    available = frozenset(WarehouseCapability) - {
        WarehouseCapability.SQL_SCHEMA_TESTS,
    }
    monkeypatch.setattr(
        "dbt_ml.compiler.adapter_capabilities",
        lambda _adapter_type: available,
    )

    with pytest.raises(ConfigError, match=r"sql_schema_tests.*model tests"):
        validate_warehouse_capabilities([model], "non_sql")


def test_bigquery_full_preflight_rejects_non_atomic_replacement() -> None:
    with pytest.raises(ConfigError, match=r"atomic_full_replace.*full materialization"):
        validate_warehouse_capabilities([_extraction("raw")], "bigquery")


@pytest.mark.parametrize("command", ["compile", "run", "build", "test"])
def test_invalid_backend_is_exit_2_before_profile_resolution(
    tmp_path: Path, command: str
) -> None:
    (tmp_path / "dbt_ml_project.yml").write_text(
        "name: invalid_backend\nprofile: deliberately_missing\n"
        "extraction:\n  default_backend: typo_backend\n"
    )
    (tmp_path / "sources").mkdir()
    (tmp_path / "sources" / "docs.yml").write_text(
        "version: 2\nsources:\n  - name: docs\n    path: data/docs\n"
    )
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "raw.yml").write_text(
        "version: 2\nmodels:\n  - name: raw_docs\n"
        "    source: ref('docs')\n    extraction: {}\n"
    )

    result = CliRunner().invoke(cli, ["--project-dir", str(tmp_path), command])

    assert result.exit_code == 2, result.output
    assert "typo_backend" in result.output
    assert "not registered" in result.output
    assert "dbt_ml_project.yml:4:20" in result.output
    assert "[extraction.default_backend]" in result.output
    assert "no profiles.yml" not in result.output


def test_explicit_unregistered_backend_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="unregistered backend 'missing'"):
        validate_project_contract(
            ProjectConfig(name="p"),
            [_source()],
            [_extraction("raw", backend="missing")],
            tmp_path,
        )


def test_unregistered_inference_provider_is_rejected(tmp_path: Path) -> None:
    model = ModelConfig(
        name="raw",
        source="ref('docs')",
        extraction={
            "backend": "llm",
            "options": {
                "provider": "missing-provider",
                "fields": [{"name": "title", "type": "string"}],
            },
        },
    )

    with pytest.raises(
        ConfigError,
        match=r"Inference provider 'missing-provider' is not registered",
    ):
        validate_project_contract(
            ProjectConfig(name="p"),
            [_source()],
            [model],
            tmp_path,
        )


@pytest.mark.parametrize("explicit_backend", [True, False])
def test_model_owned_llm_credential_error_drops_reference_from_traceback(
    tmp_path: Path,
    explicit_backend: bool,
) -> None:
    reference_name = "MODEL_CREDENTIAL_REFERENCE_SENTINEL_154"
    extraction: dict[str, object] = {
        "options": {
            "api_key_env": reference_name,
            "fields": [{"name": "title", "type": "string"}],
        }
    }
    if explicit_backend:
        extraction["backend"] = "llm"
    model = ModelConfig(
        name="raw",
        source="ref('docs')",
        extraction=extraction,
    )
    project = ProjectConfig(
        name="p",
        extraction={"default_backend": "llm"},
    )

    with pytest.raises(ConfigError) as exc_info:
        validate_project_contract(project, [_source()], [model], tmp_path)

    error = exc_info.value
    assert reference_name not in str(error)
    assert reference_name not in repr(model)
    assert reference_name not in repr(model.model_dump())
    assert error.__cause__ is None
    assert error.__context__ is None
    traceback = error.__traceback__
    while traceback is not None:
        module = traceback.tb_frame.f_globals.get("__name__", "")
        if isinstance(module, str) and module.startswith("dbt_ml"):
            assert reference_name not in repr(traceback.tb_frame.f_locals)
        traceback = traceback.tb_next


def test_loaded_model_credential_error_retains_only_position_provenance(
    tmp_path: Path,
) -> None:
    reference_name = "LOADED_MODEL_CREDENTIAL_REFERENCE_SENTINEL_154"
    (tmp_path / "dbt_ml_project.yml").write_text(
        "name: p\nextraction:\n  default_backend: llm\n"
    )
    (tmp_path / "sources").mkdir()
    (tmp_path / "sources" / "docs.yml").write_text(
        "version: 2\nsources:\n  - name: docs\n    path: data/docs\n"
    )
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "raw.yml").write_text(
        "version: 2\nmodels:\n  - name: raw\n"
        "    source: ref('docs')\n"
        "    extraction:\n"
        "      options:\n"
        f"        api_key_env: {reference_name}\n"
        "        fields:\n"
        "          - name: title\n"
    )
    project, sources, models = load_project(tmp_path)
    provenance = models[0].yaml_provenance
    assert provenance is not None
    assert provenance._document.data is None
    assert models[0].extraction is not None
    assert isinstance(
        models[0].extraction.options["api_key_env"], CredentialReference
    )
    rendered = (
        repr(models[0])
        + repr(models[0].model_dump())
        + models[0].model_dump_json()
    )
    assert reference_name not in rendered

    with pytest.raises(ConfigError) as exc_info:
        validate_project_contract(project, sources, models, tmp_path)

    traceback = exc_info.value.__traceback__
    while traceback is not None:
        module = traceback.tb_frame.f_globals.get("__name__", "")
        if isinstance(module, str) and module.startswith("dbt_ml"):
            assert reference_name not in repr(traceback.tb_frame.f_locals)
        traceback = traceback.tb_next


def test_default_llm_credential_is_protected_before_earlier_model_error(
    tmp_path: Path,
) -> None:
    reference_name = "EARLY_MODEL_ERROR_REFERENCE_SENTINEL_154"
    model = ModelConfig(
        name="raw",
        source="ref('missing')",
        extraction={
            "options": {
                "api_key_env": reference_name,
                "fields": [{"name": "title", "type": "string"}],
            }
        },
    )
    project = ProjectConfig(
        name="p",
        extraction={"default_backend": "llm"},
    )

    with pytest.raises(ConfigError, match="unknown source") as exc_info:
        validate_project_contract(project, [_source()], [model], tmp_path)

    assert model.extraction is not None
    assert isinstance(
        model.extraction.options["api_key_env"], CredentialReference
    )
    traceback = exc_info.value.__traceback__
    while traceback is not None:
        module = traceback.tb_frame.f_globals.get("__name__", "")
        if isinstance(module, str) and module.startswith("dbt_ml"):
            assert reference_name not in repr(traceback.tb_frame.f_locals)
        traceback = traceback.tb_next


def test_provider_without_native_batch_is_rejected_at_compile(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "dbt_ml.compiler.get_inference_provider",
        lambda _name: SimpleNamespace(supports_native_batch=False),
    )
    model = ModelConfig(
        name="raw",
        source="ref('docs')",
        extraction={
            "backend": "llm",
            "options": {
                "provider": "sync-only",
                "batch": True,
                "fields": [{"name": "title", "type": "string"}],
            },
        },
    )

    with pytest.raises(ConfigError, match="does not support native batch"):
        validate_project_contract(
            ProjectConfig(name="p"),
            [_source()],
            [model],
            tmp_path,
        )


def test_provider_checks_defer_to_profile_when_model_does_not_pin_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Without a model-pinned provider the effective provider is profile
    configuration, so registration and batch capability are validated in
    resolve_llm_options — not against the canonical default here."""

    def _forbidden(_name: str) -> None:
        raise AssertionError("provider registry must not be consulted")

    monkeypatch.setattr("dbt_ml.compiler.get_inference_provider", _forbidden)
    model = ModelConfig(
        name="raw",
        source="ref('docs')",
        extraction={
            "backend": "llm",
            "options": {
                "batch": True,
                "fields": [{"name": "title", "type": "string"}],
            },
        },
    )

    validate_project_contract(
        ProjectConfig(name="p"),
        [_source()],
        [model],
        tmp_path,
    )


@pytest.mark.parametrize(
    "command",
    [
        ["compile"],
        ["run"],
        ["build"],
        ["test"],
        ["run", "--watch"],
    ],
)
def test_invalid_backend_options_fail_before_profile_or_source_access(
    tmp_path: Path, command: list[str]
) -> None:
    (tmp_path / "dbt_ml_project.yml").write_text(
        "name: invalid_options\nprofile: deliberately_missing\n"
        "extraction:\n  default_backend: json\n"
    )
    (tmp_path / "sources").mkdir()
    (tmp_path / "sources" / "docs.yml").write_text(
        "version: 2\nsources:\n  - name: docs\n    path: gs://bucket/docs\n"
    )
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "raw.yml").write_text(
        "version: 2\nmodels:\n  - name: raw_docs\n"
        "    source: ref('docs')\n    extraction:\n"
        "      options:\n        include_text: true\n"
    )

    result = CliRunner().invoke(
        cli, ["--project-dir", str(tmp_path), *command]
    )

    assert result.exit_code == 2, result.output
    assert "Invalid options for extraction backend 'json'" in result.output
    assert "options.include_text" in result.output
    assert "raw.yml:7:23" in result.output
    assert "models.0.extraction.options.include_text" in result.output
    assert "no profiles.yml" not in result.output


def test_model_llm_cache_path_escape_fails_before_profile_resolution(
    tmp_path: Path,
) -> None:
    (tmp_path / "dbt_ml_project.yml").write_text(
        "name: invalid_cache\nprofile: deliberately_missing\n"
    )
    (tmp_path / "sources").mkdir()
    (tmp_path / "sources" / "docs.yml").write_text(
        "version: 2\nsources:\n  - name: docs\n    path: gs://bucket/docs\n"
    )
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "raw.yml").write_text(
        "version: 2\nmodels:\n  - name: raw_docs\n"
        "    source: ref('docs')\n    extraction:\n      backend: llm\n"
        "      options:\n        cache_path: ../outside.duckdb\n"
        "        fields:\n          - {name: title, type: string}\n"
    )

    result = CliRunner().invoke(
        cli, ["--project-dir", str(tmp_path), "run"]
    )

    assert result.exit_code == 2, result.output
    assert "llm cache_path" in result.output
    assert "outside the project directory" in result.output
    assert "raw.yml:8:21" in result.output
    assert "models.0.extraction.options.cache_path" in result.output
    assert "no profiles.yml" not in result.output


@pytest.mark.parametrize("command", [["compile"], ["run"], ["build"], ["test"]])
def test_non_executable_ml_contract_fails_before_profile_resolution(
    tmp_path: Path, command: list[str]
) -> None:
    (tmp_path / "dbt_ml_project.yml").write_text(
        "name: invalid_ml\nprofile: deliberately_missing\n"
    )
    (tmp_path / "sources").mkdir()
    (tmp_path / "sources" / "docs.yml").write_text(
        "version: 2\nsources:\n  - name: docs\n    path: data/docs\n"
    )
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "models.yml").write_text(
        "version: 2\nmodels:\n"
        "  - name: raw_docs\n    source: ref('docs')\n"
        "    extraction:\n      backend: json\n"
        "  - name: unsupported_regression\n"
        "    depends_on: [ref('raw_docs')]\n"
        "    ml:\n      task: regressor\n      text_field: text\n"
    )

    result = CliRunner().invoke(
        cli, ["--project-dir", str(tmp_path), *command]
    )

    assert result.exit_code == 2, result.output
    assert "task 'regressor' is not executable" in result.output
    assert "models.yml:10:13" in result.output
    assert "models.1.ml.task" in result.output
    assert "no profiles.yml" not in result.output


@pytest.mark.parametrize(
    ("model", "message"),
    [
        (
            ModelConfig(name="raw", extraction={"backend": "json"}),
            "must declare exactly one `source:`",
        ),
        (
            ModelConfig(
                name="raw",
                source="ref('upstream')",
                extraction={"backend": "json"},
            ),
            "source 'upstream' is a model",
        ),
        (
            ModelConfig(
                name="raw",
                source="ref('docs')",
                depends_on=["ref('upstream')"],
                extraction={"backend": "json"},
            ),
            "must use `source:`, not `depends_on:`",
        ),
        (
            ModelConfig(
                name="chunked",
                depends_on=["ref('docs')"],
                chunk={"text_field": "text"},
            ),
            "dependency 'docs' is a source",
        ),
        (
            ModelConfig(name="chunked", chunk={"text_field": "text"}),
            "exactly one `depends_on:`",
        ),
        (
            ModelConfig(
                name="features",
                ml={"task": "features", "text_field": "text"},
            ),
            "at least one `depends_on:`",
        ),
        (
            ModelConfig(
                name="derived",
                depends_on=["ref('upstream')", "upstream"],
                ml={"task": "features", "text_field": "text"},
            ),
            "duplicate dependencies",
        ),
    ],
)
def test_model_edge_contracts_fail_before_execution(
    tmp_path: Path, model: ModelConfig, message: str
) -> None:
    upstream = _extraction("upstream")
    with pytest.raises(ConfigError, match=message):
        validate_project_contract(
            ProjectConfig(name="p"), [_source()], [upstream, model], tmp_path
        )


@pytest.mark.parametrize("kind", ["transform", "ml"])
def test_non_incremental_model_kinds_reject_incremental_materialization(
    tmp_path: Path, kind: str
) -> None:
    if kind == "transform":
        model = ModelConfig(
            name="derived",
            depends_on=["ref('raw')"],
            transform={"type": "python", "module": "transforms.derived"},
            materialization="incremental",
        )
    else:
        model = ModelConfig(
            name="derived",
            depends_on=["ref('raw')"],
            ml={"task": "features", "text_field": "text"},
            materialization="incremental",
        )

    with pytest.raises(ConfigError, match="only supports `materialization: full`"):
        validate_project_contract(
            ProjectConfig(name="p"), [_source()], [_extraction("raw"), model], tmp_path
        )


@pytest.mark.parametrize(
    ("transform", "module_source", "message"),
    [
        ({"type": "sql", "module": "transforms.derived"}, None, "unsupported type"),
        ({"type": "python"}, None, "requires a `module:`"),
        (
            {"type": "python", "module": "transforms.missing"},
            None,
            "not found",
        ),
        (
            {"type": "python", "module": "transforms.derived"},
            "run = 1\n",
            "must define a top-level",
        ),
        (
            {"type": "python", "module": "transforms.derived"},
            "def run():\n    return None\n",
            "must accept either",
        ),
        (
            {"type": "python", "module": "transforms.derived"},
            "async def run(deps):\n    return None\n",
            "async transform functions are not supported",
        ),
    ],
)
def test_transform_contract_is_validated_at_compile_time(
    tmp_path: Path,
    transform: dict[str, object],
    module_source: str | None,
    message: str,
) -> None:
    if module_source is not None:
        (tmp_path / "transforms").mkdir()
        (tmp_path / "transforms" / "derived.py").write_text(module_source)
    model = ModelConfig(
        name="derived",
        depends_on=["ref('raw')"],
        transform=transform,
    )

    with pytest.raises(ConfigError, match=message):
        validate_project_contract(
            ProjectConfig(name="p"), [_source()], [_extraction("raw"), model], tmp_path
        )


def test_transform_with_deps_and_optional_context_is_valid(tmp_path: Path) -> None:
    (tmp_path / "transforms").mkdir()
    (tmp_path / "transforms" / "derived.py").write_text(
        "def run(deps, ctx=None):\n    return deps['raw']\n"
    )
    model = ModelConfig(
        name="derived",
        depends_on=["ref('raw')"],
        transform={"type": "python", "module": "transforms.derived"},
    )

    dag = validate_project_contract(
        ProjectConfig(name="p"), [_source()], [_extraction("raw"), model], tmp_path
    )

    assert dag.execution_order() == ["raw", "derived"]


@pytest.mark.parametrize(
    ("module_path", "module_source", "message"),
    [
        ("tests.missing", None, "module not found"),
        ("../outside", None, "valid dotted Python module path"),
        ("tests.custom", "value = 1\n", "must define `run"),
        (
            "tests.custom",
            "def run(connection):\n    return None\n",
            "must accept `\\(con, table_ref\\)`",
        ),
        (
            "tests.custom",
            "async def run(connection, table_ref):\n    return None\n",
            "Async custom test functions are not supported",
        ),
        ("tests.custom", "raise RuntimeError('boom')\n", "could not be imported"),
    ],
)
def test_python_test_contract_is_validated_at_compile_time(
    tmp_path: Path,
    module_path: str,
    module_source: str | None,
    message: str,
) -> None:
    if module_source is not None:
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "custom.py").write_text(module_source)
    model = _extraction("raw")
    model.tests = [{"python": module_path}]

    with pytest.raises(ConfigError, match=message):
        validate_project_contract(ProjectConfig(name="p"), [_source()], [model], tmp_path)


def test_valid_python_test_contract_compiles(tmp_path: Path) -> None:
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "custom.py").write_text(
        "def run(connection, table_ref):\n    return None\n"
    )
    model = _extraction("raw")
    model.tests = [{"python": "tests.custom"}]

    dag = validate_project_contract(ProjectConfig(name="p"), [_source()], [model], tmp_path)

    assert dag.execution_order() == ["raw"]


@pytest.mark.parametrize("kind", ["transform", "python_test"])
@pytest.mark.parametrize("link_style", ["leaf", "intermediate"])
def test_executable_module_symlink_cannot_escape_project(
    tmp_path: Path, kind: str, link_style: str
) -> None:
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "code.py").write_text(
        "def run(first, second=None):\n    return None\n"
    )
    module_dir_name = "transforms" if kind == "transform" else "tests"
    module_dir = project_dir / module_dir_name
    if link_style == "intermediate":
        module_dir.symlink_to(outside, target_is_directory=True)
        module_path = f"{module_dir_name}.code"
    else:
        module_dir.mkdir()
        (module_dir / "code.py").symlink_to(outside / "code.py")
        module_path = f"{module_dir_name}.code"

    if kind == "transform":
        model = ModelConfig(
            name="derived",
            depends_on=["ref('raw')"],
            transform={"type": "python", "module": module_path},
        )
        models = [_extraction("raw"), model]
    else:
        model = _extraction("raw")
        model.tests = [{"python": module_path}]
        models = [model]

    with pytest.raises(ConfigError, match="outside the project directory"):
        validate_project_contract(
            ProjectConfig(name="p"), [_source()], models, project_dir
        )


def test_executable_module_symlink_within_project_is_allowed(tmp_path: Path) -> None:
    (tmp_path / "shared").mkdir()
    (tmp_path / "shared" / "derived.py").write_text(
        "def run(deps):\n    return deps['raw']\n"
    )
    (tmp_path / "transforms").mkdir()
    (tmp_path / "transforms" / "derived.py").symlink_to(
        tmp_path / "shared" / "derived.py"
    )
    model = ModelConfig(
        name="derived",
        depends_on=["ref('raw')"],
        transform={"type": "python", "module": "transforms.derived"},
    )

    dag = validate_project_contract(
        ProjectConfig(name="p"), [_source()], [_extraction("raw"), model], tmp_path
    )

    assert dag.execution_order() == ["raw", "derived"]


@pytest.mark.parametrize(
    "spec",
    [
        "not_empty",
        {"not_null": ["id", "name"]},
        {"unique": "id"},
        {"min_rows": 1},
        {"python": "tests.custom_check"},
        {"matches_regex": {"column": "id", "pattern": r"^A\d+$"}},
        {"accepted_values": {"column": "status", "values": ["open", "closed"]}},
        {"accepted_range": {"column": "score", "min": 0, "max": 1}},
        {"null_rate": {"column": "name", "max": 0.1}},
        {"grounded_in": {"value": "answer", "source": "body", "method": "exact"}},
        {
            "relationships": {
                "column": "parent_id",
                "to": "ref('parent')",
                "field": "id",
            }
        },
    ],
)
def test_every_builtin_test_shape_has_a_valid_form(spec: object) -> None:
    assert parse_test_spec(spec).name


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ({"unknown": "id"}, "Unknown test"),
        ({"not_null": []}, "non-empty list"),
        ({"min_rows": -1}, "non-negative integer"),
        ({"matches_regex": {"column": "id", "pattern": "["}}, "invalid pattern"),
        ({"accepted_values": {"column": "x", "values": []}}, "non-empty values"),
        ({"accepted_range": {"column": "x"}}, "at least one"),
        ({"null_rate": {"column": "x", "max": 2}}, "between 0 and 1"),
        (
            {"grounded_in": {"value": "x", "source": "body", "method": "semantic"}},
            "method must be",
        ),
        (
            {"grounded_in": {"value": "x", "source": "body", "method": []}},
            "method must be",
        ),
        (
            {"relationships": {"column": "x", "to": "ref('parent')"}},
            "relationships requires",
        ),
        ({"unique": "id", "severity": "fatal"}, "Unknown severity"),
        ({"unique": "id", "severity": []}, "Unknown severity"),
        (
            {"null_rate": {"column": "x", 1: "not-an-option-name"}},
            "option names must be strings",
        ),
    ],
)
def test_invalid_builtin_test_shapes_are_rejected(
    spec: object, message: str
) -> None:
    with pytest.raises(SpecError, match=message):
        parse_test_spec(spec)


@pytest.mark.parametrize(
    ("target", "message"),
    [("missing", "unknown model"), ("docs", "target 'docs' is a source")],
)
def test_relationship_target_must_be_a_known_model(
    tmp_path: Path, target: str, message: str
) -> None:
    child = _extraction("child")
    child.tests = [
        {
            "relationships": {
                "column": "parent_id",
                "to": f"ref('{target}')",
                "field": "id",
            }
        }
    ]

    with pytest.raises(ConfigError, match=message):
        validate_project_contract(
            ProjectConfig(name="p"), [_source()], [child], tmp_path
        )


def test_relationship_predecessor_preserves_selectors_and_source_ancestry() -> None:
    sources = [_source("parent_docs"), _source("child_docs")]
    parent = _extraction("parent", "parent_docs")
    child = _extraction("child", "child_docs")
    child.tests = [
        {
            "relationships": {
                "column": "parent_id",
                "to": "ref('parent')",
                "field": "id",
            }
        }
    ]

    dag = ProjectDAG(sources, [child, parent])

    assert dag.execution_order() == ["parent", "child"]
    assert set(dag.select_models(select="+child")) == {"parent", "child"}
    assert set(dag.required_sources(["child"])) == {"parent_docs", "child_docs"}


def test_build_materializes_relationship_target_before_running_child_test(
    tmp_path: Path,
) -> None:
    (tmp_path / "dbt_ml_project.yml").write_text("name: relationship_order\n")
    (tmp_path / "sources").mkdir()
    (tmp_path / "sources" / "docs.yml").write_text(
        "version: 2\nsources:\n  - name: docs\n    path: data/docs\n"
        "    file_pattern: '*.json'\n"
    )
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "a_child.yml").write_text(
        "version: 2\nmodels:\n  - name: child\n"
        "    source: ref('docs')\n    extraction:\n      backend: json\n"
        "      options:\n        fields: [id, parent_id]\n"
        "    tests:\n      - relationships:\n          column: parent_id\n"
        "          to: ref('parent')\n          field: id\n"
    )
    (tmp_path / "models" / "z_parent.yml").write_text(
        "version: 2\nmodels:\n  - name: parent\n"
        "    source: ref('docs')\n    extraction:\n      backend: json\n"
        "      options:\n        fields: [id]\n"
    )
    data = tmp_path / "data" / "docs"
    data.mkdir(parents=True)
    (data / "one.json").write_text('{"id": 1, "parent_id": 1}')

    result = build_project(tmp_path)

    assert [run.model_name for run in result.run_results] == ["parent", "child"]
    relationship = next(test for test in result.test_results if test.test_name == "relationships")
    assert relationship.passed
