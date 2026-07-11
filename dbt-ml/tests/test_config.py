from __future__ import annotations

from pathlib import Path
from types import ModuleType

import pytest
from click.testing import CliRunner
from pydantic import BaseModel

from dbt_ml.cli import cli
from dbt_ml.config import ConfigError, load_project
from dbt_ml.config import model as model_config_module
from dbt_ml.config import profile as profile_config_module
from dbt_ml.config import project as project_config_module
from dbt_ml.config import source as source_config_module
from dbt_ml.config.model import ExtractionConfig, FieldConfig, ModelConfig, ModelFile


def test_load_example_project(example_project_dir: Path) -> None:
    project, sources, models = load_project(example_project_dir)
    assert project.name == "invoice_pipeline"
    assert project.duckdb.schema_name == "dbt_ml"
    assert project.extraction.default_backend == "json"
    assert {s.name for s in sources} == {"vendor_invoices"}
    assert {m.name for m in models} == {"raw_invoices", "invoice_summary", "monthly_totals"}


def test_missing_project_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match=r"dbt_ml_project\.yml"):
        load_project(tmp_path)


def test_invalid_yaml_reports_path(tmp_path: Path) -> None:
    (tmp_path / "dbt_ml_project.yml").write_text("name: x\n")
    (tmp_path / "sources").mkdir()
    (tmp_path / "sources" / "bad.yml").write_text(
        "version: 2\nsources:\n  - description: 'missing required name'\n"
    )
    with pytest.raises(ConfigError, match=r"bad\.yml"):
        load_project(tmp_path)


def test_raw_invoices_is_incremental(example_project_dir: Path) -> None:
    _, _, models = load_project(example_project_dir)
    raw = next(m for m in models if m.name == "raw_invoices")
    assert raw.materialization == "incremental"
    assert raw.extraction is not None
    assert raw.extraction.backend == "json"
    assert raw.extraction.options["fields"] == [
        "invoice_id",
        "vendor",
        "issue_date",
        "line_items",
        "total",
        "currency",
    ]


@pytest.mark.parametrize(
    ("key", "value", "expected"),
    [
        ("data_type", "integer", "integer"),
        ("data-type", "bool", "boolean"),
        ("type", "number", "float"),
        ("dtype", "varchar", "string"),
    ],
)
def test_field_data_type_aliases(
    key: str, value: str, expected: str
) -> None:
    field = FieldConfig.model_validate({"name": "value", key: value})
    assert field.data_type == expected
    assert field.model_dump()["data_type"] == expected


def test_field_rejects_unknown_data_type() -> None:
    with pytest.raises(ValueError, match="data_type"):
        FieldConfig(name="value", data_type="uuid")


def test_invoice_summary_depends_on_raw(example_project_dir: Path) -> None:
    _, _, models = load_project(example_project_dir)
    summary = next(m for m in models if m.name == "invoice_summary")
    assert summary.materialization == "full"
    assert summary.depends_on == ["ref('raw_invoices')"]
    assert summary.transform is not None
    assert summary.transform.module == "transforms.summarize"


def test_loads_classic_ml_model_config(tmp_path: Path) -> None:
    (tmp_path / "dbt_ml_project.yml").write_text(
        "\n".join(
            [
                "name: classic_ml_project",
                "version: '0.1.0'",
                "source-paths: ['sources']",
                "model-paths: ['models']",
            ]
        )
    )
    (tmp_path / "sources").mkdir()
    (tmp_path / "sources" / "tickets.yml").write_text(
        "\n".join(
            [
                "version: 2",
                "sources:",
                "  - name: tickets",
                "    path: data/tickets",
            ]
        )
    )
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "ticket_features.yml").write_text(
        "\n".join(
            [
                "version: 2",
                "models:",
                "  - name: ticket_tfidf",
                "    depends_on: [ref('raw_tickets')]",
                "    ml:",
                "      task: features",
                "      mode: fit_transform",
                "      provider: builtin.tfidf",
                "      text_field: body",
                "      artifact:",
                "        path: target/artifacts/ticket_tfidf",
                "      metrics: [vocabulary_size]",
                "      options:",
                "        ngram_range: [1, 2]",
                "        max_features: 50000",
            ]
        )
    )

    _, _, models = load_project(tmp_path)
    ml_model = models[0]
    assert ml_model.ml is not None
    assert ml_model.ml.task == "features"
    assert ml_model.ml.mode == "fit_transform"
    assert ml_model.ml.provider == "builtin.tfidf"
    assert ml_model.ml.text_field == "body"
    assert ml_model.ml.artifact.path == Path("target/artifacts/ticket_tfidf")
    assert ml_model.ml.metrics == ["vocabulary_size"]
    assert ml_model.ml.options["max_features"] == 50000


def test_multiple_kind_blocks_rejected() -> None:
    with pytest.raises(ValueError, match="multiple kind blocks"):
        ModelConfig(
            name="conflicted",
            extraction={"backend": "json"},
            transform={"type": "python", "module": "transforms.x"},
        )


def test_model_file_requires_kind_block() -> None:
    with pytest.raises(ValueError, match="missing an extraction/transform/ml/chunk block"):
        ModelFile.model_validate(
            {"version": 2, "models": [{"name": "kindless"}]}
        )


def test_bare_model_config_allowed_programmatically() -> None:
    # DAG fixtures and docs tooling build ModelConfig directly without a
    # kind block; only the YAML load path requires one.
    assert ModelConfig(name="fixture_only").kind_block_count == 0


def test_kindless_model_fails_at_load(tmp_path: Path) -> None:
    (tmp_path / "dbt_ml_project.yml").write_text("name: p\n")
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "m.yml").write_text(
        "version: 2\nmodels:\n  - name: no_kind\n"
    )
    with pytest.raises(ConfigError, match="no_kind"):
        load_project(tmp_path)


def _public_config_models(module: ModuleType) -> list[type[BaseModel]]:
    return [
        value
        for value in vars(module).values()
        if isinstance(value, type)
        and issubclass(value, BaseModel)
        and value.__module__ == module.__name__
    ]


def test_all_public_config_models_forbid_unknown_fields() -> None:
    modules = (
        model_config_module,
        profile_config_module,
        project_config_module,
        source_config_module,
    )
    models = [model for module in modules for model in _public_config_models(module)]

    assert models
    assert all(model.model_config.get("extra") == "forbid" for model in models)


def _compile_fixture(tmp_path: Path, model_yaml: str) -> Path:
    (tmp_path / "dbt_ml_project.yml").write_text("name: strict_project\n")
    (tmp_path / "sources").mkdir()
    (tmp_path / "sources" / "docs.yml").write_text(
        "version: 2\nsources:\n  - name: docs\n    path: data\n"
    )
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "strict.yml").write_text(model_yaml)
    return tmp_path


def test_compile_unknown_model_key_is_precise_exit_2(tmp_path: Path) -> None:
    project = _compile_fixture(
        tmp_path,
        "version: 2\nmodels:\n  - name: raw_docs\n"
        "    source: ref('docs')\n"
        "    extraction:\n      backend: json\n"
        "    materializtion: incremental\n",
    )

    result = CliRunner().invoke(cli, ["--project-dir", str(project), "compile"])

    assert result.exit_code == 2, result.output
    assert "strict.yml" in result.output
    assert "models.0.materializtion" in result.output
    assert "Extra inputs are not permitted" in result.output
    assert not (project / "target" / "manifest.json").exists()


def test_compile_unknown_source_key_is_precise_exit_2(tmp_path: Path) -> None:
    (tmp_path / "dbt_ml_project.yml").write_text("name: strict_project\n")
    (tmp_path / "sources").mkdir()
    (tmp_path / "sources" / "docs.yml").write_text(
        "version: 2\nsources:\n  - name: docs\n    path: data\n"
        "    file_patern: '*.json'\n"
    )

    result = CliRunner().invoke(cli, ["--project-dir", str(tmp_path), "compile"])

    assert result.exit_code == 2, result.output
    assert "docs.yml" in result.output
    assert "sources.0.file_patern" in result.output


@pytest.mark.parametrize("directory, root_key", [("sources", "sources"), ("models", "models")])
def test_yaml_schema_version_other_than_2_is_rejected(
    tmp_path: Path, directory: str, root_key: str
) -> None:
    (tmp_path / "dbt_ml_project.yml").write_text("name: versioned\n")
    path = tmp_path / directory
    path.mkdir()
    if root_key == "sources":
        item = "  - name: docs\n    path: data\n"
    else:
        item = "  - name: raw_docs\n    extraction:\n      backend: json\n"
    (path / "bad_version.yml").write_text(
        f"version: 1\n{root_key}:\n{item}"
    )

    with pytest.raises(ConfigError, match=r"(?s)version.*Input should be 2"):
        load_project(tmp_path)


@pytest.mark.parametrize(
    "options",
    [
        {"fields": ["document_id"]},
        {"fields": [{"name": "source_uri", "type": "string"}]},
        {"frontmatter_fields": ["backend_name"]},
        {"text_field": "CONTENT_HASH"},
        {"body_field": "code_version"},
        {"selectors": {"extracted_at": "h1"}},
    ],
)
def test_extraction_options_reject_reserved_lineage_fields(
    options: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="reserved dbt-ml lineage columns"):
        ExtractionConfig(backend="llm", options=options)


def test_compile_reserved_extraction_field_is_exit_2(tmp_path: Path) -> None:
    project = _compile_fixture(
        tmp_path,
        "version: 2\nmodels:\n  - name: raw_docs\n"
        "    source: ref('docs')\n"
        "    extraction:\n      backend: llm\n      options:\n"
        "        fields:\n          - name: document_id\n            type: string\n",
    )

    result = CliRunner().invoke(cli, ["--project-dir", str(project), "compile"])

    assert result.exit_code == 2, result.output
    assert "strict.yml" in result.output
    assert "document_id" in result.output
    assert "reserved dbt-ml lineage columns" in result.output


def test_model_cannot_select_operator_credential_environment_variable(
    tmp_path: Path,
) -> None:
    project = _compile_fixture(
        tmp_path,
        "version: 2\nmodels:\n  - name: raw_docs\n"
        "    source: ref('docs')\n"
        "    extraction:\n      backend: llm\n      options:\n"
        "        api_key_env: GITHUB_TOKEN\n"
        "        fields:\n          - name: title\n            type: string\n",
    )

    result = CliRunner().invoke(cli, ["--project-dir", str(project), "compile"])

    assert result.exit_code == 2, result.output
    assert "operator-owned configuration" in result.output
    assert "profiles.yml" in result.output


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("temperature", -0.1),
        ("temperature", float("nan")),
        ("max_tokens", 0),
        ("max_tokens", 65_537),
        ("max_retries", 21),
        ("max_concurrent", 0),
        ("max_concurrent", True),
        ("batch_poll_seconds", 0),
        ("batch_poll_seconds", 3601),
    ],
)
def test_extraction_config_rejects_unbounded_llm_numeric_options(
    name: str, value: object
) -> None:
    with pytest.raises(ValueError, match=f"llm option '{name}'"):
        ExtractionConfig(backend="llm", options={name: value})


def test_invalid_llm_option_fails_before_gcs_source_discovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "dbt_ml_project.yml").write_text(
        "name: llm_project\nextraction:\n  default_backend: llm\n"
    )
    (tmp_path / "sources").mkdir()
    (tmp_path / "sources" / "docs.yml").write_text(
        "version: 2\nsources:\n  - name: docs\n    path: gs://bucket/docs\n"
    )
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "raw.yml").write_text(
        "version: 2\nmodels:\n  - name: raw_docs\n"
        "    source: ref('docs')\n"
        "    extraction:\n      options:\n        max_concurrent: 0\n"
    )
    discovered = False

    def _unexpected_client(*args: object, **kwargs: object) -> None:
        nonlocal discovered
        discovered = True
        raise AssertionError("GCS discovery must not run")

    monkeypatch.setattr(
        "dbt_ml.sources.gcs.GCSDocumentSource._make_client", _unexpected_client
    )

    result = CliRunner().invoke(cli, ["--project-dir", str(tmp_path), "run"])

    assert result.exit_code == 2, result.output
    assert "max_concurrent" in result.output
    assert not discovered
