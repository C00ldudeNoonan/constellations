from __future__ import annotations

from pathlib import Path

import pytest

from dbt_ml.config import ConfigError, load_project
from dbt_ml.config.model import ModelConfig, ModelFile


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
