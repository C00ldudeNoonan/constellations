from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from dbt_ml.manifest import (
    MANIFEST_FILENAME,
    RUN_RESULTS_FILENAME,
    build_manifest,
    write_manifest,
    write_run_results,
)
from dbt_ml.runner import run_project
from dbt_ml.synth import generate_invoices


@pytest.fixture
def fresh_project(tmp_path: Path, example_project_dir: Path) -> Path:
    dst = tmp_path / "project"
    shutil.copytree(
        example_project_dir,
        dst,
        ignore=shutil.ignore_patterns("data", "target", "__pycache__"),
    )
    return dst


def test_manifest_shape(fresh_project: Path) -> None:
    m = build_manifest(fresh_project)
    assert m["manifest_version"] == 1
    assert m["project"]["name"] == "invoice_pipeline"
    assert {s["name"] for s in m["sources"]} == {"vendor_invoices"}
    assert {x["name"] for x in m["models"]} == {
        "raw_invoices",
        "invoice_summary",
        "monthly_totals",
    }
    assert m["dag"]["execution_order"][0] == "raw_invoices"
    assert ["vendor_invoices", "raw_invoices"] in m["dag"]["edges"]


def test_manifest_has_code_versions(fresh_project: Path) -> None:
    m = build_manifest(fresh_project)
    versions = {x["name"]: x["code_version"] for x in m["models"]}
    assert all(isinstance(v, str) and len(v) == 16 for v in versions.values())


def test_manifest_emits_ml_models(tmp_path: Path) -> None:
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
    (tmp_path / "models" / "raw_tickets.yml").write_text(
        "\n".join(
            [
                "version: 2",
                "models:",
                "  - name: raw_tickets",
                "    source: ref('tickets')",
                "    extraction:",
                "      backend: json",
                "      options:",
                "        fields: [body]",
            ]
        )
    )
    (tmp_path / "models" / "ticket_features.yml").write_text(
        "\n".join(
            [
                "version: 2",
                "models:",
                "  - name: ticket_tfidf",
                "    depends_on: [ref('raw_tickets')]",
                "    ml:",
                "      task: features",
                "      provider: builtin.tfidf",
                "      text_field: body",
                "      artifact:",
                "        path: target/artifacts/ticket_tfidf",
            ]
        )
    )

    manifest = build_manifest(tmp_path)
    model = next(m for m in manifest["models"] if m["name"] == "ticket_tfidf")
    assert model["kind"] == "ml"
    assert model["ml"]["task"] == "features"
    assert model["ml"]["provider"] == "builtin.tfidf"
    assert model["ml"]["artifact"]["path"] == "target/artifacts/ticket_tfidf"
    assert isinstance(model["code_version"], str)


def test_write_manifest_creates_file(fresh_project: Path) -> None:
    path = write_manifest(fresh_project)
    assert path.exists()
    assert path.name == MANIFEST_FILENAME
    payload = json.loads(path.read_text())
    assert payload["project"]["name"] == "invoice_pipeline"


def test_run_writes_run_results(fresh_project: Path) -> None:
    generate_invoices(3, fresh_project / "data" / "invoices", seed=1)
    results = run_project(fresh_project)
    path = write_run_results(fresh_project, results, elapsed_seconds=1.5)
    assert path.exists()
    assert path.name == RUN_RESULTS_FILENAME
    payload = json.loads(path.read_text())
    assert len(payload["results"]) == len(results)
    assert {r["model_name"] for r in payload["results"]} == {r.model_name for r in results}


def test_run_results_metadata_and_relations(fresh_project: Path) -> None:
    generate_invoices(3, fresh_project / "data" / "invoices", seed=1)
    results = run_project(fresh_project)
    payload = json.loads(
        write_run_results(fresh_project, results, elapsed_seconds=2.0).read_text()
    )

    meta = payload["metadata"]
    assert meta["status"] == "success"
    assert meta["invocation"] == "run"
    assert meta["elapsed_seconds"] == 2.0
    assert meta["counts"] == {
        "total": len(results),
        "success": len(results),
        "error": 0,
        "skipped": 0,
    }
    assert meta["target"]["adapter_type"] == "duckdb"
    assert meta["target"]["schema"] == "dbt_ml"

    for row in payload["results"]:
        assert row["status"] == "success"
        rel = row["relation"]
        assert rel["name"] == row["model_name"]
        assert rel["fully_qualified"].endswith("." + row["model_name"])


def test_run_results_reports_skipped(fresh_project: Path) -> None:
    generate_invoices(2, fresh_project / "data" / "invoices", seed=1)
    results = run_project(fresh_project, select="raw_invoices")
    payload = json.loads(
        write_run_results(
            fresh_project,
            results,
            invocation="build",
            skipped=["invoice_summary"],
        ).read_text()
    )
    statuses = {r["model_name"]: r["status"] for r in payload["results"]}
    assert statuses["invoice_summary"] == "skipped"
    assert payload["metadata"]["status"] == "error"
    assert payload["metadata"]["counts"]["skipped"] == 1
