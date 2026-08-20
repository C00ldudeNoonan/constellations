from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from stel.docs import generate_docs
from stel.hashing import HASH_DIGEST_SIZE
from stel.manifest import (
    MANIFEST_FILENAME,
    RUN_RESULTS_FILENAME,
    build_manifest,
    build_run_results,
    write_manifest,
    write_run_results,
)
from stel.runner import run_project
from stel.synth import generate_invoices


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
    assert all(
        isinstance(v, str) and len(v) == HASH_DIGEST_SIZE * 2
        for v in versions.values()
    )


def test_manifest_emits_safe_effective_inference_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "stel_project.yml").write_text(
        "\n".join(
            [
                "name: llm_project",
                "version: '0.1.0'",
                "profile: llm_project",
                "source-paths: ['sources']",
                "model-paths: ['models']",
            ]
        )
    )
    (tmp_path / "profiles.yml").write_text(
        "\n".join(
            [
                "llm_project:",
                "  target: dev",
                "  outputs:",
                "    dev:",
                "      warehouse:",
                "        type: duckdb",
                "        path: ./target/stel.duckdb",
                "      llm:",
                "        provider: anthropic",
                "        model: effective-model",
                "        api_key_env: PRIVATE_PROVIDER_KEY",
            ]
        )
    )
    (tmp_path / "sources").mkdir()
    (tmp_path / "sources" / "documents.yml").write_text(
        "\n".join(
            [
                "version: 2",
                "sources:",
                "  - name: documents",
                "    path: data/documents",
            ]
        )
    )
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "extract.yml").write_text(
        "\n".join(
            [
                "version: 2",
                "models:",
                "  - name: extracted_documents",
                "    source: ref('documents')",
                "    extraction:",
                "      backend: llm",
                "      options:",
                "        fields:",
                "          - name: title",
                "            type: string",
            ]
        )
    )
    monkeypatch.setenv("PRIVATE_PROVIDER_KEY", "credential-value-must-not-leak")

    manifest = build_manifest(tmp_path)
    model = next(
        item
        for item in manifest["models"]
        if item["name"] == "extracted_documents"
    )

    assert model["inference"]["provider"] == "anthropic"
    assert model["inference"]["model"] == "effective-model"
    assert model["inference"]["implementation"].startswith("provider-v")
    assert set(model["inference"]) == {"provider", "model", "implementation"}

    (tmp_path / "models" / "context.yml").write_text(
        "\n".join(
            [
                "version: 2",
                "models:",
                "  - name: agent_context",
                "    depends_on:",
                "      - ref('extracted_documents')",
                "    transform:",
                "      type: python",
                "      module: transforms.agent_context",
                "      uses_llm: true",
            ]
        )
    )
    transform_model = next(
        item for item in build_manifest(tmp_path)["models"]
        if item["name"] == "agent_context"
    )
    assert transform_model["inference"] == model["inference"]

    serialized = json.dumps(manifest)
    assert "PRIVATE_PROVIDER_KEY" not in serialized
    assert "credential-value-must-not-leak" not in serialized


def test_bigquery_credentials_are_absent_from_artifacts_and_docs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_name = "DISTINCTIVE_ARTIFACT_BQ_TOKEN_REFERENCE"
    secret = "distinctive-artifact-bq-token-value"
    monkeypatch.setenv(env_name, secret)
    (tmp_path / "stel_project.yml").write_text(
        "\n".join(
            [
                "name: bigquery_artifact_project",
                "version: '0.1.0'",
                "profile: bigquery_artifact_project",
            ]
        )
        + "\n"
    )
    (tmp_path / "profiles.yml").write_text(
        "\n".join(
            [
                "bigquery_artifact_project:",
                "  target: prod",
                "  outputs:",
                "    prod:",
                "      warehouse:",
                "        type: bigquery",
                "        project: public-project-id",
                f'        token: "{{{{ env_var(\'{env_name}\') }}}}"',
            ]
        )
        + "\n"
    )
    (tmp_path / "sources").mkdir()
    (tmp_path / "models").mkdir()

    manifest = build_manifest(tmp_path)
    run_results = build_run_results(tmp_path, [])
    write_manifest(tmp_path)
    write_run_results(tmp_path, [])
    docs = generate_docs(tmp_path)

    rendered = json.dumps({"manifest": manifest, "run_results": run_results})
    rendered += "".join(
        path.read_text()
        for path in (tmp_path / "target").rglob("*")
        if path.is_file()
    )
    assert docs.pages_written == 2
    assert env_name not in rendered
    assert secret not in rendered


def test_manifest_emits_ml_models(tmp_path: Path) -> None:
    (tmp_path / "stel_project.yml").write_text(
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
        "warnings": 0,
    }
    assert meta["target"]["adapter_type"] == "duckdb"
    assert meta["target"]["schema"] == "stel"

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
