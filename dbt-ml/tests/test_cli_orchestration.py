"""Exit-code + machine-readable output contract for orchestrators (issue #87)."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from click.testing import CliRunner

from dbt_ml.cli import cli
from dbt_ml.synth import generate_invoices


def _copy_example(tmp_path: Path, example_project_dir: Path) -> Path:
    dst = tmp_path / "project"
    shutil.copytree(
        example_project_dir,
        dst,
        ignore=shutil.ignore_patterns("data", "target", "__pycache__"),
    )
    return dst


def test_run_json_emits_parseable_payload(
    tmp_path: Path, example_project_dir: Path
) -> None:
    dst = _copy_example(tmp_path, example_project_dir)
    generate_invoices(6, dst / "data" / "invoices", seed=1)

    runner = CliRunner()
    result = runner.invoke(cli, ["--project-dir", str(dst), "run", "--json"])
    assert result.exit_code == 0, result.output

    payload = json.loads(result.output)
    meta = payload["metadata"]
    assert meta["status"] == "success"
    assert meta["invocation"] == "run"
    assert meta["elapsed_seconds"] is not None
    assert meta["counts"]["total"] == len(payload["results"])
    assert meta["target"]["adapter_type"] == "duckdb"

    raw = next(r for r in payload["results"] if r["model_name"] == "raw_invoices")
    assert raw["status"] == "success"
    assert raw["relation"]["name"] == "raw_invoices"
    assert raw["relation"]["fully_qualified"].endswith(".raw_invoices")

    # stdout must be byte-identical to the on-disk artifact.
    on_disk = (dst / "target" / "run_results.json").read_text()
    assert result.output.strip() == on_disk.strip()


def test_run_malformed_yaml_exits_2(tmp_path: Path, example_project_dir: Path) -> None:
    dst = _copy_example(tmp_path, example_project_dir)
    (dst / "dbt_ml_project.yml").write_text("name: broken\nversion: [oops\n")

    runner = CliRunner()
    result = runner.invoke(cli, ["--project-dir", str(dst), "run"])
    assert result.exit_code == 2, result.output


def test_run_missing_source_exits_2(tmp_path: Path, example_project_dir: Path) -> None:
    dst = _copy_example(tmp_path, example_project_dir)
    bad = dst / "models" / "raw_invoices.yml"
    bad.write_text(bad.read_text().replace("ref('vendor_invoices')", "ref('nope')"))

    runner = CliRunner()
    result = runner.invoke(cli, ["--project-dir", str(dst), "run"])
    # Unknown ref is a DAG/config error → the run never starts.
    assert result.exit_code == 2, result.output


def test_run_document_failure_exits_1(
    tmp_path: Path, example_project_dir: Path
) -> None:
    dst = _copy_example(tmp_path, example_project_dir)
    generate_invoices(3, dst / "data" / "invoices", seed=1)
    (dst / "data" / "invoices" / "corrupt.json").write_text("{ not valid json")

    runner = CliRunner()
    result = runner.invoke(cli, ["--project-dir", str(dst), "run", "--json"])
    # The run started but a document failed → exit 1, status error, error captured.
    assert result.exit_code == 1, result.output
    payload = json.loads(result.output)
    assert payload["metadata"]["status"] == "error"
    raw = next(r for r in payload["results"] if r["model_name"] == "raw_invoices")
    assert raw["status"] == "error"
    assert any("corrupt.json" in e for e in raw["errors"])


def test_build_json_marks_skipped_downstream(
    tmp_path: Path, example_project_dir: Path
) -> None:
    dst = _copy_example(tmp_path, example_project_dir)
    generate_invoices(5, dst / "data" / "invoices", seed=1)

    # A test that fails for every row hard-fails raw_invoices and blocks its
    # descendants, which build reports as skipped.
    raw = dst / "models" / "raw_invoices.yml"
    raw.write_text(
        raw.read_text().replace(
            "tests:",
            "tests:\n      - accepted_values: {column: currency, values: [XXX]}",
            1,
        )
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["--project-dir", str(dst), "build", "--json"])
    assert result.exit_code == 1, result.output
    payload = json.loads(result.output)
    statuses = {r["model_name"]: r["status"] for r in payload["results"]}
    assert statuses["invoice_summary"] == "skipped"
    assert payload["metadata"]["counts"]["skipped"] >= 1
