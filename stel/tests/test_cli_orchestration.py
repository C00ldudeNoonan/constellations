"""Exit-code + machine-readable output contract for orchestrators (issue #87)."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from click.testing import CliRunner

from stel.cli import cli
from stel.synth import generate_invoices


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


def test_global_context_options_work_after_subcommand(
    tmp_path: Path, example_project_dir: Path
) -> None:
    dst = _copy_example(tmp_path, example_project_dir)
    generate_invoices(2, dst / "data" / "invoices", seed=1)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "run",
            "--project-dir",
            str(dst),
            "--profiles-dir",
            str(dst),
            "--target",
            "dev",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["metadata"]["target"]["name"] == "dev"


def test_run_with_unmatched_tag_succeeds_with_empty_selection(
    tmp_path: Path, example_project_dir: Path
) -> None:
    dst = _copy_example(tmp_path, example_project_dir)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["run", "--project-dir", str(dst), "--select", "tag:not_present"],
    )

    assert result.exit_code == 0, result.output
    assert "No models selected." in result.output


def test_run_malformed_yaml_exits_2(tmp_path: Path, example_project_dir: Path) -> None:
    dst = _copy_example(tmp_path, example_project_dir)
    (dst / "stel_project.yml").write_text("name: broken\nversion: [oops\n")

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


def test_build_json_leaf_test_failure_marks_error(
    tmp_path: Path, example_project_dir: Path
) -> None:
    dst = _copy_example(tmp_path, example_project_dir)
    generate_invoices(5, dst / "data" / "invoices", seed=1)

    # A test that fails on a leaf model (invoice_summary has no descendants), so
    # nothing is skipped and the model's run has no errors — the payload status
    # must still be error, matching the exit code.
    summary = dst / "models" / "invoice_summary.yml"
    summary.write_text(
        summary.read_text().replace("- min_rows: 1", "- min_rows: 100000")
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["--project-dir", str(dst), "build", "--json"])
    assert result.exit_code == 1, result.output
    payload = json.loads(result.output)

    assert payload["metadata"]["status"] == "error"
    assert payload["metadata"]["counts"]["error"] >= 1
    assert payload["metadata"]["counts"]["skipped"] == 0

    summary_row = next(
        r for r in payload["results"] if r["model_name"] == "invoice_summary"
    )
    assert summary_row["status"] == "error"
    assert any("min_rows" in f for f in summary_row["test_failures"])
