"""Backend extraction warnings surface in run output and run_results.

Backends report non-fatal per-document issues via ExtractionResult.warnings;
the runner aggregates them per model (distinct message -> document count), the
CLI prints a capped WARNING section, and run_results.json carries the full
counts. Warnings never change the exit code.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from click.testing import CliRunner

from stel.cli import cli
from stel.manifest import build_run_results
from stel.runner import run_project

SELECTOR_WARNING = "selector '.price' for field 'price' matched nothing"


def _copy_example(tmp_path: Path, example_project_dir: Path) -> Path:
    dst = tmp_path / "project"
    shutil.copytree(
        example_project_dir,
        dst,
        ignore=shutil.ignore_patterns("data", "target", "__pycache__"),
    )
    return dst


def _add_pages_model(project: Path, count: int) -> None:
    """An html model whose selector matches nothing, so every document warns
    with the same message — the aggregation case."""
    pages = project / "data" / "pages"
    pages.mkdir(parents=True)
    for i in range(count):
        (pages / f"page_{i}.html").write_text(
            f"<html><body><p>doc {i}</p></body></html>"
        )
    (project / "sources" / "pages.yml").write_text(
        "version: 2\n"
        "sources:\n"
        "  - name: pages\n"
        '    path: "./data/pages/"\n'
        '    file_pattern: "*.html"\n'
    )
    (project / "models" / "raw_pages.yml").write_text(
        "version: 2\n"
        "models:\n"
        "  - name: raw_pages\n"
        "    source: ref('pages')\n"
        "    extraction:\n"
        "      backend: html\n"
        "      options:\n"
        "        selectors:\n"
        '          price: ".price"\n'
        "    materialization: full\n"
    )


def _add_bare_invoices(project: Path, count: int) -> None:
    """Invoices missing `currency`: the json backend's warning embeds the
    filename, so each document yields a distinct message — the cap case."""
    invoices = project / "data" / "invoices"
    invoices.mkdir(parents=True)
    for i in range(count):
        (invoices / f"inv_{i}.json").write_text(
            json.dumps(
                {
                    "invoice_id": f"inv-{i}",
                    "vendor": "acme",
                    "issue_date": "2026-01-01",
                    "line_items": [],
                    "total": 10.0,
                }
            )
        )


def test_runner_aggregates_warnings_per_model(
    tmp_path: Path, example_project_dir: Path
) -> None:
    project = _copy_example(tmp_path, example_project_dir)
    _add_pages_model(project, count=3)

    results = run_project(project, select="raw_pages")

    (r,) = results
    assert r.warnings == {SELECTOR_WARNING: 3}
    assert r.errors == []


def test_run_results_carry_warning_counts(
    tmp_path: Path, example_project_dir: Path
) -> None:
    project = _copy_example(tmp_path, example_project_dir)
    _add_pages_model(project, count=2)

    results = run_project(project, select="raw_pages")
    payload = build_run_results(project, results)

    (row,) = [r for r in payload["results"] if r["model_name"] == "raw_pages"]
    assert row["warnings"] == {SELECTOR_WARNING: 2}
    assert row["status"] == "success"
    assert payload["metadata"]["counts"]["warnings"] == 2
    assert payload["metadata"]["status"] == "success"


def test_cli_prints_warnings_and_exits_zero(
    tmp_path: Path, example_project_dir: Path
) -> None:
    project = _copy_example(tmp_path, example_project_dir)
    _add_pages_model(project, count=2)

    result = CliRunner().invoke(
        cli, ["--project-dir", str(project), "run", "--select", "raw_pages"]
    )

    assert result.exit_code == 0, result.output + result.stderr
    assert f"WARNING: {SELECTOR_WARNING} (2 documents)" in result.stderr


def test_cli_caps_warning_lines_per_model(
    tmp_path: Path, example_project_dir: Path
) -> None:
    project = _copy_example(tmp_path, example_project_dir)
    _add_bare_invoices(project, count=7)

    result = CliRunner().invoke(
        cli, ["--project-dir", str(project), "run", "--select", "raw_invoices"]
    )

    assert result.exit_code == 0, result.output + result.stderr
    assert result.stderr.count("WARNING:") == 5
    assert "2 more distinct warnings (see run_results.json)" in result.stderr

    on_disk = json.loads((project / "target" / "run_results.json").read_text())
    (row,) = [
        r for r in on_disk["results"] if r["model_name"] == "raw_invoices"
    ]
    assert len(row["warnings"]) == 7
    assert on_disk["metadata"]["counts"]["warnings"] == 7


def test_build_prints_warnings_and_keeps_exit_code(
    tmp_path: Path, example_project_dir: Path
) -> None:
    project = _copy_example(tmp_path, example_project_dir)
    _add_pages_model(project, count=2)

    result = CliRunner().invoke(
        cli, ["--project-dir", str(project), "build", "--select", "raw_pages"]
    )

    assert result.exit_code == 0, result.output + result.stderr
    assert f"WARNING: {SELECTOR_WARNING} (2 documents)" in result.stderr
