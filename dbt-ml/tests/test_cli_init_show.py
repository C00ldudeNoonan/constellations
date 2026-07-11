from __future__ import annotations

import io
import json
import shutil
from pathlib import Path

import click
import pytest
from click.testing import CliRunner

from dbt_ml.cli import _safe_console_text, cli
from dbt_ml.runner import run_project
from dbt_ml.synth import generate_invoices


def _copy_example(tmp_path: Path, example_project_dir: Path) -> Path:
    dst = tmp_path / "project"
    shutil.copytree(
        example_project_dir,
        dst,
        ignore=shutil.ignore_patterns("data", "target", "__pycache__"),
    )
    return dst


def test_test_store_failures_persists_and_reports(
    tmp_path: Path, example_project_dir: Path
) -> None:
    dst = _copy_example(tmp_path, example_project_dir)
    generate_invoices(5, dst / "data" / "invoices", seed=1)
    run_project(dst)

    # currency is USD/EUR/etc in synthetic data — XXX guarantees every row fails.
    raw = dst / "models" / "raw_invoices.yml"
    raw.write_text(
        raw.read_text().replace(
            "tests:",
            "tests:\n      - accepted_values: {column: currency, values: [XXX]}",
            1,
        )
    )

    runner = CliRunner()
    result = runner.invoke(
        cli, ["--project-dir", str(dst), "test", "--store-failures"]
    )
    assert result.exit_code == 1, result.output
    assert "stored" in result.output
    table = "dbt_ml_test_failures__raw_invoices__accepted_values__currency"
    assert table in result.output

    show = runner.invoke(cli, ["--project-dir", str(dst), "ls"])
    assert table not in show.output  # inspection table, not a model


def test_ls_lists_models(tmp_path: Path, example_project_dir: Path) -> None:
    dst = _copy_example(tmp_path, example_project_dir)
    runner = CliRunner()
    result = runner.invoke(cli, ["--project-dir", str(dst), "ls"])
    assert result.exit_code == 0, result.output
    assert "raw_invoices" in result.output
    assert "extraction" in result.output
    assert "invoice_summary" in result.output
    # sources are excluded by default
    assert "vendor_invoices" not in result.output


def test_ls_select_and_resource_type(tmp_path: Path, example_project_dir: Path) -> None:
    dst = _copy_example(tmp_path, example_project_dir)
    runner = CliRunner()
    result = runner.invoke(
        cli, ["--project-dir", str(dst), "ls", "--select", "raw_invoices"]
    )
    assert result.exit_code == 0, result.output
    assert "raw_invoices" in result.output
    assert "invoice_summary" not in result.output

    src = runner.invoke(
        cli, ["--project-dir", str(dst), "ls", "--resource-type", "source"]
    )
    assert src.exit_code == 0, src.output
    assert "vendor_invoices" in src.output


def test_ls_json_output(tmp_path: Path, example_project_dir: Path) -> None:
    dst = _copy_example(tmp_path, example_project_dir)
    runner = CliRunner()
    result = runner.invoke(
        cli, ["--project-dir", str(dst), "ls", "--output", "json"]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    names = {row["name"] for row in payload}
    assert "raw_invoices" in names
    assert all(row["resource_type"] == "model" for row in payload)


def test_init_creates_runnable_project(tmp_path: Path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as cwd:
        result = runner.invoke(cli, ["init", "scaffold"], catch_exceptions=False)
        assert result.exit_code == 0, result.output

        target = Path(cwd) / "scaffold"
        assert (target / "dbt_ml_project.yml").exists()
        assert (target / "profiles.yml").exists()
        assert (target / "sources" / "invoices.yml").exists()
        assert (target / "models" / "raw_invoices.yml").exists()
        project_text = (target / "dbt_ml_project.yml").read_text()
        profiles_text = (target / "profiles.yml").read_text()
        assert "name: scaffold" in project_text
        assert "profile: scaffold" in project_text
        assert "schema: scaffold" in profiles_text
        assert "__PROJECT_NAME__" not in project_text
        assert "__PROJECT_NAME__" not in profiles_text


def test_init_with_pdf_template(tmp_path: Path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as cwd:
        result = runner.invoke(cli, ["init", "p", "--template", "pdf"])
        assert result.exit_code == 0, result.output
        target = Path(cwd) / "p"
        assert (target / "models" / "raw_pdf_text.yml").exists()
        assert "backend: pdf" in (target / "models" / "raw_pdf_text.yml").read_text()
        assert "default_backend: pdf" in (target / "dbt_ml_project.yml").read_text()


def test_init_with_markdown_template(tmp_path: Path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as cwd:
        result = runner.invoke(cli, ["init", "p", "--template", "markdown"])
        assert result.exit_code == 0, result.output
        target = Path(cwd) / "p"
        assert (target / "models" / "raw_documents.yml").exists()


def test_init_with_html_template(tmp_path: Path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as cwd:
        result = runner.invoke(cli, ["init", "p", "--template", "html"])
        assert result.exit_code == 0, result.output
        target = Path(cwd) / "p"
        assert (target / "models" / "raw_pages.yml").exists()


def test_init_unknown_template_rejected(tmp_path: Path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli, ["init", "p", "--template", "bogus"])
        assert result.exit_code != 0


def test_init_refuses_existing_dir(tmp_path: Path) -> None:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        Path("existing").mkdir()
        result = runner.invoke(cli, ["init", "existing"])
        assert result.exit_code != 0
        assert "already exists" in result.output


def test_show_prints_rows(tmp_path: Path, example_project_dir: Path) -> None:
    import shutil

    dst = tmp_path / "project"
    shutil.copytree(
        example_project_dir,
        dst,
        ignore=shutil.ignore_patterns("data", "target", "__pycache__"),
    )
    generate_invoices(8, dst / "data" / "invoices", seed=1)
    run_project(dst)

    runner = CliRunner()
    result = runner.invoke(cli, ["--project-dir", str(dst), "show", "raw_invoices", "--limit", "3"])
    assert result.exit_code == 0, result.output
    assert "shape: (3," in result.output  # polars repr always starts with shape


def test_show_output_is_safe_for_strict_cp1252_console() -> None:
    class StrictCp1252Stream:
        encoding = "cp1252"
        errors = "strict"

    text = "shape: (1, 1)\n\u250c\u2500\u2500\u2510\n\u6771\u4eac"
    safe = _safe_console_text(text, StrictCp1252Stream())

    safe.encode("cp1252", errors="strict")
    assert safe != text
    assert "shape: (1, 1)" in safe


def test_show_output_is_safe_when_console_wrapper_reports_replace() -> None:
    class ReplacingCp1252Stream:
        encoding = "cp1252"
        errors = "replace"

    text = "shape: (1, 1)\n\u250c\u2500\u2500\u2510\n\u6771\u4eac"
    safe = _safe_console_text(text, ReplacingCp1252Stream())

    safe.encode("cp1252", errors="strict")
    assert safe != text


def test_show_writes_through_same_strict_cp1252_stream(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, example_project_dir: Path
) -> None:
    dst = _copy_example(tmp_path, example_project_dir)
    generate_invoices(1, dst / "data" / "invoices", seed=1)
    run_project(dst)

    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="cp1252", errors="strict")
    monkeypatch.setattr(click, "get_text_stream", lambda name: stream)

    result = CliRunner().invoke(
        cli, ["--project-dir", str(dst), "show", "raw_invoices", "--limit", "1"]
    )

    stream.flush()
    rendered = raw.getvalue().decode("cp1252")
    assert result.exit_code == 0, result.output
    assert "shape: (1," in rendered
    assert "\u250c" not in rendered


def test_show_missing_db(tmp_path: Path, example_project_dir: Path) -> None:
    import shutil

    dst = tmp_path / "project"
    shutil.copytree(
        example_project_dir,
        dst,
        ignore=shutil.ignore_patterns("data", "target", "__pycache__"),
    )
    runner = CliRunner()
    result = runner.invoke(cli, ["--project-dir", str(dst), "show", "raw_invoices"])
    assert result.exit_code != 0
    assert "Run `dbt-ml run`" in result.output


def test_show_unknown_model(tmp_path: Path, example_project_dir: Path) -> None:
    import shutil

    dst = tmp_path / "project"
    shutil.copytree(
        example_project_dir,
        dst,
        ignore=shutil.ignore_patterns("data", "target", "__pycache__"),
    )
    generate_invoices(2, dst / "data" / "invoices", seed=1)
    run_project(dst)

    runner = CliRunner()
    result = runner.invoke(cli, ["--project-dir", str(dst), "show", "no_such_model"])
    assert result.exit_code != 0
    assert "not found" in result.output
