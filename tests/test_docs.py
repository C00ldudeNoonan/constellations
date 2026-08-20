from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from stel.docs import generate_docs
from stel.manifest import write_run_results
from stel.runner import run_project
from stel.synth import generate_invoices, generate_support_tickets


@pytest.fixture
def fresh_project(tmp_path: Path, example_project_dir: Path) -> Path:
    dst = tmp_path / "project"
    shutil.copytree(
        example_project_dir,
        dst,
        ignore=shutil.ignore_patterns("data", "target", "__pycache__"),
    )
    return dst


def test_generate_docs_basic(fresh_project: Path) -> None:
    generate_invoices(5, fresh_project / "data" / "invoices", seed=1)
    run_project(fresh_project)
    result = generate_docs(fresh_project)

    assert result.pages_written >= 4
    assert result.output_dir.exists()
    files = {p.name for p in result.output_dir.glob("*.html")}
    assert "index.html" in files
    assert "lineage.html" in files
    assert "model_raw_invoices.html" in files
    assert "model_invoice_summary.html" in files


def test_generate_docs_creates_manifest_if_missing(fresh_project: Path) -> None:
    """If you run docs generate before any compile/run, it should still work."""
    result = generate_docs(fresh_project)
    assert (result.output_dir / "index.html").exists()
    assert (fresh_project / "target" / "manifest.json").exists()


def test_index_includes_project_name(fresh_project: Path) -> None:
    generate_invoices(2, fresh_project / "data" / "invoices", seed=1)
    run_project(fresh_project)
    result = generate_docs(fresh_project)
    text = (result.output_dir / "index.html").read_text()
    assert "invoice_pipeline" in text
    assert "raw_invoices" in text


def test_model_page_renders_with_run_data(fresh_project: Path) -> None:
    generate_invoices(4, fresh_project / "data" / "invoices", seed=1)
    results = run_project(fresh_project)
    write_run_results(fresh_project, results)
    result = generate_docs(fresh_project)
    raw_page = (result.output_dir / "model_raw_invoices.html").read_text()
    assert "Last run" in raw_page
    assert "rows written" in raw_page


def test_model_page_renders_ml_artifact_metadata(tmp_path: Path) -> None:
    src = Path(__file__).resolve().parents[1] / "examples" / "classic_text_ml"
    project = tmp_path / "classic_text_ml"
    shutil.copytree(src, project, ignore=shutil.ignore_patterns("data", "target"))
    generate_support_tickets(4, project / "data" / "tickets", seed=15)
    results = run_project(project)
    write_run_results(project, results)

    result = generate_docs(project)
    page = (result.output_dir / "model_ticket_tfidf.html").read_text()
    assert "artifact version" in page
    assert "artifact metadata" in page
    assert "artifact_files_hash" in page
