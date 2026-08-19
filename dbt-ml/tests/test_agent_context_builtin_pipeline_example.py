"""End-to-end proof that examples/agent_context_from_builtin_pipeline (issue
#300) makes a built-in extraction:/chunk: pipeline agent_context/v1-
discoverable through the same catalog logic the MCP server uses."""
from __future__ import annotations

import shutil
from pathlib import Path

import duckdb

from dbt_ml.checks import run_project_tests
from dbt_ml.manifest import write_manifest
from dbt_ml.mcp_server.catalog import ArtifactCatalog
from dbt_ml.runner import run_project


def _project(tmp_path: Path) -> Path:
    repo = Path(__file__).resolve().parents[1]
    src = repo / "examples" / "agent_context_from_builtin_pipeline"
    dst = tmp_path / "agent_context_from_builtin_pipeline"
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("target", "__pycache__"))
    return dst


def _row_count(project: Path, table: str) -> int:
    con = duckdb.connect(str(project / "target" / "context.duckdb"), read_only=True)
    try:
        row = con.execute(f'SELECT COUNT(*) FROM "context".context.{table}').fetchone()
        assert row is not None
        return row[0]
    finally:
        con.close()


def test_builtin_pipeline_example_all_checks_pass(tmp_path: Path) -> None:
    project = _project(tmp_path)
    run_project(project)
    results = run_project_tests(project)
    failed = [r for r in results if not r.passed]
    assert failed == [], f"unexpected failures: {[(r.test_name, r.message) for r in failed]}"


def test_builtin_pipeline_example_chunks_are_real_multi_chunk_splits(tmp_path: Path) -> None:
    project = _project(tmp_path)
    run_project(project)

    documents = _row_count(project, "raw_research_notes")
    real_chunks = _row_count(project, "research_note_chunks")
    contract_chunks = _row_count(project, "document_chunks")

    # The real chunk: splitter produces several chunks per document, not one —
    # proving this isn't the metric_evidence_agent shortcut of one chunk per doc.
    assert real_chunks > documents
    # document_chunks projects exactly one contract row per real chunk row.
    assert contract_chunks == real_chunks


def test_builtin_pipeline_example_is_mcp_discoverable(tmp_path: Path) -> None:
    project = _project(tmp_path)
    run_project(project)
    write_manifest(project)

    catalog = ArtifactCatalog.load(project)
    resource = catalog.get("context_search")
    assert resource is not None

    summary = resource.summary()
    assert summary.contract == "agent_context/v1"
    assert summary.grain == "document_chunks"
    assert summary.access == "governed"
