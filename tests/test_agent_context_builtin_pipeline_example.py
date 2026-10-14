"""End-to-end proof that examples/agent_context_from_builtin_pipeline (issue
#300) makes a built-in extraction:/chunk: pipeline agent_context/v1-
discoverable through the same catalog logic the MCP server uses."""
from __future__ import annotations

import logging
import shutil
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pytest

from stel.agent_context import FreshnessStatus
from stel.checks import run_project_tests
from stel.manifest import write_manifest
from stel.mcp_server.catalog import ArtifactCatalog
from stel.mcp_server.repository import WarehouseContextRepository
from stel.mcp_server.serving_state import freshness, read_activations
from stel.runner import run_project
from stel.search import SearchSession

# Runs a whole project or opens a retrieval store, so it belongs to the
# `e2e` tier (issue #518). `test_test_tiers.py` fails if a file that
# does either is missing this.
pytestmark = pytest.mark.e2e


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

    # Live state, supplied by the server: this test is about the descriptor
    # the catalog builds, and the activation read has its own test below.
    summary = resource.summary(
        freshness=FreshnessStatus.UNKNOWN,
        last_successful_materialization=None,
    )
    assert summary.contract == "agent_context/v1"
    assert summary.grain == "document_chunks"
    assert summary.access == "governed"


def test_a_published_index_reports_its_activation_through_a_real_warehouse(
    tmp_path: Path,
) -> None:
    """The catalog's ledger read, against a real warehouse (issue #682).

    Every other test of this path uses a repository double, which cannot
    catch the things that actually break it: a predicate naming a column the
    ledger does not have, a relation name that does not resolve in the
    adapter's schema, a scope fingerprint the manifest spells differently
    from the coordinator. Here the project really publishes, so the row this
    reads is the one `mark_ready` wrote.
    """
    project = _project(tmp_path)
    run_project(project)
    write_manifest(project)

    catalog = ArtifactCatalog.load(project)
    resource = catalog.get("context_search")
    assert resource is not None

    session = SearchSession(project, target=None, profiles_dir=None)
    repository = WarehouseContextRepository(session)
    try:
        activations = read_activations(repository, [resource])
    finally:
        repository.close()

    activated = activations.get("context_search")
    assert activated is not None, (
        "the index was just published, so the ledger must date it"
    )
    assert activated <= datetime.now(UTC)
    # No policy is declared, so a dated index is fresh and nothing is unknown.
    assert (
        freshness(activated, stale_after_seconds=None, now=datetime.now(UTC))
        is FreshnessStatus.FRESH
    )


def test_a_ledger_without_the_activation_column_reads_as_no_activation(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The upgrade path, against a real ledger (issue #682).

    `activated_epoch` is added by `_ensure_tables`, which runs in a publisher
    -- not in the MCP server. So the first server to start after an upgrade
    reads a ledger that genuinely lacks the column, and the projection naming
    it must read as "no activation on record" rather than fail the listing.
    Only a real adapter can show that: a repository double decides for itself
    what a missing column does.

    Quietly, too. Without the column check this still answers "unknown" --
    the projection raises, and the read is wrapped and logged -- but it warns
    that the ledger could not be read on every single listing until the next
    publish. A ledger that is merely older than a column is an expected state,
    not a fault to report, so the absence of that warning is the property
    under test as much as the empty result is.
    """
    project = _project(tmp_path)
    run_project(project)
    write_manifest(project)

    con = duckdb.connect(str(project / "target" / "context.duckdb"))
    try:
        con.execute(
            'ALTER TABLE "context".context.stel_serving_ledger '
            "DROP COLUMN activated_epoch"
        )
    finally:
        con.close()

    catalog = ArtifactCatalog.load(project)
    resource = catalog.get("context_search")
    assert resource is not None

    session = SearchSession(project, target=None, profiles_dir=None)
    repository = WarehouseContextRepository(session)
    try:
        with caplog.at_level(logging.WARNING, logger="stel.mcp_server.serving_state"):
            activations = read_activations(repository, [resource])
    finally:
        repository.close()

    assert activations == {}
    assert caplog.records == [], (
        "a ledger older than the column is expected, not a read failure"
    )
    assert (
        freshness(None, stale_after_seconds=None, now=datetime.now(UTC))
        is FreshnessStatus.UNKNOWN
    )
