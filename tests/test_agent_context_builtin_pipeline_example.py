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
import yaml
from mcp import ClientSession
from mcp.shared.memory import create_connected_server_and_client_session

from stel.agent_context import FreshnessStatus
from stel.checks import run_project_tests
from stel.manifest import write_manifest
from stel.mcp_server.authorization import Principal, StaticPrincipalResolver
from stel.mcp_server.catalog import ArtifactCatalog
from stel.mcp_server.repository import WarehouseContextRepository
from stel.mcp_server.server import create_mcp_server
from stel.mcp_server.service import ContextService
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


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_mcp_array_membership_filters_the_published_index(tmp_path: Path) -> None:
    """Issue #723: discover, validate and execute overlap through real MCP.

    The search input tags each chunk with its document's title. A filter
    must select that document and exclude other searchable documents, in
    every retrieval mode, while policy fields remain caller-inaccessible.
    """
    project = _project(tmp_path)
    config_path = project / "models" / "context_search.yml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["models"][0]["search"]["attributes"].append(
        {
            "name": "topics",
            "data_type": "array[string]",
            "filter_role": "user",
            "returned": True,
        }
    )
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    run_project(project, exclude="context_search")
    # Prepare the search input directly: embed currently scalarizes list
    # passthrough values, a separate issue from the MCP filter contract.
    connection = duckdb.connect(str(project / "target" / "context.duckdb"))
    try:
        connection.execute(
            'ALTER TABLE "context".context.chunk_embeddings ADD COLUMN topics VARCHAR[]'
        )
        connection.execute('UPDATE "context".context.chunk_embeddings SET topics = [title]')
    finally:
        connection.close()
    run_project(project, select="context_search")
    write_manifest(project)
    service = ContextService.from_project(
        project,
        principal_resolver=StaticPrincipalResolver(
            Principal(
                "test-reader",
                tenant_id="economic-data",
                access_groups=("research-desk",),
            )
        ),
    )
    wanted = "Labor market cooling continues into the second quarter"
    try:
        async with create_connected_server_and_client_session(
            create_mcp_server(service),
            raise_exceptions=True,
        ) as session:
            catalog = await session.call_tool("list_context_models", {})
            assert not catalog.isError
            assert catalog.structuredContent is not None
            fields = catalog.structuredContent["models"][0]["retrieval"]["filter_fields"]
            assert next(field for field in fields if field["field"] == "topics") == {
                "field": "topics",
                "data_type": "array[string]",
                "operators": ["array_contains_any"],
            }
            for mode in ("text", "vector", "hybrid"):
                await _assert_array_membership_search(session, mode, wanted)
            for field, operator, value in (
                ("topics", "eq", wanted),
                ("tenant_id", "array_contains_any", ["economic-data"]),
            ):
                refused = await session.call_tool(
                    "search_context",
                    {
                        "model": "context_search",
                        "query": "growth",
                        "mode": "text",
                        "filters": [{"field": field, "operator": operator, "value": value}],
                    },
                )
                assert refused.structuredContent is not None
                assert refused.structuredContent["results"] == []
                assert refused.structuredContent["error"]["code"] == "capability_unavailable"
    finally:
        service.close()


async def _assert_array_membership_search(session: ClientSession, mode: str, wanted: str) -> None:
    arguments = {"model": "context_search", "query": "growth", "mode": mode}
    baseline = await session.call_tool("search_context", arguments)
    assert baseline.structuredContent is not None
    assert baseline.structuredContent["error"] is None
    assert len({hit["document_id"] for hit in baseline.structuredContent["results"]}) > 1
    for values, expected in ((["absent", wanted], True), (["absent"], False)):
        response = await session.call_tool("search_context", {
            **arguments,
            "filters": [{"field": "topics", "operator": "array_contains_any", "value": values}],
        })
        assert not response.isError
        assert response.structuredContent is not None
        assert response.structuredContent["error"] is None
        hits = response.structuredContent["results"]
        assert bool(hits) is expected
        assert all(hit["attributes"]["topics"] == [wanted] for hit in hits)


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
