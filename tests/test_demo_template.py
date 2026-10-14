"""`stel init --template demo`: a complete offline project in one command (ALE-46).

The template is the first thing a new user runs, so what its README promises
is pinned here: the numbers it prints, the search hits it describes, the
incremental rebuild it walks through, and that every `stel` command it shows
actually runs.
"""
from __future__ import annotations

import asyncio
import importlib.util
import re
import shlex
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import duckdb
import polars as pl
import pytest
from click.testing import CliRunner
from mcp.shared.memory import create_connected_server_and_client_session

import stel
from stel.checks import run_project_tests
from stel.cli import cli
from stel.concept_cloud import export_concept_cloud
from stel.config.vocabulary import Vocabulary
from stel.mcp_server.authorization import (
    ClaimAuthorizationProvider,
    Principal,
    StaticPrincipalResolver,
)
from stel.mcp_server.server import create_mcp_server
from stel.mcp_server.service import ContextService
from stel.runner import run_project
from stel.search import (
    SearchFilter,
    SearchFilterOperator,
    SearchMode,
    SearchRequest,
    search,
)
from stel.transforms import TransformContext

# Runs a whole project or opens a retrieval store, so it belongs to the
# `e2e` tier (issue #518). `test_test_tiers.py` fails if a file that
# does either is missing this.
pytestmark = pytest.mark.e2e

NAME = "acme"


def _scaffold(root: Path) -> Path:
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=root) as cwd:
        result = runner.invoke(cli, ["init", NAME, "--template", "demo"])
        assert result.exit_code == 0, result.output
        return Path(cwd) / NAME


def _rows(project: Path, sql: str) -> list[tuple[Any, ...]]:
    con = duckdb.connect(str(project / "target" / "warehouse.duckdb"), read_only=True)
    try:
        return con.execute(sql).fetchall()
    finally:
        con.close()


@pytest.fixture(scope="module")
def first_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, list[str]]:
    """Scaffolded and run once; tests that change the project build their own."""
    project = _scaffold(tmp_path_factory.mktemp("demo"))
    failed = [r.model_name for r in run_project(project) if r.errors]
    return project, failed


@pytest.fixture
def built(first_run: tuple[Path, list[str]]) -> Path:
    return first_run[0]


def test_init_names_every_file_after_the_project(tmp_path: Path) -> None:
    project = _scaffold(tmp_path)
    leftovers = [
        path.name
        for path in project.rglob("*")
        if path.is_file() and "__PROJECT_NAME__" in path.read_text(encoding="utf-8")
    ]
    assert leftovers == []
    readme = (project / "README.md").read_text(encoding="utf-8")
    assert f"from {NAME}.system_health" in readme


def test_every_model_builds_and_every_test_passes(
    first_run: tuple[Path, list[str]],
) -> None:
    """The README says `stel build` runs 13 models and their tests."""
    built, failed_models = first_run
    assert failed_models == []
    models = {
        row[0]
        for row in _rows(
            built,
            "select table_name from information_schema.tables "
            f"where table_schema = '{NAME}' and table_name not like 'stel_%'",
        )
    }
    # Twelve tables; the thirteenth model is the search index, which lives in
    # the same file under the retrieval store's own name.
    assert len(models) == 12, sorted(models)
    failed = [r for r in run_project_tests(built) if not r.passed]
    assert failed == [], [(r.test_name, r.message) for r in failed]


def test_system_health_finds_the_system_with_tickets_and_no_runbook(built: Path) -> None:
    rows = _rows(
        built,
        f"select system, postmortems, tickets, has_runbook from {NAME}.system_health",
    )
    by_system = {row[0]: row[1:] for row in rows}
    assert by_system["Webhook Dispatcher"] == (2, 3, True)
    assert by_system["Payments API"] == (1, 2, True)
    assert by_system["Auth Gateway"] == (0, 1, False)
    assert [s for s, (_, _, has_runbook) in by_system.items() if not has_runbook] == [
        "Auth Gateway"
    ]


def test_search_finds_the_root_cause_and_filters_to_tickets(built: Path) -> None:
    pool = search(
        built,
        SearchRequest(
            model="knowledge_search",
            query="connection pool exhausted",
            mode=SearchMode.TEXT,
            limit=3,
        ),
    ).results
    assert pool[0].metadata["title"] == "Postmortem: checkout outage, 2026-03-11"
    assert pool[0].metadata["section"] == "Root cause"
    assert pool[1].metadata["title"] == "Runbook: Payments API"

    tickets = search(
        built,
        SearchRequest(
            model="knowledge_search",
            query="webhooks late",
            mode=SearchMode.TEXT,
            limit=10,
            filters=(SearchFilter("kind", SearchFilterOperator.EQUAL, "ticket"),),
        ),
    ).results
    assert tickets
    assert {hit.metadata["kind"] for hit in tickets} == {"ticket"}
    assert tickets[0].metadata["title"] == "T-1188: Refund webhooks arriving hours late"


def test_changing_one_document_reprocesses_only_its_passages(tmp_path: Path) -> None:
    """The README's step 3, with the counts it prints."""
    project = _scaffold(tmp_path)
    run_project(project)
    doc = project / "data" / "docs" / "release-notes-2026-08.md"
    with doc.open("a", encoding="utf-8") as handle:
        handle.write("\n## Update\n\nThe Auth Gateway now has its own runbook.\n")

    results = {r.model_name: r for r in run_project(project)}

    assert (results["raw_docs"].documents_processed, results["raw_docs"].documents_skipped) == (
        1,
        7,
    )
    assert results["raw_tickets"].documents_processed == 0
    for name in ("chunk_embeddings", "knowledge_search"):
        result = results[name]
        assert (result.documents_processed, result.documents_skipped) == (2, 29), name


def test_star_map_names_and_colours_systems_from_the_vocabulary(built: Path) -> None:
    export = export_concept_cloud(
        built,
        linking_model="entity_links",
        relation_model="entity_relations",
        names_model="system_names",
        dimension_specs={"class": "system_names.class"},
    )
    concepts = {c.canonical_id: c for c in export.concepts}
    assert len(concepts) == 8
    # Named by the vocabulary, not by the most frequent mention ("webhooks").
    assert concepts["Webhook Dispatcher"].display == "Webhook Dispatcher"
    assert concepts["Webhook Dispatcher"].dimensions == {"class": "service"}
    assert concepts["Postgres"].dimensions == {"class": "datastore"}
    assert export.concept_edges


def test_the_longest_declared_name_claims_an_overlapping_span() -> None:
    """Template code a user extends with their own vocabulary: when one term's
    name contains another's, the longer one wins the span, whatever order
    the terms are declared in."""
    path = Path(stel.__file__).parent / "templates" / "demo" / "transforms" / "entity_mentions.py"
    spec = importlib.util.spec_from_file_location("demo_entity_mentions", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    vocabulary = Vocabulary.model_validate(
        {"terms": [{"label": "Search"}, {"label": "Search Service"}]}
    )
    ctx = SimpleNamespace(options={"vocabulary": "v"}, vocabularies={"v": vocabulary})
    chunks = pl.DataFrame({"context_id": ["c1"], "text": ["The Search Service is slow."]})

    mentions = module.run({"document_chunks": chunks}, cast(TransformContext, ctx))

    assert mentions["entity_text"].to_list() == ["Search Service"]


def _call(project: Path, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
    async def go() -> dict[str, Any]:
        service = ContextService.from_project(
            project,
            principal_resolver=StaticPrincipalResolver(
                Principal(subject_id="me", tenant_id="acme")
            ),
            authorization=ClaimAuthorizationProvider(),
        )
        app = create_mcp_server(service)
        async with create_connected_server_and_client_session(
            app, raise_exceptions=True
        ) as session:
            result = await session.call_tool(tool, arguments)
            assert result.structuredContent is not None
            return result.structuredContent

    return asyncio.run(go())


def test_an_agent_can_scope_a_search_to_a_declared_class(built: Path) -> None:
    """The README's step 6: `entity_scope` over the declared vocabulary, with
    a citation back to the section each passage came from."""
    found = _call(
        built,
        "search_context",
        {
            "model": "knowledge_search",
            "query": "slow",
            "mode": "text",
            "limit": 5,
            "entity_scope": {"class": "datastore"},
        },
    )
    assert found["results"]
    datastores = {"Postgres", "Kafka", "Redis", "PgBouncer"}
    for hit in found["results"]:
        linked = {entity["entity_key"][1] for entity in hit["entities"]}
        assert linked & datastores, hit["citation"]
        assert hit["citation"]["source_uri"].startswith("data/")


def _readme_commands(readme: str) -> list[list[str]]:
    """Every `stel` command in the README's bash blocks, continuation lines
    joined. Commands that serve until interrupted are left out."""
    commands: list[list[str]] = []
    for block in re.findall(r"```bash\n(.*?)```", readme, flags=re.DOTALL):
        for line in block.replace("\\\n", " ").splitlines():
            argv = shlex.split(line)
            if not argv or argv[0] != "stel":
                continue
            if "serve" in argv or "--serve" in argv:
                continue
            commands.append(argv[1:])
    return commands


def test_every_stel_command_in_the_readme_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _scaffold(tmp_path)
    # The README is read from inside the project, so its relative paths are.
    monkeypatch.chdir(project)
    readme = (project / "README.md").read_text(encoding="utf-8")
    commands = _readme_commands(readme)
    assert ["build"] in commands
    assert any(argv[0] == "concept-cloud" for argv in commands)

    runner = CliRunner()
    for argv in commands:
        result = runner.invoke(cli, ["--project-dir", str(project), *argv])
        assert result.exit_code == 0, (argv, result.output)
