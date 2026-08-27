"""examples/agent_transcripts served from DuckDB instead of LanceDB (#371).

The acceptance criterion this covers is a specific one: an example switches
from LanceDB to DuckDB-native retrieval **by changing target configuration
rather than model logic**. So these tests run the same project, the same
models, and the same queries under two targets and compare — a test that
exercised only the DuckDB target would prove the store runs, not that it is
a drop-in for the one it replaces.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import duckdb
import pytest

from stel.runner import run_project
from stel.search import (
    SearchFilter,
    SearchFilterOperator,
    SearchMode,
    SearchRequest,
    search,
)

TENANT = "local-dev"
QUERY = "rounding test"


def _example(tmp_path: Path, name: str) -> Path:
    repo = Path(__file__).resolve().parents[1]
    destination = tmp_path / name
    shutil.copytree(
        repo / "examples" / "agent_transcripts",
        destination,
        ignore=shutil.ignore_patterns("target", "__pycache__"),
    )
    return destination


def _search(project: Path, target: str, mode: SearchMode) -> list[str]:
    results = search(
        project,
        SearchRequest(model="transcript_search", query=QUERY, mode=mode, limit=5),
        target=target,
        policy_filters=[
            SearchFilter("tenant_id", SearchFilterOperator.EQUAL, TENANT)
        ],
    )
    return [result.record_id for result in results]


@pytest.fixture(scope="module")
def both_targets(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Build the example once per target; a full run is not cheap."""
    root = tmp_path_factory.mktemp("duckdb_example")
    built: dict[str, Path] = {}
    for target in ("dev", "dev_duckdb"):
        project = _example(root, target)
        run_project(project, target=target)
        built[target] = project
    return built


# ─── the criterion: a config-only switch ────────────────────────────────────


@pytest.mark.parametrize(
    "mode", [SearchMode.VECTOR, SearchMode.TEXT, SearchMode.HYBRID]
)
def test_duckdb_and_lancedb_return_the_same_results(
    both_targets: dict[str, Path], mode: SearchMode
) -> None:
    """Same corpus, same query, same ranking — including hybrid, which is
    core's RRF over two legs rather than anything either store computes."""
    lancedb_ids = _search(both_targets["dev"], "dev", mode)
    duckdb_ids = _search(both_targets["dev_duckdb"], "dev_duckdb", mode)

    assert lancedb_ids
    assert duckdb_ids == lancedb_ids


def test_the_projects_differ_only_by_target(both_targets: dict[str, Path]) -> None:
    """The switch must not have needed a model edit. If any file under models/
    differs between the two trees, the store is not a drop-in and the example
    is not demonstrating what it claims."""
    lancedb_models = both_targets["dev"] / "models"
    duckdb_models = both_targets["dev_duckdb"] / "models"

    for source in sorted(lancedb_models.rglob("*")):
        if not source.is_file():
            continue
        mirrored = duckdb_models / source.relative_to(lancedb_models)
        assert mirrored.read_bytes() == source.read_bytes(), source.name


def test_duckdb_target_stands_up_no_second_system(
    both_targets: dict[str, Path],
) -> None:
    """The point of the store: retrieval lives in the warehouse file, so the
    DuckDB target should not have produced a LanceDB directory at all."""
    assert not (both_targets["dev_duckdb"] / "target" / "lancedb").exists()
    assert (both_targets["dev"] / "target" / "lancedb").exists()


def test_published_collection_is_a_table_in_the_warehouse_file(
    both_targets: dict[str, Path],
) -> None:
    database = both_targets["dev_duckdb"] / "target" / "transcripts.duckdb"
    connection = duckdb.connect(str(database), read_only=True)
    try:
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT table_name FROM duckdb_tables() WHERE schema_name = 'main'"
            ).fetchall()
        }
        collection = "agent_transcripts__dev_duckdb__agent_transcripts"

        assert collection in tables
        rows = connection.execute(
            f'SELECT count(*) FROM "{collection}"'
        ).fetchone()
        assert rows is not None and rows[0] > 0
    finally:
        connection.close()


def test_policy_filter_is_enforced_on_the_duckdb_target(
    both_targets: dict[str, Path],
) -> None:
    """A governed model must not answer for a tenant the caller has no claim
    to, whichever store is underneath."""
    results = search(
        both_targets["dev_duckdb"],
        SearchRequest(model="transcript_search", query=QUERY, mode=SearchMode.TEXT),
        target="dev_duckdb",
        policy_filters=[
            SearchFilter("tenant_id", SearchFilterOperator.EQUAL, "someone-else")
        ],
    )

    assert [result.record_id for result in results] == []


# ─── republication ──────────────────────────────────────────────────────────


def test_republishing_converges_rather_than_duplicating(tmp_path: Path) -> None:
    """Incremental publication is keyed by stable id. A second run over an
    unchanged corpus must leave the collection the same size — an append would
    double it while every query still looked plausible."""
    project = _example(tmp_path, "republish")
    run_project(project, target="dev_duckdb")
    database = project / "target" / "transcripts.duckdb"
    collection = "agent_transcripts__dev_duckdb__agent_transcripts"

    def _count() -> int:
        connection = duckdb.connect(str(database), read_only=True)
        try:
            row = connection.execute(f'SELECT count(*) FROM "{collection}"').fetchone()
            return int(row[0]) if row else 0
        finally:
            connection.close()

    first = _count()
    run_project(project, target="dev_duckdb")
    second = _count()

    assert first > 0
    assert second == first


def test_full_refresh_republishes_the_whole_corpus(tmp_path: Path) -> None:
    project = _example(tmp_path, "refresh")
    run_project(project, target="dev_duckdb")
    database = project / "target" / "transcripts.duckdb"
    collection = "agent_transcripts__dev_duckdb__agent_transcripts"

    run_project(project, target="dev_duckdb", full_refresh=True)

    connection = duckdb.connect(str(database), read_only=True)
    try:
        row = connection.execute(f'SELECT count(*) FROM "{collection}"').fetchone()
        assert row is not None and row[0] > 0
    finally:
        connection.close()

    assert _search(project, "dev_duckdb", SearchMode.HYBRID)
