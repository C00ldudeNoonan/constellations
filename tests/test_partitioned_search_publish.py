"""A search publish that owns one partition of its upstream reconciles stale
keys against that partition (issue #764).

A segmented group's member reads only its own rows (ADR-0037). Judged against
the whole upstream relation, a row re-dated into another partition still
exists, so it is never stale and stays in the collection it left. These tests
publish through `run_project` with a partition injected where the group
declaration will supply one (ALE-572).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
import pytest

from stel import runner
from stel.adapters.base import ReadPredicate, ReadPredicateOperator
from stel.config import load_project
from stel.profile import resolve_profile
from stel.retrieval import StoreRole, create_store
from stel.runner import run_project
from tests.support_retrieval import materialize_upstream, write_project

# Runs a whole project and opens a retrieval store, so it belongs to the `e2e`
# tier (issue #518).
pytestmark = pytest.mark.e2e

_FILED_IN_2024 = (ReadPredicate("filing_year", ReadPredicateOperator.EQUAL, 2024),)


def _upstream(years: dict[str, int]) -> pl.DataFrame:
    ids = list(years)
    return pl.DataFrame(
        {
            "chunk_id": ids,
            "document_id": [f"d{chunk_id[1:]}" for chunk_id in ids],
            "text": [f"text of {chunk_id}" for chunk_id in ids],
            "embedding": [[1.0, 0.0] for _ in ids],
            "category": ["prices" for _ in ids],
            "title": [f"title {chunk_id}" for chunk_id in ids],
            "filing_year": [years[chunk_id] for chunk_id in ids],
        }
    )


def _publish_partition(
    monkeypatch: pytest.MonkeyPatch, partition: tuple[ReadPredicate, ...]
) -> None:
    """Hand the search publish a partition, as a group member's will."""
    real = runner._run_search_model

    def partitioned(**kwargs: Any) -> Any:
        kwargs["partition"] = partition
        return real(**kwargs)

    monkeypatch.setattr(runner, "_run_search_model", partitioned)


def _published_ids(project_dir: Path) -> set[str]:
    project, _, _ = load_project(project_dir)
    resolved = resolve_profile(project, project_dir)
    assert resolved.retrieval is not None
    with create_store(
        resolved.retrieval.stores["primary"],
        project_name=project.name,
        target_name=resolved.target_name,
        alias="primary",
        role=StoreRole.PUBLISH,
    ) as store:
        name = store.physical_collection("context")
        return {
            record_id
            for page in store.iter_record_ids(name, id_field="chunk_id", page_size=10)
            for record_id in page
        }


def test_a_row_that_moves_out_of_the_partition_is_removed_on_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """c1 is re-dated from 2024 into 2025: still upstream, no longer this
    member's. c3 moves the other way and c2 does not move."""
    write_project(tmp_path)
    _publish_partition(monkeypatch, _FILED_IN_2024)
    materialize_upstream(tmp_path, _upstream({"c1": 2024, "c2": 2024, "c3": 2025}))

    run_project(tmp_path, select="context_search")
    assert _published_ids(tmp_path) == {"c1", "c2"}

    materialize_upstream(tmp_path, _upstream({"c1": 2025, "c2": 2024, "c3": 2024}))
    second = run_project(tmp_path, select="context_search")

    assert _published_ids(tmp_path) == {"c2", "c3"}
    assert second[0].documents_deleted == 1
    assert second[0].rows_inserted == 1
    # c2 stayed inside the partition and unchanged: not touched.
    assert second[0].documents_skipped == 1


def test_an_unpartitioned_publish_keeps_a_row_that_changes_year(
    tmp_path: Path,
) -> None:
    """The unpredicated path is unchanged: with no partition declared, the
    column is just data and a re-dated row is still the same row."""
    write_project(tmp_path)
    materialize_upstream(tmp_path, _upstream({"c1": 2024, "c2": 2024}))
    run_project(tmp_path, select="context_search")

    materialize_upstream(tmp_path, _upstream({"c1": 2025, "c2": 2024}))
    second = run_project(tmp_path, select="context_search")

    assert _published_ids(tmp_path) == {"c1", "c2"}
    assert second[0].documents_deleted == 0
