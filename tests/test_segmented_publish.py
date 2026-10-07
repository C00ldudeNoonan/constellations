"""A search publish whose snapshot reports segment progress, into a collection
that does not exist yet (issue #658).

A segmented read (issue #614) reports how far it has got on every page, and the
publish records that on the collection so a later attempt can resume. The
first page arrives before the first write creates the collection, so on a first
build and on every private generation the record went to a collection the store
had never created, and the publish failed on its first page. The bundled DuckDB
adapter reports no progress, which is how the suite missed it: these tests make
its snapshot report progress the way BigQuery's segmented read does.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl
import pyarrow as pa
import pytest

from stel.adapters.base import SnapshotResumePoint, TableReadSnapshot
from stel.adapters.duckdb import DuckDBAdapter
from stel.config import load_project
from stel.execution import search as search_execution
from stel.profile import resolve_profile
from stel.retrieval import LanceDBStore, RetrievalError, StoreRole, create_store
from stel.runner import RunError, run_project
from tests.support_retrieval import materialize_upstream, write_project

# Runs a whole project and opens a retrieval store, so it belongs to the `e2e`
# tier (issue #518).
pytestmark = pytest.mark.e2e

# Four rows read two to a page: two pages, so a second page's progress has a
# collection to land on once the first page has created it.
_ROWS = pl.DataFrame(
    {
        "chunk_id": ["c1", "c2", "c3", "c4"],
        "document_id": ["d1", "d2", "d3", "d4"],
        "text": ["inflation slowed", "employment increased", "output grew", "rates held"],
        "embedding": [[1.0, 0.0], [0.0, 1.0], [0.7, 0.3], [0.3, 0.7]],
        "category": ["prices", "labor", "growth", "rates"],
        "title": ["CPI", "Payrolls", "GDP", "FOMC"],
    }
)


def _report_segment_progress(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every DuckDB table snapshot report progress like a segmented read.

    Two segments split at `c3`; a segment counts as complete once a later page
    has been pulled, as BigQuery's read reports it.
    """
    real_open = DuckDBAdapter._open_table_snapshot
    instant = datetime.now(UTC).isoformat()

    def opened(self: DuckDBAdapter, request: Any) -> TableReadSnapshot:
        snapshot = real_open(self, request)
        pulled = {"pages": 0, "rows": 0}
        batches = snapshot._batches

        def counted() -> Iterator[pa.RecordBatch]:
            for batch in batches:
                pulled["pages"] += 1
                pulled["rows"] += batch.num_rows
                yield batch

        def progress() -> SnapshotResumePoint:
            return SnapshotResumePoint(
                snapshot_time=instant,
                key_type="STRING",
                boundaries=("c3",),
                completed=min(max(pulled["pages"] - 1, 0), 2),
                rows=pulled["rows"],
                generation="generation-a",
            )

        snapshot._batches = counted()
        snapshot._progress = progress
        return snapshot

    monkeypatch.setattr(DuckDBAdapter, "_open_table_snapshot", opened)


def _project(tmp_path: Path) -> None:
    write_project(tmp_path)
    materialize_upstream(tmp_path, _ROWS)


def _store(project_dir: Path) -> Any:
    project, _, _ = load_project(project_dir)
    resolved = resolve_profile(project, project_dir)
    assert resolved.retrieval is not None
    return create_store(
        resolved.retrieval.stores["primary"],
        project_name=project.name,
        target_name=resolved.target_name,
        alias="primary",
        role=StoreRole.PUBLISH,
    )


def test_a_first_build_through_a_segmented_read_publishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The first case #658 hit: no collection, and progress on page one."""
    _project(tmp_path)
    _report_segment_progress(monkeypatch)
    # An in-place publish then looks for stale rows as of the instant the read
    # pinned, which needs time travel DuckDB does not have and refuses. That
    # step is BigQuery's and outside #658, so the double reports no pin to it.
    monkeypatch.setattr(search_execution, "_pinned_instant", lambda snapshot: None)

    run_project(tmp_path, select="context_search")

    with _store(tmp_path) as store:
        published = store.inspect_collection(store.physical_collection("context"))
    assert published is not None
    assert published.row_count == len(_ROWS)


def test_a_private_generation_through_a_segmented_read_publishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The case #658 was found on: a rebuild into a private generation over a
    live collection, which starts as empty as a first build does."""
    _project(tmp_path)
    run_project(tmp_path, select="context_search")
    _report_segment_progress(monkeypatch)

    run_project(tmp_path, select="context_search", full_refresh=True)

    with _store(tmp_path) as store:
        generations = [name for name in store.list_collections() if "__g" in name]
        assert len(generations) == 1
        rebuilt = store.inspect_collection(generations[0])
    assert rebuilt is not None
    assert rebuilt.row_count == len(_ROWS)
    # Every segment published, so the progress is cleared with the rows stamped
    # complete; it must not outlive the publish it describes.
    assert rebuilt.source_progress is None


def test_progress_from_before_the_collection_existed_is_stamped_for_a_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Deferring the record must not lose it: a rebuild that dies on its second
    page leaves a generation stamped with the progress its pages reached, which
    is what a retry resumes from (issue #614)."""
    _project(tmp_path)
    run_project(tmp_path, select="context_search")
    _report_segment_progress(monkeypatch)
    writes = {"count": 0}
    real_append = LanceDBStore.append

    def append_once(self: LanceDBStore, *args: Any, **kwargs: Any) -> Any:
        writes["count"] += 1
        if writes["count"] > 1:
            raise RetrievalError("simulated failure writing the second page")
        return real_append(self, *args, **kwargs)

    monkeypatch.setattr(LanceDBStore, "append", append_once)
    with pytest.raises(RunError):
        run_project(tmp_path, select="context_search", full_refresh=True)

    assert writes["count"] == 2
    with _store(tmp_path) as store:
        generations = [name for name in store.list_collections() if "__g" in name]
        assert len(generations) == 1
        left = store.inspect_collection(generations[0])
    assert left is not None
    assert left.source_progress is not None
    assert json.loads(left.source_progress)["completed"] == 1
