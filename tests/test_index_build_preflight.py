"""The ANN build's memory is predictable before it is paid (issue #476).

The #473 retest's predecessor appended 3.6M rows for hours and would then have
died building `HnswFlat` at ~3x the vectors' bytes on a 20 GiB container, with
nothing having mentioned the cost. These pin the estimate, the headroom line,
the wording, and that the publish path speaks before the build rather than
after it.
"""
from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from stel.retrieval.servability import (
    index_build_advisory,
    index_build_advisory_row_threshold,
    index_build_peak_bytes,
)

# The corpus from issue #473: 3,613,979 rows x 768 float32 on a 20 GiB box.
_ROWS = 3_613_979
_DIMS = 768
_LIMIT = 20 * 1024**3


def test_the_reported_corpus_lands_where_it_was_measured() -> None:
    flat = index_build_peak_bytes(rows=_ROWS, dimensions=_DIMS, vector_index="ivf_hnsw_flat")
    sq = index_build_peak_bytes(rows=_ROWS, dimensions=_DIMS, vector_index="ivf_hnsw_sq")
    pq = index_build_peak_bytes(rows=_ROWS, dimensions=_DIMS, vector_index="ivf_pq")
    assert 34e9 < flat < 37e9
    assert 19e9 < sq < 21e9
    assert 12e9 < pq < 14e9


@pytest.mark.parametrize("vector_index", ["ivf_hnsw_flat", "ivf_hnsw_sq"])
def test_the_hnsw_family_is_warned_about_on_the_reported_box(vector_index: str) -> None:
    advisory = index_build_advisory(
        collection="sec_chunk_search", rows=_ROWS, dimensions=_DIMS,
        vector_index=vector_index, limit_bytes=_LIMIT,
    )
    assert advisory is not None
    assert vector_index in advisory and "20.0 GiB" in advisory
    assert "index: ivf_pq" in advisory and "on_index_change: online" in advisory


def test_pq_fits_the_reported_box_and_stays_quiet() -> None:
    assert index_build_advisory(
        collection="sec_chunk_search", rows=_ROWS, dimensions=_DIMS,
        vector_index="ivf_pq", limit_bytes=_LIMIT,
    ) is None


def test_an_oversized_pq_build_is_told_to_grow_the_container() -> None:
    """When even the smallest build does not fit, the remedy is not a type."""
    advisory = index_build_advisory(
        collection="c", rows=_ROWS, dimensions=_DIMS, vector_index="ivf_pq",
        limit_bytes=8 * 1024**3,
    )
    assert advisory is not None
    assert "already the smallest build" in advisory
    # The preamble quotes the declared config; the *remedy* must not tell the
    # operator to switch to the type they already have.
    assert "Declare `index: ivf_pq`" not in advisory


def test_sq_at_the_ceiling_is_caught_by_the_headroom_line() -> None:
    """SQ estimates ~20.0 GB against a 21.5 GB ceiling: under the ceiling,
    over the 75% a build may plan on. The estimate is a lower bound sitting on
    the publish's own residency, so the marginal case must speak."""
    assert index_build_peak_bytes(rows=_ROWS, dimensions=_DIMS, vector_index="ivf_hnsw_sq") < _LIMIT
    assert index_build_advisory(
        collection="c", rows=_ROWS, dimensions=_DIMS, vector_index="ivf_hnsw_sq", limit_bytes=_LIMIT
    ) is not None


def test_no_ceiling_means_nothing_to_compare_against() -> None:
    assert index_build_advisory(
        collection="c", rows=_ROWS, dimensions=_DIMS, vector_index="ivf_hnsw_flat", limit_bytes=None
    ) is None


def test_a_small_collection_says_nothing() -> None:
    assert index_build_advisory(
        collection="c", rows=20_000, dimensions=_DIMS, vector_index="ivf_hnsw_flat",
        limit_bytes=_LIMIT,
    ) is None


def test_in_progress_wording_claims_a_floor() -> None:
    advisory = index_build_advisory(
        collection="c", rows=2_000_000, dimensions=_DIMS, vector_index="ivf_hnsw_flat",
        limit_bytes=_LIMIT, in_progress=True,
    )
    assert advisory is not None
    assert "published so far" in advisory and "at least" in advisory


def test_the_row_threshold_is_the_inverse_of_the_estimate() -> None:
    threshold = index_build_advisory_row_threshold(
        dimensions=_DIMS, vector_index="ivf_hnsw_flat", limit_bytes=_LIMIT
    )
    assert index_build_advisory(
        collection="c", rows=threshold, dimensions=_DIMS, vector_index="ivf_hnsw_flat",
        limit_bytes=_LIMIT,
    ) is not None
    assert index_build_advisory(
        collection="c", rows=threshold - 1, dimensions=_DIMS, vector_index="ivf_hnsw_flat",
        limit_bytes=_LIMIT,
    ) is None


def test_unknown_or_degenerate_inputs_estimate_nothing() -> None:
    assert index_build_peak_bytes(rows=_ROWS, dimensions=_DIMS, vector_index="ivf_rq") == 0
    assert index_build_peak_bytes(rows=0, dimensions=_DIMS, vector_index="ivf_pq") == 0
    assert (
        index_build_advisory_row_threshold(dimensions=0, vector_index="ivf_pq", limit_bytes=_LIMIT)
        == 0
    )


# ─── the publish path ───────────────────────────────────────────────────────


def _model() -> Any:
    return SimpleNamespace(name="sec_chunk_search", search=SimpleNamespace(access="governed"))


def _spec(*, vector_search: str, vector_index: str | None, dimensions: int | None = _DIMS) -> Any:
    return SimpleNamespace(
        logical_name="sec_chunk_search", vector_search=vector_search,
        vector_dimensions=dimensions, vector_index=vector_index,
    )


def test_the_publish_path_warns_before_a_build_that_will_not_fit(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from stel.execution.search import _warn_on_index_build_memory

    with caplog.at_level(logging.WARNING, logger="stel.execution.search"):
        warned = _warn_on_index_build_memory(
            _model(), _spec(vector_search="approximate", vector_index="ivf_hnsw_flat"),
            _ROWS, limit_bytes=_LIMIT,
        )
    assert warned is True
    assert "sec_chunk_search" in caplog.text and "ivf_hnsw_flat" in caplog.text


@pytest.mark.parametrize(
    "vector_search,vector_index,limit",
    [
        ("exact", None, _LIMIT),            # no index, no build
        ("approximate", "ivf_pq", _LIMIT),  # fits
        ("approximate", "ivf_hnsw_flat", None),  # no container ceiling
    ],
)
def test_the_publish_path_stays_quiet_when_there_is_nothing_to_warn_about(
    caplog: pytest.LogCaptureFixture,
    vector_search: str,
    vector_index: str | None,
    limit: int | None,
) -> None:
    from stel.execution.search import _warn_on_index_build_memory

    with caplog.at_level(logging.WARNING, logger="stel.execution.search"):
        warned = _warn_on_index_build_memory(
            _model(), _spec(vector_search=vector_search, vector_index=vector_index), _ROWS,
            limit_bytes=limit,
        )
    assert warned is False
    assert caplog.text == ""


def test_an_existing_collection_is_warned_about_before_the_first_row_is_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    """End to end: with a ceiling small enough that even the two-row fixture's
    build crosses the headroom line, an existing collection is warned about
    at minute zero — before the first upstream row is even read, which is the
    only moment worth speaking at when the rows take hours. The in-progress
    site alone would speak a batch later, so this pins the earlier one."""
    from stel.execution import search as search_module
    from stel.runner import run_project
    from tests.test_online_publication import _prepare

    _prepare(tmp_path)  # publishes exact, then flips the config to approximate
    # 2 rows x 2 dims x 4 B x 3.2 = 51.2 B of estimated build; 75% of 60 B is 45 B.
    monkeypatch.setattr(search_module, "container_memory_limit_bytes", lambda: 60)
    order: list[str] = []
    real_indexed_rows = search_module._indexed_rows

    def indexed_rows(*args: Any, **kwargs: Any) -> Any:
        order.append("read")
        return real_indexed_rows(*args, **kwargs)

    monkeypatch.setattr(search_module, "_indexed_rows", indexed_rows)

    class _Mark(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            if "peaks at roughly" in record.getMessage():
                order.append("warn")

    logging.getLogger("stel.execution.search").addHandler(handler := _Mark())
    try:
        with caplog.at_level(logging.WARNING, logger="stel.execution.search"):
            run_project(tmp_path, select="context_search")
    finally:
        logging.getLogger("stel.execution.search").removeHandler(handler)

    assert order.count("warn") == 1
    assert "read" in order
    assert order.index("warn") < order.index("read")
    assert "ivf_hnsw_flat" in caplog.text and "index: ivf_pq" in caplog.text


def test_a_first_publish_speaks_as_the_streamed_count_crosses_the_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    """No collection exists yet, so the existing-collection site has nothing
    to say; the in-progress site must speak from the running count, before
    the build, and claim a floor rather than a total."""
    from stel.execution import search as search_module
    from stel.retrieval import LanceDBStore
    from stel.runner import run_project
    from tests.test_retrieval import _materialize_upstream, _rows, _write_project

    _write_project(tmp_path)
    model_path = tmp_path / "models" / "retrieval.yml"
    model_path.write_text(
        model_path.read_text(encoding="utf-8").replace(
            "        search: exact\n", "        search: approximate\n"
        ),
        encoding="utf-8",
    )
    _materialize_upstream(tmp_path, _rows())
    monkeypatch.setattr(search_module, "container_memory_limit_bytes", lambda: 60)
    order: list[str] = []
    real_ensure = LanceDBStore.ensure_indexes

    def ensure(self: Any, spec: Any) -> Any:
        order.append("build")
        return real_ensure(self, spec)

    monkeypatch.setattr(LanceDBStore, "ensure_indexes", ensure)

    class _Mark(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            if "peaks at roughly" in record.getMessage():
                order.append("warn")

    logging.getLogger("stel.execution.search").addHandler(handler := _Mark())
    try:
        with caplog.at_level(logging.WARNING, logger="stel.execution.search"):
            run_project(tmp_path, select="context_search")
    finally:
        logging.getLogger("stel.execution.search").removeHandler(handler)

    assert order.count("warn") == 1
    assert order.index("warn") < order.index("build")
    assert "published so far" in caplog.text

