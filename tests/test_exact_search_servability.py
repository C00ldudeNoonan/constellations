"""What an `exact` vector collection costs to query, said before it is paid.

The failure this covers (issue #461) was not a crash. A 3.6M-row, 768-dimension
collection with `search: exact` built, validated, published and reported
`ready`, and every governed query against it then timed out — because `exact`
builds no vector index, so each query read the whole ~11GB vector column, and
the context server's `timeout_seconds` ceiling sits below that cost. Nothing in
the build path said a word.

So the tests here are about *signal*, not about the arithmetic being right to
the second: the estimate is anchored to one measurement and admits as much.
What must hold is that a collection large enough to be unqueryable says so, a
small one stays quiet, and the governed case is named as unservable rather than
merely slow.
"""
from __future__ import annotations

import logging
from typing import Any

import pytest

from stel.retrieval.servability import (
    DEFAULT_CONTEXT_TIMEOUT_SECONDS,
    MAX_CONTEXT_TIMEOUT_SECONDS,
    estimated_exact_scan_seconds,
    exact_scan_bytes,
    exact_search_advisory,
)

# The corpus from the issue, and the one that was fine before it.
_REPORTED_ROWS = 3_613_979
_REPORTED_DIMENSIONS = 768


def test_the_reported_corpus_is_estimated_in_the_measured_range() -> None:
    """Anchor. The report measured ~275s for this shape, warm and cold; an
    estimator that put it at 3s or 3000s would warn at the wrong sizes."""
    seconds = estimated_exact_scan_seconds(
        rows=_REPORTED_ROWS, dimensions=_REPORTED_DIMENSIONS
    )

    assert 200 <= seconds <= 400
    assert exact_scan_bytes(
        rows=_REPORTED_ROWS, dimensions=_REPORTED_DIMENSIONS
    ) == pytest.approx(11.1e9, rel=0.02)


def test_a_small_collection_says_nothing() -> None:
    """`exact` is a legitimate default and most collections are small. An
    advisory on every publish would be noise, and noise is how the one that
    mattered would have been missed anyway."""
    assert (
        exact_search_advisory(
            collection="ctx", rows=20_000, dimensions=768, access="public"
        )
        is None
    )


def test_an_empty_collection_says_nothing() -> None:
    """A first publish inspects a collection before it has rows."""
    assert (
        exact_search_advisory(
            collection="ctx", rows=0, dimensions=768, access="governed"
        )
        is None
    )


def test_a_large_collection_names_the_cost_and_the_remedy() -> None:
    advisory = exact_search_advisory(
        collection="sec_chunk_search",
        rows=_REPORTED_ROWS,
        dimensions=_REPORTED_DIMENSIONS,
        access="public",
    )

    assert advisory is not None
    assert "sec_chunk_search" in advisory
    assert "3,613,979" in advisory
    # The remedy has to be the cheap one, or the warning just relocates the
    # problem: `approximate` plus `online` is an index build, not a re-embed.
    assert "search: approximate" in advisory
    assert "on_index_change: online" in advisory


def test_the_reported_governed_index_is_named_as_a_serving_problem() -> None:
    """The case that actually happened. The scan alone estimates at ~278s,
    under the 600s ceiling — so an advisory keyed only on the ceiling would
    have stayed silent on the very index that could not be served. What makes
    it unservable is that it is governed: the default timeout is 30s, and the
    governed path pays warehouse reads and per-row authorization on top."""
    advisory = exact_search_advisory(
        collection="sec_chunk_search",
        rows=_REPORTED_ROWS,
        dimensions=_REPORTED_DIMENSIONS,
        access="governed",
    )

    assert advisory is not None
    assert "governed" in advisory
    assert f"{DEFAULT_CONTEXT_TIMEOUT_SECONDS:.0f}s" in advisory
    assert f"{MAX_CONTEXT_TIMEOUT_SECONDS:.0f}s" in advisory


def test_a_scan_over_the_ceiling_is_unanswerable_by_any_setting() -> None:
    """The top tier, and the only one that may say a query cannot be answered:
    past the ceiling there is no permitted `timeout_seconds` large enough."""
    advisory = exact_search_advisory(
        collection="ctx", rows=20_000_000, dimensions=1536, access="governed"
    )

    assert advisory is not None
    assert "no permitted setting is large enough" in advisory


def test_a_public_index_is_called_expensive_rather_than_unservable() -> None:
    """A public index is queried directly rather than through the context
    server, so neither timeout binds it. Slow is the honest word there, and
    an operator is entitled to choose slow."""
    advisory = exact_search_advisory(
        collection="ctx",
        rows=_REPORTED_ROWS,
        dimensions=_REPORTED_DIMENSIONS,
        access="public",
    )

    assert advisory is not None
    assert "no permitted setting is large enough" not in advisory
    assert "governed" not in advisory


# ─── the publish path actually emits it ─────────────────────────────────────


def _model(access: str) -> Any:
    from types import SimpleNamespace

    return SimpleNamespace(
        name="sec_chunk_search", search=SimpleNamespace(access=access)
    )


def _spec(*, vector_search: str, dimensions: int | None) -> Any:
    from types import SimpleNamespace

    return SimpleNamespace(
        logical_name="sec_chunk_search",
        vector_search=vector_search,
        vector_dimensions=dimensions,
    )


def test_the_publish_path_warns_on_a_large_exact_collection(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from stel.execution.search import _warn_on_exact_vector_scan

    with caplog.at_level(logging.WARNING, logger="stel.execution.search"):
        warned = _warn_on_exact_vector_scan(
            _model("governed"),
            _spec(vector_search="exact", dimensions=768),
            _REPORTED_ROWS,
        )

    assert warned is True
    assert "sec_chunk_search" in caplog.text
    assert "search: approximate" in caplog.text


@pytest.mark.parametrize(
    "vector_search,dimensions,rows",
    [
        # Already approximate: an ANN index answers this, nothing to say.
        ("approximate", 768, _REPORTED_ROWS),
        # No vector at all: a text-only collection scans nothing per query.
        ("exact", None, _REPORTED_ROWS),
        ("exact", 768, 20_000),
    ],
)
def test_the_publish_path_stays_quiet_when_there_is_nothing_to_warn_about(
    caplog: pytest.LogCaptureFixture,
    vector_search: str,
    dimensions: int | None,
    rows: int,
) -> None:
    from stel.execution.search import _warn_on_exact_vector_scan

    with caplog.at_level(logging.WARNING, logger="stel.execution.search"):
        warned = _warn_on_exact_vector_scan(
            _model("governed"), _spec(vector_search=vector_search, dimensions=dimensions), rows
        )

    assert warned is False
    assert caplog.text == ""


# ─── a timeout no retry can satisfy (issue #461) ────────────────────────────


def test_a_timeout_at_the_ceiling_is_not_reported_as_retryable() -> None:
    """`retryable: true` told the caller to try again on a call this server is
    configured never to finish. A deterministic full scan fails identically on
    every attempt, and at the ceiling there is no larger `timeout_seconds` left
    to set, so `true` was an instruction to retry forever."""
    from stel.mcp_server.contracts import MCPErrorCode
    from stel.mcp_server.service import timeout_error

    error = timeout_error(MAX_CONTEXT_TIMEOUT_SECONDS)

    assert error.code is MCPErrorCode.TIMEOUT
    assert error.retryable is False
    assert "retrying will not help" in error.message


def test_a_timeout_below_the_ceiling_is_still_retryable() -> None:
    """The complement. Under the ceiling a timeout can be contention, and the
    operator still has a lever — raising `timeout_seconds` — so telling the
    caller to give up would be as wrong as telling it to keep trying."""
    from stel.mcp_server.service import timeout_error

    error = timeout_error(DEFAULT_CONTEXT_TIMEOUT_SECONDS)

    assert error.retryable is True
    assert "retrying will not help" not in error.message


def test_the_limiter_reports_the_timeout_through_that_rule() -> None:
    """The helper is only worth anything if the serving path uses it."""
    import time

    from stel.mcp_server.service import ContextServiceError, _OperationLimiter

    limiter = _OperationLimiter(
        max_concurrency=1, max_requests_per_minute=100, timeout_seconds=0.01
    )
    try:
        with pytest.raises(ContextServiceError) as error:
            limiter.run(lambda: time.sleep(5))
    finally:
        limiter.close()

    assert error.value.retryable is True
