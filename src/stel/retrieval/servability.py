"""Whether an `exact` vector collection can still be answered (issue #461).

`search: exact` builds no vector index, so every query reads the whole vector
column. That is invisible at small scale and fatal at large: a 3.6M-row,
768-dimension collection published successfully, passed its post-publication
readiness checks, and was then unqueryable through stel's own governed MCP
server, whose `timeout_seconds` ceiling sits below the cost of a single scan.
Nothing warned, and by the time it surfaced the remedy looked like a full
republish under a new collection name.

This module is the arithmetic that makes that cost predictable before it is
paid. It estimates rather than measures, deliberately: the value is in saying
"this will not be servable" while a cheaper choice is still cheap, not in
predicting a latency. The estimate is anchored to one real measurement and is
stated as such wherever it is reported.
"""

from __future__ import annotations

from math import ceil
from typing import Final

# The largest `timeout_seconds` a context server will accept. Defined here
# rather than in `mcp_server` because it is what makes an index unservable
# *at publish time*, which is where the useful warning lives; the server
# imports it back for its own field bound so the two cannot drift.
MAX_CONTEXT_TIMEOUT_SECONDS: Final = 600.0

# What a context server allows itself unless an operator raises it. The
# distinction matters below: a scan that outlasts this is already failing in
# the shipped configuration, while a scan that outlasts the ceiling cannot be
# rescued by any permitted setting at all.
DEFAULT_CONTEXT_TIMEOUT_SECONDS: Final = 30.0

# Effective end-to-end throughput of a full vector scan, from the measurement
# reported in issue #461: 3,613,979 rows x 768 float32 (~11.1 GB) in ~275s
# against a LanceDB store on GCS, cold and warm alike. Object storage is the
# case worth warning about. A local NVMe store is faster, so this errs toward
# warning early rather than late -- the right direction for an advisory whose
# alternative is discovering the problem after a four-hour publish.
_SCAN_BYTES_PER_SECOND: Final = 40_000_000.0

_FLOAT32_BYTES: Final = 4

# Under this a full scan is an ordinary query cost, not a design problem. It
# is the default serving timeout on purpose: a scan that outlasts it is one
# the shipped configuration already cannot answer.
_ADVISORY_SECONDS: Final = DEFAULT_CONTEXT_TIMEOUT_SECONDS


def exact_scan_bytes(*, rows: int, dimensions: int) -> int:
    """Bytes an exact vector query reads: the whole float32 vector column."""
    return rows * dimensions * _FLOAT32_BYTES


def estimated_exact_scan_seconds(*, rows: int, dimensions: int) -> float:
    """Rough wall clock for one exact vector query over `rows` x `dimensions`."""
    return exact_scan_bytes(rows=rows, dimensions=dimensions) / _SCAN_BYTES_PER_SECOND


def exact_advisory_row_threshold(*, dimensions: int) -> int:
    """Rows at which an exact scan first becomes worth warning about.

    The inverse of the estimate, so a publish still streaming its first
    collection can warn the moment it crosses the line instead of waiting for
    a final row count it does not have yet (Codex review, #461).
    """
    if dimensions <= 0:
        return 0
    per_row = dimensions * _FLOAT32_BYTES
    return ceil(_ADVISORY_SECONDS * _SCAN_BYTES_PER_SECOND / per_row)


def exact_search_advisory(
    *,
    collection: str,
    rows: int,
    dimensions: int,
    access: str,
    in_progress: bool = False,
) -> str | None:
    """The warning an `exact` collection of this size deserves, if any.

    Returns None when the scan is cheap enough not to be worth a line.

    Three tiers, because the remedies differ. Above the context server's
    absolute ceiling nothing can answer the query, governed or not. Below it
    but governed, the index is reachable only through that server, whose
    default timeout is far lower and whose authorization work lands on top of
    the scan — the measured incident sat here, at an estimated 278s. Below it
    and public, the query is merely expensive, which is a choice an operator is
    entitled to make.

    `in_progress` says `rows` is a running count from a publish that is still
    streaming, so the wording claims a floor rather than a total. The estimate
    is then a lower bound too, which only strengthens the warning.
    """
    if rows <= 0 or dimensions <= 0:
        return None
    seconds = estimated_exact_scan_seconds(rows=rows, dimensions=dimensions)
    if seconds < _ADVISORY_SECONDS:
        return None
    gigabytes = exact_scan_bytes(rows=rows, dimensions=dimensions) / 1_000_000_000
    scanned = (
        f"the {rows:,} rows published so far -- this run is still streaming"
        if in_progress
        else f"all {rows:,} rows"
    )
    at_least = "at least " if in_progress else ""
    measured = (
        f"collection '{collection}' declares `search: exact`, which builds no "
        f"vector index, so every query scans {scanned} x {dimensions} "
        f"dimensions -- {at_least}about {gigabytes:.1f} GB, an estimated "
        f"{at_least}{seconds:.0f}s per query on object storage (issue #461; a "
        "local store will be faster)"
    )
    remedy = (
        "Declare `search: approximate` to build an ANN index. With "
        "`on_index_change: online` the switch is an index build over the "
        "vectors already published -- no re-embed and no new collection name."
    )
    if seconds >= MAX_CONTEXT_TIMEOUT_SECONDS:
        return (
            f"{measured}. No context server can answer this: `timeout_seconds` "
            f"cannot be set above {MAX_CONTEXT_TIMEOUT_SECONDS:.0f}s, so no "
            f"permitted setting is large enough. The index will publish and "
            f"report ready, and every search_context call against it will time "
            f"out. {remedy}"
        )
    if access == "governed":
        return (
            f"{measured}. This index is governed, so it is queryable only "
            f"through a context server, whose `timeout_seconds` defaults to "
            f"{DEFAULT_CONTEXT_TIMEOUT_SECONDS:.0f}s and cannot exceed "
            f"{MAX_CONTEXT_TIMEOUT_SECONDS:.0f}s — and the governed path adds "
            f"warehouse reads and per-row authorization on top of the scan, so "
            f"there is less headroom than the estimate alone suggests. "
            f"{remedy}"
        )
    return f"{measured}. {remedy}"
