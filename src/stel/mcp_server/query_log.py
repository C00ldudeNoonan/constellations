"""Buffered, off-request-path writing for the MCP query log (issue #528).

The log itself shipped with issue #329, and it works, but it wrote one row
per served query through its own warehouse connection, inline in
`search_context` before the response returned. On BigQuery that is a fresh
connect plus a Parquet load job per query -- together several seconds, on a
path the rest of #519's work spent a day taking from 39.7s to 13.7s. Enabling
the log would have given most of that back, which is why #528 asked for the
sink to be buffered and off the request path rather than for more fields.

So `log_query` now hands a row to a queue and returns. One background thread
drains it, batches rows, and writes a batch per flush -- one connect and one
load job for many queries rather than one of each per query.

Fail-open in both directions, because a log line must never cost an answer:

- A full queue **drops** rows rather than blocking the caller. A server busy
  enough to fill it is a server whose queries matter more than its telemetry.
- A failed write is warned about and discarded, never retried into a growing
  backlog and never raised. `append_log.write_rows` already owns that
  contract; this module preserves it across the thread boundary.
"""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Mapping
from typing import Any, Protocol

log = logging.getLogger(__name__)

# Sized to bound memory, not to guarantee delivery. At the default flush
# interval a server would have to sustain thousands of queries per second to
# reach it, and one that does has a bigger problem than a lossy log.
DEFAULT_MAX_PENDING = 10_000


class BatchWriter(Protocol):
    """Writes one batch of log rows. Must not raise; see the module docstring."""

    def __call__(self, rows: list[Mapping[str, Any]]) -> None: ...


class BufferedQueryLog:
    """Collects query-log rows and writes them in batches off the request path.

    Not started until the first row arrives, so a server whose target has no
    query log configured never creates the thread.
    """

    def __init__(
        self,
        write_batch: BatchWriter,
        *,
        max_rows_per_flush: int,
        flush_interval_seconds: float,
        max_pending: int = DEFAULT_MAX_PENDING,
    ) -> None:
        self._write_batch = write_batch
        self._max_rows_per_flush = max_rows_per_flush
        self._flush_interval_seconds = flush_interval_seconds
        self._queue: queue.Queue[Mapping[str, Any] | None] = queue.Queue(
            maxsize=max_pending
        )
        self._lock = threading.Lock()
        self._worker: threading.Thread | None = None
        self._closed = False
        self._dropped = 0

    @property
    def dropped_rows(self) -> int:
        """Rows discarded because the queue was full. Diagnostics only."""
        return self._dropped

    def submit(self, row: Mapping[str, Any]) -> None:
        """Queue one row. Never blocks, never raises."""
        with self._lock:
            if self._closed:
                return
            self._ensure_worker()
        try:
            self._queue.put_nowait(row)
        except queue.Full:
            # Counted rather than logged per row: a full queue means many
            # rows in a row, and a warning each would be its own load problem.
            self._dropped += 1
            if self._dropped == 1 or self._dropped % 1000 == 0:
                log.warning(
                    "MCP query log is falling behind; %d row(s) dropped. "
                    "Responses are unaffected",
                    self._dropped,
                )

    def close(self, *, timeout_seconds: float = 5.0) -> None:
        """Stop accepting rows and flush what is queued.

        Bounded rather than unbounded: shutdown is not allowed to hang on a
        warehouse that has stopped answering. Rows still queued when the
        timeout expires are lost, which is the same trade `submit` makes.
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
            worker = self._worker
        if worker is None:
            return
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            # The sentinel cannot get in, so the worker is already saturated;
            # it exits on the closed flag once it drains.
            pass
        worker.join(timeout=timeout_seconds)
        if worker.is_alive():
            log.warning(
                "MCP query log did not finish flushing within %.1fs; "
                "queued rows are discarded",
                timeout_seconds,
            )

    def _ensure_worker(self) -> None:
        """Start the drain thread. Caller holds the lock."""
        if self._worker is not None:
            return
        # Daemon: a stuck warehouse must not keep the process alive. `close`
        # is the orderly path and is what the service calls.
        self._worker = threading.Thread(
            target=self._drain,
            name="stel-mcp-query-log",
            daemon=True,
        )
        self._worker.start()

    def _drain(self) -> None:
        batch: list[Mapping[str, Any]] = []
        while True:
            try:
                row = self._queue.get(timeout=self._flush_interval_seconds)
            except queue.Empty:
                # The interval elapsed with nothing new: flush what is held so
                # a quiet server's last query still reaches the warehouse
                # rather than waiting for the next one.
                self._flush(batch)
                batch = []
                if self._closed:
                    return
                continue
            if row is None:
                self._flush(batch)
                return
            batch.append(row)
            if len(batch) >= self._max_rows_per_flush:
                self._flush(batch)
                batch = []

    def _flush(self, batch: list[Mapping[str, Any]]) -> None:
        if not batch:
            return
        try:
            self._write_batch(batch)
        except Exception as error:
            # The writer is contracted not to raise, so this is a defect in it
            # rather than a warehouse failure -- but a defect in a log writer
            # must still not take down the drain thread and stop all logging.
            log.warning(
                "MCP query log batch failed [%s]; %d row(s) discarded",
                type(error).__name__,
                len(batch),
            )
