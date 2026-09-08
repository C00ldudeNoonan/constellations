from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from threading import Lock
from typing import Any, Protocol

from ..adapters import create_adapter
from ..adapters.base import AdapterError, ReadPredicate
from ..append_log import QUERY_LOG_SCHEMA, write_rows
from ..search import SearchSession
from .query_log import BufferedQueryLog

log = logging.getLogger(__name__)


class ContextRepositoryError(Exception):
    pass


class ContextRepositoryLimitError(ContextRepositoryError):
    pass


class ContextRepository(Protocol):
    def read_rows(
        self,
        relation: str,
        *,
        predicates: Sequence[ReadPredicate],
        max_rows: int,
        columns: Sequence[str] | None = None,
    ) -> tuple[Mapping[str, Any], ...]: ...

    def log_query(self, row: Mapping[str, Any]) -> None:
        """Append one served query to the MCP query log (issue #329).

        A default no-op, so an in-memory or test repository needs no logging
        implementation to satisfy the protocol.
        """
        return None

    def query_log_captures_text(self) -> bool:
        """Whether this target opted into storing raw query text (issue #329)."""
        return False

    def warm_up(self) -> None:
        """Eagerly validate warehouse access at server startup (issue #365).

        A default no-op, so an in-memory or test repository needs no warm-up
        implementation to satisfy the protocol.
        """
        return None

    def close(self) -> None:
        """Flush anything buffered and release resources (issue #528).

        A default no-op, for the same reason as `warm_up`: a repository that
        buffers nothing has nothing to flush.
        """
        return None


class WarehouseContextRepository:
    """Warehouse reads for the MCP service, through the serving session.

    The session is shared with `PortableContextSearch` so the re-read of each
    hit's row, the query-log write and the query's own lease all use one
    warehouse connection when the adapter allows one to be held (issue
    #523); before that, every one of them opened its own.
    """

    def __init__(self, session: SearchSession) -> None:
        self._session = session
        # Resolved now, not on first request: a bad profile should fail the
        # service's construction, before a transport starts.
        self._resolved = session.resolve(None).profile
        self._query_log_lock = Lock()
        self._query_log_buffer: BufferedQueryLog | None = None

    def query_log_captures_text(self) -> bool:
        config = self._resolved.mcp_query_log
        return config is not None and config.enabled and config.capture_query_text

    def warm_up(self) -> None:
        """Open the warehouse once, exactly as a request would.

        Credentials resolve lazily inside the adapter open; under stdio
        serving a hang or failure there surfaces only as a per-call "timeout"
        with no diagnostics (issue #365). Warming up at startup makes a broken
        auth setup fail loudly at boot instead. When the session holds its
        connection, this is the open the first request would otherwise pay.
        """
        with self._session.warehouse(None):
            pass

    def log_query(self, row: Mapping[str, Any]) -> None:
        """Queue a served query for the log, if this target enabled one.

        Returns as soon as the row is buffered (issue #528). #523 took the
        per-query *connect* out of this path by sharing the session's
        warehouse; the append itself was still one write per query, inline
        before the response returned, and on BigQuery a write is a Parquet
        load job. Batching removes the rest.

        Best-effort by contract (see `append_log`): serving an answer must
        never fail, or wait, because its log line could not be written.
        """
        config = self._resolved.mcp_query_log
        if config is None or not config.enabled:
            return
        self._query_log().submit(row)

    def _query_log(self) -> BufferedQueryLog:
        """The buffer for this target, created on first use."""
        config = self._resolved.mcp_query_log
        assert config is not None
        with self._query_log_lock:
            if self._query_log_buffer is None:
                self._query_log_buffer = BufferedQueryLog(
                    self._write_query_log_batch,
                    max_rows_per_flush=config.flush_max_rows,
                    flush_interval_seconds=config.flush_interval_seconds,
                )
            return self._query_log_buffer

    def _write_query_log_batch(self, rows: list[Mapping[str, Any]]) -> None:
        """Append one batch. Runs on the drain thread, so it never raises.

        Opens its own adapter rather than borrowing the session's held
        connection. The session guards that connection, so a batch written
        through it would block whichever request wanted it next for the length
        of a load job -- moving the cost off the logging call and onto an
        unrelated query, which is the problem this is here to remove. One
        connect per batch is the price of not contending, and a batch covers
        many queries.
        """
        config = self._resolved.mcp_query_log
        if config is None or not config.enabled:
            return
        try:
            with create_adapter(
                self._resolved.warehouse,
                project_dir=self._session.project_dir,
            ) as adapter:
                write_rows(
                    adapter,
                    config,
                    [dict(row) for row in rows],
                    schema=QUERY_LOG_SCHEMA,
                    what="the MCP query log",
                )
        except Exception as error:
            log.warning(
                "Could not open the warehouse to write the MCP query log [%s]; "
                "%d row(s) discarded, responses unaffected",
                type(error).__name__,
                len(rows),
            )

    def close(self) -> None:
        """Flush any buffered query-log rows. Safe to call more than once."""
        with self._query_log_lock:
            buffer = self._query_log_buffer
        if buffer is not None:
            buffer.close()

    def read_rows(
        self,
        relation: str,
        *,
        predicates: Sequence[ReadPredicate],
        max_rows: int,
        columns: Sequence[str] | None = None,
    ) -> tuple[Mapping[str, Any], ...]:
        if max_rows < 1:
            raise ValueError("max_rows must be positive")
        rows: list[Mapping[str, Any]] = []
        try:
            with self._session.warehouse(None) as adapter:
                if relation not in adapter.list_tables():
                    return ()
                with adapter.table_snapshot(
                    relation,
                    columns=columns,
                    batch_size=min(max_rows + 1, 1000),
                    predicate=predicates,
                ) as snapshot:
                    for batch in snapshot:
                        for row in batch.to_pylist():
                            rows.append(row)
                            if len(rows) > max_rows:
                                raise ContextRepositoryLimitError(
                                    "The governed context read exceeded its scan limit"
                                )
        except ContextRepositoryLimitError:
            raise
        except AdapterError:
            raise ContextRepositoryError(
                "The governed context relation could not be read"
            ) from None
        return tuple(rows)
