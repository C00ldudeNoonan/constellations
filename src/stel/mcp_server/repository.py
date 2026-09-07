from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from ..adapters.base import AdapterError, ReadPredicate
from ..append_log import QUERY_LOG_SCHEMA, write_rows
from ..search import SearchSession

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
        """Append a served query to the log, if this target enabled one.

        Best-effort by contract (see `append_log`): serving an answer must
        never fail because its log line could not be written.
        """
        config = self._resolved.mcp_query_log
        if config is None or not config.enabled:
            return
        try:
            with self._session.warehouse(None) as adapter:
                written = write_rows(
                    adapter,
                    config,
                    [dict(row)],
                    schema=QUERY_LOG_SCHEMA,
                    what="the MCP query log",
                )
                if written < 1:
                    # `write_rows` keeps its best-effort contract by swallowing
                    # the adapter's error, so a broken held connection would
                    # otherwise survive to fail the next request. Retiring it
                    # costs one reconnect; keeping it costs a served answer.
                    self._session.retire_warehouse(adapter)
        except Exception as error:
            log.warning(
                "Could not open the warehouse to write the MCP query log [%s]; "
                "the response is unaffected",
                type(error).__name__,
            )

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
