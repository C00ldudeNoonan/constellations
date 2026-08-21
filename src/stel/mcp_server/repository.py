from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from ..adapters import create_adapter
from ..adapters.base import AdapterError, ReadPredicate
from ..append_log import QUERY_LOG_SCHEMA, write_rows
from ..config import load_project
from ..profile import resolve_profile

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


class WarehouseContextRepository:
    def __init__(
        self,
        project_dir: Path,
        *,
        target: str | None = None,
        profiles_dir: Path | None = None,
    ) -> None:
        self._project_dir = project_dir
        project, _, _ = load_project(project_dir)
        self._resolved = resolve_profile(
            project,
            project_dir,
            target=target,
            profiles_dir=profiles_dir,
        )

    def query_log_captures_text(self) -> bool:
        config = self._resolved.mcp_query_log
        return config is not None and config.enabled and config.capture_query_text

    def log_query(self, row: Mapping[str, Any]) -> None:
        """Append a served query to the log, if this target enabled one.

        Best-effort by contract (see `append_log`): serving an answer must
        never fail because its log line could not be written.
        """
        config = self._resolved.mcp_query_log
        if config is None or not config.enabled:
            return
        try:
            with create_adapter(
                self._resolved.warehouse,
                project_dir=self._project_dir,
            ) as adapter:
                write_rows(
                    adapter,
                    config,
                    [dict(row)],
                    schema=QUERY_LOG_SCHEMA,
                    what="the MCP query log",
                )
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
            with create_adapter(
                self._resolved.warehouse,
                project_dir=self._project_dir,
            ) as adapter:
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
