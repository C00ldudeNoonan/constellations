from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from ..adapters import create_adapter
from ..adapters.base import AdapterError, ReadPredicate
from ..config import load_project
from ..profile import resolve_profile


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
