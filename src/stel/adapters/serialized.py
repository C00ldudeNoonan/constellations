"""One warehouse connection shared across threads, one statement at a time.

Neither DuckDB connection is thread-safe, and a BigQuery client is safest
treated the same way, so every caller that hands one adapter to concurrent
threads wraps it here: the runner under `--threads N` (one connection, one
model per thread) and a serving session holding its connection across MCP
requests (issue #523), whose tools run on the SDK's worker threads.
"""
from __future__ import annotations

import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

from .base import TableReadSnapshot, WarehouseAdapter


class SerializedAdapter:
    """Serializes every adapter method call behind a lock so independent callers
    can run on separate threads while sharing one warehouse connection. Property
    access (schema_ref, catalog, …) passes through untouched; only callables are
    guarded, which covers all the read/write paths the runner and the serving
    path use."""

    def __init__(self, adapter: WarehouseAdapter, lock: threading.Lock) -> None:
        self._adapter = adapter
        self._lock = lock

    @contextmanager
    def table_snapshot(
        self,
        table: str,
        *,
        columns: Sequence[str] | None = None,
        batch_size: int = 10_000,
        predicate: Any = None,
        key_column: str | None = None,
    ) -> Iterator[TableReadSnapshot]:
        """Guard the open; stream unlocked (issue #432).

        Holding the lock across the whole context serialized `--threads N` down
        to one model at a time, because a streaming stage keeps its snapshot
        open for its entire run — provider calls and publishes included. That is
        not serialized I/O, it is serialized execution, and it arrived with the
        bounded-memory work: before #411 and #423, embed and chunk read through
        `read_table`, which takes the generic per-call lock and releases it.

        Streaming unlocked is safe for the same reason `state_page_reader` has
        always been: both adapters open a **dedicated cursor** for the snapshot
        (`DuckDBAdapter._cursor()`, BigQuery's own query job), so the read does
        not share the session the lock protects. Creating that cursor does touch
        the shared connection, so the open stays guarded.
        """
        with self._lock:
            manager = self._adapter.table_snapshot(
                table,
                columns=columns,
                batch_size=batch_size,
                predicate=predicate,
                key_column=key_column,
            )
            snapshot = manager.__enter__()
        try:
            yield snapshot
        except BaseException as error:
            with self._lock:
                if not manager.__exit__(type(error), error, error.__traceback__):
                    raise
        else:
            with self._lock:
                manager.__exit__(None, None, None)

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._adapter, name)
        if not callable(attr):
            return attr

        def guarded(*args: Any, **kwargs: Any) -> Any:
            with self._lock:
                return attr(*args, **kwargs)

        return guarded
