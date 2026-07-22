"""A capture adapter that intercepts a model's output frame instead of writing
it to a real warehouse.

In embedded mode dbt — not dbt-ml — owns materialization: the dbt Python model
returns a frame and dbt runs the CREATE TABLE. So dbt-ml must *produce* the
model's output frame without writing the final target. This adapter subclasses
the reference DuckDB adapter and points it at an in-memory database, so all of
dbt-ml's lifecycle/state/read machinery still works, but the three materialize
entry points stash the frame and skip the write.

Prototype limitation (#177): this serves extraction/transform models whose only
warehouse interaction is the final write. Upstream reads for transform models —
and the bidirectional `dbt_ref` source — are handled by feeding dbt-managed
relations in through the API layer, not by this adapter reading a real warehouse.
"""
from __future__ import annotations

import tempfile
from collections.abc import Iterable
from pathlib import Path

import polars as pl
from pydantic import BaseModel

from ..adapters.duckdb import DuckDBAdapter, DuckDBWarehouseConfig


class CaptureAdapter(DuckDBAdapter):
    """DuckDB adapter that captures the frame handed to each materialize call.

    Backed by a throwaway temp DuckDB file so state tables, schema creation, and
    typed empty relations all behave exactly as the reference adapter — only the
    final target write is intercepted. The real target database (owned by dbt) is
    never touched. The captured frame is read back by the API layer and returned
    to dbt.
    """

    def __init__(self, *, schema: str = "main") -> None:
        # An absolute temp path resolves to itself (no project_dir join) and is
        # deleted on close; nothing dbt-ml-owned is persisted to the dbt database.
        self._scratch_dir = tempfile.TemporaryDirectory(prefix="dbt_ml_embed_")
        scratch_db = Path(self._scratch_dir.name).resolve() / "capture.duckdb"
        config = DuckDBWarehouseConfig(path=scratch_db, schema_name=schema)
        super().__init__(config, project_dir=None)
        self._captured: pl.DataFrame | None = None

    def _close(self) -> None:
        super()._close()
        self._scratch_dir.cleanup()

    @property
    def captured(self) -> pl.DataFrame:
        if self._captured is None:
            raise RuntimeError(
                "No frame was captured — the model produced no materialize call. "
                "The embedded prototype supports extraction/transform models only."
            )
        return self._captured

    def materialize_full(
        self, table: str, df: pl.DataFrame, *, options: BaseModel | None = None
    ) -> int:
        self._captured = df
        return df.height

    def materialize_incremental(
        self,
        table: str,
        df: pl.DataFrame,
        *,
        key_col: str,
        on_schema_change: str = "fail",
        options: BaseModel | None = None,
    ) -> int:
        # Embedded mode has no dbt-ml-side incremental state: dbt owns the table
        # and its incremental strategy. We always hand dbt the full produced frame
        # and let dbt's own materialization decide how to merge it.
        self._captured = df
        return df.height

    def materialize_full_chunks(
        self,
        table: str,
        chunks: Iterable[pl.DataFrame],
        *,
        options: BaseModel | None = None,
    ) -> int:
        frames = [chunk for chunk in chunks if chunk.height or chunk.width]
        combined = pl.concat(frames, how="diagonal") if frames else pl.DataFrame()
        self._captured = combined
        return combined.height
