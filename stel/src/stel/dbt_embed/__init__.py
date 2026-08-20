"""Embed stel inside a dbt-duckdb project (design issue #177, prototype).

stel normally runs as a standalone CLI that owns its own DAG, materialization,
and state. This package exposes stel's extraction/transform engine as a
*library* so a dbt-duckdb Python model can invoke it and hand the resulting frame
back to dbt, which owns materialization, tests, docs, and lineage.

Prototype scope (see #177): dbt-duckdb only; extraction/transform models; the
frame is produced in-process and returned to dbt. stel-side materialization,
per-document incremental state, and non-DuckDB warehouses are out of scope here.

This is an optional integration: it imports the stel engine lazily and is only
meaningful when invoked from within a dbt Python model.
"""
from __future__ import annotations

from .api import materialize

__all__ = ["materialize"]
