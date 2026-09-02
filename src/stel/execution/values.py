from __future__ import annotations

import json
from typing import Any


def scalarize(value: Any) -> Any:
    if isinstance(value, dict | list):
        return json.dumps(value, default=str)
    return value


def warehouse_key_cast_matches_python(dtype: Any) -> bool:
    """Whether `CAST(col AS VARCHAR/STRING)` reproduces Python's `str(value)`.

    State keys are written as `str(value)`, so removal detection may only be
    pushed into the warehouse for columns where the engine's own cast lands on
    the same text. It does for strings trivially and for integers, which is
    the typed-id contract embed actually supports.

    It does **not** for booleans: Python writes `True`, DuckDB and BigQuery
    both cast to `true`, and an anti-join on that mismatch reports every
    unchanged row as absent — which is a deletion, not a slow query (PR #457
    review). Floats, decimals and temporals are excluded on the same
    principle rather than by testing a few values: their text form is a
    formatting decision each engine makes for itself, and the failure is
    silent and destructive.

    Anything else falls back to reconciling in Python, where both sides go
    through the same `str()` that wrote the key.
    """
    import polars as pl

    # Named explicitly rather than via a polars dtype group: the groups are
    # deprecated, and the point here is that this list is a *decision* about
    # which casts are provably identical, not a category lookup.
    return dtype in {
        pl.String,
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
    }
