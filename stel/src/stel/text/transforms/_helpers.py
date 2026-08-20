"""Shared helpers for built-in text transforms."""
from __future__ import annotations

import polars as pl


def upstream_df(deps: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """The single upstream dataframe. Raises if a transform somehow has 0 or
    >1 dependency — built-in text transforms expect exactly one input."""
    if len(deps) != 1:
        raise ValueError(
            f"Built-in text transforms expect exactly one upstream dependency; "
            f"got {len(deps)}: {sorted(deps)}"
        )
    return next(iter(deps.values()))


def require_text_column(df: pl.DataFrame, text_field: str) -> None:
    if text_field not in df.columns:
        raise ValueError(
            f"Expected text column '{text_field}' in upstream; got: "
            f"{sorted(df.columns)}"
        )
