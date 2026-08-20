"""Emit derived text statistics as new columns.

YAML usage:

    transform:
      type: python
      module: stel.text.transforms.text_stats
      options:
        text_field: body          # which column to analyze (required)
        emit: [word_count, char_count, sentence_count]  # default: all four
        prefix: ""                # optional column prefix
"""
from __future__ import annotations

import polars as pl

from ...transforms import TransformContext
from ..stats import text_stats
from ._helpers import require_text_column, upstream_df

_ALL_FIELDS = ("char_count", "word_count", "sentence_count", "paragraph_count")


def run(deps: dict[str, pl.DataFrame], ctx: TransformContext) -> pl.DataFrame:
    df = upstream_df(deps)
    text_field = ctx.options.get("text_field", "text")
    require_text_column(df, text_field)

    emit = tuple(ctx.options.get("emit", _ALL_FIELDS))
    prefix = ctx.options.get("prefix", "")

    stats = [text_stats(t or "").to_dict() for t in df[text_field].to_list()]
    new_cols = {
        f"{prefix}{f}": pl.Series([s[f] for s in stats]) for f in emit
    }
    return df.with_columns(**new_cols)
