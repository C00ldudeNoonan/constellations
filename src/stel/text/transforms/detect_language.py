"""Emit a 2-letter language code per row.

YAML:

    transform:
      type: python
      module: stel.text.transforms.detect_language
      options:
        text_field: body          # which column to inspect (required)
        output_field: language    # emit column (default: "language")
        default: en               # fallback when detection fails / text too short
"""
from __future__ import annotations

import polars as pl

from ...transforms import TransformContext
from ..language import detect_language
from ._helpers import require_text_column, upstream_df


def run(deps: dict[str, pl.DataFrame], ctx: TransformContext) -> pl.DataFrame:
    df = upstream_df(deps)
    text_field = ctx.options.get("text_field", "text")
    out_field = ctx.options.get("output_field", "language")
    default = ctx.options.get("default")
    require_text_column(df, text_field)

    langs = [detect_language(t or "", default=default) for t in df[text_field].to_list()]
    return df.with_columns(pl.Series(out_field, langs))
