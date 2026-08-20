"""Replace one column's contents with ftfy-cleaned text.

YAML:

    transform:
      type: python
      module: stel.text.transforms.clean_encoding
      options:
        text_field: body         # which column to clean (required)
        output_field: body       # where to write (defaults to text_field, i.e. in-place)
"""
from __future__ import annotations

import polars as pl

from ...transforms import TransformContext
from ..encoding import clean_encoding
from ._helpers import require_text_column, upstream_df


def run(deps: dict[str, pl.DataFrame], ctx: TransformContext) -> pl.DataFrame:
    df = upstream_df(deps)
    text_field = ctx.options.get("text_field", "text")
    out_field = ctx.options.get("output_field", text_field)
    require_text_column(df, text_field)

    cleaned = [clean_encoding(t or "") for t in df[text_field].to_list()]
    return df.with_columns(pl.Series(out_field, cleaned))
