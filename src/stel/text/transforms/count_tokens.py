"""Emit a token-count column for budgeting LLM calls / chunking decisions.

YAML:

    transform:
      type: python
      module: stel.text.transforms.count_tokens
      options:
        text_field: body
        output_field: token_count   # default
        model: gpt-4o                # encoding name OR family (default: cl100k_base)
"""
from __future__ import annotations

import polars as pl

from ...transforms import TransformContext
from ..tokens import count_tokens
from ._helpers import require_text_column, upstream_df


def run(deps: dict[str, pl.DataFrame], ctx: TransformContext) -> pl.DataFrame:
    df = upstream_df(deps)
    text_field = ctx.options.get("text_field", "text")
    out_field = ctx.options.get("output_field", "token_count")
    model = ctx.options.get("model", "cl100k_base")
    require_text_column(df, text_field)

    counts = [count_tokens(t or "", model=model) for t in df[text_field].to_list()]
    return df.with_columns(pl.Series(out_field, counts).cast(pl.Int64))
