"""Flag near-duplicate rows via MinHash + LSH.

YAML:

    transform:
      type: python
      module: stel.text.transforms.find_duplicates
      options:
        text_field: body
        output_field: duplicate_group  # default; null for unique rows, else cluster id
        threshold: 0.85                # Jaccard similarity threshold (default 0.8)
        num_perm: 128                  # MinHash permutations (default 128)
        shingle_size: 5                # k-word shingles (default 5)
"""
from __future__ import annotations

import polars as pl

from ...transforms import TransformContext
from ..dedup import near_duplicates
from ._helpers import require_text_column, upstream_df


def run(deps: dict[str, pl.DataFrame], ctx: TransformContext) -> pl.DataFrame:
    df = upstream_df(deps)
    text_field = ctx.options.get("text_field", "text")
    out_field = ctx.options.get("output_field", "duplicate_group")
    threshold = float(ctx.options.get("threshold", 0.8))
    num_perm = int(ctx.options.get("num_perm", 128))
    k = int(ctx.options.get("shingle_size", 5))
    require_text_column(df, text_field)

    texts = [t or "" for t in df[text_field].to_list()]
    clusters = near_duplicates(
        texts, threshold=threshold, num_perm=num_perm, k=k
    )
    # Assign cluster ids: first cluster -> 0, second -> 1, ... ; non-clustered -> None
    group_by_idx: dict[int, int] = {}
    for gid, cluster in enumerate(clusters):
        for idx in cluster:
            group_by_idx[idx] = gid
    col = [group_by_idx.get(i) for i in range(len(texts))]
    return df.with_columns(pl.Series(out_field, col, dtype=pl.Int64))
