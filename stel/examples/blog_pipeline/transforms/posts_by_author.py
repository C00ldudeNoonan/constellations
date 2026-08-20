from __future__ import annotations

import polars as pl


def run(deps: dict[str, pl.DataFrame]) -> pl.DataFrame:
    raw = deps["raw_posts"]
    return (
        raw.group_by("author")
        .agg(
            pl.len().alias("post_count"),
            pl.col("word_count").sum().alias("total_words"),
        )
        .sort("total_words", descending=True)
    )
