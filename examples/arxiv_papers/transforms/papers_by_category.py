from __future__ import annotations

import polars as pl


def run(deps: dict[str, pl.DataFrame]) -> pl.DataFrame:
    raw = deps["raw_papers"]
    return (
        raw.group_by("primary_category")
        .agg(
            pl.len().alias("paper_count"),
            pl.col("n_authors").mean().round(2).alias("avg_authors"),
        )
        .sort("paper_count", descending=True)
    )
