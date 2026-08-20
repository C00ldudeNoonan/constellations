from __future__ import annotations

import polars as pl


def run(deps: dict[str, pl.DataFrame]) -> pl.DataFrame:
    raw = deps["raw_tickets"]
    open_only = raw.filter(~pl.col("status").is_in(["resolved", "closed"]))
    return (
        open_only.group_by("assigned_team")
        .agg(
            pl.len().alias("open_total"),
            (pl.col("priority") == "urgent").sum().alias("urgent_open"),
            (pl.col("priority") == "high").sum().alias("high_open"),
            (pl.col("priority") == "medium").sum().alias("medium_open"),
            (pl.col("priority") == "low").sum().alias("low_open"),
        )
        .sort("open_total", descending=True)
    )
