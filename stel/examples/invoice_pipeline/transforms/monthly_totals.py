from __future__ import annotations

import polars as pl


def run(deps: dict[str, pl.DataFrame]) -> pl.DataFrame:
    raw = deps["raw_invoices"]
    return (
        raw.with_columns(
            pl.col("issue_date").str.strptime(pl.Date, format="%Y-%m-%d").alias("_d")
        )
        .with_columns(pl.col("_d").dt.truncate("1mo").alias("month"))
        .group_by("month")
        .agg(
            pl.len().alias("invoice_count"),
            pl.col("total").sum().alias("total_spend"),
        )
        .sort("month")
    )
