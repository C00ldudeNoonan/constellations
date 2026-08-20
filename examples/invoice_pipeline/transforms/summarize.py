from __future__ import annotations

import polars as pl


def run(deps: dict[str, pl.DataFrame]) -> pl.DataFrame:
    raw = deps["raw_invoices"]
    return (
        raw.group_by("vendor")
        .agg(
            pl.len().alias("invoice_count"),
            pl.col("total").sum().alias("total_spend"),
        )
        .sort("total_spend", descending=True)
    )
