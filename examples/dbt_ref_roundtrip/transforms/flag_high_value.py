"""Flags above-median-spend vendors in a dbt-built table read via `dbt_ref`."""

from __future__ import annotations

import polars as pl


def run(deps: dict[str, pl.DataFrame]) -> pl.DataFrame:
    facts = deps["invoice_facts"]
    threshold = facts["total_spend"].median()
    return facts.with_columns((pl.col("total_spend") >= threshold).alias("flagged"))
