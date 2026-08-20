from __future__ import annotations

from datetime import UTC, datetime

import polars as pl


def run(deps: dict[str, pl.DataFrame]) -> pl.DataFrame:
    raw = deps["raw_tickets"]
    now = datetime.now(UTC)
    return (
        raw.filter(~pl.col("status").is_in(["resolved", "closed"]))
        .with_columns(
            pl.col("created_at")
            .str.to_datetime(strict=False, time_zone="UTC")
            .alias("_created")
        )
        .with_columns(
            (
                (pl.lit(now) - pl.col("_created")).dt.total_seconds() / 3600.0
            ).round(2).alias("age_hours")
        )
        .select(
            "ticket_id",
            "priority",
            "status",
            "assigned_team",
            "customer_tier",
            "age_hours",
        )
        .sort("age_hours", descending=True)
    )
