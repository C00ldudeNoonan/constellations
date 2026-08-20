from __future__ import annotations

from datetime import UTC, datetime

import polars as pl


def run(deps: dict[str, pl.DataFrame]) -> pl.DataFrame:
    raw = deps["raw_tickets"]
    now = datetime.now(UTC)
    parsed = raw.with_columns(
        pl.col("created_at")
        .str.to_datetime(strict=False, time_zone="UTC")
        .alias("_created"),
        pl.col("first_response_at")
        .str.to_datetime(strict=False, time_zone="UTC")
        .alias("_first_response"),
    ).with_columns(
        # For tickets with no first response yet, age = now - created;
        # for responded tickets, response latency = first_response - created.
        pl.when(pl.col("_first_response").is_null())
        .then((pl.lit(now) - pl.col("_created")).dt.total_seconds() / 3600.0)
        .otherwise(
            (pl.col("_first_response") - pl.col("_created")).dt.total_seconds() / 3600.0
        )
        .alias("response_age_hours")
    )

    breaches = parsed.filter(
        pl.col("response_age_hours") > pl.col("sla_target_hours")
    )

    return (
        breaches.with_columns(
            (pl.col("response_age_hours") - pl.col("sla_target_hours"))
            .round(2)
            .alias("breach_hours")
        )
        .with_columns(pl.col("response_age_hours").round(2))
        .select(
            "ticket_id",
            "priority",
            "customer_tier",
            "assigned_team",
            "sla_target_hours",
            "response_age_hours",
            "breach_hours",
        )
        .sort("breach_hours", descending=True)
    )
