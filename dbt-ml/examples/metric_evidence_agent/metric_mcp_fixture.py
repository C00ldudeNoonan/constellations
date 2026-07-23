from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, ConfigDict


class GroupByParam(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    grain: str | None = None


class OrderByParam(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    descending: bool = False


def create_metric_mcp_server() -> Any:
    app = FastMCP(
        "dbt-semantic-layer-fixture",
        instructions="Deterministic offline fixture for the dbt MCP query_metrics tool.",
    )

    @app.tool()
    async def query_metrics(
        metrics: list[str],
        group_by: list[GroupByParam] | None = None,
        order_by: list[OrderByParam] | None = None,
        where: str | None = None,
        limit: int | None = None,
    ) -> str:
        """Return the governed refund-rate metric fixture as CSV."""
        del order_by
        expected_grouping = [
            GroupByParam(name="metric_time", grain="quarter"),
            GroupByParam(name="customer_segment"),
        ]
        if metrics != ["refund_rate"]:
            raise ValueError("This fixture only exposes the refund_rate metric")
        if group_by != expected_grouping:
            raise ValueError("refund_rate must be grouped by quarter and customer segment")
        if where != "customer_segment = 'enterprise'":
            raise ValueError("This example requires the governed enterprise segment filter")
        if limit is not None and limit < 2:
            raise ValueError("The example requires both comparison quarters")
        return (
            "metric_time__quarter,customer_segment,refund_rate\n"
            "2026-04-01,enterprise,0.07\n"
            "2026-01-01,enterprise,0.04\n"
        )

    return app
