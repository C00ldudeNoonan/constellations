from __future__ import annotations

import argparse
import asyncio
import csv
import io
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.shared.memory import create_connected_server_and_client_session
from metric_mcp_fixture import create_metric_mcp_server

from dbt_ml.agent_context import canonical_entity_key, make_entity_id
from dbt_ml.mcp_server.authorization import (
    ClaimAuthorizationProvider,
    Principal,
    StaticPrincipalResolver,
)
from dbt_ml.mcp_server.server import create_mcp_server
from dbt_ml.mcp_server.service import ContextService

_MODEL = "context_search"
_METRIC_ARGUMENTS = {
    "metrics": ["refund_rate"],
    "group_by": [
        {"name": "metric_time", "grain": "quarter"},
        {"name": "customer_segment"},
    ],
    "where": "customer_segment = 'enterprise'",
    "limit": 2,
}
_SEARCH_ARGUMENTS = {
    "model": _MODEL,
    "query": "enterprise refund policy",
    "mode": "text",
    "limit": 10,
    "filters": [
        {"field": "customer_segment", "operator": "eq", "value": "enterprise"},
        {"field": "effective_date", "operator": "ge", "value": "2026-04-01"},
        {"field": "effective_date", "operator": "le", "value": "2026-06-30"},
    ],
}


def _tool_json(result: Any) -> dict[str, Any]:
    if result.isError:
        message = result.content[0].text if result.content else "unknown MCP error"
        raise RuntimeError(message)
    if result.structuredContent is None:
        raise RuntimeError("MCP tool returned no structured content")
    return dict(result.structuredContent)


def _tool_text(result: Any) -> str:
    if result.isError:
        message = result.content[0].text if result.content else "unknown MCP error"
        raise RuntimeError(message)
    for content in result.content:
        if content.type == "text":
            return str(content.text)
    raise RuntimeError("MCP tool returned no text content")


def _utc_timestamp(value: str) -> str:
    return datetime.fromisoformat(value).astimezone(UTC).isoformat().replace("+00:00", "Z")


async def _query_metric() -> tuple[dict[str, Any], dict[str, Any]]:
    dbt_mcp_url = os.environ.get("DBT_MCP_URL")
    if dbt_mcp_url:
        token = os.environ.get("DBT_MCP_ACCESS_TOKEN")
        headers = {"Authorization": f"Bearer {token}"} if token else None
        async with streamablehttp_client(dbt_mcp_url, headers=headers) as streams:
            async with ClientSession(streams[0], streams[1]) as session:
                await session.initialize()
                result = await session.call_tool("query_metrics", _METRIC_ARGUMENTS)
    else:
        app = create_metric_mcp_server()
        async with create_connected_server_and_client_session(
            app,
            raise_exceptions=True,
        ) as session:
            result = await session.call_tool("query_metrics", _METRIC_ARGUMENTS)
    rows = list(csv.DictReader(io.StringIO(_tool_text(result))))
    baseline = float(rows[0]["refund_rate"])
    current = float(rows[1]["refund_rate"])
    entity_key = canonical_entity_key("enterprise")
    metric = {
        "name": "refund_rate",
        "filters": {
            "customer_segment": "enterprise",
            "metric_time": ["2026-01-01", "2026-06-30"],
        },
        "customer_segment": "enterprise",
        "entity": {
            "namespace": "economic_data",
            "name": "customer_segment",
            "entity_key": entity_key,
            "entity_id": make_entity_id(
                "economic_data",
                "customer_segment",
                entity_key,
            ),
        },
        "baseline_quarter": rows[0]["metric_time__quarter"],
        "baseline_value": baseline,
        "current_quarter": rows[1]["metric_time__quarter"],
        "current_value": current,
        "change_percentage_points": round((current - baseline) * 100, 1),
        "data_as_of": "2026-06-30T23:59:59Z",
    }
    trace = {
        "server": "dbt-mcp",
        "tool": "query_metrics",
        "arguments": _METRIC_ARGUMENTS,
        "result": {"row_count": len(rows), "metric": "refund_rate"},
    }
    return metric, trace


async def _query_context(
    project_dir: Path,
    *,
    subject_id: str,
    access_groups: tuple[str, ...],
    scenario: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    service = ContextService.from_project(
        project_dir,
        principal_resolver=StaticPrincipalResolver(
            Principal(
                subject_id=subject_id,
                tenant_id="economic-data",
                access_groups=access_groups,
            )
        ),
        authorization=ClaimAuthorizationProvider(),
    )
    trace: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    try:
        app = create_mcp_server(service)
        async with create_connected_server_and_client_session(
            app,
            raise_exceptions=True,
        ) as session:
            listed = _tool_json(await session.call_tool("list_context_models", {}))
            trace.append(
                {
                    "server": "dbt-ml",
                    "tool": "list_context_models",
                    "scenario": scenario,
                    "arguments": {},
                    "result": {
                        "models": [model["name"] for model in listed["models"]],
                    },
                }
            )

            searched = _tool_json(await session.call_tool("search_context", _SEARCH_ARGUMENTS))
            if searched.get("error") is not None:
                raise RuntimeError(str(searched["error"]))
            results = sorted(
                searched["results"],
                key=lambda row: (row["interval"]["valid_from"], row["document_id"]),
            )
            trace.append(
                {
                    "server": "dbt-ml",
                    "tool": "search_context",
                    "scenario": scenario,
                    "arguments": _SEARCH_ARGUMENTS,
                    "result": {
                        "result_count": len(results),
                        "document_ids": [row["document_id"] for row in results],
                    },
                }
            )

            for row in results:
                document_arguments = {
                    "model": _MODEL,
                    "document_id": row["document_id"],
                    "document_version_id": row["document_version_id"],
                }
                document = _tool_json(await session.call_tool("get_document", document_arguments))
                if document.get("error") is not None:
                    raise RuntimeError(str(document["error"]))
                lineage_arguments = {
                    "model": _MODEL,
                    "reference_type": "context",
                    "reference_id": row["context_id"],
                }
                lineage = _tool_json(
                    await session.call_tool(
                        "get_context_lineage",
                        lineage_arguments,
                    )
                )
                if lineage.get("error") is not None:
                    raise RuntimeError(str(lineage["error"]))
                chunk = document["chunks"][0]
                record = lineage["record"]
                evidence.append(
                    {
                        "document_id": document["document_id"],
                        "document_version_id": document["document_version_id"],
                        "context_id": chunk["context_id"],
                        "chunk_id": chunk["chunk_id"],
                        "source_uri": document["source"]["source_uri"],
                        "source_version": document["source"]["source_version"],
                        "valid_from": _utc_timestamp(document["interval"]["valid_from"]),
                        "recorded_from": _utc_timestamp(
                            document["interval"]["recorded_from"]
                        ),
                        "freshness": document["freshness"]["status"],
                        "text": chunk["text"],
                        "citation": chunk["citation"],
                        "entities": chunk["entities"],
                        "provenance_fingerprint": record["lineage"]["provenance_fingerprint"],
                    }
                )
                trace.extend(
                    [
                        {
                            "server": "dbt-ml",
                            "tool": "get_document",
                            "scenario": scenario,
                            "arguments": document_arguments,
                            "result": {
                                "source_uri": document["source"]["source_uri"],
                                "chunk_count": len(document["chunks"]),
                            },
                        },
                        {
                            "server": "dbt-ml",
                            "tool": "get_context_lineage",
                            "scenario": scenario,
                            "arguments": lineage_arguments,
                            "result": {
                                "resources": record["lineage"]["resources"],
                                "provenance_fingerprint": record["lineage"][
                                    "provenance_fingerprint"
                                ],
                            },
                        },
                    ]
                )
    finally:
        service.close()
    return evidence, trace


def _answer(
    metric: dict[str, Any],
    evidence: list[dict[str, Any]],
    *,
    access_limited: bool,
) -> dict[str, Any]:
    entity_id = metric["entity"]["entity_id"]
    if any(
        entity_id not in {entity["entity_id"] for entity in row["entities"]}
        for row in evidence
    ):
        raise RuntimeError("Documentary evidence did not join to the metric entity")
    return {
        "finding": (
            "Enterprise refund rate increased from 4.0% in Q1 2026 to 7.0% "
            "in Q2 2026, a 3.0 percentage-point increase."
        ),
        "interpretation": (
            "The retrieved Q2 policy changes are consistent with the increase, "
            "but this evidence does not establish causation."
        ),
        "access_limited": access_limited,
        "metric": metric,
        "evidence": evidence,
    }


async def run_demo(project_dir: Path) -> dict[str, Any]:
    metric, metric_trace = await _query_metric()
    authorized_evidence, authorized_trace = await _query_context(
        project_dir,
        subject_id="policy-analyst",
        access_groups=("policy-reviewers",),
        scenario="authorized",
    )
    reduced_evidence, reduced_trace = await _query_context(
        project_dir,
        subject_id="external-analyst",
        access_groups=(),
        scenario="reduced",
    )
    return {
        "question": (
            "Why did the enterprise refund rate increase in Q2 2026, and what "
            "governed policy evidence was effective during that quarter?"
        ),
        "authorized_answer": _answer(
            metric,
            authorized_evidence,
            access_limited=False,
        ),
        "reduced_answer": _answer(
            metric,
            reduced_evidence,
            access_limited=True,
        ),
        "tool_trace": [metric_trace, *authorized_trace, *reduced_trace],
    }


def _build(project_dir: Path) -> None:
    executable = Path(sys.executable).with_name("dbt-ml")
    subprocess.run(
        [str(executable), "--project-dir", str(project_dir), "build"],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the metric-plus-evidence demo")
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()
    project_dir = args.project_dir.resolve()
    if not args.skip_build:
        _build(project_dir)
    result = asyncio.run(run_demo(project_dir))
    output_path = project_dir / "target" / "demo_result.json"
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
