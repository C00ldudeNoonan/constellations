from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def _answer_snapshot(result: dict[str, Any]) -> dict[str, Any]:
    authorized = result["authorized_answer"]
    reduced = result["reduced_answer"]
    return {
        "question": result["question"],
        "finding": authorized["finding"],
        "interpretation": authorized["interpretation"],
        "metric_data_as_of": authorized["metric"]["data_as_of"],
        "metric_entity_id": authorized["metric"]["entity"]["entity_id"],
        "authorized_sources": [row["source_uri"] for row in authorized["evidence"]],
        "reduced_sources": [row["source_uri"] for row in reduced["evidence"]],
    }


def _trace_snapshot(result: dict[str, Any]) -> list[dict[str, Any]]:
    snapshots: list[dict[str, Any]] = []
    for event in result["tool_trace"]:
        tool = event["tool"]
        payload = event["result"]
        if tool == "query_metrics":
            result_count = payload["row_count"]
        elif tool == "list_context_models":
            result_count = len(payload["models"])
        elif tool == "search_context":
            result_count = payload["result_count"]
        elif tool == "get_document":
            result_count = payload["chunk_count"]
        else:
            result_count = 1
        snapshots.append(
            {
                "server": event["server"],
                "tool": tool,
                "scenario": event.get("scenario"),
                "result_count": result_count,
            }
        )
    return snapshots


def test_metric_evidence_agent_matches_reviewed_snapshots(tmp_path: Path) -> None:
    source = Path(__file__).parents[1] / "examples" / "metric_evidence_agent"
    project = tmp_path / "metric_evidence_agent"
    shutil.copytree(
        source,
        project,
        ignore=shutil.ignore_patterns("target", "__pycache__"),
    )

    environment = os.environ.copy()
    environment.pop("DBT_MCP_URL", None)
    environment.pop("DBT_MCP_ACCESS_TOKEN", None)
    completed = subprocess.run(
        [
            sys.executable,
            str(project / "run_demo.py"),
            "--project-dir",
            str(project),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads((project / "target" / "demo_result.json").read_text())
    expected_answer = json.loads((project / "expected_answer.json").read_text())
    expected_trace = json.loads((project / "expected_tool_trace.json").read_text())

    assert _answer_snapshot(result) == expected_answer
    assert _trace_snapshot(result) == expected_trace
    assert result["authorized_answer"]["access_limited"] is False
    assert result["reduced_answer"]["access_limited"] is True
    for item in result["authorized_answer"]["evidence"]:
        assert len(item["document_id"]) == 32
        assert len(item["document_version_id"]) == 32
        assert len(item["context_id"]) == 32
        assert len(item["chunk_id"]) == 32
        assert item["citation"]["source_uri"] == item["source_uri"]
        assert item["citation"]["section_path"]
        assert item["valid_from"].endswith("Z")
        assert item["recorded_from"].endswith("Z")
        assert item["freshness"] == "fresh"
        assert item["provenance_fingerprint"]

    reduced_payload = json.dumps(
        {
            "answer": result["reduced_answer"],
            "trace": [
                event for event in result["tool_trace"] if event.get("scenario") == "reduced"
            ],
        }
    )
    restricted = result["authorized_answer"]["evidence"][1]
    for field in (
        "document_id",
        "document_version_id",
        "context_id",
        "chunk_id",
        "source_uri",
        "text",
        "provenance_fingerprint",
    ):
        assert restricted[field] not in reduced_payload
