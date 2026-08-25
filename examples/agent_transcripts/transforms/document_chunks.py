from __future__ import annotations

import json
import re
from typing import Any

import polars as pl

from stel.agent_context import empty_agent_context_frame, project_document_chunk_row
from stel.transforms import IncrementalContract, ReferenceDep

# Extra columns this wrapper adds on top of the agent_context/v1 contract.
# `upstream_document_id` is the incremental parent key; the exchange
# attributes are what make the search index filterable by what a session
# actually did (issue #360 §4-5).
_EXTRA_FIELDS: dict[str, pl.DataType] = {
    "title": pl.String(),
    "harness": pl.String(),
    "upstream_document_id": pl.String(),
    "exchange_ordinal": pl.Int64(),
    "exchange_heading": pl.String(),
    "tools_used": pl.String(),
    "files_touched": pl.String(),
    "tool_errors": pl.Int64(),
}

_ORDINAL_IN_SECTION = re.compile(r"^\[(\d+)\]")


def declared_incremental_contract(options: dict[str, Any]) -> IncrementalContract:
    """Sessions are the parents; the registry is a keyed reference (issue
    #364), so ingesting one new session projects only that session's chunks."""
    return IncrementalContract(
        parent_key="upstream_document_id",
        child_key="chunk_id",
        parent_source="transcript_chunks",
        parent_source_key="document_id",
        reference_deps=(
            ReferenceDep("document_registry", join_key="upstream_document_id"),
        ),
    )


def _schema() -> dict[str, pl.DataType]:
    # Contract schema, never inference (issue #366): chunks from a session
    # with no attributable section would otherwise infer Null-typed optional
    # citation columns.
    return {
        **dict(empty_agent_context_frame("document_chunks").schema),
        **_EXTRA_FIELDS,
    }


def _exchange_index(registry_row: dict[str, Any]) -> dict[int, dict[str, Any]]:
    raw = registry_row.get("exchanges")
    parsed = json.loads(raw) if isinstance(raw, str) and raw else []
    index: dict[int, dict[str, Any]] = {}
    if isinstance(parsed, list):
        for exchange in parsed:
            if isinstance(exchange, dict) and isinstance(exchange.get("ordinal"), int):
                index[exchange["ordinal"]] = exchange
    return index


def run(deps: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Wrap each transcript chunk in an agent_context/v1 document_chunks row,
    joining the chunk's `section` heading back to its exchange metadata via
    the `[<ordinal>]` prefix the converter renders (issue #360 §1)."""
    registry_by_doc = {
        str(row["upstream_document_id"]): row
        for row in deps["document_registry"].iter_rows(named=True)
    }
    rows: list[dict[str, Any]] = []
    for chunk in deps["transcript_chunks"].iter_rows(named=True):
        parent = registry_by_doc[str(chunk["document_id"])]
        section = chunk.get("section")
        ordinal: int | None = None
        exchange: dict[str, Any] = {}
        if isinstance(section, str):
            match = _ORDINAL_IN_SECTION.match(section)
            if match is not None:
                ordinal = int(match.group(1))
                exchange = _exchange_index(parent).get(ordinal, {})
        row = project_document_chunk_row(
            parent,
            chunk_index=int(chunk["chunk_index"]),
            text=str(chunk["text"]),
            upstream_unique_id="model.agent_transcripts.transcript_chunks",
            invocation_id="agent-transcripts-v1",
            chunker_identity=f"{chunk['chunk_strategy']}:700:80:v1",
            citation_section_path=[section] if isinstance(section, str) else None,
            parser_identity="transcript/v1",
            transform_identity="agent-transcripts-chunks:v1",
            schema_fingerprint="agent-transcripts-chunks-v1",
        )
        row["title"] = str(parent["title"])
        row["harness"] = str(parent["harness"])
        row["upstream_document_id"] = str(chunk["document_id"])
        row["exchange_ordinal"] = ordinal
        row["exchange_heading"] = section if isinstance(section, str) else None
        row["tools_used"] = json.dumps(exchange.get("tools_used") or [])
        row["files_touched"] = json.dumps(exchange.get("files_touched") or [])
        row["tool_errors"] = (
            exchange["tool_errors"]
            if isinstance(exchange.get("tool_errors"), int)
            else None
        )
        rows.append(row)
    return pl.DataFrame(rows, schema=_schema())
