"""Wrap each passage in an agent_context/v1 chunk row.

`project_document_chunk_row` copies the parent document's provenance and
policy onto the chunk and computes its stable ids, so a search hit can be
cited back to the exact document version and section it came from.
"""
from __future__ import annotations

from typing import Any

import polars as pl

from stel.agent_context import empty_agent_context_frame, project_document_chunk_row
from stel.transforms import IncrementalContract, ReferenceDep

_EXTRA_FIELDS: dict[str, pl.DataType] = {
    "title": pl.String(),
    "kind": pl.String(),
    "section": pl.String(),
    "source_key": pl.String(),
}


def _schema() -> dict[str, pl.DataType]:
    return {
        **dict(empty_agent_context_frame("document_chunks").schema),
        **_EXTRA_FIELDS,
    }


def declared_incremental_contract(options: dict[str, Any]) -> IncrementalContract:
    """Documents are the parents and the registry a keyed reference, so a
    changed document re-projects its own chunks and nobody else's."""
    return IncrementalContract(
        parent_key="source_key",
        child_key="chunk_id",
        parent_source="passages",
        parent_source_key="source_key",
        reference_deps=(ReferenceDep("document_registry", join_key="source_key"),),
    )


def run(deps: dict[str, pl.DataFrame]) -> pl.DataFrame:
    registry = {
        str(row["source_key"]): row for row in deps["document_registry"].iter_rows(named=True)
    }
    rows: list[dict[str, Any]] = []
    for chunk in deps["passages"].iter_rows(named=True):
        parent = registry[str(chunk["source_key"])]
        section = chunk.get("section")
        row = project_document_chunk_row(
            parent,
            chunk_index=int(chunk["chunk_index"]),
            text=str(chunk["text"]),
            upstream_unique_id="model.demo.passages",
            invocation_id="demo-v1",
            chunker_identity=f"{chunk['chunk_strategy']}:600:0:v1",
            citation_section_path=[str(section)] if section else None,
            parser_identity=parent["parser_identity"],
            transform_identity="demo-chunks:v1",
            schema_fingerprint="demo-chunks-v1",
        )
        row.update(
            title=str(parent["title"]),
            kind=str(parent["kind"]),
            section=str(section) if section else None,
            source_key=str(chunk["source_key"]),
        )
        rows.append(row)
    return pl.DataFrame(rows, schema=_schema())
