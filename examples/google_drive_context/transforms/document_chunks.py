"""agent_context/v1 document_chunks rows from the heading-attributed chunks.

The chunk model carried `source_path` and `section` onto every chunk, so the
join to the registry is by `source_path` (the registry's `source_key`) and the
citation names the document and the heading — a Doc's section, or the slide
of a deck — that the chunk fell under.
"""
from __future__ import annotations

from typing import Any

import polars as pl

from stel.agent_context import empty_agent_context_frame, project_document_chunk_row
from stel.transforms import IncrementalContract, ReferenceDep

_EXTRA_FIELDS: dict[str, pl.DataType] = {
    "title": pl.String(),
    "section": pl.String(),
    "source_key": pl.String(),
}


def _schema() -> dict[str, pl.DataType]:
    return {**dict(empty_agent_context_frame("document_chunks").schema), **_EXTRA_FIELDS}


def declared_incremental_contract(options: dict[str, Any]) -> IncrementalContract:
    """Chunks are the parents, keyed by their document's path; the registry is
    a reference keyed the same way, so a changed Drive file reprojects only
    its own chunks (issue #364)."""
    return IncrementalContract(
        parent_key="source_key",
        child_key="chunk_id",
        parent_source="drive_chunks",
        parent_source_key="source_path",
        reference_deps=(ReferenceDep("document_registry", join_key="source_key"),),
    )


def run(deps: dict[str, pl.DataFrame]) -> pl.DataFrame:
    registry_by_key = {
        str(row["source_key"]): row for row in deps["document_registry"].iter_rows(named=True)
    }
    rows: list[dict[str, Any]] = []
    for chunk in deps["drive_chunks"].iter_rows(named=True):
        source_key = str(chunk["source_path"])
        parent = registry_by_key[source_key]
        title = str(parent["title"])
        section = chunk.get("section")
        section_path = [title, str(section)] if section else [title]
        row = project_document_chunk_row(
            parent,
            chunk_index=int(chunk["chunk_index"]),
            text=str(chunk["text"]),
            upstream_unique_id="model.google_drive_context.drive_chunks",
            invocation_id="google-drive-context-v1",
            chunker_identity=f"{chunk['chunk_strategy']}:600:60:v1",
            citation_section_path=section_path,
            parser_identity="markdown:v1",
            transform_identity="google-drive-context-chunks:v1",
            schema_fingerprint="google-drive-context-chunks-v1",
        )
        row["title"] = title
        row["section"] = str(section) if section else None
        row["source_key"] = source_key
        rows.append(row)
    return pl.DataFrame(rows, schema=_schema())
