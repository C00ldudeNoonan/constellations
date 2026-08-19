from __future__ import annotations

from typing import Any

import polars as pl

from dbt_ml.agent_context import project_document_chunk_row


def run(deps: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Wrap the built-in chunk: model's real splitter output in agent_context/v1
    document_chunks rows, using the project_document_chunk_row helper. See
    models/document_chunks.yml.

    research_note_chunks["document_id"] is the built-in extraction pipeline's
    internal id (versioning.compute_document_id) — a different id space than
    the agent_context contract's document_id (make_document_id), which the
    document_registry transform computed. The two share a column name by
    coincidence, so the join to the parent registry row goes through
    source_key, never through document_id.
    """
    registry_by_source_key = {
        str(row["source_key"]): row
        for row in deps["document_registry"].iter_rows(named=True)
    }
    rows: list[dict[str, Any]] = []
    for chunk in deps["research_note_chunks"].iter_rows(named=True):
        parent = registry_by_source_key[str(chunk["source_key"])]
        row = project_document_chunk_row(
            parent,
            chunk_index=int(chunk["chunk_index"]),
            text=str(chunk["text"]),
            upstream_unique_id="model.agent_context_from_builtin_pipeline.research_note_chunks",
            invocation_id="agent-context-builtin-pipeline-v1",
            chunker_identity=f"{chunk['chunk_strategy']}:400:50:v1",
            citation_section_path=[str(parent["title"])],
            parser_identity="json:v1",
            transform_identity="agent-context-builtin-pipeline-chunks:v1",
            schema_fingerprint="agent-context-builtin-pipeline-chunks-v1",
        )
        row["title"] = str(parent["title"])
        rows.append(row)
    return pl.DataFrame(rows)
