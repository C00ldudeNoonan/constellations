"""Publish matched entity links in the agent_context/v1 link grain.

This is what lets an agent ask for "passages about the Payments API" or
"every datastore" over MCP: the server reads these rows, and matches their
`(entity_namespace, entity_key)` against the declared vocabulary.
"""
from __future__ import annotations

from typing import Any

import polars as pl

from stel.agent_context import empty_agent_context_frame, project_entity_link


def run(deps: dict[str, pl.DataFrame]) -> pl.DataFrame:
    recorded = {
        str(row["context_id"]): row["recorded_from"]
        for row in deps["document_chunks"].iter_rows(named=True)
    }
    rows: dict[str, dict[str, Any]] = {}
    for link in deps["entity_links"].iter_rows(named=True):
        if link["status"] != "matched":
            continue
        row = project_entity_link(
            context_id=str(link["document_id"]),
            entity_namespace=str(link["entity_namespace"]),
            entity_name="system",
            canonical_id=str(link["canonical_id"]),
            relationship_type="mentions",
            link_method=f"{link['resolver']}:{link['resolver_version']}",
            recorded_from=recorded[str(link["document_id"])],
        )
        # A passage that names a system twice is still one link.
        rows[row["context_entity_link_id"]] = row
    schema = dict(empty_agent_context_frame("context_entity_links").schema)
    return pl.DataFrame(list(rows.values()), schema=schema)
