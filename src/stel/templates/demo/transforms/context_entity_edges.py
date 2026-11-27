"""Publish asserted relations in the agent_context/v1 edge grain.

This is what lets an agent ask "what is the Payments API related to, and where
does it say so?" over MCP: `get_entity_neighbors` reads these rows, walks them
outward, and cites the passage each one came from.

The demo configures `entity_mentions` and the extractors with
`context_id` as their document id, so a relation's `document_id` *is* the
passage that asserts it. Another project maps its own identifier here, and
`project_context_entity_edge` takes `context_id` explicitly for that reason.
"""
from __future__ import annotations

from typing import Any

import polars as pl

from stel.agent_context import empty_agent_context_frame, project_context_entity_edge


def run(deps: dict[str, pl.DataFrame]) -> pl.DataFrame:
    recorded = {
        str(row["context_id"]): row["recorded_from"]
        for row in deps["document_chunks"].iter_rows(named=True)
    }
    mentions = {
        str(link["mention_id"]): link
        for link in deps["entity_links"].iter_rows(named=True)
        if link["status"] == "matched"
    }
    rows: dict[str, dict[str, Any]] = {}
    for relation in deps["entity_relations"].iter_rows(named=True):
        if relation["status"] != "asserted":
            continue
        subject = mentions.get(str(relation["subject_mention_id"]))
        obj = mentions.get(str(relation["object_mention_id"]))
        if subject is None or obj is None:
            continue
        if (subject["entity_namespace"], subject["canonical_id"]) == (
            obj["entity_namespace"],
            obj["canonical_id"],
        ):
            continue
        row = project_context_entity_edge(
            context_id=str(relation["document_id"]),
            relation_id=str(relation["relation_id"]),
            relation_type=str(relation["relation_type"]),
            directed=bool(relation["directed"]),
            subject_namespace=str(subject["entity_namespace"]),
            subject_name="system",
            subject_canonical_id=str(subject["canonical_id"]),
            object_namespace=str(obj["entity_namespace"]),
            object_name="system",
            object_canonical_id=str(obj["canonical_id"]),
            confidence=relation["confidence"],
            recorded_from=recorded[str(relation["document_id"])],
        )
        rows[row["context_entity_edge_id"]] = row
    schema = dict(empty_agent_context_frame("context_entity_edges").schema)
    return pl.DataFrame(list(rows.values()), schema=schema)
