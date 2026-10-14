"""Wrap every extracted doc and ticket in an agent_context/v1 registry row.

The contract carries what an agent needs to cite and trust a passage: a
stable document id, a version, validity dates, and access policy. The
`project_document_registry_row` helper computes the ids and fingerprints;
this file decides what each source's fields mean.
"""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import PurePath
from typing import Any

import polars as pl

from stel.agent_context import empty_agent_context_frame, project_document_registry_row

# Columns added on top of the contract. `text` is what `passages` chunks.
_EXTRA_FIELDS: dict[str, pl.DataType] = {
    "title": pl.String(),
    "kind": pl.String(),
    "team": pl.String(),
    "published_at": pl.String(),
    "text": pl.String(),
}


def _schema() -> dict[str, pl.DataType]:
    # Built from the contract rather than inferred, so an optional column
    # that happens to be all-null keeps its declared type.
    return {
        **dict(empty_agent_context_frame("document_registry").schema),
        **_EXTRA_FIELDS,
    }


def _instant(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _row(
    *,
    source_system: str,
    source_key: str,
    source_version: str,
    source_uri: str,
    dated: str,
    text: str,
) -> dict[str, Any]:
    at = _instant(dated)
    return project_document_registry_row(
        text=text,
        source_system=source_system,
        source_key=source_key,
        source_version=source_version,
        source_uri=source_uri,
        upstream_unique_id=f"model.demo.raw_{source_system}",
        invocation_id="demo-v1",
        recorded_from=at,
        ingested_at=at,
        materialized_at=at,
        valid_from=at,
        # Everything in the demo is readable by anyone. A governed index
        # would set tenants and access groups here, per document.
        tenant_id="acme",
        is_public=True,
        authorization_resolved=True,
        classification="public",
        policy_ref=f"{source_system}:{source_key}",
        policy_version="v1",
        source_updated_at=at,
        source_observed_at=at,
        parser_identity=f"{source_system}:v1",
        transform_identity="demo-registry:v1",
        schema_fingerprint="demo-registry-v1",
    )


def run(deps: dict[str, pl.DataFrame]) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for doc in deps["raw_docs"].iter_rows(named=True):
        name = PurePath(str(doc["source_path"])).name
        row = _row(
            source_system="docs",
            source_key=PurePath(name).stem,
            source_version=str(doc["content_hash"]),
            source_uri=f"data/docs/{name}",
            dated=str(doc["published_at"]),
            text=str(doc["body"]),
        )
        row.update(
            title=str(doc["title"]),
            kind=str(doc["kind"]),
            team=str(doc["team"]),
            published_at=str(doc["published_at"]),
            text=str(doc["body"]),
        )
        rows.append(row)
    for ticket in deps["raw_tickets"].iter_rows(named=True):
        ticket_id = str(ticket["ticket_id"])
        text = (
            f"# {ticket['subject']}\n\n"
            f"Opened by {ticket['customer']} ({ticket['priority']} priority).\n\n"
            f"{ticket['body']}"
        )
        row = _row(
            source_system="tickets",
            source_key=ticket_id,
            source_version=str(ticket["content_hash"]),
            source_uri=f"data/tickets/{ticket_id.lower()}.json",
            dated=str(ticket["opened_at"]),
            text=text,
        )
        row.update(
            title=f"{ticket_id}: {ticket['subject']}",
            kind="ticket",
            team="support",
            published_at=str(ticket["opened_at"])[:10],
            text=text,
        )
        rows.append(row)
    return pl.DataFrame(rows, schema=_schema())
