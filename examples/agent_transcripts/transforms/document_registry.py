from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import polars as pl

from stel.agent_context import empty_agent_context_frame, project_document_registry_row

# Deterministic fallback when a transcript carries no parseable timestamps.
_FALLBACK_AT = datetime(2026, 8, 20, tzinfo=UTC)

# Extra columns this wrapper adds on top of the agent_context/v1 contract.
# `upstream_document_id` is the extraction pipeline's internal document id —
# the chunks wrapper's incremental contract joins on it (issue #364) — and
# `exchanges` carries the transcript/v1 exchange metadata forward as JSON for
# the chunks wrapper to attribute sections.
_EXTRA_FIELDS: dict[str, pl.DataType] = {
    "title": pl.String(),
    "harness": pl.String(),
    "session_id": pl.String(),
    "upstream_document_id": pl.String(),
    "exchanges": pl.String(),
}


def _schema() -> dict[str, pl.DataType]:
    # Always construct with the contract schema, never inference: a batch
    # that is all-null in an optional column would otherwise infer a
    # Null-typed column and drift the schema per batch (issue #366).
    return {
        **dict(empty_agent_context_frame("document_registry").schema),
        **_EXTRA_FIELDS,
    }


def _timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value).astimezone(UTC)
    except ValueError:
        return None


def _json_text(value: Any) -> str:
    """The extraction backend scalarizes list/object fields to JSON strings;
    normalize either representation to one JSON string column."""
    if isinstance(value, str):
        return value
    return json.dumps(value if value is not None else [])


def run(deps: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Wrap each transcript/v1 session in an agent_context/v1 document_registry
    row. The session is the document (issue #360 §4)."""
    rows: list[dict[str, Any]] = []
    for source in deps["raw_transcripts"].iter_rows(named=True):
        harness = str(source["harness"])
        session_id = str(source["session_id"])
        source_key = f"{harness}-{session_id}"
        started_at = _timestamp(source.get("started_at"))
        ended_at = _timestamp(source.get("ended_at"))
        recorded = ended_at or started_at or _FALLBACK_AT
        row = project_document_registry_row(
            text=str(source["text"]),
            source_system="agent-transcripts",
            source_key=source_key,
            source_version=str(source["content_hash"]),
            source_uri=f"transcript://{harness}/{session_id}",
            upstream_unique_id="model.agent_transcripts.raw_transcripts",
            invocation_id="agent-transcripts-v1",
            recorded_from=recorded,
            ingested_at=recorded,
            materialized_at=recorded,
            valid_from=started_at or recorded,
            tenant_id="local-dev",
            is_public=False,
            access_groups=["operators"],
            classification="internal",
            policy_ref=f"transcript:{source_key}",
            policy_version="v1",
            authorization_resolved=True,
            source_updated_at=ended_at or recorded,
            source_observed_at=recorded,
            parser_identity="transcript/v1",
            transform_identity="agent-transcripts-registry:v1",
            schema_fingerprint="agent-transcripts-registry-v1",
        )
        row["title"] = f"{harness} session {session_id[:8]}"
        row["harness"] = harness
        row["session_id"] = session_id
        row["upstream_document_id"] = str(source["document_id"])
        row["exchanges"] = _json_text(source.get("exchanges"))
        rows.append(row)
    return pl.DataFrame(rows, schema=_schema())
