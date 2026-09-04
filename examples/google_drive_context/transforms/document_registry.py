"""agent_context/v1 document_registry rows from the Drive extraction model.

Every field the contract needs is already on the extraction row: `source_path`
is the folder-relative name (the identity), `content_hash` is the md5 or the
change token (the version), `source_uri` pins the Drive file and version, and
`source_metadata` carries the file's modified time.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import PurePosixPath
from typing import Any

import polars as pl

from stel.agent_context import empty_agent_context_frame, project_document_registry_row

# Fixed so the example is reproducible; a production wrapper would stamp the
# run's own clock.
_RECORDED_AT = datetime(2026, 9, 1, tzinfo=UTC)
_EXTRA_FIELDS: dict[str, pl.DataType] = {"title": pl.String()}


def _schema() -> dict[str, pl.DataType]:
    # The contract schema, never inference: an all-null optional column would
    # otherwise infer Null and fail or drift on a later batch (issue #366).
    return {**dict(empty_agent_context_frame("document_registry").schema), **_EXTRA_FIELDS}


def _modified_time(source_metadata: str | None) -> datetime:
    if source_metadata:
        modified = json.loads(source_metadata).get("modified_time")
        if modified:
            return datetime.fromisoformat(str(modified).replace("Z", "+00:00")).astimezone(UTC)
    return _RECORDED_AT


def run(deps: dict[str, pl.DataFrame]) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for source in deps["drive_documents"].iter_rows(named=True):
        source_path = str(source["source_path"])
        modified = _modified_time(source.get("source_metadata"))
        row = project_document_registry_row(
            text=str(source["body"]),
            source_system="google_drive",
            source_key=source_path,
            source_version=str(source["content_hash"]),
            source_uri=str(source["source_uri"]),
            upstream_unique_id="source.google_drive_context.drive_folder",
            invocation_id="google-drive-context-v1",
            recorded_from=_RECORDED_AT,
            ingested_at=_RECORDED_AT,
            materialized_at=_RECORDED_AT,
            valid_from=modified,
            tenant_id="drive",
            is_public=True,
            access_groups=[],
            classification="internal",
            policy_ref=f"drive:{source_path}",
            policy_version="1",
            authorization_resolved=True,
            source_updated_at=modified,
            source_observed_at=_RECORDED_AT,
            parser_identity="markdown:v1",
            transform_identity="google-drive-context-registry:v1",
            schema_fingerprint="google-drive-context-registry-v1",
        )
        # The document's name without the synthesized `.md`.
        row["title"] = PurePosixPath(source_path).stem
        rows.append(row)
    return pl.DataFrame(rows, schema=_schema())
