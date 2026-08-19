from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import polars as pl

from dbt_ml.agent_context import project_document_registry_row

_RECORDED_AT = datetime(2026, 7, 20, tzinfo=UTC)


def run(deps: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Wrap the built-in extraction: model's output in an agent_context/v1
    document_registry row, using the project_document_registry_row helper for
    id/fingerprint computation. See models/document_registry.yml."""
    rows: list[dict[str, Any]] = []
    for source in deps["raw_research_notes"].iter_rows(named=True):
        source_key = str(source["source_key"])
        effective_date = datetime.fromisoformat(str(source["effective_date"])).astimezone(UTC)
        row = project_document_registry_row(
            text=str(source["text"]),
            source_system="research_notes",
            source_key=source_key,
            source_version=str(source["source_version"]),
            source_uri=str(source["canonical_uri"]),
            upstream_unique_id="source.agent_context_from_builtin_pipeline.research_notes",
            invocation_id="agent-context-builtin-pipeline-v1",
            recorded_from=_RECORDED_AT,
            ingested_at=_RECORDED_AT,
            materialized_at=_RECORDED_AT,
            valid_from=effective_date,
            tenant_id="economic-data",
            is_public=bool(source["is_public"]),
            access_groups=source["access_groups"] or [],
            classification=str(source["classification"]),
            policy_ref=f"research:{source_key}",
            policy_version=str(source["source_version"]),
            authorization_resolved=True,
            source_updated_at=effective_date,
            source_observed_at=_RECORDED_AT,
            parser_identity="json:v1",
            transform_identity="agent-context-builtin-pipeline-registry:v1",
            schema_fingerprint="agent-context-builtin-pipeline-registry-v1",
        )
        row["title"] = str(source["title"])
        rows.append(row)
    return pl.DataFrame(rows)
