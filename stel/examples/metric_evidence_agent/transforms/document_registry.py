from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import polars as pl

from stel.agent_context import (
    content_hash,
    make_document_id,
    make_document_version_id,
    make_provenance_fingerprint,
)

_RECORDED_AT = datetime(2026, 7, 1, 12, tzinfo=UTC)
_REFRESH_DUE_AT = datetime(2027, 1, 1, tzinfo=UTC)
_STALE_AFTER = datetime(2027, 4, 1, tzinfo=UTC)


def run(deps: dict[str, pl.DataFrame]) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for source in deps["raw_policy_documents"].iter_rows(named=True):
        text = str(source["text"])
        source_key = str(source["source_key"])
        source_version = str(source["source_version"])
        effective_date = datetime.fromisoformat(str(source["effective_date"])).astimezone(UTC)
        document_id = make_document_id("policy_repository", source_key)
        source_content_hash = content_hash(text)
        document_version_id = make_document_version_id(
            document_id,
            source_version,
            source_content_hash,
        )
        raw_access_groups = source["access_groups"]
        access_groups = sorted(
            str(value)
            for value in (
                json.loads(raw_access_groups)
                if isinstance(raw_access_groups, str)
                else raw_access_groups or []
            )
        )
        rows.append(
            {
                "document_id": document_id,
                "document_version_id": document_version_id,
                "source_system": "policy_repository",
                "source_key": source_key,
                "source_uri": str(source["canonical_uri"]),
                "source_version": source_version,
                "source_content_hash": source_content_hash,
                "validity_known": True,
                "valid_from": effective_date,
                "valid_to": None,
                "recorded_from": _RECORDED_AT,
                "recorded_to": None,
                "tenant_id": "economic-data",
                "is_public": bool(source["is_public"]),
                "access_groups": access_groups,
                "classification": str(source["classification"]),
                "policy_ref": f"policy:{source_key}",
                "policy_version": source_version,
                "authorization_resolved": True,
                "source_updated_at": effective_date,
                "source_observed_at": _RECORDED_AT,
                "ingested_at": _RECORDED_AT,
                "materialized_at": _RECORDED_AT,
                "freshness_checked_at": _RECORDED_AT,
                "refresh_due_at": _REFRESH_DUE_AT,
                "stale_after": _STALE_AFTER,
                "upstream_unique_id": "source.metric_evidence_agent.policy_documents",
                "invocation_id": "metric-evidence-example-v1",
                "parser_identity": "json:v1",
                "transform_identity": "metric-evidence-registry:v1",
                "prompt_fingerprint": None,
                "schema_fingerprint": "metric-evidence-registry-v1",
                "provider_identity": None,
                "model_identity": None,
                "provenance_fingerprint": make_provenance_fingerprint(
                    {
                        "source_key": source_key,
                        "source_version": source_version,
                        "transform": "metric-evidence-registry:v1",
                    }
                ),
                "title": str(source["title"]),
                "section": str(source["section"]),
                "customer_segment": str(source["customer_segment"]),
                "effective_date": effective_date.date().isoformat(),
                "text": text,
            }
        )
    return pl.DataFrame(rows)
