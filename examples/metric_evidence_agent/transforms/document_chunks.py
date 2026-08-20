from __future__ import annotations

import json
from typing import Any

import polars as pl

from stel.agent_context import (
    content_hash,
    make_chunk_id,
    make_context_id,
    make_provenance_fingerprint,
)

_CARRIED_FIELDS = (
    "validity_known",
    "valid_from",
    "valid_to",
    "recorded_from",
    "recorded_to",
    "tenant_id",
    "is_public",
    "access_groups",
    "classification",
    "policy_ref",
    "policy_version",
    "authorization_resolved",
    "source_updated_at",
    "source_observed_at",
    "ingested_at",
    "materialized_at",
    "freshness_checked_at",
    "refresh_due_at",
    "stale_after",
)


def run(deps: dict[str, pl.DataFrame]) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for document in deps["document_registry"].iter_rows(named=True):
        text = str(document["text"])
        chunk_id = make_chunk_id(str(document["document_id"]), 0, text)
        rows.append(
            {
                "context_id": make_context_id(
                    str(document["document_version_id"]),
                    chunk_id,
                ),
                "chunk_id": chunk_id,
                "document_id": str(document["document_id"]),
                "document_version_id": str(document["document_version_id"]),
                "chunk_index": 0,
                "text": text,
                "chunk_content_hash": content_hash(text),
                "source_uri": str(document["source_uri"]),
                "citation_page_number": None,
                "citation_section_path": [str(document["section"])],
                "citation_char_start": 0,
                "citation_char_end": len(text),
                "citation_speaker": None,
                "citation_start_seconds": None,
                "citation_end_seconds": None,
                "citation_locator": json.dumps(
                    {"section": str(document["section"])},
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                **{name: document[name] for name in _CARRIED_FIELDS},
                "chunker_identity": "whole-document:v1",
                "upstream_unique_id": "model.metric_evidence_agent.document_registry",
                "invocation_id": "metric-evidence-example-v1",
                "parser_identity": "json:v1",
                "transform_identity": "metric-evidence-chunks:v1",
                "prompt_fingerprint": None,
                "schema_fingerprint": "metric-evidence-chunks-v1",
                "provider_identity": None,
                "model_identity": None,
                "provenance_fingerprint": make_provenance_fingerprint(
                    {
                        "document_version_id": str(document["document_version_id"]),
                        "transform": "metric-evidence-chunks:v1",
                    }
                ),
                "title": str(document["title"]),
                "source_version": str(document["source_version"]),
                "customer_segment": str(document["customer_segment"]),
                "effective_date": document["effective_date"],
            }
        )
    return pl.DataFrame(rows)
