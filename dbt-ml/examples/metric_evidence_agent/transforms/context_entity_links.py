from __future__ import annotations

from typing import Any

import polars as pl

from dbt_ml.agent_context import (
    canonical_entity_key,
    make_context_entity_link_id,
    make_entity_id,
    make_provenance_fingerprint,
)


def run(deps: dict[str, pl.DataFrame]) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for chunk in deps["document_chunks"].iter_rows(named=True):
        entity_key = canonical_entity_key(str(chunk["customer_segment"]))
        entity_id = make_entity_id(
            "economic_data",
            "customer_segment",
            entity_key,
        )
        rows.append(
            {
                "context_entity_link_id": make_context_entity_link_id(
                    str(chunk["context_id"]),
                    entity_id,
                    "applies_to",
                ),
                "context_id": str(chunk["context_id"]),
                "entity_namespace": "economic_data",
                "entity_name": "customer_segment",
                "entity_id": entity_id,
                "entity_key": entity_key,
                "dbt_unique_id": "semantic_model.metric_evidence_semantic.refunds",
                "relationship_type": "applies_to",
                "link_method": "exact_source_field:v1",
                "confidence": None,
                "recorded_from": chunk["recorded_from"],
                "recorded_to": None,
                "link_provenance_fingerprint": make_provenance_fingerprint(
                    {
                        "context_id": str(chunk["context_id"]),
                        "entity_id": entity_id,
                        "method": "exact_source_field:v1",
                    }
                ),
            }
        )
    return pl.DataFrame(rows)
