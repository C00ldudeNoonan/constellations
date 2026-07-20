from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from pydantic import ValidationError

from dbt_ml.adapters import create_adapter, parse_warehouse_config
from dbt_ml.agent_context import (
    AGENT_CONTEXT_CONTRACT,
    AgentContextGrain,
    AgentContextValidationError,
    FreshnessStatus,
    active_as_of,
    canonical_entity_key,
    citation_locator,
    content_hash,
    contract_descriptor,
    freshness_status,
    interval_contains,
    is_publishable_context,
    make_chunk_id,
    make_context_entity_link_id,
    make_context_id,
    make_document_id,
    make_document_version_id,
    make_entity_id,
    make_provenance_fingerprint,
    policy_fingerprint,
    retrieval_projection_fingerprint,
    validate_agent_context_frame,
    validate_agent_context_relations,
)
from dbt_ml.config.model import ModelConfig
from dbt_ml.dbt_export import build_dbt_sources
from dbt_ml.docs import generate_docs
from dbt_ml.manifest import build_manifest
from dbt_ml.runner import RunError, _validate_agent_context_output
from dbt_ml.versioning import compute_code_version

_T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _document(
    *,
    source_version: str = "v1",
    body: str = "Enterprise refund policy",
    recorded_from: datetime = _T0,
    recorded_to: datetime | None = None,
    access_groups: tuple[str, ...] = ("analyst",),
    authorization_resolved: bool = True,
) -> dict[str, Any]:
    source_system = "policy-repository"
    source_key = "refunds/enterprise"
    document_id = make_document_id(source_system, source_key)
    source_content_hash = content_hash(body)
    document_version_id = make_document_version_id(document_id, source_version, source_content_hash)
    return {
        "document_id": document_id,
        "document_version_id": document_version_id,
        "source_system": source_system,
        "source_key": source_key,
        "source_uri": "https://policies.example.test/refunds/enterprise",
        "source_version": source_version,
        "source_content_hash": source_content_hash,
        "validity_known": True,
        "valid_from": _T0,
        "valid_to": None,
        "recorded_from": recorded_from,
        "recorded_to": recorded_to,
        "tenant_id": "economic-data",
        "is_public": False,
        "access_groups": json.dumps(list(access_groups)),
        "classification": "internal",
        "policy_ref": "policy/refunds",
        "policy_version": "3",
        "authorization_resolved": authorization_resolved,
        "source_updated_at": _T0,
        "source_observed_at": _T0 + timedelta(hours=1),
        "ingested_at": _T0 + timedelta(hours=2),
        "materialized_at": _T0 + timedelta(hours=3),
        "freshness_checked_at": _T0 + timedelta(hours=3),
        "refresh_due_at": _T0 + timedelta(days=2),
        "stale_after": _T0 + timedelta(days=30),
        "upstream_unique_id": "source.economic_data.policy_documents",
        "invocation_id": "invocation-001",
        "parser_identity": "markdown/1",
        "transform_identity": "policy-normalization/1",
        "prompt_fingerprint": None,
        "schema_fingerprint": "schema-fingerprint",
        "provider_identity": None,
        "model_identity": None,
        "provenance_fingerprint": make_provenance_fingerprint(
            {
                "source": "source.economic_data.policy_documents",
                "parser": "markdown/1",
                "transform": "policy-normalization/1",
            }
        ),
    }


def _chunk(document: dict[str, Any], *, index: int = 0) -> dict[str, Any]:
    text = "Refunds require approval for enterprise accounts."
    chunk_id = make_chunk_id(document["document_id"], index, text)
    context_id = make_context_id(document["document_version_id"], chunk_id)
    carried = {
        key: document[key]
        for key in (
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
    }
    return {
        "context_id": context_id,
        "chunk_id": chunk_id,
        "document_id": document["document_id"],
        "document_version_id": document["document_version_id"],
        "chunk_index": index,
        "text": text,
        "chunk_content_hash": content_hash(text),
        "source_uri": document["source_uri"],
        "citation_page_number": 2,
        "citation_section_path": json.dumps(["Refunds", "Enterprise"]),
        "citation_char_start": 100,
        "citation_char_end": 151,
        "citation_speaker": None,
        "citation_start_seconds": None,
        "citation_end_seconds": None,
        "citation_locator": json.dumps({"paragraph": 3}),
        **carried,
        "chunker_identity": "recursive/1000/100",
        "upstream_unique_id": "model.economic_data.document_registry",
        "invocation_id": "invocation-001",
        "parser_identity": "markdown/1",
        "transform_identity": "policy-normalization/1",
        "prompt_fingerprint": None,
        "schema_fingerprint": "schema-fingerprint",
        "provider_identity": None,
        "model_identity": None,
        "provenance_fingerprint": make_provenance_fingerprint(
            {
                "source": "model.economic_data.document_registry",
                "chunker": "recursive/1000/100",
            }
        ),
    }


def _link(
    chunk: dict[str, Any],
    *,
    entity_key_value: Any,
    relationship_type: str,
) -> dict[str, Any]:
    entity_key = canonical_entity_key(entity_key_value)
    entity_id = make_entity_id("economic_data", "customer_segment", entity_key)
    return {
        "context_entity_link_id": make_context_entity_link_id(
            chunk["context_id"], entity_id, relationship_type
        ),
        "context_id": chunk["context_id"],
        "entity_namespace": "economic_data",
        "entity_name": "customer_segment",
        "entity_id": entity_id,
        "entity_key": entity_key,
        "dbt_unique_id": "semantic_model.economic_data.orders",
        "relationship_type": relationship_type,
        "link_method": "sql_exact_key",
        "confidence": None,
        "recorded_from": _T0,
        "recorded_to": None,
        "link_provenance_fingerprint": make_provenance_fingerprint(
            {"model": "model.economic_data.context_entity_links"}
        ),
    }


def test_contract_descriptor_is_versioned_and_machine_readable() -> None:
    descriptor = contract_descriptor(AgentContextGrain.DOCUMENT_CHUNKS)

    assert descriptor["contract"] == AGENT_CONTEXT_CONTRACT
    assert descriptor["grain"] == "document_chunks"
    assert descriptor["primary_key"] == ["context_id"]
    assert descriptor["foreign_keys"] == {
        "document_version_id": "document_registry.document_version_id"
    }
    fields = {field["name"]: field for field in descriptor["fields"]}
    assert fields["citation_page_number"]["nullable"] is True
    assert fields["authorization_resolved"]["nullable"] is False


def test_model_config_and_code_version_include_agent_context(tmp_path: Path) -> None:
    model = ModelConfig.model_validate(
        {
            "name": "document_registry",
            "transform": {"type": "python", "module": "transforms.context"},
            "agent_context": {"grain": "document_registry"},
        }
    )
    assert model.agent_context is not None
    assert model.agent_context.contract == AGENT_CONTEXT_CONTRACT
    assert model.agent_context.grain is AgentContextGrain.DOCUMENT_REGISTRY

    without_contract = compute_code_version(
        extraction=None,
        transform=model.transform,
        project_dir=tmp_path,
    )
    with_contract = compute_code_version(
        extraction=None,
        transform=model.transform,
        agent_context=model.agent_context,
        project_dir=tmp_path,
    )
    assert with_contract != without_contract

    with pytest.raises(ValidationError, match="warehouse transform model"):
        ModelConfig.model_validate(
            {
                "name": "builtin_chunks",
                "chunk": {},
                "agent_context": {"grain": "document_chunks"},
            }
        )

    with pytest.raises(ValidationError, match="upstream warehouse model"):
        ModelConfig.model_validate(
            {
                "name": "served_context",
                "materialization": "full",
                "search": {
                    "collection": "context",
                    "id_field": "context_id",
                    "text_fields": ["text"],
                    "query": {"modes": ["filter"]},
                },
                "agent_context": {"grain": "document_chunks"},
            }
        )


def test_runtime_validation_runs_before_contract_materialization() -> None:
    model = ModelConfig.model_validate(
        {
            "name": "document_registry",
            "transform": {"type": "python", "module": "transforms.context"},
            "agent_context": {"grain": "document_registry"},
        }
    )
    with pytest.raises(RunError, match="missing columns"):
        _validate_agent_context_output(pl.DataFrame({"document_id": ["bad"]}), model)

    empty = _validate_agent_context_output(pl.DataFrame(), model)
    assert empty.columns == [
        field["name"] for field in contract_descriptor("document_registry")["fields"]
    ]


def test_contract_is_discoverable_in_manifest_dbt_sources_and_docs(
    tmp_path: Path,
) -> None:
    (tmp_path / "dbt_ml_project.yml").write_text(
        "name: context_project\nprofile: context_project\n"
    )
    (tmp_path / "profiles.yml").write_text(
        "context_project:\n"
        "  target: dev\n"
        "  outputs:\n"
        "    dev:\n"
        "      warehouse:\n"
        "        type: duckdb\n"
        "        path: ./target/context.duckdb\n"
        "        schema: context\n"
    )
    (tmp_path / "sources").mkdir()
    (tmp_path / "sources" / "documents.yml").write_text(
        "version: 2\nsources:\n  - name: documents\n    path: data/documents\n"
    )
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "context.yml").write_text(
        "version: 2\n"
        "models:\n"
        "  - name: document_chunks\n"
        "    depends_on: [ref('documents')]\n"
        "    transform:\n"
        "      type: python\n"
        "      module: transforms.context\n"
        "    agent_context:\n"
        "      contract: agent_context/v1\n"
        "      grain: document_chunks\n"
    )

    manifest = build_manifest(tmp_path)
    assert manifest["manifest_version"] == 2
    model = manifest["models"][0]
    descriptor = model["agent_context"]
    assert descriptor["contract"] == AGENT_CONTEXT_CONTRACT
    assert descriptor["grain"] == "document_chunks"
    assert descriptor["unique_id"] == "model.context_project.document_chunks"
    assert descriptor["relation"]["fully_qualified"].endswith(".context.document_chunks")
    assert model["depends_on"] == ["ref('documents')"]

    dbt_source = build_dbt_sources(tmp_path)["sources"][0]
    table = dbt_source["tables"][0]
    assert table["meta"]["dbt_ml"]["agent_context"] == {
        "contract": AGENT_CONTEXT_CONTRACT,
        "grain": "document_chunks",
        "primary_key": ["context_id"],
        "foreign_keys": {"document_version_id": "document_registry.document_version_id"},
    }
    columns = {column["name"]: column for column in table["columns"]}
    assert columns["context_id"]["data_type"] == "string"
    assert columns["citation_section_path"]["data_type"] == "string"
    assert columns["authorization_resolved"]["meta"]["dbt_ml"]["agent_context"]["nullable"] is False

    docs = generate_docs(tmp_path)
    page = (docs.output_dir / "model_document_chunks.html").read_text()
    assert AGENT_CONTEXT_CONTRACT in page
    assert "context.document_chunks" in page
    assert "citation_page_number" in page
    assert "document_registry.document_version_id" in page


def test_stable_ids_are_deterministic_and_version_sensitive() -> None:
    first = _document()
    same = _document()
    revised = _document(source_version="v2", body="Revised refund policy")

    assert first["document_id"] == same["document_id"] == revised["document_id"]
    assert first["document_version_id"] == same["document_version_id"]
    assert first["document_version_id"] != revised["document_version_id"]
    assert make_document_id("a:b", "c") != make_document_id("a", "b:c")
    assert make_entity_id("economic_data", "account", canonical_entity_key(1)) != make_entity_id(
        "economic_data", "account", canonical_entity_key("1")
    )


def test_contract_relations_validate_many_to_many_links_and_policy_carry() -> None:
    document = _document()
    chunks = [_chunk(document), _chunk(document, index=1)]
    links = [
        _link(chunks[0], entity_key_value="enterprise", relationship_type="applies_to"),
        _link(chunks[0], entity_key_value="regulated", relationship_type="mentions"),
        _link(chunks[1], entity_key_value="enterprise", relationship_type="applies_to"),
    ]

    validate_agent_context_relations(
        pl.DataFrame([document]),
        pl.DataFrame(chunks),
        pl.DataFrame(links),
    )

    changed_policy = dict(chunks[0], access_groups=json.dumps(["executive"]))
    with pytest.raises(AgentContextValidationError, match="policy fields must match"):
        validate_agent_context_relations(
            pl.DataFrame([document]),
            pl.DataFrame([changed_policy, chunks[1]]),
            pl.DataFrame(links),
        )


def test_temporal_selection_uses_half_open_intervals_and_unknown_is_distinct() -> None:
    first = _document(recorded_to=_T0 + timedelta(days=5))
    second = _document(
        source_version="v2",
        body="Revised refund policy",
        recorded_from=_T0 + timedelta(days=5),
    )
    unknown = dict(
        _document(source_version="unknown", body="Undated notice"),
        validity_known=False,
        valid_from=None,
        valid_to=None,
    )

    at_boundary = active_as_of(
        [first, second, unknown],
        valid_at=_T0 + timedelta(days=5),
        recorded_at=_T0 + timedelta(days=5),
    )
    assert [row["document_version_id"] for row in at_boundary] == [second["document_version_id"]]
    assert interval_contains(_T0, _T0, _T0 + timedelta(days=1))
    assert not interval_contains(_T0 + timedelta(days=1), _T0, _T0 + timedelta(days=1))

    with_unknown = active_as_of(
        [unknown],
        valid_at=_T0,
        recorded_at=_T0,
        include_unknown_validity=True,
    )
    assert with_unknown == (unknown,)

    overlapping = _document(source_version="v2", body="Overlapping version")
    with pytest.raises(AgentContextValidationError, match="must not overlap"):
        validate_agent_context_frame(
            pl.DataFrame([first, overlapping]),
            AgentContextGrain.DOCUMENT_REGISTRY,
        )


def test_policy_and_freshness_fail_closed_and_invalidate_acl_changes() -> None:
    document = _document()
    chunk = _chunk(document)
    unresolved = dict(chunk, authorization_resolved=False)
    changed_acl = dict(chunk, access_groups=json.dumps(["executive"]))

    assert is_publishable_context(chunk)
    assert not is_publishable_context(unresolved)
    assert policy_fingerprint(chunk) == policy_fingerprint(dict(chunk, access_groups=["analyst"]))
    assert policy_fingerprint(chunk) != policy_fingerprint(changed_acl)
    assert retrieval_projection_fingerprint(chunk) != (
        retrieval_projection_fingerprint(changed_acl)
    )
    assert freshness_status(chunk, now=_T0 + timedelta(days=1)) is (FreshnessStatus.FRESH)
    assert freshness_status(chunk, now=_T0 + timedelta(days=3)) is (FreshnessStatus.PIPELINE_STALE)

    source_stale = dict(chunk, refresh_due_at=None)
    assert freshness_status(source_stale, now=_T0 + timedelta(days=31)) is (
        FreshnessStatus.SOURCE_STALE
    )
    unchecked = dict(
        chunk,
        freshness_checked_at=None,
        refresh_due_at=_T0 + timedelta(days=2),
    )
    assert freshness_status(unchecked, now=_T0 + timedelta(days=1)) is (FreshnessStatus.UNKNOWN)


def test_invalid_ids_intervals_citations_and_provenance_are_rejected() -> None:
    document = _document()
    invalid_version = dict(document, document_version_id="0" * 32)
    with pytest.raises(AgentContextValidationError, match="v1 algorithm"):
        validate_agent_context_frame(
            pl.DataFrame([invalid_version]), AgentContextGrain.DOCUMENT_REGISTRY
        )

    invalid_interval = dict(document, valid_to=_T0)
    with pytest.raises(AgentContextValidationError, match="valid_to must be later"):
        validate_agent_context_frame(
            pl.DataFrame([invalid_interval]), AgentContextGrain.DOCUMENT_REGISTRY
        )

    chunk = _chunk(document)
    invalid_citation = dict(chunk, citation_char_end=None)
    with pytest.raises(AgentContextValidationError, match="offsets must be paired"):
        validate_agent_context_frame(
            pl.DataFrame([invalid_citation]), AgentContextGrain.DOCUMENT_CHUNKS
        )

    invalid_document_id = dict(chunk, document_id="not-a-v1-id")
    with pytest.raises(AgentContextValidationError, match="document_id"):
        validate_agent_context_frame(
            pl.DataFrame([invalid_document_id]),
            AgentContextGrain.DOCUMENT_CHUNKS,
        )

    missing_provenance = dict(document, provenance_fingerprint=None)
    with pytest.raises(AgentContextValidationError, match="provenance_fingerprint"):
        validate_agent_context_frame(
            pl.DataFrame([missing_provenance]), AgentContextGrain.DOCUMENT_REGISTRY
        )

    link = _link(chunk, entity_key_value="enterprise", relationship_type="mentions")
    noncanonical_key = dict(link, entity_key='["string", "enterprise"]')
    with pytest.raises(AgentContextValidationError, match="canonical JSON"):
        validate_agent_context_frame(
            pl.DataFrame([noncanonical_key]),
            AgentContextGrain.CONTEXT_ENTITY_LINKS,
        )


def test_citation_round_trip_and_duckdb_as_of_fixture(tmp_path: Path) -> None:
    first = _document(recorded_to=_T0 + timedelta(days=5))
    second = _document(
        source_version="v2",
        body="Revised refund policy",
        recorded_from=_T0 + timedelta(days=5),
    )
    chunk = _chunk(second)
    link = _link(chunk, entity_key_value="enterprise", relationship_type="applies_to")

    warehouse = parse_warehouse_config(
        {
            "type": "duckdb",
            "path": str(tmp_path / "warehouse.duckdb"),
            "schema": "context",
        }
    )
    with create_adapter(warehouse) as adapter:
        adapter.materialize_full("document_registry", pl.DataFrame([first, second]))
        adapter.materialize_full("document_chunks", pl.DataFrame([chunk]))
        adapter.materialize_full("context_entity_links", pl.DataFrame([link]))

        versions = adapter.query_df(
            'SELECT document_version_id FROM "context".document_registry '
            "WHERE recorded_from <= ? AND "
            "(recorded_to IS NULL OR ? < recorded_to)",
            [_T0 + timedelta(days=5), _T0 + timedelta(days=5)],
        )
        joined = adapter.query_df(
            "SELECT c.context_id, d.source_uri, l.entity_id "
            'FROM "context".document_chunks c '
            'JOIN "context".document_registry d USING (document_version_id) '
            'JOIN "context".context_entity_links l USING (context_id)'
        )

    assert versions.rows() == [(second["document_version_id"],)]
    assert joined.rows() == [(chunk["context_id"], second["source_uri"], link["entity_id"])]
    assert citation_locator(chunk) == {
        "source_uri": second["source_uri"],
        "chunk_index": 0,
        "page_number": 2,
        "section_path": ["Refunds", "Enterprise"],
        "char_start": 100,
        "char_end": 151,
        "extra": {"paragraph": 3},
    }
