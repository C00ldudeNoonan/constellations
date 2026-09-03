from __future__ import annotations

import base64
import contextlib
import json
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest

from stel.adapters.base import ReadPredicate, ReadPredicateOperator
from stel.agent_context import AgentContextGrain, contract_descriptor
from stel.mcp_server.authorization import (
    ClaimAuthorizationProvider,
    PolicyAttribute,
    Principal,
    StaticPrincipalResolver,
)
from stel.mcp_server.catalog import ArtifactCatalog
from stel.mcp_server.contracts import (
    BusinessFilter,
    FilterOperator,
    GetContextLineageRequest,
    GetDocumentRequest,
    LineageReferenceType,
    ListContextModelsRequest,
    MCPErrorCode,
    SearchContextRequest,
)
from stel.mcp_server.server import create_mcp_server
from stel.mcp_server.service import (
    ContextSearch,
    ContextServerSettings,
    ContextService,
)
from stel.search import (
    SearchFilter,
    SearchProvenance,
    SearchRequest,
    SearchResult,
)

DOC_ALLOWED = "a" * 32
VERSION_ALLOWED = "b" * 32
CONTEXT_ALLOWED_1 = "c" * 32
CHUNK_ALLOWED_1 = "d" * 32
CONTEXT_ALLOWED_2 = "e" * 32
CHUNK_ALLOWED_2 = "f" * 32
DOC_HIDDEN = "1" * 32
VERSION_HIDDEN = "2" * 32
CONTEXT_HIDDEN = "3" * 32
CHUNK_HIDDEN = "4" * 32
MISSING_ID = "0" * 32


class FakeRepository:
    def __init__(
        self,
        rows: Mapping[str, Sequence[Mapping[str, Any]]],
        *,
        capture_query_text: bool = False,
    ) -> None:
        self.rows = rows
        self.calls: list[tuple[str, tuple[ReadPredicate, ...]]] = []
        # Query-log rows the service handed us (issue #329), so tests can
        # assert what a served query records without a warehouse.
        self.logged: list[Mapping[str, Any]] = []
        self._capture_query_text = capture_query_text
        self.warm_ups = 0

    def log_query(self, row: Mapping[str, Any]) -> None:
        self.logged.append(row)

    def query_log_captures_text(self) -> bool:
        return self._capture_query_text

    def warm_up(self) -> None:
        self.warm_ups += 1

    def read_rows(
        self,
        relation: str,
        *,
        predicates: Sequence[ReadPredicate],
        max_rows: int,
        columns: Sequence[str] | None = None,
    ) -> tuple[Mapping[str, Any], ...]:
        del columns
        self.calls.append((relation, tuple(predicates)))
        selected = tuple(
            row
            for row in self.rows.get(relation, ())
            if all(_matches(row, predicate) for predicate in predicates)
        )
        if len(selected) > max_rows:
            raise AssertionError("fixture exceeded service scan limit")
        return selected


class FakeSearch(ContextSearch):
    def __init__(self) -> None:
        self.request: SearchRequest | None = None
        self.policy_filters: tuple[SearchFilter, ...] = ()

    def execute(
        self,
        request: SearchRequest,
        *,
        policy_filters: Sequence[SearchFilter],
    ) -> Sequence[SearchResult]:
        self.request = request
        self.policy_filters = tuple(policy_filters)
        return (
            _hit(CONTEXT_ALLOWED_1, DOC_ALLOWED, CHUNK_ALLOWED_1, rank=1),
            _hit(CONTEXT_HIDDEN, DOC_HIDDEN, CHUNK_HIDDEN, rank=2),
        )


def _matches(row: Mapping[str, Any], predicate: ReadPredicate) -> bool:
    value = row.get(predicate.column)
    if predicate.operator is ReadPredicateOperator.EQUAL:
        return value == predicate.value
    if predicate.operator is ReadPredicateOperator.IN:
        assert isinstance(predicate.value, tuple)
        return value in predicate.value
    if predicate.operator is ReadPredicateOperator.IS_NULL:
        return value is None
    raise AssertionError(f"unsupported fixture predicate {predicate.operator}")


def _artifact_catalog() -> ArtifactCatalog:
    registry_id = "model.context_demo.document_registry"
    chunks_id = "model.context_demo.document_chunks"
    links_id = "model.context_demo.context_entity_links"
    embed_id = "model.context_demo.context_embeddings"
    search_id = "search_index.context_demo.context_search"
    manifest = {
        "manifest_version": 2,
        "target": {"name": "dev", "warehouse": {}},
        "models": [
            _context_model(
                "document_registry",
                registry_id,
                AgentContextGrain.DOCUMENT_REGISTRY,
            ),
            _context_model(
                "document_chunks",
                chunks_id,
                AgentContextGrain.DOCUMENT_CHUNKS,
            ),
            {
                "name": "context_embeddings",
                "unique_id": embed_id,
                "resource_type": "model",
                "output": {
                    "type": "warehouse_relation",
                    "relation": {"name": "context_embeddings"},
                },
            },
            _context_model(
                "context_entity_links",
                links_id,
                AgentContextGrain.CONTEXT_ENTITY_LINKS,
            ),
            {
                "name": "context_search",
                "unique_id": search_id,
                "resource_type": "search_index",
                "description": "Governed economic document context",
                "access": "governed",
                "output": {
                    "type": "serving_resource",
                    "serving_resource": {
                        "store_type": "fake",
                        "id_field": "context_id",
                        "attributes": [
                            {
                                "name": "tenant_id",
                                "data_type": "string",
                                "filter_role": "policy",
                            },
                            {
                                "name": "category",
                                "data_type": "string",
                                "filter_role": "user",
                            },
                            {
                                "name": "classification",
                                "data_type": "string",
                                "filter_role": "policy",
                            },
                        ],
                        "query": {
                            "modes": ["hybrid", "text", "vector"],
                            "consistency": "strong",
                        },
                    },
                },
            },
        ],
        "dag": {
            "execution_order": [
                "source.context_demo.documents",
                registry_id,
                chunks_id,
                embed_id,
                links_id,
                search_id,
            ],
            "edges": [
                ["source.context_demo.documents", registry_id],
                [registry_id, chunks_id],
                [chunks_id, embed_id],
                [chunks_id, links_id],
                [embed_id, search_id],
            ],
        },
    }
    run_results = {
        "metadata": {"generated_at": "2026-07-20T12:00:00+00:00"},
        "results": [{"model_name": "context_search", "status": "success"}],
    }
    return ArtifactCatalog.from_payloads(manifest, run_results=run_results)


def _context_model(
    name: str,
    unique_id: str,
    grain: AgentContextGrain,
) -> dict[str, Any]:
    return {
        "name": name,
        "unique_id": unique_id,
        "resource_type": "model",
        "agent_context": {
            **contract_descriptor(grain),
            "unique_id": unique_id,
            "relation": {"name": name},
        },
        "output": {
            "type": "warehouse_relation",
            "relation": {"name": name},
        },
    }


def _fixture_rows() -> dict[str, tuple[Mapping[str, Any], ...]]:
    return {
        "document_registry": (
            _registry(DOC_ALLOWED, VERSION_ALLOWED, tenant="research", source_version="v2"),
            _registry(DOC_HIDDEN, VERSION_HIDDEN, tenant="restricted", source_version="v9"),
        ),
        "document_chunks": (
            _chunk(
                DOC_ALLOWED,
                VERSION_ALLOWED,
                CONTEXT_ALLOWED_1,
                CHUNK_ALLOWED_1,
                index=0,
                tenant="research",
                text=(
                    "Consumer prices slowed while inflation remained above target. "
                    "The release also revised the prior month."
                ),
            ),
            _chunk(
                DOC_ALLOWED,
                VERSION_ALLOWED,
                CONTEXT_ALLOWED_2,
                CHUNK_ALLOWED_2,
                index=1,
                tenant="research",
                text="Payroll growth was steady in the latest employment report.",
            ),
            _chunk(
                DOC_HIDDEN,
                VERSION_HIDDEN,
                CONTEXT_HIDDEN,
                CHUNK_HIDDEN,
                index=0,
                tenant="restricted",
                text="This hidden tenant content must never leave the service.",
            ),
        ),
        "context_entity_links": (
            {
                "context_entity_link_id": "8" * 32,
                "context_id": CONTEXT_ALLOWED_1,
                "entity_namespace": "economic_data",
                "entity_name": "series",
                "entity_id": "9" * 32,
                "entity_key": '[["string","CPIAUCSL"]]',
                "dbt_unique_id": "semantic_model.economic_data.series",
                "relationship_type": "applies_to",
                "link_method": "deterministic:v1",
                "confidence": None,
                "recorded_from": datetime(2026, 7, 1, tzinfo=UTC),
                "recorded_to": None,
            },
            {
                "context_entity_link_id": "5" * 32,
                "context_id": CONTEXT_HIDDEN,
                "entity_namespace": "economic_data",
                "entity_name": "confidential_series",
                "entity_id": "6" * 32,
                "entity_key": '[["string","HIDDEN"]]',
                "dbt_unique_id": None,
                "relationship_type": "applies_to",
                "link_method": "deterministic:v1",
                "confidence": None,
                "recorded_from": datetime(2026, 7, 1, tzinfo=UTC),
                "recorded_to": None,
            },
        ),
    }


def _policy(tenant: str) -> dict[str, Any]:
    return {
        "tenant_id": tenant,
        "is_public": False,
        "access_groups": [],
        "classification": "internal",
        "policy_ref": "policy:economic-context",
        "policy_version": "1",
        "authorization_resolved": True,
    }


def _temporal() -> dict[str, Any]:
    observed = datetime(2026, 7, 1, tzinfo=UTC)
    return {
        "validity_known": True,
        "valid_from": datetime(2026, 1, 1, tzinfo=UTC),
        "valid_to": None,
        "recorded_from": observed,
        "recorded_to": None,
        "source_updated_at": observed,
        "source_observed_at": observed,
        "ingested_at": observed,
        "materialized_at": observed,
        "freshness_checked_at": observed,
        "refresh_due_at": datetime(2027, 1, 1, tzinfo=UTC),
        "stale_after": datetime(2027, 1, 1, tzinfo=UTC),
    }


def _provenance() -> dict[str, Any]:
    return {
        "upstream_unique_id": "source.context_demo.documents",
        "invocation_id": "invocation-2026-07-20",
        "parser_identity": "pdf:v1",
        "transform_identity": "economic-context:v1",
        "prompt_fingerprint": None,
        "schema_fingerprint": "schema-v1",
        "provider_identity": None,
        "model_identity": None,
        "provenance_fingerprint": "7" * 32,
    }


def _registry(
    document_id: str,
    version_id: str,
    *,
    tenant: str,
    source_version: str,
) -> Mapping[str, Any]:
    return {
        "document_id": document_id,
        "document_version_id": version_id,
        "source_system": "fred",
        "source_key": f"release:{document_id}",
        "source_uri": f"https://example.test/releases/{document_id}",
        "source_version": source_version,
        "source_content_hash": "6" * 32,
        **_temporal(),
        **_policy(tenant),
        **_provenance(),
    }


def _chunk(
    document_id: str,
    version_id: str,
    context_id: str,
    chunk_id: str,
    *,
    index: int,
    tenant: str,
    text: str,
) -> Mapping[str, Any]:
    return {
        "document_id": document_id,
        "document_version_id": version_id,
        "context_id": context_id,
        "chunk_id": chunk_id,
        "chunk_index": index,
        "text": text,
        "source_uri": f"https://example.test/releases/{document_id}",
        "citation_page_number": index + 1,
        "citation_section_path": ["Economic releases"],
        "citation_char_start": index * 100,
        "citation_char_end": index * 100 + len(text),
        "citation_speaker": None,
        "citation_start_seconds": None,
        "citation_end_seconds": None,
        "citation_locator": '{"format":"release"}',
        **_temporal(),
        **_policy(tenant),
        **_provenance(),
    }


def _hit(
    context_id: str,
    document_id: str,
    chunk_id: str,
    *,
    rank: int,
) -> SearchResult:
    return SearchResult(
        record_id=context_id,
        document_id=document_id,
        chunk_id=chunk_id,
        rank=rank,
        score=1 / rank,
        raw_score=None,
        raw_score_kind=None,
        text={"text": "retrieval content is not trusted by the MCP boundary"},
        metadata={},
        display={},
        contributing_ranks={"text": rank},
        provenance=SearchProvenance(
            project="context_demo",
            model="context_search",
            unique_id="search_index.context_demo.context_search",
            target="dev",
            store_type="fake",
            logical_collection="context_search",
            physical_collection="context_search__generation",
            upstream="model.context_demo.context_embeddings",
            embedding=None,
        ),
    )


def _service(
    *,
    principal: Principal | None = None,
    settings: ContextServerSettings | None = None,
    repository: FakeRepository | None = None,
) -> tuple[ContextService, FakeSearch]:
    fake_search = FakeSearch()
    service = ContextService(
        catalog=_artifact_catalog(),
        repository=repository or FakeRepository(_fixture_rows()),
        context_search=fake_search,
        principal_resolver=StaticPrincipalResolver(
            principal
            if principal is not None
            else Principal(
                "local-user",
                tenant_id="research",
                policy_claims={"classification": "internal"},
            )
        ),
        authorization=ClaimAuthorizationProvider(),
        settings=settings,
    )
    return service, fake_search


def test_context_models_are_artifact_backed_and_principal_scoped() -> None:
    service, _ = _service()
    try:
        response = service.list_context_models(ListContextModelsRequest())
    finally:
        service.close()

    assert response.error is None
    assert [model.name for model in response.models] == ["context_search"]
    model = response.models[0]
    assert model.contract == "agent_context/v1"
    assert model.last_successful_materialization == datetime(
        2026, 7, 20, 12, tzinfo=UTC
    )
    assert model.retrieval.modes == ("hybrid", "text", "vector")
    assert [field.field for field in model.retrieval.filter_fields] == ["category"]
    assert model.entity_types == ("series",)


def test_missing_principal_fails_closed_before_discovery() -> None:
    service = ContextService(
        catalog=_artifact_catalog(),
        repository=FakeRepository(_fixture_rows()),
        context_search=FakeSearch(),
        principal_resolver=StaticPrincipalResolver(None),
        authorization=ClaimAuthorizationProvider(),
    )
    try:
        response = service.list_context_models(ListContextModelsRequest())
    finally:
        service.close()

    assert response.models == ()
    assert response.error is not None
    assert response.error.code is MCPErrorCode.MISSING_PRINCIPAL


def test_shared_group_cannot_cross_a_tenant_boundary() -> None:
    provider = ClaimAuthorizationProvider()
    row = {
        **_policy("restricted"),
        "access_groups": ["shared-reviewers"],
    }

    assert provider.can_read(
        Principal(
            "local-user",
            tenant_id="research",
            access_groups=("shared-reviewers",),
        ),
        row,
    ) is False


def test_declared_policy_attributes_are_required_on_each_row() -> None:
    provider = ClaimAuthorizationProvider()
    row = {**_policy("research"), "classification": None}

    assert provider.can_read(
        Principal(
            "local-user",
            tenant_id="research",
            policy_claims={"classification": "internal"},
        ),
        row,
        attributes=(PolicyAttribute("classification", "string"),),
    ) is False


def test_request_rate_limit_returns_a_retryable_busy_error() -> None:
    service, _ = _service(
        settings=ContextServerSettings(max_requests_per_minute=1)
    )
    try:
        first = service.list_context_models(ListContextModelsRequest())
        second = service.list_context_models(ListContextModelsRequest())
    finally:
        service.close()

    assert first.error is None
    assert second.error is not None
    assert second.error.code is MCPErrorCode.BUSY
    assert second.error.retryable is True


def test_search_uses_server_policy_and_rechecks_returned_rows() -> None:
    service, fake_search = _service(
        settings=ContextServerSettings(max_snippet_bytes=64)
    )
    try:
        response = service.search_context(
            SearchContextRequest(
                model="context_search",
                query="inflation and employment",
                mode="text",
            )
        )
    finally:
        service.close()

    assert response.error is None
    assert len(response.results) == 1
    result = response.results[0]
    assert result.context_id == CONTEXT_ALLOWED_1
    assert result.document_version_id == VERSION_ALLOWED
    assert result.source_version == "v2"
    assert result.snippet_truncated is True
    assert result.entities[0].name == "series"
    assert result.citation.page_number == 1
    assert result.lineage.search_index_unique_id == (
        "search_index.context_demo.context_search"
    )
    assert fake_search.policy_filters[0].field == "tenant_id"
    assert fake_search.policy_filters[0].value == "research"
    assert fake_search.request is not None
    assert fake_search.request.filters == ()


def test_policy_scope_is_not_a_tool_argument_or_business_filter() -> None:
    assert "tenant_id" not in SearchContextRequest.model_fields
    assert "access_groups" not in SearchContextRequest.model_fields
    assert "policy_filters" not in SearchContextRequest.model_fields

    service, _ = _service()
    try:
        response = service.search_context(
            SearchContextRequest(
                model="context_search",
                query="inflation",
                mode="text",
                filters=(
                    BusinessFilter(
                        field="tenant_id",
                        operator=FilterOperator.EQUAL,
                        value="restricted",
                    ),
                ),
            )
        )
    finally:
        service.close()

    assert response.results == ()
    assert response.error is not None
    assert response.error.code is MCPErrorCode.CAPABILITY_UNAVAILABLE


def test_document_fetch_rechecks_permissions_and_paginates() -> None:
    service, _ = _service()
    try:
        first = service.get_document(
            GetDocumentRequest(
                model="context_search",
                document_id=DOC_ALLOWED,
                document_version_id=VERSION_ALLOWED,
                limit=1,
            )
        )
        assert first.error is None
        assert [chunk.context_id for chunk in first.chunks] == [CONTEXT_ALLOWED_1]
        assert first.next_cursor is not None
        second = service.get_document(
            GetDocumentRequest(
                model="context_search",
                document_id=DOC_ALLOWED,
                document_version_id=VERSION_ALLOWED,
                limit=1,
                cursor=first.next_cursor,
            )
        )
        hidden = service.get_document(
            GetDocumentRequest(
                model="context_search",
                document_id=DOC_HIDDEN,
                document_version_id=VERSION_HIDDEN,
            )
        )
        absent = service.get_document(
            GetDocumentRequest(
                model="context_search",
                document_id=MISSING_ID,
                document_version_id=MISSING_ID,
            )
        )
    finally:
        service.close()

    assert second.error is None
    assert [chunk.context_id for chunk in second.chunks] == [CONTEXT_ALLOWED_2]
    assert second.next_cursor is None
    assert hidden.error is not None and absent.error is not None
    assert hidden.error == absent.error
    assert hidden.error.code is MCPErrorCode.NOT_FOUND_OR_DENIED


def test_document_cursor_past_the_last_chunk_returns_an_empty_page() -> None:
    cursor_payload = {
        "kind": "document",
        "model": "context_search",
        "document_id": DOC_ALLOWED,
        "document_version_id": VERSION_ALLOWED,
        "after": [999, "f" * 32],
    }
    cursor = base64.urlsafe_b64encode(
        json.dumps(cursor_payload, sort_keys=True, separators=(",", ":")).encode()
    ).decode().rstrip("=")
    service, _ = _service()
    try:
        empty = service.get_document(
            GetDocumentRequest(
                model="context_search",
                document_id=DOC_ALLOWED,
                document_version_id=VERSION_ALLOWED,
                limit=1,
                cursor=cursor,
            )
        )
    finally:
        service.close()

    assert empty.error is None
    assert empty.chunks == ()
    assert empty.next_cursor is None
    assert empty.lineage is not None


def test_lineage_is_authorized_and_uses_safe_artifact_identities() -> None:
    service, _ = _service()
    try:
        response = service.get_context_lineage(
            GetContextLineageRequest(
                model="context_search",
                reference_type=LineageReferenceType.CONTEXT,
                reference_id=CONTEXT_ALLOWED_1,
            )
        )
        hidden = service.get_context_lineage(
            GetContextLineageRequest(
                model="context_search",
                reference_type=LineageReferenceType.CONTEXT,
                reference_id=CONTEXT_HIDDEN,
            )
        )
    finally:
        service.close()

    assert response.error is None and response.record is not None
    assert response.record.source.source_system == "fred"
    assert response.record.lineage.resources == (
        "source.context_demo.documents",
        "model.context_demo.document_registry",
        "model.context_demo.document_chunks",
        "model.context_demo.context_embeddings",
        "search_index.context_demo.context_search",
    )
    assert hidden.record is None
    assert hidden.error is not None
    assert hidden.error.code is MCPErrorCode.NOT_FOUND_OR_DENIED


@pytest.mark.anyio
async def test_mcp_protocol_discovers_exactly_four_read_only_tools() -> None:
    from mcp.shared.memory import create_connected_server_and_client_session

    service, _ = _service()
    app = create_mcp_server(service)
    try:
        async with create_connected_server_and_client_session(
            app,
            raise_exceptions=True,
        ) as session:
            tools = await session.list_tools()
            assert {tool.name for tool in tools.tools} == {
                "list_context_models",
                "search_context",
                "get_document",
                "get_context_lineage",
            }
            search_tool = next(
                tool for tool in tools.tools if tool.name == "search_context"
            )
            input_properties = search_tool.inputSchema["properties"]
            assert "tenant_id" not in input_properties
            assert "access_groups" not in input_properties
            result = await session.call_tool("list_context_models", {})
            assert result.isError is False
            assert result.structuredContent is not None
            assert result.structuredContent["models"][0]["name"] == "context_search"
    finally:
        service.close()


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


# ─── MCP query log (issue #329 phase 1) ─────────────────────────────────────


def _searched(repository: FakeRepository, query: str = "inflation and employment"):
    service, _ = _service(repository=repository)
    try:
        return service.search_context(
            SearchContextRequest(model="context_search", query=query, mode="text")
        )
    finally:
        service.close()


def test_a_served_query_is_logged_with_a_fingerprint_not_its_text() -> None:
    """The privacy default: fingerprint always, text never unless opted in.

    "Which questions repeat, and which return nothing" is answerable without
    storing what anyone typed.
    """
    from stel.append_log import query_fingerprint

    repository = FakeRepository(_fixture_rows())

    response = _searched(repository)

    assert response.error is None
    assert len(repository.logged) == 1
    row = repository.logged[0]
    assert row["query_fingerprint"] == query_fingerprint("inflation and employment")
    assert row["query_text"] is None
    assert "inflation" not in json.dumps(row)


def test_capture_query_text_is_a_separate_opt_in() -> None:
    repository = FakeRepository(_fixture_rows(), capture_query_text=True)

    _searched(repository)

    assert repository.logged[0]["query_text"] == "inflation and employment"


def test_the_log_records_the_principal_and_what_was_served() -> None:
    repository = FakeRepository(_fixture_rows())

    _searched(repository)

    row = repository.logged[0]
    assert row["principal_id"] == "local-user"
    assert row["tenant_id"] == "research"
    assert row["model_name"] == "context_search"
    assert row["result_count"] == 1
    assert row["zero_results"] is False
    # Post-authorization: only chunks this principal was allowed to see.
    assert row["returned_chunk_ids"] == [CHUNK_ALLOWED_1]


class EmptySearch(FakeSearch):
    def execute(
        self,
        request: SearchRequest,
        *,
        policy_filters: Sequence[SearchFilter],
    ) -> Sequence[SearchResult]:
        del request, policy_filters
        return ()


def test_a_zero_result_query_is_logged_as_such() -> None:
    """The cheapest quality signal there is, materialized as a column.

    A question the index cannot answer is exactly what a chunking or metadata
    gap looks like from outside, so it has to be a `WHERE` clause rather than
    something reconstructed from an empty id list.
    """
    repository = FakeRepository(_fixture_rows())
    service = ContextService(
        catalog=_artifact_catalog(),
        repository=repository,
        context_search=EmptySearch(),
        principal_resolver=StaticPrincipalResolver(
            Principal(
                "local-user",
                tenant_id="research",
                policy_claims={"classification": "internal"},
            )
        ),
        authorization=ClaimAuthorizationProvider(),
    )
    try:
        response = service.search_context(
            SearchContextRequest(
                model="context_search", query="nothing answers this", mode="text"
            )
        )
    finally:
        service.close()

    assert response.error is None
    assert response.results == ()
    row = repository.logged[0]
    assert row["zero_results"] is True
    assert row["result_count"] == 0
    assert row["returned_chunk_ids"] == []
    assert row["top_score"] is None


def test_a_denied_request_logs_nothing() -> None:
    # Logging happens after authorization, so a refused request leaves no row
    # — and cannot be used to probe which models exist.
    service, _ = _service(
        principal=Principal("intruder", tenant_id="other"),
        repository=(repository := FakeRepository(_fixture_rows())),
    )
    try:
        response = service.search_context(
            SearchContextRequest(model="context_search", query="x", mode="text")
        )
    finally:
        service.close()

    assert response.error is not None
    assert repository.logged == []


def test_a_slow_log_write_cannot_time_out_a_served_answer() -> None:
    """Logging happens outside the request deadline.

    The limiter times the guarded operation; a synchronous append inside it
    meant a stalled warehouse could turn an otherwise successful search into
    a TIMEOUT, contradicting the best-effort guarantee (Codex review, #333).
    """

    class SlowLogRepository(FakeRepository):
        def log_query(self, row: Mapping[str, Any]) -> None:
            time.sleep(0.3)
            super().log_query(row)

    repository = SlowLogRepository(_fixture_rows())
    # A deadline far shorter than the log write: if the write were inside it,
    # this search would fail.
    service, _ = _service(
        repository=repository,
        settings=ContextServerSettings(timeout_seconds=0.15),
    )
    try:
        response = service.search_context(
            SearchContextRequest(
                model="context_search", query="inflation and employment", mode="text"
            )
        )
    finally:
        service.close()

    assert response.error is None
    assert len(response.results) == 1
    assert len(repository.logged) == 1


def test_warm_up_delegates_to_repository() -> None:
    repository = FakeRepository(_fixture_rows())
    service, _ = _service(repository=repository)
    try:
        service.warm_up()
    finally:
        service.close()

    assert repository.warm_ups == 1


def test_serve_stdio_warms_up_before_the_transport_starts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Credential problems must fail loudly at boot, not as per-call
    "timeout" errors once the stdio transport owns the pipes (issue #365)."""
    from stel.mcp_server import server as server_module

    events: list[str] = []

    class StubService:
        def warm_up(self) -> None:
            events.append("warm_up")

        def close(self) -> None:
            events.append("close")

    class StubApp:
        def run(self, transport: str) -> None:
            events.append(f"run:{transport}")

    monkeypatch.setattr(
        server_module.ContextService,
        "from_project",
        classmethod(lambda cls, project_dir, **kwargs: StubService()),
    )
    monkeypatch.setattr(
        server_module,
        "create_mcp_server",
        lambda service, token_verifier=None, **auth: StubApp(),
    )

    server_module.serve_stdio(Path("unused-project-dir"))

    assert events == ["warm_up", "run:stdio", "close"]


# ─── deployable transports (issue #392) ─────────────────────────────────────


def test_a_network_transport_refuses_a_process_wide_principal(
    tmp_path: Path,
) -> None:
    """The footgun has to be impossible, not discouraged.

    The environment resolver is right for stdio — the operator running the
    process is the principal — and inverted over a network, where every caller
    would be served as whichever identity the process started with, filtered by
    that identity's tenant. Nothing about the responses would look wrong.
    """
    from stel.mcp_server.authorization import (
        EnvironmentPrincipalResolver,
        Principal,
        StaticPrincipalResolver,
    )
    from stel.mcp_server.server import serve_network

    for resolver in (
        EnvironmentPrincipalResolver(),
        StaticPrincipalResolver(Principal(subject_id="a", access_groups=("g",))),
    ):
        with pytest.raises(ValueError, match="one identity for the whole process"):
            serve_network(
                tmp_path,
                transport="streamable-http",
                host="127.0.0.1",
                port=8000,
                principal_resolver=resolver,
            )


def test_an_unknown_network_transport_is_refused(tmp_path: Path) -> None:
    from stel.mcp_server.authorization import TrustedHeaderPrincipalResolver
    from stel.mcp_server.server import serve_network

    with pytest.raises(ValueError, match="Unknown network transport"):
        serve_network(
            tmp_path,
            transport="websocket",
            host="127.0.0.1",
            port=8000,
            principal_resolver=TrustedHeaderPrincipalResolver(),
        )


def test_the_header_resolver_reads_the_request_in_flight() -> None:
    """Identity is per call, not per process — that is the whole point."""
    from stel.mcp_server.authorization import TrustedHeaderPrincipalResolver

    resolver = TrustedHeaderPrincipalResolver()

    class _Request:
        def __init__(self, headers: dict[str, str]) -> None:
            self.headers = headers

    alice = _Request(
        {
            "x-stel-principal-id": "alice",
            "x-stel-tenant-id": "acme",
            "x-stel-access-groups": "analysts, admins",
        }
    )
    with _request_in_flight(alice):
        principal = resolver.resolve()
    assert principal is not None
    assert principal.subject_id == "alice"
    assert principal.tenant_id == "acme"
    # Principal normalizes groups to a sorted, de-duplicated tuple.
    assert principal.access_groups == ("admins", "analysts")

    bob = _Request({"x-stel-principal-id": "bob", "x-stel-tenant-id": "globex"})
    with _request_in_flight(bob):
        other = resolver.resolve()
    assert other is not None
    assert other.tenant_id == "globex"


def test_the_header_resolver_yields_nothing_without_a_request() -> None:
    """Off a request — stdio, or a bad wiring — there is no identity to infer,
    and inventing one is how a server ends up unauthenticated by default."""
    from stel.mcp_server.authorization import TrustedHeaderPrincipalResolver

    assert TrustedHeaderPrincipalResolver().resolve() is None


def test_an_unidentified_request_yields_no_principal() -> None:
    """A request that reaches the server without the proxy's header is not
    anonymous-but-allowed; it has no principal at all."""
    from stel.mcp_server.authorization import TrustedHeaderPrincipalResolver

    class _Request:
        def __init__(self) -> None:
            self.headers = {"x-stel-tenant-id": "acme"}

    with _request_in_flight(_Request()):
        assert TrustedHeaderPrincipalResolver().resolve() is None


@contextlib.contextmanager
def _request_in_flight(request: Any) -> Any:
    """Stand in for the SDK's per-request contextvar."""
    from types import SimpleNamespace
    from typing import cast

    from mcp.server.lowlevel.server import request_ctx

    token = request_ctx.set(cast(Any, SimpleNamespace(request=request)))
    try:
        yield
    finally:
        request_ctx.reset(token)


def test_the_cli_refuses_a_network_transport_without_a_per_request_identity(
    tmp_path: Path,
) -> None:
    """Refusing by default is the guardrail: forgetting the flag must not
    quietly fall back to the stdio identity model."""
    from click.testing import CliRunner

    from stel.cli import cli

    result = CliRunner().invoke(
        cli,
        ["--project-dir", str(tmp_path), "mcp", "serve", "--transport", "streamable-http"],
    )

    assert result.exit_code != 0
    assert "needs its own identity" in result.output


def test_the_cli_refuses_proxy_headers_on_stdio(tmp_path: Path) -> None:
    """Accepting the flag on stdio would imply the headers were doing something
    there; they are not, and a silently ignored security flag is worse than a
    rejected one."""
    from click.testing import CliRunner

    from stel.cli import cli

    result = CliRunner().invoke(
        cli,
        [
            "--project-dir",
            str(tmp_path),
            "mcp",
            "serve",
            "--trust-proxy-principal-headers",
        ],
    )

    assert result.exit_code != 0
    assert "applies to a network transport" in result.output


def test_a_request_scoped_resolver_sees_the_calling_thread_context() -> None:
    """The network transport resolves identity from a contextvar the SDK sets
    per request (issue #392). Every operation is submitted to a
    ThreadPoolExecutor, and contextvars do not cross that boundary on their
    own — so without propagation the resolver finds nothing and every network
    call is refused as MISSING_PRINCIPAL, no matter who the caller is
    (Codex review).
    """
    import contextvars

    current: contextvars.ContextVar[Principal | None] = contextvars.ContextVar(
        "test_request_principal", default=None
    )

    class RequestScopedResolver:
        def resolve(self) -> Principal | None:
            return current.get()

    service = ContextService(
        catalog=_artifact_catalog(),
        repository=FakeRepository(_fixture_rows()),
        context_search=FakeSearch(),
        principal_resolver=cast(Any, RequestScopedResolver()),
        authorization=ClaimAuthorizationProvider(),
    )
    token = current.set(
        Principal(
            "network-caller",
            tenant_id="research",
            policy_claims={"classification": "internal"},
        )
    )
    try:
        response = service.list_context_models(ListContextModelsRequest())
    finally:
        current.reset(token)
        service.close()

    assert response.error is None, response.error
    assert [model.name for model in response.models] == ["context_search"]
