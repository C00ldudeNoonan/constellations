from __future__ import annotations

import base64
import json
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import BoundedSemaphore, Lock
from time import monotonic
from typing import Any, Protocol, TypeVar, cast

from pydantic import BaseModel, ConfigDict, Field

from ..adapters.base import ReadPredicate, ReadPredicateOperator
from ..agent_context import citation_locator, freshness_status
from ..search import (
    SearchError,
    SearchFilter,
    SearchFilterOperator,
    SearchMode,
    SearchRequest,
    SearchResult,
    search,
)
from .authorization import (
    AuthorizationError,
    AuthorizationProvider,
    ClaimAuthorizationProvider,
    EnvironmentPrincipalResolver,
    Principal,
    PrincipalResolver,
)
from .catalog import ArtifactCatalog, ContextResource
from .contracts import (
    BusinessFilter,
    CitationDescriptor,
    CompactLineage,
    ContextEntity,
    ContextInterval,
    ContextLineageRecord,
    DocumentChunk,
    DocumentSource,
    FreshnessDescriptor,
    GetContextLineageRequest,
    GetContextLineageResponse,
    GetDocumentRequest,
    GetDocumentResponse,
    LineageReferenceType,
    ListContextModelsRequest,
    ListContextModelsResponse,
    MCPErrorCode,
    SearchContextRequest,
    SearchContextResponse,
    SearchContextResult,
    ToolError,
)
from .repository import (
    ContextRepository,
    ContextRepositoryError,
    ContextRepositoryLimitError,
    WarehouseContextRepository,
)


class ContextServerSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    max_results: int = Field(default=20, ge=1, le=100)
    max_document_chunks: int = Field(default=50, ge=1, le=100)
    max_entities_per_context: int = Field(default=50, ge=1, le=1000)
    max_snippet_bytes: int = Field(default=2000, ge=64, le=100_000)
    max_chunk_bytes: int = Field(default=16_000, ge=64, le=1_000_000)
    max_response_bytes: int = Field(default=256_000, ge=1024, le=10_000_000)
    max_scan_rows: int = Field(default=10_000, ge=1, le=1_000_000)
    timeout_seconds: float = Field(default=30.0, gt=0, le=600)
    max_concurrency: int = Field(default=4, ge=1, le=64)
    max_requests_per_minute: int = Field(default=120, ge=1, le=100_000)


class ContextSearch(Protocol):
    def execute(
        self,
        request: SearchRequest,
        *,
        policy_filters: Sequence[SearchFilter],
    ) -> Sequence[SearchResult]: ...


class PortableContextSearch:
    def __init__(
        self,
        project_dir: Path,
        *,
        target: str | None,
        profiles_dir: Path | None,
    ) -> None:
        self._project_dir = project_dir
        self._target = target
        self._profiles_dir = profiles_dir

    def execute(
        self,
        request: SearchRequest,
        *,
        policy_filters: Sequence[SearchFilter],
    ) -> Sequence[SearchResult]:
        return search(
            self._project_dir,
            request,
            target=self._target,
            profiles_dir=self._profiles_dir,
            policy_filters=policy_filters,
        )


class ContextServiceError(Exception):
    def __init__(
        self,
        code: MCPErrorCode,
        message: str,
        *,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable


T = TypeVar("T")
ResponseT = TypeVar("ResponseT", bound=BaseModel)


class _OperationLimiter:
    def __init__(
        self,
        *,
        max_concurrency: int,
        max_requests_per_minute: int,
        timeout_seconds: float,
    ) -> None:
        self._semaphore = BoundedSemaphore(max_concurrency)
        self._executor = ThreadPoolExecutor(
            max_workers=max_concurrency,
            thread_name_prefix="dbt-ml-mcp",
        )
        self._max_requests_per_minute = max_requests_per_minute
        self._request_times: deque[float] = deque()
        self._request_lock = Lock()
        self._timeout_seconds = timeout_seconds

    def run(self, operation: Callable[[], T]) -> T:
        self._check_rate_limit()
        if not self._semaphore.acquire(blocking=False):
            raise ContextServiceError(
                MCPErrorCode.BUSY,
                "The context server is at its concurrency limit",
                retryable=True,
            )

        def guarded() -> T:
            try:
                return operation()
            finally:
                self._semaphore.release()

        try:
            future = self._executor.submit(guarded)
        except BaseException:
            self._semaphore.release()
            raise
        try:
            return future.result(timeout=self._timeout_seconds)
        except TimeoutError:
            raise ContextServiceError(
                MCPErrorCode.TIMEOUT,
                "The context operation exceeded its time limit",
                retryable=True,
            ) from None

    def _check_rate_limit(self) -> None:
        now = monotonic()
        cutoff = now - 60
        with self._request_lock:
            while self._request_times and self._request_times[0] <= cutoff:
                self._request_times.popleft()
            if len(self._request_times) >= self._max_requests_per_minute:
                raise ContextServiceError(
                    MCPErrorCode.BUSY,
                    "The context server is at its request rate limit",
                    retryable=True,
                )
            self._request_times.append(now)

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


@dataclass(frozen=True, slots=True)
class _AuthorizedResource:
    principal: Principal
    resource: ContextResource
    policy_filters: tuple[SearchFilter, ...]


class ContextService:
    def __init__(
        self,
        *,
        catalog: ArtifactCatalog,
        repository: ContextRepository,
        context_search: ContextSearch,
        principal_resolver: PrincipalResolver,
        authorization: AuthorizationProvider,
        settings: ContextServerSettings | None = None,
    ) -> None:
        self._catalog = catalog
        self._repository = repository
        self._search = context_search
        self._principal_resolver = principal_resolver
        self._authorization = authorization
        self._settings = settings or ContextServerSettings()
        self._limiter = _OperationLimiter(
            max_concurrency=self._settings.max_concurrency,
            max_requests_per_minute=self._settings.max_requests_per_minute,
            timeout_seconds=self._settings.timeout_seconds,
        )

    @classmethod
    def from_project(
        cls,
        project_dir: Path,
        *,
        target: str | None = None,
        profiles_dir: Path | None = None,
        principal_resolver: PrincipalResolver | None = None,
        authorization: AuthorizationProvider | None = None,
        settings: ContextServerSettings | None = None,
    ) -> ContextService:
        return cls(
            catalog=ArtifactCatalog.load(project_dir, expected_target=target),
            repository=WarehouseContextRepository(
                project_dir,
                target=target,
                profiles_dir=profiles_dir,
            ),
            context_search=PortableContextSearch(
                project_dir,
                target=target,
                profiles_dir=profiles_dir,
            ),
            principal_resolver=principal_resolver or EnvironmentPrincipalResolver(),
            authorization=authorization or ClaimAuthorizationProvider(),
            settings=settings,
        )

    def close(self) -> None:
        self._limiter.close()

    def list_context_models(
        self,
        request: ListContextModelsRequest,
    ) -> ListContextModelsResponse:
        return self._respond(
            lambda: self._list_context_models(request),
            lambda error: ListContextModelsResponse(error=error),
        )

    def search_context(self, request: SearchContextRequest) -> SearchContextResponse:
        return self._respond(
            lambda: self._search_context(request),
            lambda error: SearchContextResponse(error=error),
        )

    def get_document(self, request: GetDocumentRequest) -> GetDocumentResponse:
        return self._respond(
            lambda: self._get_document(request),
            lambda error: GetDocumentResponse(error=error),
        )

    def get_context_lineage(
        self,
        request: GetContextLineageRequest,
    ) -> GetContextLineageResponse:
        return self._respond(
            lambda: self._get_context_lineage(request),
            lambda error: GetContextLineageResponse(error=error),
        )

    def _respond(
        self,
        operation: Callable[[], ResponseT],
        error_response: Callable[[ToolError], ResponseT],
    ) -> ResponseT:
        try:
            response = self._limiter.run(operation)
            if len(response.model_dump_json().encode()) > self._settings.max_response_bytes:
                raise ContextServiceError(
                    MCPErrorCode.RESPONSE_LIMIT,
                    "The context response exceeded its configured size limit",
                )
            return response
        except ContextServiceError as error:
            return error_response(
                ToolError(
                    code=error.code,
                    message=error.message,
                    retryable=error.retryable,
                )
            )
        except ContextRepositoryLimitError:
            return error_response(
                ToolError(
                    code=MCPErrorCode.RESPONSE_LIMIT,
                    message="The governed context read exceeded its configured scan limit",
                )
            )
        except ContextRepositoryError:
            return error_response(
                ToolError(
                    code=MCPErrorCode.CAPABILITY_UNAVAILABLE,
                    message="The governed context relation could not be read",
                    retryable=True,
                )
            )
        except SearchError as error:
            return error_response(
                ToolError(
                    code=MCPErrorCode.CAPABILITY_UNAVAILABLE,
                    message=str(error),
                    retryable=True,
                )
            )
        except Exception:
            return error_response(
                ToolError(
                    code=MCPErrorCode.INTERNAL,
                    message="The context operation failed",
                    retryable=True,
                )
            )

    def _principal(self) -> Principal:
        try:
            principal = self._principal_resolver.resolve()
        except AuthorizationError:
            principal = None
        if principal is None:
            raise ContextServiceError(
                MCPErrorCode.MISSING_PRINCIPAL,
                "No authenticated caller principal is available",
            )
        return principal

    def _authorize_resource(self, name: str) -> _AuthorizedResource:
        principal = self._principal()
        resource = self._catalog.get(name)
        if resource is None:
            raise _not_found_or_denied()
        try:
            policy_filters = self._authorization.search_policy_filters(
                principal,
                access=resource.access,
                attributes=resource.policy_attributes,
            )
        except AuthorizationError:
            raise _not_found_or_denied() from None
        return _AuthorizedResource(principal, resource, policy_filters)

    def _list_context_models(
        self,
        request: ListContextModelsRequest,
    ) -> ListContextModelsResponse:
        principal = self._principal()
        available: list[ContextResource] = []
        for resource in self._catalog.all():
            try:
                self._authorization.search_policy_filters(
                    principal,
                    access=resource.access,
                    attributes=resource.policy_attributes,
                )
            except AuthorizationError:
                continue
            available.append(resource)
        after = _decode_list_cursor(request.cursor)
        if after is not None:
            available = [resource for resource in available if resource.name > after]
        page_limit = min(request.limit, self._settings.max_results)
        page = available[:page_limit]
        next_cursor = (
            _encode_cursor({"kind": "models", "after": page[-1].name})
            if len(available) > len(page) and page
            else None
        )
        return ListContextModelsResponse(
            models=tuple(
                resource.summary(
                    entity_types=self._entity_types(resource, principal)
                )
                for resource in page
            ),
            next_cursor=next_cursor,
        )

    def _search_context(self, request: SearchContextRequest) -> SearchContextResponse:
        authorized = self._authorize_resource(request.model)
        resource = authorized.resource
        if request.limit > self._settings.max_results:
            raise ContextServiceError(
                MCPErrorCode.INVALID_REQUEST,
                f"limit must not exceed {self._settings.max_results}",
            )
        if request.mode not in resource.modes:
            raise ContextServiceError(
                MCPErrorCode.CAPABILITY_UNAVAILABLE,
                "The context model does not support the requested retrieval mode",
            )
        filters = tuple(
            self._business_filter(resource, item) for item in request.filters
        )
        hits = self._search.execute(
            SearchRequest(
                model=resource.name,
                query=request.query,
                mode=SearchMode(request.mode),
                limit=request.limit,
                filters=filters,
            ),
            policy_filters=authorized.policy_filters,
        )
        chunk_rows = self._chunks_for_hits(resource, hits)
        registry_rows = self._registry_by_versions(
            resource,
            {
                str(row["document_version_id"])
                for row in chunk_rows.values()
                if isinstance(row.get("document_version_id"), str)
            },
        )
        readable: list[tuple[SearchResult, Mapping[str, Any], Mapping[str, Any]]] = []
        for hit in hits:
            row = chunk_rows.get(self._hit_key(resource, hit))
            if row is None:
                continue
            registry = registry_rows.get(str(row.get("document_version_id", "")))
            if registry is None or not self._can_read_pair(
                resource,
                authorized.principal,
                registry,
                row,
            ):
                continue
            readable.append((hit, row, registry))
        links = self._entity_links(
            resource,
            {str(row["context_id"]) for _, row, _ in readable},
        )
        results = tuple(
            self._search_result(resource, hit, row, registry, links)
            for hit, row, registry in readable[: request.limit]
        )
        return SearchContextResponse(results=results)

    def _get_document(self, request: GetDocumentRequest) -> GetDocumentResponse:
        authorized = self._authorize_resource(request.model)
        if request.limit > self._settings.max_document_chunks:
            raise ContextServiceError(
                MCPErrorCode.INVALID_REQUEST,
                f"limit must not exceed {self._settings.max_document_chunks}",
            )
        resource = authorized.resource
        registry_rows = self._repository.read_rows(
            resource.registry_relation,
            predicates=(
                _eq("document_id", request.document_id),
                _eq("document_version_id", request.document_version_id),
            ),
            max_rows=2,
        )
        if len(registry_rows) != 1 or not self._authorization.can_read(
            authorized.principal,
            registry_rows[0],
            attributes=resource.policy_attributes,
        ):
            raise _not_found_or_denied()
        registry = registry_rows[0]
        rows = self._repository.read_rows(
            resource.context_relation,
            predicates=(
                _eq("document_id", request.document_id),
                _eq("document_version_id", request.document_version_id),
            ),
            max_rows=self._settings.max_scan_rows,
        )
        if not rows or any(
            not self._authorization.can_read(
                authorized.principal,
                row,
                attributes=resource.policy_attributes,
            )
            for row in rows
        ):
            raise _not_found_or_denied()
        all_ordered = sorted(rows, key=_chunk_sort_key)
        ordered = all_ordered
        after = _decode_document_cursor(request)
        if after is not None:
            ordered = [row for row in ordered if _chunk_sort_key(row) > after]
        page = ordered[: request.limit]
        links = self._entity_links(
            resource,
            {str(row["context_id"]) for row in page},
        )
        next_cursor = (
            _encode_cursor(
                {
                    "kind": "document",
                    "model": resource.name,
                    "document_id": request.document_id,
                    "document_version_id": request.document_version_id,
                    "after": list(_chunk_sort_key(page[-1])),
                }
            )
            if len(ordered) > len(page) and page
            else None
        )
        return GetDocumentResponse(
            document_id=request.document_id,
            document_version_id=request.document_version_id,
            source=_source(registry),
            interval=_interval(registry),
            freshness=_freshness(registry),
            lineage=_lineage(resource, all_ordered[0]),
            chunks=tuple(self._document_chunk(row, links) for row in page),
            next_cursor=next_cursor,
        )

    def _get_context_lineage(
        self,
        request: GetContextLineageRequest,
    ) -> GetContextLineageResponse:
        authorized = self._authorize_resource(request.model)
        resource = authorized.resource
        field = {
            LineageReferenceType.DOCUMENT: "document_id",
            LineageReferenceType.DOCUMENT_VERSION: "document_version_id",
            LineageReferenceType.CONTEXT: "context_id",
            LineageReferenceType.CHUNK: "chunk_id",
            LineageReferenceType.SEARCH_RESULT: resource.id_field,
        }[request.reference_type]
        if field not in {"document_id", "document_version_id", "context_id", "chunk_id"}:
            raise ContextServiceError(
                MCPErrorCode.CAPABILITY_UNAVAILABLE,
                "The search result ID cannot be resolved to an agent-context record",
            )
        rows = self._repository.read_rows(
            resource.context_relation,
            predicates=(_eq(field, request.reference_id),),
            max_rows=self._settings.max_scan_rows,
        )
        rows = tuple(sorted(rows, key=_lineage_sort_key, reverse=True))
        for row in rows:
            version_id = row.get("document_version_id")
            if not isinstance(version_id, str):
                continue
            registry = self._registry_by_versions(resource, {version_id}).get(version_id)
            if registry is None or not self._can_read_pair(
                resource,
                authorized.principal,
                registry,
                row,
            ):
                continue
            context_id = _required_string(row, "context_id")
            links = self._entity_links(resource, {context_id})
            return GetContextLineageResponse(
                record=ContextLineageRecord(
                    document_id=_required_string(row, "document_id"),
                    document_version_id=version_id,
                    context_id=context_id,
                    chunk_id=_required_string(row, "chunk_id"),
                    source=_source(registry),
                    citation=_citation(row),
                    entities=links.get(context_id, ()),
                    lineage=_lineage(resource, row),
                )
            )
        raise _not_found_or_denied()

    def _business_filter(
        self,
        resource: ContextResource,
        value: BusinessFilter,
    ) -> SearchFilter:
        capability = next(
            (item for item in resource.business_filters if item.field == value.field),
            None,
        )
        if capability is None or value.operator not in capability.operators:
            raise ContextServiceError(
                MCPErrorCode.CAPABILITY_UNAVAILABLE,
                "The context model does not support the requested business filter",
            )
        return SearchFilter(
            value.field,
            SearchFilterOperator(value.operator.value),
            value.value,
        )

    def _chunks_for_hits(
        self,
        resource: ContextResource,
        hits: Sequence[SearchResult],
    ) -> dict[str, Mapping[str, Any]]:
        if resource.id_field not in {"context_id", "chunk_id"}:
            raise ContextServiceError(
                MCPErrorCode.CAPABILITY_UNAVAILABLE,
                "The search index must use context_id or chunk_id for MCP retrieval",
            )
        values = tuple(sorted({self._hit_key(resource, hit) for hit in hits}))
        if not values:
            return {}
        rows = self._repository.read_rows(
            resource.context_relation,
            predicates=(_in(resource.id_field, values),),
            max_rows=self._settings.max_scan_rows,
        )
        return {
            str(row[resource.id_field]): row
            for row in rows
            if isinstance(row.get(resource.id_field), str)
        }

    @staticmethod
    def _hit_key(resource: ContextResource, hit: SearchResult) -> str:
        if resource.id_field == "context_id":
            return hit.record_id
        return hit.chunk_id or hit.record_id

    def _registry_by_versions(
        self,
        resource: ContextResource,
        versions: set[str],
    ) -> dict[str, Mapping[str, Any]]:
        if not versions:
            return {}
        rows = self._repository.read_rows(
            resource.registry_relation,
            predicates=(_in("document_version_id", tuple(sorted(versions))),),
            max_rows=self._settings.max_scan_rows,
        )
        return {
            str(row["document_version_id"]): row
            for row in rows
            if isinstance(row.get("document_version_id"), str)
        }

    def _entity_links(
        self,
        resource: ContextResource,
        context_ids: set[str],
    ) -> dict[str, tuple[ContextEntity, ...]]:
        grouped: dict[str, list[ContextEntity]] = {}
        if not context_ids:
            return {}
        for relation in resource.entity_relations:
            rows = self._repository.read_rows(
                relation,
                predicates=(
                    _in("context_id", tuple(sorted(context_ids))),
                    _is_null("recorded_to"),
                ),
                max_rows=self._settings.max_scan_rows,
            )
            for row in rows:
                context_id = row.get("context_id")
                if not isinstance(context_id, str) or context_id not in context_ids:
                    continue
                entities = grouped.setdefault(context_id, [])
                if len(entities) >= self._settings.max_entities_per_context:
                    continue
                entities.append(_entity(row))
        return {
            context_id: tuple(sorted(entities, key=lambda item: item.entity_id))
            for context_id, entities in grouped.items()
        }

    def _entity_types(
        self,
        resource: ContextResource,
        principal: Principal,
    ) -> tuple[str, ...]:
        links: list[Mapping[str, Any]] = []
        for relation in resource.entity_relations:
            links.extend(
                self._repository.read_rows(
                    relation,
                    predicates=(_is_null("recorded_to"),),
                    max_rows=self._settings.max_scan_rows,
                    columns=("context_id", "entity_name"),
                )
            )
            if len(links) > self._settings.max_scan_rows:
                raise ContextRepositoryLimitError(
                    "Entity-type discovery exceeded its scan limit"
                )
        context_ids = {
            str(row["context_id"])
            for row in links
            if isinstance(row.get("context_id"), str)
        }
        if not context_ids:
            return ()
        chunks = self._repository.read_rows(
            resource.context_relation,
            predicates=(_in("context_id", tuple(sorted(context_ids))),),
            max_rows=self._settings.max_scan_rows,
        )
        readable = {
            str(row["context_id"])
            for row in chunks
            if isinstance(row.get("context_id"), str)
            and self._authorization.can_read(
                principal,
                row,
                attributes=resource.policy_attributes,
            )
        }
        return tuple(
            sorted(
                {
                    str(row["entity_name"])
                    for row in links
                    if row.get("context_id") in readable
                    and isinstance(row.get("entity_name"), str)
                }
            )
        )

    def _search_result(
        self,
        resource: ContextResource,
        hit: SearchResult,
        row: Mapping[str, Any],
        registry: Mapping[str, Any],
        links: Mapping[str, tuple[ContextEntity, ...]],
    ) -> SearchContextResult:
        snippet, truncated = _truncate_text(
            _required_string(row, "text"),
            self._settings.max_snippet_bytes,
        )
        context_id = _required_string(row, "context_id")
        return SearchContextResult(
            rank=hit.rank,
            score=hit.score,
            document_id=_required_string(row, "document_id"),
            document_version_id=_required_string(row, "document_version_id"),
            context_id=context_id,
            chunk_id=_required_string(row, "chunk_id"),
            snippet=snippet,
            snippet_truncated=truncated,
            source_version=_required_string(registry, "source_version"),
            entities=links.get(context_id, ()),
            interval=_interval(row),
            freshness=_freshness(row),
            citation=_citation(row),
            lineage=_lineage(resource, row),
        )

    def _document_chunk(
        self,
        row: Mapping[str, Any],
        links: Mapping[str, tuple[ContextEntity, ...]],
    ) -> DocumentChunk:
        text, truncated = _truncate_text(
            _required_string(row, "text"),
            self._settings.max_chunk_bytes,
        )
        context_id = _required_string(row, "context_id")
        return DocumentChunk(
            context_id=context_id,
            chunk_id=_required_string(row, "chunk_id"),
            chunk_index=int(row["chunk_index"]),
            text=text,
            text_truncated=truncated,
            citation=_citation(row),
            entities=links.get(context_id, ()),
        )

    def _can_read_pair(
        self,
        resource: ContextResource,
        principal: Principal,
        registry: Mapping[str, Any],
        chunk: Mapping[str, Any],
    ) -> bool:
        return self._authorization.can_read(
            principal,
            registry,
            attributes=resource.policy_attributes,
        ) and self._authorization.can_read(
            principal,
            chunk,
            attributes=resource.policy_attributes,
        )


def _not_found_or_denied() -> ContextServiceError:
    return ContextServiceError(
        MCPErrorCode.NOT_FOUND_OR_DENIED,
        "The requested governed context is unavailable",
    )


def _eq(column: str, value: str) -> ReadPredicate:
    return ReadPredicate(column, ReadPredicateOperator.EQUAL, value)


def _in(column: str, values: tuple[str, ...]) -> ReadPredicate:
    return ReadPredicate(column, ReadPredicateOperator.IN, values)


def _is_null(column: str) -> ReadPredicate:
    return ReadPredicate(column, ReadPredicateOperator.IS_NULL)


def _required_string(row: Mapping[str, Any], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        raise ContextServiceError(
            MCPErrorCode.INTERNAL,
            "A governed context relation contains an invalid contract row",
        )
    return value


def _source(row: Mapping[str, Any]) -> DocumentSource:
    source_uri = row.get("source_uri")
    return DocumentSource(
        source_system=_required_string(row, "source_system"),
        source_key=_required_string(row, "source_key"),
        source_uri=source_uri if isinstance(source_uri, str) else None,
        source_version=_required_string(row, "source_version"),
        source_content_hash=_required_string(row, "source_content_hash"),
    )


def _interval(row: Mapping[str, Any]) -> ContextInterval:
    return ContextInterval(
        validity_known=row.get("validity_known") is True,
        valid_from=row.get("valid_from"),
        valid_to=row.get("valid_to"),
        recorded_from=row.get("recorded_from"),
        recorded_to=row.get("recorded_to"),
    )


def _freshness(row: Mapping[str, Any]) -> FreshnessDescriptor:
    return FreshnessDescriptor(
        status=freshness_status(row).value,
        source_updated_at=row.get("source_updated_at"),
        source_observed_at=row.get("source_observed_at"),
        freshness_checked_at=row.get("freshness_checked_at"),
        refresh_due_at=row.get("refresh_due_at"),
        stale_after=row.get("stale_after"),
    )


def _citation(row: Mapping[str, Any]) -> CitationDescriptor:
    locator = citation_locator(row)
    return CitationDescriptor.model_validate(locator)


def _lineage(resource: ContextResource, row: Mapping[str, Any]) -> CompactLineage:
    return CompactLineage(
        resources=resource.lineage_resources,
        upstream_unique_id=_required_string(row, "upstream_unique_id"),
        invocation_id=_required_string(row, "invocation_id"),
        parser_identity=_optional_string(row.get("parser_identity")),
        transform_identity=_optional_string(row.get("transform_identity")),
        provider_identity=_optional_string(row.get("provider_identity")),
        model_identity=_optional_string(row.get("model_identity")),
        prompt_fingerprint=_optional_string(row.get("prompt_fingerprint")),
        schema_fingerprint=_optional_string(row.get("schema_fingerprint")),
        provenance_fingerprint=_required_string(row, "provenance_fingerprint"),
        search_index_unique_id=resource.unique_id,
        store_type=resource.store_type,
    )


def _entity(row: Mapping[str, Any]) -> ContextEntity:
    key = row.get("entity_key")
    if isinstance(key, str):
        try:
            key = json.loads(key)
        except json.JSONDecodeError:
            pass
    return ContextEntity(
        namespace=_required_string(row, "entity_namespace"),
        name=_required_string(row, "entity_name"),
        entity_id=_required_string(row, "entity_id"),
        entity_key=cast(Any, key),
        dbt_unique_id=_optional_string(row.get("dbt_unique_id")),
        relationship_type=_required_string(row, "relationship_type"),
        confidence=(
            float(row["confidence"]) if row.get("confidence") is not None else None
        ),
    )


def _optional_string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _truncate_text(value: str, limit: int) -> tuple[str, bool]:
    encoded = value.encode()
    if len(encoded) <= limit:
        return value, False
    return encoded[:limit].decode("utf-8", errors="ignore"), True


def _chunk_sort_key(row: Mapping[str, Any]) -> tuple[int, str]:
    return int(row.get("chunk_index", -1)), str(row.get("context_id", ""))


def _lineage_sort_key(row: Mapping[str, Any]) -> tuple[str, str, int, str]:
    return (
        str(row.get("recorded_from", "")),
        str(row.get("document_version_id", "")),
        int(row.get("chunk_index", -1)),
        str(row.get("context_id", "")),
    )


def _encode_cursor(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _decode_cursor(value: str) -> Mapping[str, Any]:
    try:
        padded = value + "=" * (-len(value) % 4)
        decoded = json.loads(base64.urlsafe_b64decode(padded).decode())
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
        raise ContextServiceError(
            MCPErrorCode.INVALID_REQUEST,
            "The pagination cursor is invalid",
        ) from None
    if not isinstance(decoded, dict):
        raise ContextServiceError(
            MCPErrorCode.INVALID_REQUEST,
            "The pagination cursor is invalid",
        )
    return decoded


def _decode_list_cursor(value: str | None) -> str | None:
    if value is None:
        return None
    payload = _decode_cursor(value)
    after = payload.get("after")
    if payload.get("kind") != "models" or not isinstance(after, str):
        raise ContextServiceError(
            MCPErrorCode.INVALID_REQUEST,
            "The pagination cursor is invalid",
        )
    return after


def _decode_document_cursor(
    request: GetDocumentRequest,
) -> tuple[int, str] | None:
    if request.cursor is None:
        return None
    payload = _decode_cursor(request.cursor)
    after = payload.get("after")
    if (
        payload.get("kind") != "document"
        or payload.get("model") != request.model
        or payload.get("document_id") != request.document_id
        or payload.get("document_version_id") != request.document_version_id
        or not isinstance(after, list)
        or len(after) != 2
        or not isinstance(after[0], int)
        or not isinstance(after[1], str)
    ):
        raise ContextServiceError(
            MCPErrorCode.INVALID_REQUEST,
            "The pagination cursor is invalid",
        )
    return after[0], after[1]
