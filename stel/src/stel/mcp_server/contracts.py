from __future__ import annotations

import re
from datetime import date, datetime
from enum import StrEnum
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MCP_CONTEXT_SCHEMA_VERSION: Final = "mcp_context/v1"
_STABLE_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
_STRICT_CONFIG = ConfigDict(extra="forbid", frozen=True)

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]
type FilterScalar = str | int | float | bool | date | datetime


class MCPErrorCode(StrEnum):
    MISSING_PRINCIPAL = "missing_principal"
    NOT_FOUND_OR_DENIED = "not_found_or_denied"
    ARTIFACT_UNAVAILABLE = "artifact_unavailable"
    CAPABILITY_UNAVAILABLE = "capability_unavailable"
    INVALID_REQUEST = "invalid_request"
    RESPONSE_LIMIT = "response_limit"
    TIMEOUT = "timeout"
    BUSY = "busy"
    INTERNAL = "internal"


class ToolError(BaseModel):
    model_config = _STRICT_CONFIG

    code: MCPErrorCode
    message: str
    retryable: bool = False


class FilterOperator(StrEnum):
    EQUAL = "eq"
    NOT_EQUAL = "ne"
    LESS_THAN = "lt"
    LESS_THAN_OR_EQUAL = "le"
    GREATER_THAN = "gt"
    GREATER_THAN_OR_EQUAL = "ge"
    IN = "in"


class BusinessFilter(BaseModel):
    model_config = _STRICT_CONFIG

    field: str = Field(min_length=1, max_length=128)
    operator: FilterOperator = FilterOperator.EQUAL
    value: FilterScalar | tuple[FilterScalar, ...]

    @model_validator(mode="after")
    def _validate_value_shape(self) -> BusinessFilter:
        if self.operator is FilterOperator.IN:
            if not isinstance(self.value, tuple) or not self.value:
                raise ValueError("in filters require one or more values")
        elif isinstance(self.value, tuple):
            raise ValueError("scalar filters accept exactly one value")
        return self


class ContextField(BaseModel):
    model_config = _STRICT_CONFIG

    name: str
    data_type: str
    nullable: bool
    description: str


class FilterCapability(BaseModel):
    model_config = _STRICT_CONFIG

    field: str
    data_type: str
    operators: tuple[FilterOperator, ...]


class RetrievalCapability(BaseModel):
    model_config = _STRICT_CONFIG

    modes: tuple[str, ...]
    consistency: str
    filter_fields: tuple[FilterCapability, ...]


class ContextModelSummary(BaseModel):
    model_config = _STRICT_CONFIG

    name: str
    description: str | None
    contract: Literal["agent_context/v1"]
    grain: Literal["document_chunks"]
    access: Literal["public", "governed"]
    schema_fields: tuple[ContextField, ...]
    retrieval: RetrievalCapability
    freshness: str
    last_successful_materialization: datetime | None
    entity_types: tuple[str, ...] = ()


class ListContextModelsRequest(BaseModel):
    model_config = _STRICT_CONFIG

    schema_version: Literal["mcp_context/v1"] = MCP_CONTEXT_SCHEMA_VERSION
    limit: int = Field(default=20, ge=1, le=100)
    cursor: str | None = Field(default=None, max_length=4096)


class ListContextModelsResponse(BaseModel):
    model_config = _STRICT_CONFIG

    schema_version: Literal["mcp_context/v1"] = MCP_CONTEXT_SCHEMA_VERSION
    models: tuple[ContextModelSummary, ...] = ()
    next_cursor: str | None = None
    error: ToolError | None = None


class SearchContextRequest(BaseModel):
    model_config = _STRICT_CONFIG

    schema_version: Literal["mcp_context/v1"] = MCP_CONTEXT_SCHEMA_VERSION
    model: str = Field(min_length=1, max_length=128)
    query: str = Field(min_length=1, max_length=32_768)
    mode: Literal["vector", "text", "hybrid"] = "hybrid"
    limit: int = Field(default=10, ge=1, le=100)
    filters: tuple[BusinessFilter, ...] = ()


class ContextEntity(BaseModel):
    model_config = _STRICT_CONFIG

    namespace: str
    name: str
    entity_id: str
    entity_key: JsonValue
    dbt_unique_id: str | None
    relationship_type: str
    confidence: float | None


class ContextInterval(BaseModel):
    model_config = _STRICT_CONFIG

    validity_known: bool
    valid_from: datetime | None
    valid_to: datetime | None
    recorded_from: datetime
    recorded_to: datetime | None


class FreshnessDescriptor(BaseModel):
    model_config = _STRICT_CONFIG

    status: str
    source_updated_at: datetime | None
    source_observed_at: datetime | None
    freshness_checked_at: datetime | None
    refresh_due_at: datetime | None
    stale_after: datetime | None


class CitationDescriptor(BaseModel):
    model_config = _STRICT_CONFIG

    source_uri: str | None = None
    chunk_index: int | None = None
    page_number: int | None = None
    section_path: tuple[str, ...] | None = None
    char_start: int | None = None
    char_end: int | None = None
    speaker: str | None = None
    start_seconds: float | None = None
    end_seconds: float | None = None
    extra: JsonValue = None


class CompactLineage(BaseModel):
    model_config = _STRICT_CONFIG

    resources: tuple[str, ...]
    upstream_unique_id: str
    invocation_id: str
    parser_identity: str | None
    transform_identity: str | None
    provider_identity: str | None
    model_identity: str | None
    prompt_fingerprint: str | None
    schema_fingerprint: str | None
    provenance_fingerprint: str
    search_index_unique_id: str
    store_type: str


class SearchContextResult(BaseModel):
    model_config = _STRICT_CONFIG

    rank: int
    score: float
    document_id: str
    document_version_id: str
    context_id: str
    chunk_id: str
    snippet: str
    snippet_truncated: bool
    source_version: str
    entities: tuple[ContextEntity, ...]
    interval: ContextInterval
    freshness: FreshnessDescriptor
    citation: CitationDescriptor
    lineage: CompactLineage


class SearchContextResponse(BaseModel):
    model_config = _STRICT_CONFIG

    schema_version: Literal["mcp_context/v1"] = MCP_CONTEXT_SCHEMA_VERSION
    results: tuple[SearchContextResult, ...] = ()
    error: ToolError | None = None


class GetDocumentRequest(BaseModel):
    model_config = _STRICT_CONFIG

    schema_version: Literal["mcp_context/v1"] = MCP_CONTEXT_SCHEMA_VERSION
    model: str = Field(min_length=1, max_length=128)
    document_id: str
    document_version_id: str
    limit: int = Field(default=20, ge=1, le=100)
    cursor: str | None = Field(default=None, max_length=4096)

    @field_validator("document_id", "document_version_id")
    @classmethod
    def _validate_stable_id(cls, value: str) -> str:
        if not _STABLE_ID_PATTERN.fullmatch(value):
            raise ValueError("stable IDs must be 32 lowercase hexadecimal characters")
        return value


class DocumentSource(BaseModel):
    model_config = _STRICT_CONFIG

    source_system: str
    source_key: str
    source_uri: str | None
    source_version: str
    source_content_hash: str


class DocumentChunk(BaseModel):
    model_config = _STRICT_CONFIG

    context_id: str
    chunk_id: str
    chunk_index: int
    text: str
    text_truncated: bool
    citation: CitationDescriptor
    entities: tuple[ContextEntity, ...]


class GetDocumentResponse(BaseModel):
    model_config = _STRICT_CONFIG

    schema_version: Literal["mcp_context/v1"] = MCP_CONTEXT_SCHEMA_VERSION
    document_id: str | None = None
    document_version_id: str | None = None
    source: DocumentSource | None = None
    interval: ContextInterval | None = None
    freshness: FreshnessDescriptor | None = None
    lineage: CompactLineage | None = None
    chunks: tuple[DocumentChunk, ...] = ()
    next_cursor: str | None = None
    error: ToolError | None = None


class LineageReferenceType(StrEnum):
    DOCUMENT = "document"
    DOCUMENT_VERSION = "document_version"
    CONTEXT = "context"
    CHUNK = "chunk"
    SEARCH_RESULT = "search_result"


class GetContextLineageRequest(BaseModel):
    model_config = _STRICT_CONFIG

    schema_version: Literal["mcp_context/v1"] = MCP_CONTEXT_SCHEMA_VERSION
    model: str = Field(min_length=1, max_length=128)
    reference_type: LineageReferenceType
    reference_id: str

    @field_validator("reference_id")
    @classmethod
    def _validate_reference_id(cls, value: str) -> str:
        if not _STABLE_ID_PATTERN.fullmatch(value):
            raise ValueError("stable IDs must be 32 lowercase hexadecimal characters")
        return value


class ContextLineageRecord(BaseModel):
    model_config = _STRICT_CONFIG

    document_id: str
    document_version_id: str
    context_id: str
    chunk_id: str
    source: DocumentSource
    citation: CitationDescriptor
    entities: tuple[ContextEntity, ...]
    lineage: CompactLineage


class GetContextLineageResponse(BaseModel):
    model_config = _STRICT_CONFIG

    schema_version: Literal["mcp_context/v1"] = MCP_CONTEXT_SCHEMA_VERSION
    record: ContextLineageRecord | None = None
    error: ToolError | None = None
