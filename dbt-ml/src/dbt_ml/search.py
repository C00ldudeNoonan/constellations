from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum
from math import isfinite
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pyarrow as pa

from .compiler import validate_project_contract, validate_retrieval_capabilities
from .config import load_project
from .config.model import ModelConfig, SearchAttributeConfig
from .embedding import (
    effective_search_config,
    embed_query,
    resolve_search_embedding_identity,
)
from .profile import resolve_profile
from .providers import ProviderError
from .retrieval import (
    RetrievalCapabilityError,
    RetrievalError,
    RetrievalFeature,
    RetrievalPredicate,
    RetrievalPredicateOperator,
    collection_config_fingerprint,
    create_store,
)


class SearchError(Exception):
    """A query failure that is safe to display without request contents."""


class SearchMode(StrEnum):
    VECTOR = "vector"
    TEXT = "text"
    HYBRID = "hybrid"


class SearchFilterOperator(StrEnum):
    EQUAL = "eq"
    NOT_EQUAL = "ne"
    LESS_THAN = "lt"
    LESS_THAN_OR_EQUAL = "le"
    GREATER_THAN = "gt"
    GREATER_THAN_OR_EQUAL = "ge"
    IN = "in"


SearchScalar = str | int | float | bool | date | datetime


@dataclass(frozen=True, slots=True, repr=False)
class SearchFilter:
    field: str
    operator: SearchFilterOperator
    value: SearchScalar | tuple[SearchScalar, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.field, str) or not self.field:
            raise ValueError("search filter field must not be empty")
        try:
            operator = SearchFilterOperator(self.operator)
        except ValueError:
            raise ValueError("search filter operator is invalid") from None
        object.__setattr__(self, "operator", operator)
        if operator == SearchFilterOperator.IN:
            values = tuple(self.value) if isinstance(self.value, list | tuple) else ()
            if not values:
                raise ValueError("search IN filters require at least one value")
            object.__setattr__(self, "value", values)
        elif isinstance(self.value, list | tuple):
            raise ValueError("search scalar filters require one value")

    def __repr__(self) -> str:
        return (
            f"SearchFilter(field={self.field!r}, operator={self.operator.value!r}, "
            "value=<redacted>)"
        )


@dataclass(frozen=True, slots=True, repr=False)
class SearchRequest:
    model: str
    query: str | None = None
    vector: tuple[float, ...] | None = None
    mode: SearchMode = SearchMode.HYBRID
    limit: int = 10
    candidate_limit: int | None = None
    filters: tuple[SearchFilter, ...] = ()
    fields: tuple[str, ...] = ()
    consistency: str = "strong"

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model:
            raise ValueError("search model must not be empty")
        try:
            mode = SearchMode(self.mode)
        except ValueError:
            raise ValueError("search mode is invalid") from None
        object.__setattr__(self, "mode", mode)
        if self.query is not None and (
            not isinstance(self.query, str)
            or not self.query
            or len(self.query.encode()) > 32_768
        ):
            raise ValueError("search query must be a non-empty string within 32768 bytes")
        if isinstance(self.limit, bool) or not 1 <= self.limit <= 1000:
            raise ValueError("search limit must be between 1 and 1000")
        if self.candidate_limit is not None and (
            isinstance(self.candidate_limit, bool)
            or not self.limit <= self.candidate_limit <= 1000
        ):
            raise ValueError("search candidate_limit must be between limit and 1000")
        if self.vector is not None:
            if any(isinstance(value, bool) for value in self.vector):
                raise ValueError("search vector must contain finite numeric values")
            vector = tuple(float(value) for value in self.vector)
            if not vector or any(not isfinite(value) for value in vector):
                raise ValueError("search vector must contain finite numeric values")
            object.__setattr__(self, "vector", vector)
        filters = tuple(self.filters)
        if any(not isinstance(item, SearchFilter) for item in filters):
            raise ValueError("search filters must be SearchFilter values")
        object.__setattr__(self, "filters", filters)
        fields = tuple(self.fields)
        if any(not isinstance(field, str) or not field for field in fields):
            raise ValueError("search fields must be non-empty strings")
        if len(fields) != len(set(fields)):
            raise ValueError("search fields must not contain duplicates")
        object.__setattr__(self, "fields", fields)
        if not isinstance(self.consistency, str) or not self.consistency:
            raise ValueError("search consistency must not be empty")

    def __repr__(self) -> str:
        return (
            f"SearchRequest(model={self.model!r}, mode={self.mode.value!r}, "
            f"limit={self.limit!r}, candidate_limit={self.candidate_limit!r}, "
            f"filters={self.filters!r}, fields={self.fields!r}, "
            f"consistency={self.consistency!r}, query=<redacted>, vector=<redacted>)"
        )


@dataclass(frozen=True, slots=True)
class SearchProvenance:
    project: str
    model: str
    unique_id: str
    target: str
    store_type: str
    logical_collection: str
    physical_collection: str
    upstream: str
    embedding: Mapping[str, Any] | None

    def __post_init__(self) -> None:
        if self.embedding is not None:
            object.__setattr__(self, "embedding", _frozen_mapping(self.embedding))

    def to_dict(self) -> dict[str, Any]:
        return {
            "project": self.project,
            "model": self.model,
            "unique_id": self.unique_id,
            "target": self.target,
            "store_type": self.store_type,
            "logical_collection": self.logical_collection,
            "physical_collection": self.physical_collection,
            "upstream": self.upstream,
            "embedding": _json_value(self.embedding),
        }


@dataclass(frozen=True, slots=True)
class SearchResult:
    record_id: str
    document_id: str | None
    chunk_id: str | None
    rank: int
    score: float
    raw_score: float | None
    raw_score_kind: str | None
    text: Mapping[str, Any]
    metadata: Mapping[str, Any]
    display: Mapping[str, Any]
    contributing_ranks: Mapping[str, int]
    provenance: SearchProvenance

    def __post_init__(self) -> None:
        object.__setattr__(self, "text", _frozen_mapping(self.text))
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata))
        object.__setattr__(self, "display", _frozen_mapping(self.display))
        object.__setattr__(
            self, "contributing_ranks", MappingProxyType(dict(self.contributing_ranks))
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "rank": self.rank,
            "score": self.score,
            "raw_score": self.raw_score,
            "raw_score_kind": self.raw_score_kind,
            "text": _json_value(self.text),
            "metadata": _json_value(self.metadata),
            "display": _json_value(self.display),
            "contributing_ranks": dict(self.contributing_ranks),
            "provenance": self.provenance.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class _RankedRow:
    record_id: str
    values: Mapping[str, Any]
    raw_score: float | None
    raw_score_kind: str | None


@dataclass(frozen=True, slots=True)
class _ScoredRow:
    record_id: str
    values: Mapping[str, Any]
    score: float
    raw_score: float | None
    raw_score_kind: str | None
    contributing_ranks: Mapping[str, int]


def search(
    project: str | Path,
    request: SearchRequest,
    *,
    target: str | None = None,
    profiles_dir: Path | None = None,
) -> list[SearchResult]:
    project_dir = Path(project).resolve()
    project_config, sources, models = load_project(project_dir)
    validate_project_contract(project_config, sources, models, project_dir)
    model = next((item for item in models if item.name == request.model), None)
    if model is None or model.search is None:
        raise SearchError(f"Search index '{request.model}' was not found")
    resolved = resolve_profile(
        project_config,
        project_dir,
        target=target,
        profiles_dir=profiles_dir,
    )
    validate_retrieval_capabilities([model], project_config, resolved)
    search_config = model.search
    if search_config.access != "public":
        raise SearchError(
            "Governed search indexes require the trusted authorization and read-lease "
            "runtime from #152"
        )
    if resolved.retrieval is None:
        raise SearchError("The active profile has no retrieval configuration")
    alias = search_config.store or resolved.retrieval.default
    store_config = resolved.retrieval.stores.get(alias)
    if store_config is None:
        raise SearchError("The search index selects an unavailable retrieval store")
    if request.mode.value not in search_config.query.modes:
        raise SearchError(
            f"Search index '{model.name}' does not allow {request.mode.value} queries"
        )
    store = create_store(
        store_config,
        project_name=project_config.name,
        target_name=resolved.target_name,
        alias=alias,
    )
    _validate_capabilities(
        model,
        request,
        store.capabilities(),
        store_type=store_config.type,
    )
    predicates = _resolve_predicates(model, request.filters)
    included_fields = _resolve_result_fields(model, request.fields)
    models_by_name = {item.name: item for item in models}
    effective_config = effective_search_config(model, models_by_name)
    expected_fingerprint = collection_config_fingerprint(
        effective_config,
        store_type=store_config.type,
    )
    query_vector = _resolve_query_vector(model, models_by_name, request)
    logical_collection = search_config.collection or model.name
    physical_collection = store.physical_collection(logical_collection)
    candidate_limit = request.candidate_limit or min(max(request.limit * 4, 50), 1000)

    try:
        with store:
            metadata = store.inspect_collection(physical_collection)
            if metadata is None:
                raise SearchError(
                    f"Search index '{model.name}' has not been published; run `dbt-ml run`"
                )
            if metadata.config_fingerprint != expected_fingerprint:
                raise SearchError(
                    f"Search index '{model.name}' is stale or incompatible; republish it"
                )
            ranked = _execute_query(
                model,
                store,
                physical_collection,
                request,
                query_vector,
                predicates,
                candidate_limit,
                included_fields,
            )
    except SearchError:
        raise
    except RetrievalError as error:
        raise SearchError(str(error)) from None

    vector_config = effective_config.get("vector")
    embedding = vector_config.get("embedding") if isinstance(vector_config, dict) else None
    upstream = (model.depends_on or [""])[0]
    provenance = SearchProvenance(
        project=project_config.name,
        model=model.name,
        unique_id=f"search_index.{project_config.name}.{model.name}",
        target=resolved.target_name,
        store_type=store_config.type,
        logical_collection=logical_collection,
        physical_collection=physical_collection,
        upstream=upstream,
        embedding=embedding if isinstance(embedding, Mapping) else None,
    )
    return [
        _to_result(
            model,
            item,
            rank=rank,
            included_fields=included_fields,
            provenance=provenance,
        )
        for rank, item in enumerate(ranked[: request.limit], 1)
    ]


def _validate_capabilities(
    model: ModelConfig,
    request: SearchRequest,
    capabilities: Any,
    *,
    store_type: str,
) -> None:
    search_config = model.search
    assert search_config is not None
    required: dict[RetrievalFeature, str] = {}
    if request.mode in {SearchMode.VECTOR, SearchMode.HYBRID}:
        vector = search_config.vector
        if vector is None:
            raise SearchError("The search index has no vector configuration")
        required[
            RetrievalFeature.APPROXIMATE_VECTOR_SEARCH
            if vector.search == "approximate"
            else RetrievalFeature.EXACT_VECTOR_SEARCH
        ] = f"{request.mode.value} query"
    if request.mode in {SearchMode.TEXT, SearchMode.HYBRID}:
        if search_config.full_text is None:
            raise SearchError("The search index has no full-text configuration")
        if request.query is None:
            raise SearchError("Text and hybrid queries require query text")
        required[RetrievalFeature.FULL_TEXT_SEARCH] = f"{request.mode.value} query"
    if request.filters:
        required[RetrievalFeature.METADATA_FILTERING] = "query filters"
    try:
        capabilities.require(required, store_type=store_type)
    except RetrievalCapabilityError as error:
        raise SearchError(str(error)) from None
    if request.consistency not in capabilities.consistency_modes:
        raise SearchError(
            f"Retrieval store does not support consistency mode '{request.consistency}'"
        )


def _resolve_query_vector(
    model: ModelConfig,
    models_by_name: Mapping[str, ModelConfig],
    request: SearchRequest,
) -> tuple[float, ...] | None:
    if request.mode not in {SearchMode.VECTOR, SearchMode.HYBRID}:
        return None
    search_config = model.search
    assert search_config is not None and search_config.vector is not None
    if request.vector is not None:
        vector = request.vector
    else:
        if request.query is None:
            raise SearchError("Vector queries require query text or a precomputed vector")
        try:
            identity = resolve_search_embedding_identity(model, models_by_name)
        except (ValueError, ProviderError) as error:
            raise SearchError(str(error)) from None
        if identity is None:
            raise SearchError(
                "This index uses an external embedding identity; provide a precomputed "
                "query vector"
            )
        try:
            vector = embed_query(request.query, identity)
        except (ValueError, ProviderError) as error:
            raise SearchError(f"Query embedding failed: {error}") from None
    if len(vector) != search_config.vector.dimensions:
        raise SearchError("Query vector dimensions do not match the search index")
    return vector


def _resolve_predicates(
    model: ModelConfig,
    filters: Sequence[SearchFilter],
) -> tuple[RetrievalPredicate, ...]:
    search_config = model.search
    assert search_config is not None
    attributes = {attribute.name: attribute for attribute in search_config.attributes}
    predicates: list[RetrievalPredicate] = []
    for item in filters:
        attribute = attributes.get(item.field)
        if attribute is None or attribute.filter_role not in {"user", "user_and_policy"}:
            raise SearchError(f"Field '{item.field}' is not available for user filtering")
        value = _coerce_filter_value(item.value, attribute)
        if item.operator not in {
            SearchFilterOperator.EQUAL,
            SearchFilterOperator.NOT_EQUAL,
            SearchFilterOperator.IN,
        } and attribute.data_type in {"boolean", "array[string]"}:
            raise SearchError(
                f"Operator '{item.operator.value}' is not valid for field '{item.field}'"
            )
        predicates.append(
            RetrievalPredicate(
                item.field,
                RetrievalPredicateOperator(item.operator.value),
                value,
            )
        )
    return tuple(predicates)


def _coerce_filter_value(
    value: SearchScalar | tuple[SearchScalar, ...],
    attribute: SearchAttributeConfig,
) -> SearchScalar | tuple[SearchScalar, ...]:
    values = value if isinstance(value, tuple) else (value,)
    converted = tuple(_coerce_scalar(item, attribute.data_type) for item in values)
    return converted if isinstance(value, tuple) else converted[0]


def _coerce_scalar(value: SearchScalar, data_type: str) -> SearchScalar:
    try:
        if data_type in {"string", "array[string]"}:
            if not isinstance(value, str):
                raise ValueError
            return value
        if data_type == "integer":
            if isinstance(value, int) and not isinstance(value, bool):
                return value
            if isinstance(value, str):
                return int(value)
            raise ValueError
        if data_type == "float":
            if not isinstance(value, str | int | float) or isinstance(value, bool):
                raise ValueError
            converted = float(value)
            if not isfinite(converted):
                raise ValueError
            return converted
        if data_type == "boolean":
            if isinstance(value, bool):
                return value
            if isinstance(value, str) and value.lower() in {"true", "false"}:
                return value.lower() == "true"
            raise ValueError
        if data_type == "date":
            if isinstance(value, datetime):
                raise ValueError
            return value if isinstance(value, date) else date.fromisoformat(str(value))
        if data_type == "timestamp":
            return value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        raise SearchError("Search filter value does not match the declared field type") from None
    raise SearchError("Search filter uses an unsupported field type")


def _resolve_result_fields(model: ModelConfig, requested: Sequence[str]) -> tuple[str, ...]:
    search_config = model.search
    assert search_config is not None
    identity_fields = (
        search_config.id_field,
        search_config.document_id_field,
        search_config.chunk_id_field,
    )
    ordered: list[str] = []
    for field in (
        *identity_fields,
        *search_config.return_text_fields,
        *(item.name for item in search_config.attributes if item.returned),
        *search_config.display_fields,
    ):
        if field is not None and field not in ordered:
            ordered.append(field)
    if not requested:
        return tuple(ordered)
    unavailable = sorted(set(requested) - set(ordered))
    if unavailable:
        raise SearchError(
            "Requested fields are not declared for return: " + ", ".join(unavailable)
        )
    required = {field for field in identity_fields if field is not None}
    return tuple(
        field
        for field in ordered
        if field in requested or field in required
    )


def _execute_query(
    model: ModelConfig,
    store: Any,
    collection: str,
    request: SearchRequest,
    vector: tuple[float, ...] | None,
    predicates: Sequence[RetrievalPredicate],
    candidate_limit: int,
    included_fields: Sequence[str],
) -> list[_ScoredRow]:
    search_config = model.search
    assert search_config is not None
    vector_rows: list[_ScoredRow] = []
    if request.mode in {SearchMode.VECTOR, SearchMode.HYBRID}:
        assert vector is not None and search_config.vector is not None
        table = store.vector_search(
            collection,
            vector,
            vector_field=search_config.vector.field,
            limit=candidate_limit,
            columns=included_fields,
            predicates=predicates,
        )
        vector_rows = _score_single(
            _rank_table(table, search_config.id_field),
            label="vector",
        )

    text_rows: list[_ScoredRow] = []
    if request.mode in {SearchMode.TEXT, SearchMode.HYBRID}:
        assert request.query is not None and search_config.full_text is not None
        by_field: dict[str, list[_ScoredRow]] = {}
        for field in search_config.full_text.fields:
            table = store.text_search(
                collection,
                request.query,
                text_field=field,
                limit=candidate_limit,
                columns=included_fields,
                predicates=predicates,
            )
            by_field[f"text:{field}"] = _score_single(
                _rank_table(table, search_config.id_field),
                label=f"text:{field}",
            )
        text_rows = (
            next(iter(by_field.values()))
            if len(by_field) == 1
            else _rrf(by_field)
        )

    if request.mode == SearchMode.VECTOR:
        return vector_rows
    if request.mode == SearchMode.TEXT:
        return text_rows
    return _rrf({"vector": vector_rows, "text": text_rows})


def _rank_table(table: pa.Table, id_field: str) -> list[_RankedRow]:
    rows: list[_RankedRow] = []
    seen: set[str] = set()
    for raw in table.to_pylist():
        record_id = raw.get(id_field)
        if not isinstance(record_id, str) or not record_id:
            raise SearchError("Retrieval store returned an invalid record ID")
        if record_id in seen:
            continue
        seen.add(record_id)
        raw_score: float | None = None
        raw_score_kind: str | None = None
        for name in ("_distance", "_score", "_relevance_score"):
            candidate = raw.get(name)
            if isinstance(candidate, int | float) and not isinstance(candidate, bool):
                converted = float(candidate)
                if isfinite(converted):
                    raw_score = converted
                    raw_score_kind = name.removeprefix("_")
                    break
        rows.append(
            _RankedRow(
                record_id=record_id,
                values={key: value for key, value in raw.items() if not key.startswith("_")},
                raw_score=raw_score,
                raw_score_kind=raw_score_kind,
            )
        )
    return rows


def _score_single(rows: Sequence[_RankedRow], *, label: str) -> list[_ScoredRow]:
    return [
        _ScoredRow(
            record_id=row.record_id,
            values=row.values,
            score=1.0 / rank,
            raw_score=row.raw_score,
            raw_score_kind=row.raw_score_kind,
            contributing_ranks={label: rank},
        )
        for rank, row in enumerate(rows, 1)
    ]


def _rrf(
    ranked: Mapping[str, Sequence[_ScoredRow]],
    *,
    rank_constant: int = 60,
) -> list[_ScoredRow]:
    nonempty = {name: rows for name, rows in ranked.items() if rows}
    if not nonempty:
        return []
    scores: dict[str, float] = {}
    values: dict[str, dict[str, Any]] = {}
    ranks: dict[str, dict[str, int]] = {}
    for name in sorted(nonempty):
        for rank, row in enumerate(nonempty[name], 1):
            scores[row.record_id] = scores.get(row.record_id, 0.0) + 1.0 / (
                rank_constant + rank
            )
            values.setdefault(row.record_id, {}).update(row.values)
            ranks.setdefault(row.record_id, {})[name] = rank
    maximum = len(nonempty) / (rank_constant + 1)
    ordered = sorted(scores, key=lambda record_id: (-scores[record_id], record_id))
    return [
        _ScoredRow(
            record_id=record_id,
            values=values[record_id],
            score=scores[record_id] / maximum,
            raw_score=None,
            raw_score_kind=None,
            contributing_ranks=ranks[record_id],
        )
        for record_id in ordered
    ]


def _to_result(
    model: ModelConfig,
    row: _ScoredRow,
    *,
    rank: int,
    included_fields: Sequence[str],
    provenance: SearchProvenance,
) -> SearchResult:
    search_config = model.search
    assert search_config is not None
    included = set(included_fields)
    text = {
        field: row.values.get(field)
        for field in search_config.return_text_fields
        if field in included
    }
    metadata = {
        item.name: row.values.get(item.name)
        for item in search_config.attributes
        if item.returned and item.name in included
    }
    display = {
        field: row.values.get(field)
        for field in search_config.display_fields
        if field in included
    }
    document_id = (
        row.values.get(search_config.document_id_field)
        if search_config.document_id_field is not None
        else None
    )
    chunk_id = (
        row.values.get(search_config.chunk_id_field)
        if search_config.chunk_id_field is not None
        else None
    )
    return SearchResult(
        record_id=row.record_id,
        document_id=document_id if isinstance(document_id, str) else None,
        chunk_id=chunk_id if isinstance(chunk_id, str) else None,
        rank=rank,
        score=round(row.score, 12),
        raw_score=row.raw_score,
        raw_score_kind=row.raw_score_kind,
        text=text,
        metadata=metadata,
        display=display,
        contributing_ranks=row.contributing_ranks,
        provenance=provenance,
    )


def _frozen_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(dict(value))


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, date | datetime):
        return value.isoformat()
    return value
