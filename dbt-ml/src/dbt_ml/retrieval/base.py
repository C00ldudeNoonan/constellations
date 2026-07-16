from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum
from math import isfinite
from pathlib import Path
from types import MappingProxyType
from typing import Any, Self

import pyarrow as pa
from pydantic import BaseModel, ConfigDict


class RetrievalError(Exception):
    """Artifact-safe retrieval failure."""


class RetrievalConfigError(RetrievalError):
    pass


class RetrievalCapabilityError(RetrievalError):
    pass


class RetrievalStoreConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    type: str

    def absolutize(self, project_dir: Path) -> RetrievalStoreConfig:
        return self


class RetrievalFeature(StrEnum):
    EXACT_VECTOR_SEARCH = "exact_vector_search"
    APPROXIMATE_VECTOR_SEARCH = "approximate_vector_search"
    METADATA_FILTERING = "metadata_filtering"
    FULL_TEXT_SEARCH = "full_text_search"
    KEYED_UPSERT = "keyed_upsert"
    KEYED_DELETE = "keyed_delete"
    INDEX_READINESS = "index_readiness"
    DURABLE_WRITE_ACK = "durable_write_ack"
    ATOMIC_BATCH_MUTATION = "atomic_batch_mutation"


class RetrievalPredicateOperator(StrEnum):
    EQUAL = "eq"
    NOT_EQUAL = "ne"
    LESS_THAN = "lt"
    LESS_THAN_OR_EQUAL = "le"
    GREATER_THAN = "gt"
    GREATER_THAN_OR_EQUAL = "ge"
    IN = "in"


RetrievalScalar = str | int | float | bool | date | datetime


@dataclass(frozen=True, repr=False)
class RetrievalPredicate:
    field: str
    operator: RetrievalPredicateOperator
    value: RetrievalScalar | tuple[RetrievalScalar, ...]

    def __post_init__(self) -> None:
        if not self.field:
            raise RetrievalError("Retrieval predicate field must not be empty")
        if not isinstance(self.operator, RetrievalPredicateOperator):
            raise RetrievalError("Retrieval predicate operator is invalid")
        if self.operator == RetrievalPredicateOperator.IN:
            if not isinstance(self.value, tuple) or not self.value:
                raise RetrievalError("Retrieval IN predicates require a non-empty tuple")
            first = type(self.value[0])
            if any(type(item) is not first for item in self.value):
                raise RetrievalError("Retrieval IN values must share one type")
            values = self.value
        else:
            if isinstance(self.value, tuple):
                raise RetrievalError("Retrieval scalar predicate received a tuple")
            values = (self.value,)
        if any(not _is_retrieval_scalar(value) for value in values):
            raise RetrievalError("Retrieval predicate contains an unsupported value")

    def __repr__(self) -> str:
        return (
            f"RetrievalPredicate(field={self.field!r}, "
            f"operator={self.operator.value!r}, value=<redacted>)"
        )


def _is_retrieval_scalar(value: Any) -> bool:
    if not isinstance(value, str | int | float | bool | date | datetime):
        return False
    return not isinstance(value, float) or isfinite(value)


@dataclass(frozen=True)
class RetrievalCapabilities:
    features: frozenset[RetrievalFeature]
    distance_metrics: frozenset[str]
    consistency_modes: frozenset[str]
    max_batch_size: int
    max_id_bytes: int | None
    max_dimensions: int | None

    def __post_init__(self) -> None:
        if self.max_batch_size <= 0:
            raise ValueError("Retrieval max_batch_size must be positive")
        if self.max_id_bytes is not None and self.max_id_bytes <= 0:
            raise ValueError("Retrieval max_id_bytes must be positive")
        if self.max_dimensions is not None and self.max_dimensions <= 0:
            raise ValueError("Retrieval max_dimensions must be positive")

    def require(self, required: Mapping[RetrievalFeature, str], *, store_type: str) -> None:
        missing = sorted(set(required) - self.features, key=lambda item: item.value)
        if not missing:
            return
        detail = ", ".join(f"{feature.value} ({required[feature]})" for feature in missing)
        raise RetrievalCapabilityError(
            f"Retrieval store '{store_type}' is missing capabilities: {detail}"
        )


@dataclass(frozen=True)
class SafeRetrievalTarget:
    store_type: str
    safe_target_identity: str


@dataclass(frozen=True)
class StateRetrievalTarget:
    store_type: str
    routing_identity_fingerprint: str
    physical_collection: str

    def descriptor(self) -> dict[str, str]:
        return {
            "store_type": self.store_type,
            "routing_identity_fingerprint": self.routing_identity_fingerprint,
            "physical_collection": self.physical_collection,
        }


@dataclass(frozen=True)
class CollectionSpec:
    logical_name: str
    physical_name: str
    id_field: str
    text_fields: tuple[str, ...]
    full_text_fields: tuple[str, ...]
    attribute_fields: tuple[str, ...]
    scalar_index_fields: tuple[str, ...]
    display_fields: tuple[str, ...]
    vector_field: str | None
    vector_dimensions: int | None
    distance_metric: str | None
    vector_search: str | None
    config_fingerprint: str
    arrow_schema: pa.Schema


@dataclass(frozen=True, repr=False)
class IndexedRow:
    record_id: str
    values: Mapping[str, Any]
    input_fingerprint: str

    def __post_init__(self) -> None:
        if not self.record_id:
            raise RetrievalError("Indexed row ID must not be empty")
        if not self.input_fingerprint:
            raise RetrievalError("Indexed row fingerprint must not be empty")
        object.__setattr__(self, "values", MappingProxyType(dict(self.values)))


@dataclass(frozen=True)
class MutationOutcome:
    status: str
    safe_error_code: str | None = None


@dataclass(frozen=True)
class MutationReceipt:
    mutation_digest: str
    atomic: bool
    outcomes: tuple[MutationOutcome, ...]

    @property
    def acknowledged(self) -> bool:
        return self.atomic and all(
            outcome.status in {"applied", "deleted", "absent"} for outcome in self.outcomes
        )


@dataclass(frozen=True)
class CollectionMetadata:
    physical_name: str
    config_fingerprint: str | None
    physical_generation: str
    row_count: int
    schema: pa.Schema


class RetrievalStore(ABC):
    def __init__(
        self,
        config: RetrievalStoreConfig,
        *,
        project_name: str,
        target_name: str,
        alias: str,
    ) -> None:
        self.config = config
        self.project_name = project_name
        self.target_name = target_name
        self.alias = alias

    @classmethod
    @abstractmethod
    def store_type(cls) -> str: ...

    @classmethod
    @abstractmethod
    def config_model(cls) -> type[RetrievalStoreConfig]: ...

    @classmethod
    @abstractmethod
    def capabilities(cls) -> RetrievalCapabilities: ...

    @classmethod
    @abstractmethod
    def implementation_identity(cls) -> str: ...

    @abstractmethod
    def __enter__(self) -> Self: ...

    @abstractmethod
    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None: ...

    @abstractmethod
    def safe_descriptor(self) -> SafeRetrievalTarget: ...

    @abstractmethod
    def state_descriptor(self, collection: str) -> StateRetrievalTarget: ...

    @abstractmethod
    def physical_collection(self, logical_name: str) -> str: ...

    @abstractmethod
    def inspect_collection(self, name: str) -> CollectionMetadata | None: ...

    @abstractmethod
    def create_collection(self, spec: CollectionSpec) -> CollectionMetadata: ...

    @abstractmethod
    def upsert(
        self,
        collection: str,
        rows: Sequence[IndexedRow],
        *,
        id_field: str,
        mutation_digest: str,
    ) -> MutationReceipt: ...

    @abstractmethod
    def delete(
        self,
        collection: str,
        record_ids: Sequence[str],
        *,
        id_field: str,
        mutation_digest: str,
    ) -> MutationReceipt: ...

    @abstractmethod
    def ensure_indexes(self, spec: CollectionSpec) -> CollectionMetadata: ...

    @abstractmethod
    def vector_search(
        self,
        collection: str,
        vector: Sequence[float],
        *,
        vector_field: str,
        limit: int,
        columns: Sequence[str] | None = None,
        predicates: Sequence[RetrievalPredicate] = (),
    ) -> pa.Table: ...

    @abstractmethod
    def text_search(
        self,
        collection: str,
        query: str,
        *,
        text_field: str,
        limit: int,
        columns: Sequence[str] | None = None,
        predicates: Sequence[RetrievalPredicate] = (),
    ) -> pa.Table: ...
