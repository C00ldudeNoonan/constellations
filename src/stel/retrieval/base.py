from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum
from math import isfinite
from pathlib import Path
from types import MappingProxyType, TracebackType
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
    # In-place widening of a published collection (issue #344). Advertised
    # only by a store that can add columns to a live collection without
    # rewriting its rows. It permits exactly the changes classification
    # calls compatible: the design doc is explicit that a broad flag cannot
    # make an incompatible dimension or type change safe.
    ONLINE_SCHEMA_EVOLUTION = "online_schema_evolution"
    # Store-side publisher fencing proofs (issue #152). A warehouse fencing
    # token cannot stop a partitioned process from calling an independent
    # store SDK, so publication requires the store to advertise exactly how
    # stale writers are excluded: an OS-enforced single-host lock, provider
    # conditional writes, or immutable generations with conditional
    # activation. The latter two are reserved for distributed adapters.
    SINGLE_HOST_PUBLISHER_LOCK = "single_host_publisher_lock"
    PROVIDER_ENFORCED_FENCING = "provider_enforced_fencing"
    IMMUTABLE_GENERATION_ACTIVATION = "immutable_generation_activation"


PUBLISHER_FENCING_FEATURES = frozenset(
    {
        RetrievalFeature.SINGLE_HOST_PUBLISHER_LOCK,
        RetrievalFeature.PROVIDER_ENFORCED_FENCING,
        RetrievalFeature.IMMUTABLE_GENERATION_ACTIVATION,
    }
)


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
    # Canonical JSON of the semantic descriptor (issue #344). Persisted with
    # the collection so a later publish can name which field changed rather
    # than only observing that the digest moved.
    descriptor: str
    # The pre-#344 digest for this same config, used only to recognize a
    # collection stamped before the descriptor existed and prove it unchanged.
    # Removable with the rest of the legacy path (#321 category 1).
    legacy_config_fingerprint: str
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
    # None for a collection published before #344, which carries only the
    # legacy digest. Such a collection is re-stamped in place once its
    # configuration is proven unchanged; it is never rebuilt for this.
    descriptor: str | None
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
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...

    @abstractmethod
    def safe_descriptor(self) -> SafeRetrievalTarget: ...

    def publisher_fence(self, collection: str) -> AbstractContextManager[None]:
        """Store-enforced exclusive publisher fence for one collection.

        Adapters advertising SINGLE_HOST_PUBLISHER_LOCK must return a context
        manager that excludes every other publisher process for the fence's
        lifetime and raises RetrievalError when the fence is already held.
        Adapters relying on PROVIDER_ENFORCED_FENCING or
        IMMUTABLE_GENERATION_ACTIVATION enforce their proof inside mutation
        and activation instead and may keep this default."""
        del collection
        capabilities = self.capabilities()
        if RetrievalFeature.SINGLE_HOST_PUBLISHER_LOCK in capabilities.features:
            raise RetrievalError(
                "Retrieval store advertises single_host_publisher_lock but does "
                "not implement publisher_fence()"
            )
        return nullcontext()

    @abstractmethod
    def state_descriptor(self, collection: str) -> StateRetrievalTarget: ...

    @abstractmethod
    def physical_collection(self, logical_name: str) -> str: ...

    @abstractmethod
    def inspect_collection(self, name: str) -> CollectionMetadata | None: ...

    @abstractmethod
    def create_collection(self, spec: CollectionSpec) -> CollectionMetadata: ...

    @abstractmethod
    def evolve_collection(self, spec: CollectionSpec, added: Sequence[str]) -> None:
        """Widen a live collection to `spec`, in place, without rewriting rows.

        Called only for a change classification has found compatible and only
        when the store advertises `ONLINE_SCHEMA_EVOLUTION`. `added` names the
        columns to introduce; they arrive null and are filled by the republish
        that follows, which costs an index rewrite but no provider calls.

        Implementations must also re-stamp the collection, so a run that
        evolves and then fails does not leave a widened collection still
        describing its previous shape.
        """

    @abstractmethod
    def restamp_collection(self, spec: CollectionSpec) -> None:
        """Rewrite a collection's stored descriptor, leaving its rows alone.

        Called only after the caller has proven the configuration is unchanged
        and the stamp is merely in an older format (issue #344). Stores that
        cannot rewrite a stamp in place should raise rather than silently
        succeed: a caller that believes a collection was re-stamped will not
        try again.
        """

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
