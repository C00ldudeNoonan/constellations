from __future__ import annotations

import re
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
    # Filtering an array-valued column by overlap with a set (issue #397).
    # Declared separately from METADATA_FILTERING because a store can filter
    # scalars perfectly well and have no way to express array overlap. A store
    # that cannot must fail closed: silently dropping a policy filter turns an
    # unusable governed model into an unfiltered one, which is far worse.
    ARRAY_CONTAINMENT_FILTERS = "array_containment_filters"
    SINGLE_HOST_PUBLISHER_LOCK = "single_host_publisher_lock"
    PROVIDER_ENFORCED_FENCING = "provider_enforced_fencing"
    IMMUTABLE_GENERATION_ACTIVATION = "immutable_generation_activation"
    # Private generation build (issue #355). The store can create a collection
    # under a caller-chosen physical name that receives no production queries,
    # and drop one later. Deliberately weaker than
    # IMMUTABLE_GENERATION_ACTIVATION: with the active generation resolved
    # through the warehouse-owned serving ledger, the store is never asked to
    # swap anything, so it needs no alias primitive and no conditional write.
    # The swap is a fenced warehouse row update; the store only builds and
    # drops. A store with a fixed collection namespace cannot do even this,
    # which is what the flag exists to catch.
    PRIVATE_GENERATION_BUILD = "private_generation_build"


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
    # "this array column shares at least one element with these values"
    # (issue #397). Distinct from IN, which asks whether a *scalar* column is
    # one of several values -- the inverse relation, and not expressible for
    # an array-valued attribute.
    ARRAY_CONTAINS_ANY = "array_contains_any"


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
        if self.operator in _TUPLE_VALUED_OPERATORS:
            if not isinstance(self.value, tuple) or not self.value:
                # An empty set would mean "matches nothing", but an operator
                # that silently matches nothing is indistinguishable from a
                # dropped filter to everything downstream. Callers must decide
                # explicitly rather than encode it as an empty tuple.
                raise RetrievalError(
                    f"Retrieval {self.operator.value} predicates require a "
                    "non-empty tuple"
                )
            first = type(self.value[0])
            if any(type(item) is not first for item in self.value):
                raise RetrievalError(
                    f"Retrieval {self.operator.value} values must share one type"
                )
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


_TUPLE_VALUED_OPERATORS = frozenset(
    {RetrievalPredicateOperator.IN, RetrievalPredicateOperator.ARRAY_CONTAINS_ANY}
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


_GENERATION_RE = re.compile(r"^[a-z0-9]{1,16}$")

# Collections built for a generation are named `<base>__g<token>` by
# `physical_collection(..., generation=...)`. The marker plus the exact token
# shape is what a retirement sweep matches against, so the shape is reserved:
# no *base* collection name may end this way (see
# `reject_generation_shaped_collection_name`).
GENERATION_MARKER = "__g"
_GENERATION_SUFFIX_RE = re.compile(rf"{GENERATION_MARKER}[a-z0-9]{{1,16}}$")


def validate_generation_token(value: str) -> str:
    """Validate a generation token before it is rendered into a collection name.

    Deliberately narrow: the token crosses into a physical collection name, so
    it is restricted to lowercase alphanumerics rather than merely escaped.
    """
    if not _GENERATION_RE.fullmatch(value):
        raise RetrievalError(
            "Retrieval generation token must be 1-16 lowercase alphanumeric "
            "characters"
        )
    return value


def reject_generation_shaped_collection_name(physical: str) -> str:
    """Refuse a base collection name that looks like a generation collection.

    The generation retirement sweep (issue #355) classifies any collection
    named `<base>__g<token>` as a retired generation of `<base>` and deletes
    it. A *logical* collection whose resolved base name ends the same way —
    say logical `ctx__garchive` next to logical `ctx` — would be
    indistinguishable from a generation of its sibling and swept with it, so
    the shape is reserved at name-resolution time. Every store must route its
    unsuffixed base names through this check before appending a generation
    suffix.
    """
    if _GENERATION_SUFFIX_RE.search(physical):
        raise RetrievalError(
            f"Retrieval collection name '{physical}' ends with the reserved "
            f"generation suffix `{GENERATION_MARKER}<token>`; rename the "
            "logical collection so it cannot be mistaken for a retired "
            "generation of a sibling collection"
        )
    return physical


@dataclass(frozen=True)
class SafeRetrievalTarget:
    store_type: str
    safe_target_identity: str


@dataclass(frozen=True)
class StateRetrievalTarget:
    """Identity of the serving scope for one logical collection (issue #355).

    `descriptor()` keys on the *logical* collection, not the physical one.
    That is what lets the serving ledger row stay reachable while the physical
    collection behind it is replaced: a reader resolves the scope from the
    logical name alone, then follows `active_generation` to the physical
    collection. Keying on the physical name instead would make the ledger
    unreadable the moment generations exist — you would have to know the
    active generation to compute the scope that names it.

    `physical_collection` is retained for artifact reporting only; it is
    deliberately excluded from the descriptor.
    """

    store_type: str
    routing_identity_fingerprint: str
    physical_collection: str
    logical_collection: str

    def descriptor(self) -> dict[str, str]:
        return {
            "store_type": self.store_type,
            "routing_identity_fingerprint": self.routing_identity_fingerprint,
            "logical_collection": self.logical_collection,
        }

    def legacy_descriptor(self) -> dict[str, str]:
        """The pre-#355 physical-keyed descriptor.

        Only `serving migrate-scope` uses this, to locate rows written under
        the old identity so they can be rewritten under the new one.
        """
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
    def physical_collection(
        self, logical_name: str, *, generation: str | None = None
    ) -> str:
        """The physical collection backing `logical_name`.

        With `generation=None` this must return the name used before issue
        #355 existed, unchanged — that is the collection every already
        published index still lives in, and renaming it would strand the data.
        A generation token names a distinct, privately built collection that
        activation may later point the logical name at.
        """

    def list_collections(self) -> tuple[str, ...]:
        """Every physical collection name visible in this store's namespace.

        Used to find generations the serving ledger no longer points at. The
        names are unfiltered by ownership: `drop_collection` is what refuses a
        collection stel does not own, and it should stay the single place that
        decides that.
        """
        capabilities = self.capabilities()
        if RetrievalFeature.PRIVATE_GENERATION_BUILD in capabilities.features:
            raise RetrievalError(
                "Retrieval store advertises private_generation_build but does "
                "not implement list_collections()"
            )
        raise RetrievalCapabilityError(
            f"Retrieval store '{self.store_type()}' cannot list collections"
        )

    def drop_collection(self, name: str) -> bool:
        """Remove a physical collection, returning whether one was removed.

        Used to retire a superseded generation and to clean up a private build
        that failed before activation. Never called on the active generation;
        the caller resolves that through the serving ledger first.
        """
        del name
        capabilities = self.capabilities()
        if RetrievalFeature.PRIVATE_GENERATION_BUILD in capabilities.features:
            raise RetrievalError(
                "Retrieval store advertises private_generation_build but does "
                "not implement drop_collection()"
            )
        raise RetrievalCapabilityError(
            f"Retrieval store '{self.store_type()}' cannot drop collections"
        )

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
