from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from datetime import date, datetime, timedelta
from functools import partial
from math import isfinite
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, Self

import pyarrow as pa
from pydantic import ConfigDict, Field, field_validator

from ..credentials import CredentialReference
from ..hashing import canonical_fingerprint
from ..memory import container_memory_limit_bytes
from ..optional_dependencies import (
    import_optional_dependency,
    optional_dependency_version,
)
from .base import (
    CollectionMetadata,
    CollectionSpec,
    IndexedRow,
    MutationOutcome,
    MutationReceipt,
    RetrievalCapabilities,
    RetrievalError,
    RetrievalFeature,
    RetrievalPredicate,
    RetrievalPredicateOperator,
    RetrievalStore,
    RetrievalStoreConfig,
    SafeRetrievalTarget,
    StateRetrievalTarget,
    StoreRole,
    reject_generation_shaped_collection_name,
    sanitized_retrieval_cause,
    validate_generation_token,
)
from .locks import PublisherLock, default_host_lock_base
from .registry import register

log = logging.getLogger(__name__)

_COLLECTION_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")

_MB = 1024 * 1024

# Per-role cache budgets in MiB, as (index, metadata), used when the profile
# names none (issue #479).
#
# `SERVE` is absent on purpose rather than set to a large number: leaving the
# serving path on LanceDB's own defaults keeps ANN query latency exactly as it
# is today, and this change is not the place to relitigate it. An operator who
# needs serving bounded — the same 20 GiB container hosting a query process —
# sets the fields explicitly, which wins in every role.
#
# The publisher numbers are bounds, not measurements of need: publish-side
# session occupancy measured 5 MB on a 600k-row collection (issue #475), so
# 256 MB is already generous for what the publisher demonstrably uses, while
# leaving the ceiling to the merge and the index build that actually need it.
_ROLE_CACHE_BUDGET_MB: dict[StoreRole, tuple[int, int]] = {
    StoreRole.PUBLISH: (256, 64),
    StoreRole.INSPECT: (32, 16),
}

# LanceDB's own defaults, as its Session docs give them.
_LANCEDB_DEFAULT_BUDGET_MB = (6 * 1024, 1024)

# Share of a detected container ceiling the caches may take between them. The
# serving process is the one this binds: its default is LanceDB's ~7 GB, which
# is most of a small container before the query process holds anything. Half
# leaves the other half to the process itself, and on a container large enough
# for LanceDB's defaults it does not bind at all — a 20 GiB box allows 10 GiB
# and the default asks for 7.
_CEILING_CACHE_SHARE = 0.5


def session_cache_budget(
    config: LanceDBConfig, role: StoreRole
) -> tuple[int, int] | None:
    """Cache sizes in bytes as `(index, metadata)`, or None for LanceDB's own.

    An explicit profile setting wins in every role; a role default fills in
    only what the profile left unset, so bounding one cache does not silently
    unbound the other.

    A detected container ceiling clamps a *default* but never an explicit
    setting: the operator may know something the cgroup does not, and this is
    advisory the same way #412's DuckDB detection is.
    """
    default = _ROLE_CACHE_BUDGET_MB.get(role)
    index_mb = config.index_cache_size_mb
    metadata_mb = config.metadata_cache_size_mb
    configured = index_mb is not None and metadata_mb is not None
    if default is None and index_mb is None and metadata_mb is None:
        # SERVE, unconfigured: LanceDB's defaults, unless a container ceiling
        # makes them absurd for the box we are actually on.
        clamped = _clamped_to_ceiling(_LANCEDB_DEFAULT_BUDGET_MB)
        return None if clamped is None else _to_bytes(clamped)
    fallback = default or (0, 0)
    resolved_index = index_mb if index_mb is not None else fallback[0]
    resolved_metadata = metadata_mb if metadata_mb is not None else fallback[1]
    if not resolved_index or not resolved_metadata:
        # One side configured under a role with no default (SERVE): fill the
        # other from LanceDB's documented defaults rather than from zero, which
        # would disable a cache nobody asked to disable.
        resolved_index = resolved_index or _LANCEDB_DEFAULT_BUDGET_MB[0]
        resolved_metadata = resolved_metadata or _LANCEDB_DEFAULT_BUDGET_MB[1]
    resolved = (resolved_index, resolved_metadata)
    if not configured:
        resolved = _clamped_to_ceiling(resolved) or resolved
    return _to_bytes(resolved)


def _to_bytes(budget_mb: tuple[int, int]) -> tuple[int, int]:
    return (budget_mb[0] * _MB, budget_mb[1] * _MB)


def _clamped_to_ceiling(budget_mb: tuple[int, int]) -> tuple[int, int] | None:
    """`budget_mb` reduced to fit a detected container ceiling, or None.

    None means "nothing to say": no ceiling detected, or one roomy enough that
    the budget already fits. The two caches are scaled together so their
    relative sizing — which is LanceDB's shape, not ours to reinterpret —
    survives the clamp, and neither is driven below 1 MB.
    """
    ceiling = container_memory_limit_bytes()
    if ceiling is None:
        return None
    allowance_mb = int(ceiling * _CEILING_CACHE_SHARE) // _MB
    total_mb = budget_mb[0] + budget_mb[1]
    if total_mb <= allowance_mb:
        return None
    scale = allowance_mb / total_mb
    return (
        max(int(budget_mb[0] * scale), 1),
        max(int(budget_mb[1] * scale), 1),
    )

# Arrow schema metadata written into every collection stel creates. These
# names live in existing LanceDB collections, so they are data, not code.
# `_OWNER_VALUE` in particular is an adoption gate, not a label: a collection
# whose owner metadata does not match exactly is refused as external, so
# changing this value makes every collection already published unreadable.
_OWNER_KEY = b"dbt_ml.owner"
_OWNER_VALUE = b"dbt-ml"
_CONTRACT_KEY = b"dbt_ml.record_contract"
_CONFIG_KEY = b"dbt_ml.config_fingerprint"
# The #344 semantic descriptor. Held as metadata on the collection's id
# field rather than on the schema because LanceDB can rewrite field
# metadata in place (`update_field_metadata`) and cannot rewrite
# schema-level metadata at all — re-stamping a pre-#344 collection has to
# be possible without rewriting its rows.
_DESCRIPTOR_KEY = b"stel.collection_descriptor"
# The descriptor's digest, written beside it. A re-stamped collection has
# to report its *new* identity everywhere, and the schema-level
# `_CONFIG_KEY` cannot be rewritten — leaving post-publication validation
# comparing a v1 stamp against a v2 fingerprint forever (Codex review, #344).
_FINGERPRINT_KEY = b"stel.config_fingerprint"
# The digest the stored rows were fingerprinted under (issue #495). Absent on
# a collection stamped before it existed, where that digest was necessarily
# the config fingerprint.
_ROW_FINGERPRINT_KEY = b"stel.row_fingerprint"
# Rows per batch when seeding a generation from another collection. Chosen so
# one batch of 768-dim float32 vectors is tens of MB rather than gigabytes;
# the point is bounded residency, not throughput (issue #495).
_SEED_BATCH_ROWS = 10_000
# Recorded on published search indexes and compared on read to invalidate
# state when the store implementation changes.
_IMPLEMENTATION_IDENTITY_PREFIX = "dbt_ml.retrieval.lancedb:v1"

# Object-store schemes LanceDB connects to natively (Rust object_store cores).
# A `path` carrying one of these bypasses local-filesystem resolution and the
# local mkdir, and flows straight into `lancedb.connect()`.
_CLOUD_SCHEMES = ("s3", "s3a", "gs", "gcs", "az", "abfs", "abfss")
# Aliases that address the same physical object store are folded to one
# canonical scheme so the store identity and the publisher lock treat, e.g.,
# `s3://b/p` and `s3a://b/p` as the same target rather than two.
_SCHEME_ALIASES = {"s3a": "s3", "gcs": "gs", "abfss": "abfs"}
_CLOUD_URI_RE = re.compile(rf"^({'|'.join(_CLOUD_SCHEMES)})://", re.IGNORECASE)
# scheme://authority[/path] — the authority (bucket/container/account) must be
# present. Rejects `s3://` and `gs:///prefix` at parse time instead of failing
# late inside lancedb.connect() after a publication lease is already held.
_CLOUD_URI_AUTHORITY_RE = re.compile(
    rf"^({'|'.join(_CLOUD_SCHEMES)})://([^/]+)(/.*)?$", re.IGNORECASE
)
# storage_options is non-secret routing only; keys that name a credential must
# use storage_options_env (a reference) instead of being pasted as a value.
_SECRET_OPTION_KEY_RE = re.compile(
    r"(secret|token|password|access_key|account_key|sas|credential|api_key)",
    re.IGNORECASE,
)


class LanceDBConfig(RetrievalStoreConfig):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    type: Literal["lancedb"] = "lancedb"
    # A local filesystem path or a cloud object-store URI (s3://, gs://, az://).
    path: str
    # Non-secret routing passthrough for cloud backends (region, endpoint,
    # allow_http, …), forwarded to lancedb.connect(storage_options=...). These
    # values are part of the store's physical identity, so they are folded into
    # the safe descriptor; secrets must NOT go here (see storage_options_env).
    storage_options: dict[str, str] = Field(default_factory=dict)
    # Credential-bearing storage options as references: option-key -> the
    # environment-variable NAME holding the secret. The value is resolved only
    # at connect() (in __enter__), mirroring the repo's CredentialReference
    # contract; the reference is redacted from dumps and never enters the
    # identity, fingerprints, logs, or artifacts.
    storage_options_env: dict[str, CredentialReference] = Field(default_factory=dict)
    # Local directory hosting the publisher lock. Optional for local stores
    # (defaults to the data dir); for a cloud store it names a host-shared path
    # so the single-host publisher lock is actually shared across every
    # publisher on the host (see `_lock_dir`).
    publisher_lock_dir: str | None = None
    collection_template: str = "{project}__{target}__{collection}"
    timeout_seconds: float = Field(default=30.0, gt=0, le=3600)
    minimum_consistency: Literal["strong"] = "strong"
    # Cache budgets handed to LanceDB's `Session` (issue #479). Unset means
    # "the role's default" (see `session_cache_budget`), not "LanceDB's":
    # LanceDB's own are ~6 GB index + 1 GB metadata, which is 7 GB of
    # unasked-for budget on a memory-limited publisher. These are execution
    # settings, deliberately absent from `routing_options()` and the safe
    # descriptor, so changing one cannot reclassify a published collection.
    index_cache_size_mb: int | None = Field(default=None, ge=1, le=1_048_576)
    metadata_cache_size_mb: int | None = Field(default=None, ge=1, le=1_048_576)
    # Bounded retry for each index build (issue #491). The build is the last
    # step of a publish that may have written rows for hours, and one
    # transient object-store error there used to discard the whole run.
    # Retrying is safe: `create_index(replace=True)` is idempotent, and a
    # partial build is re-entered by the `num_unindexed_rows` check. The
    # delay doubles after each failed attempt. Execution settings, not
    # identity: neither reaches the store descriptor.
    index_build_attempts: int = Field(default=3, ge=1, le=10)
    index_build_retry_seconds: float = Field(default=5.0, ge=0, le=600)

    @field_validator("path")
    @classmethod
    def _validate_path(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("path must be a non-empty local path or cloud URI")
        if _CLOUD_URI_RE.match(value) and not _CLOUD_URI_AUTHORITY_RE.match(value):
            raise ValueError(
                "cloud object-store path must include a bucket/container, e.g. "
                "gs://bucket/prefix or s3://bucket/prefix"
            )
        return value

    @field_validator("storage_options")
    @classmethod
    def _validate_storage_options(cls, value: dict[str, str]) -> dict[str, str]:
        secret_keys = sorted(k for k in value if _SECRET_OPTION_KEY_RE.search(k))
        if secret_keys:
            raise ValueError(
                "storage_options is for non-secret routing only; move credential "
                f"option(s) {secret_keys} to storage_options_env as an "
                "environment-variable reference"
            )
        return value

    @field_validator("collection_template")
    @classmethod
    def _validate_template(cls, value: str) -> str:
        if not value or any(separator in value for separator in ("/", "\\", "..")):
            raise ValueError("collection_template must be a safe collection name template")
        try:
            rendered = value.format(project="project", target="target", collection="collection")
        except (KeyError, ValueError):
            raise ValueError(
                "collection_template may use only {project}, {target}, and {collection}"
            ) from None
        if not _COLLECTION_RE.fullmatch(rendered):
            raise ValueError("collection_template renders an invalid collection name")
        return value

    @property
    def is_cloud_uri(self) -> bool:
        return bool(_CLOUD_URI_RE.match(self.path))

    def connect_target(self) -> str:
        """The string handed to `lancedb.connect()`."""
        return self.path

    def local_data_path(self) -> Path:
        """Filesystem path for local-disk operations (mkdir, co-located lock).

        Only meaningful for a local store; callers must gate on
        :attr:`is_cloud_uri` first."""
        if self.is_cloud_uri:
            raise RetrievalError(
                "LanceDB cloud-backed store has no local data path "
                "(code=lancedb_cloud_no_local_path)"
            )
        return Path(self.path)

    def identity_key(self) -> str:
        """Stable, secret-free identity for the store *location*. Local paths are
        posix-normalized so the fingerprint is stable across platforms and
        matches the pre-cloud-URI behavior; cloud URIs are canonicalized so
        equivalent scheme aliases (s3://≡s3a://, gs://≡gcs://) map to one target.
        Credentials are excluded; non-secret routing is added separately by the
        store's safe descriptor so a changed endpoint yields a distinct scope."""
        if self.is_cloud_uri:
            return _canonical_cloud_uri(self.path)
        return Path(self.path).as_posix()

    def routing_options(self) -> dict[str, str]:
        """Non-secret storage options in a stable order, for the store identity.
        Empty for local stores and cloud stores without routing, which keeps
        their descriptor fingerprint byte-identical to the pre-routing shape."""
        return {key: self.storage_options[key] for key in sorted(self.storage_options)}

    def absolutize(self, project_dir: Path) -> LanceDBConfig:
        if self.is_cloud_uri:
            return self
        path = Path(self.path)
        resolved = path if path.is_absolute() else project_dir / path
        return self.model_copy(update={"path": str(resolved.resolve())})


# `search.vector.index` -> the `lancedb.index` config class that builds it, and
# the `index_type` string LanceDB reports for it afterwards. The report name is
# what `ensure_indexes` compares against, so a type switch rebuilds rather than
# leaving the previous structure to answer under the new declaration.
_VECTOR_INDEX_CONFIGS = {
    "ivf_hnsw_flat": "HnswFlat",
    "ivf_hnsw_sq": "HnswSq",
    "ivf_pq": "IvfPq",
}
_VECTOR_INDEX_TYPES = {
    "ivf_hnsw_flat": "IvfHnswFlat",
    "ivf_hnsw_sq": "IvfHnswSq",
    "ivf_pq": "IvfPq",
}
# Product quantization trains 2**num_bits centroids per sub-vector, and LanceDB
# (8-bit codes) refuses a corpus with fewer rows than that: "Not enough rows to
# train PQ. Requires 256 rows" — at any dimension, measured on 0.34. Checked
# here, before the build, because that native message is sanitized away on the
# way out and the operator would otherwise see only `lancedb_index_failed`.
_PQ_MINIMUM_ROWS = 256

# Indirection so tests can observe the backoff without waiting it out.
_sleep = time.sleep


class _IndexBuildExhausted(Exception):
    """Every attempt at one index build failed; carries the last native error."""

    def __init__(self, attempts: int, error: Exception) -> None:
        super().__init__(f"index build failed after {attempts} attempts")
        self.attempts = attempts
        self.error = error


@register
class LanceDBStore(RetrievalStore):
    def __init__(
        self,
        config: RetrievalStoreConfig,
        *,
        project_name: str,
        target_name: str,
        alias: str,
        role: StoreRole,
    ) -> None:
        if not isinstance(config, LanceDBConfig):
            raise RetrievalError("LanceDB store received incompatible configuration")
        super().__init__(
            config,
            project_name=project_name,
            target_name=target_name,
            alias=alias,
            role=role,
        )
        self._config = config
        self._db: Any | None = None

    @classmethod
    def store_type(cls) -> str:
        return "lancedb"

    @classmethod
    def config_model(cls) -> type[RetrievalStoreConfig]:
        return LanceDBConfig

    @classmethod
    def implementation_identity(cls) -> str:
        version = optional_dependency_version("lancedb")
        return f"{_IMPLEMENTATION_IDENTITY_PREFIX}:lancedb-{version}"

    @classmethod
    def capabilities(cls) -> RetrievalCapabilities:
        return RetrievalCapabilities(
            features=frozenset(
                {
                    RetrievalFeature.EXACT_VECTOR_SEARCH,
                    RetrievalFeature.APPROXIMATE_VECTOR_SEARCH,
                    RetrievalFeature.METADATA_FILTERING,
                    RetrievalFeature.FULL_TEXT_SEARCH,
                    RetrievalFeature.KEYED_UPSERT,
                    RetrievalFeature.KEYED_DELETE,
                    RetrievalFeature.INDEX_READINESS,
                    RetrievalFeature.DURABLE_WRITE_ACK,
                    RetrievalFeature.ATOMIC_BATCH_MUTATION,
                    # DataFusion's `array_has_any` expresses set overlap
                    # against a list column (issue #397).
                    RetrievalFeature.ARRAY_CONTAINMENT_FILTERS,
                    RetrievalFeature.SINGLE_HOST_PUBLISHER_LOCK,
                    # Build under a private name, drop later; the swap itself
                    # is a warehouse row update, not a LanceDB operation
                    # (issue #355).
                    RetrievalFeature.PRIVATE_GENERATION_BUILD,
                    RetrievalFeature.COLLECTION_SEEDING,
                }
            ),
            distance_metrics=frozenset({"cosine", "euclidean", "dot"}),
            consistency_modes=frozenset({"strong"}),
            max_batch_size=100_000,
            max_id_bytes=8192,
            max_dimensions=16_384,
        )

    def __enter__(self) -> Self:
        lancedb = import_optional_dependency(
            "lancedb", extra="lancedb", feature="LanceDB retrieval"
        )
        connect_kwargs: dict[str, Any] = {}
        if self._config.is_cloud_uri:
            # No local .mkdir(): the object store owns the namespace, and
            # Path.mkdir() has no concept of a bucket. lancedb.connect() creates
            # the prefix lazily on first write.
            storage_options = self._resolve_storage_options()
            if storage_options:
                connect_kwargs["storage_options"] = storage_options
        else:
            self._config.local_data_path().mkdir(parents=True, exist_ok=True)
        budget = session_cache_budget(self._config, self.role)
        if budget is not None:
            # Without a Session, LanceDB takes its own defaults (~6 GB index +
            # 1 GB metadata), which a container ceiling is invisible to
            # (issue #479, the #412 shape one store over).
            index_bytes, metadata_bytes = budget
            connect_kwargs["session"] = lancedb.Session(
                index_cache_size_bytes=index_bytes,
                metadata_cache_size_bytes=metadata_bytes,
            )
        failure: RetrievalError | None = None
        try:
            self._db = lancedb.connect(
                self._config.connect_target(), **connect_kwargs
            )
        except Exception as error:
            failure = _operation_failed("connect", "lancedb_connect_failed", error)
        if failure is not None:
            raise failure
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._db = None

    def _resolve_storage_options(self) -> dict[str, str]:
        """Assemble the storage_options passed to lancedb.connect(): non-secret
        routing verbatim, plus credential references resolved from the
        environment here at the SDK boundary — never earlier, never stored."""
        resolved = dict(self._config.storage_options)
        for option_key, reference in self._config.storage_options_env.items():
            if not reference.is_available():
                raise RetrievalError(
                    f"Storage credential for '{option_key}' is not set "
                    "(code=lancedb_storage_credential_missing)"
                )
            resolved[option_key] = reference.resolve().reveal()
        return resolved

    def _connection(self) -> Any:
        if self._db is None:
            raise RetrievalError("LanceDB store is not open")
        return self._db

    def _open_owned_table(self, name: str) -> Any:
        if not _COLLECTION_RE.fullmatch(name):
            raise RetrievalError("LanceDB collection name is invalid")
        table = self._connection().open_table(name)
        if (table.schema.metadata or {}).get(_OWNER_KEY) != _OWNER_VALUE:
            raise RetrievalError(
                "LanceDB collection is not owned by stel "
                "(code=lancedb_external_collection)"
            )
        return table

    def publisher_fence(self, collection: str) -> AbstractContextManager[None]:
        """OS-enforced exclusive publisher lock, valid on one host only.

        For a local store the lock file lives next to the LanceDB data, so every
        stel publisher on the host contends on the same inode/handle. A
        cloud-backed store has no local data directory, so the lock lives in a
        host-stable local directory keyed by the store URI (see `_lock_dir`) —
        this keeps the same single-host guarantee and no more. This is the
        documented boundary from #152: the lock cannot fence a publisher on
        another machine, whether they share a network filesystem or the same
        object-store prefix. Cross-host publisher exclusion would need
        provider-enforced fencing, which this store does not advertise."""
        if not _COLLECTION_RE.fullmatch(collection):
            raise RetrievalError("LanceDB collection name is invalid")
        lock_dir = self._lock_dir()
        lock_dir.mkdir(parents=True, exist_ok=True)
        return PublisherLock(
            lock_dir / f"{collection}.stel-publisher.lock",
            store_type=self.store_type(),
        )

    def _lock_dir(self) -> Path:
        """Directory holding this store's publisher lock files.

        Local store, no override: co-located with the data (unchanged). Anything
        else is keyed by the store URI so publishers of the *same* store share a
        directory while different stores don't collide.

        For a cloud store the default base is deliberately NOT
        `tempfile.gettempdir()`: TMPDIR is per-process/per-container, so two
        publishers on one host could resolve to different directories and their
        `flock`s would never contend — silently voiding the single-host lock.
        The default is a fixed host-machine location instead. Publishers running
        in isolated mount namespaces (separate containers) don't share any local
        path automatically; those deployments must set `publisher_lock_dir` to a
        volume mounted into every publisher, which is the only way a local file
        lock can honestly back the single-host capability there.

        The digest is the store's physical-target identity (canonical URI plus
        non-secret routing), so scheme aliases (s3://≡s3a://) and a changed
        endpoint route to the correct — shared or distinct — lock, matching the
        safe descriptor exactly."""
        override = self._config.publisher_lock_dir
        if override is None and not self._config.is_cloud_uri:
            return self._config.local_data_path()
        base = (
            Path(override)
            if override is not None
            else default_host_lock_base(self.store_type())
        )
        digest = self.safe_descriptor().safe_target_identity[:32]
        return base / digest

    def safe_descriptor(self) -> SafeRetrievalTarget:
        payload: dict[str, Any] = {
            "store_type": self.store_type(),
            "path": self._config.identity_key(),
        }
        # Add routing only when present so local stores (and cloud stores without
        # routing) keep the exact pre-routing fingerprint — existing state scopes
        # stay valid. A changed endpoint/region now yields a distinct identity,
        # so state from one physical store can't be misread against another.
        routing = self._config.routing_options()
        if routing:
            payload["routing"] = routing
        identity = canonical_fingerprint(
            payload, domain="dbt-ml-safe-retrieval-target"
        )
        return SafeRetrievalTarget(self.store_type(), identity)

    def state_descriptor(self, collection: str) -> StateRetrievalTarget:
        return StateRetrievalTarget(
            self.store_type(),
            self.safe_descriptor().safe_target_identity,
            self.physical_collection(collection),
            collection,
        )

    def physical_collection(
        self, logical_name: str, *, generation: str | None = None
    ) -> str:
        values = {
            "project": _identifier_piece(self.project_name),
            "target": _identifier_piece(self.target_name),
            "collection": _identifier_piece(logical_name),
        }
        physical = reject_generation_shaped_collection_name(
            self._config.collection_template.format(**values)
        )
        if generation is not None:
            # Suffixed only when a generation is asked for, so the unsuffixed
            # name keeps addressing collections published before #355.
            physical = f"{physical}__g{validate_generation_token(generation)}"
        if not _COLLECTION_RE.fullmatch(physical):
            raise RetrievalError("Resolved LanceDB collection name is invalid")
        return physical

    def list_collections(self) -> tuple[str, ...]:
        db = self._connection()
        failure: RetrievalError | None = None
        try:
            return tuple(sorted(db.list_tables().tables))
        except Exception as error:
            failure = _operation_failed("list", "lancedb_list_failed", error)
        raise failure

    def drop_collection(self, name: str) -> bool:
        if not _COLLECTION_RE.fullmatch(name):
            raise RetrievalError("LanceDB collection name is invalid")
        db = self._connection()
        failure: RetrievalError | None = None
        try:
            if name not in db.list_tables().tables:
                return False
            # Opening through the owned-table path first refuses to drop a
            # collection stel did not create, so a mistyped or externally
            # managed name cannot be destroyed here.
            self._open_owned_table(name)
            db.drop_table(name)
            return True
        except RetrievalError:
            raise
        except Exception as error:
            failure = _operation_failed("drop", "lancedb_drop_failed", error)
        raise failure

    def inspect_collection(self, name: str) -> CollectionMetadata | None:
        db = self._connection()
        failure: RetrievalError | None = None
        try:
            if name not in db.list_tables().tables:
                return None
            table = self._open_owned_table(name)
            schema = table.schema
            metadata = schema.metadata or {}
            generation = canonical_fingerprint(
                {
                    "name": name,
                    "version": getattr(table, "version", None),
                    "rows": table.count_rows(),
                },
                domain="dbt-ml-lancedb-generation",
            )
            return CollectionMetadata(
                physical_name=name,
                config_fingerprint=_read_fingerprint(schema, metadata),
                descriptor=_read_field_value(schema, _DESCRIPTOR_KEY),
                physical_generation=generation,
                row_count=int(table.count_rows()),
                schema=schema,
                row_fingerprint=_read_field_value(schema, _ROW_FINGERPRINT_KEY),
            )
        except RetrievalError:
            raise
        except Exception as error:
            failure = _operation_failed("inspect", "lancedb_inspect_failed", error)
        raise failure

    def seed_collection(self, spec: CollectionSpec, *, source: str) -> int:
        """Copy `source`'s rows into this generation, one batch at a time.

        Streamed rather than materialized: the corpus this exists for is ~11GB
        of float32 vectors, and pulling it into one Arrow table to move it
        would trade the 4.2h warehouse round trip (issue #495) for the OOM of
        issue #473. `_SEED_BATCH_ROWS` bounds the residency instead.

        Columns are projected onto the target schema by name. The two
        collections share a contract, but a positional copy that transposed two
        same-typed columns would corrupt the index silently, and by the time it
        surfaced the generation would already be live.
        """
        failure: RetrievalError | None = None
        seeded = 0
        try:
            origin = self._open_owned_table(source)
            target = self._open_owned_table(spec.physical_name)
            names = [field.name for field in spec.arrow_schema]
            for batch in origin.search(None).to_batches(_SEED_BATCH_ROWS):
                projected = pa.Table.from_batches([batch], schema=batch.schema)
                target.add(projected.select(names))
                seeded += batch.num_rows
        except Exception as error:
            failure = _operation_failed("seed collection", "lancedb_seed_failed", error)
        if failure is not None:
            raise failure
        return seeded

    def create_collection(self, spec: CollectionSpec) -> CollectionMetadata:
        db = self._connection()
        metadata = dict(spec.arrow_schema.metadata or {})
        metadata.update(
            {
                _OWNER_KEY: _OWNER_VALUE,
                _CONTRACT_KEY: b"1",
                _CONFIG_KEY: spec.config_fingerprint.encode(),
            }
        )
        schema = _with_descriptor(
            spec.arrow_schema.with_metadata(metadata), spec
        )
        failure: RetrievalError | None = None
        try:
            db.create_table(spec.physical_name, schema=schema)
        except Exception as error:
            failure = _operation_failed("create collection", "lancedb_create_failed", error)
        if failure is not None:
            raise failure
        created = self.inspect_collection(spec.physical_name)
        if created is None:
            raise RetrievalError("LanceDB collection creation was not observable")
        return created

    def restamp_collection(self, spec: CollectionSpec) -> None:
        table = self._open_owned_table(spec.physical_name)
        failure: RetrievalError | None = None
        try:
            table.update_field_metadata(
                {
                    "path": spec.id_field,
                    "metadata": {
                        _DESCRIPTOR_KEY.decode(): spec.descriptor,
                        _FINGERPRINT_KEY.decode(): spec.config_fingerprint,
                        _ROW_FINGERPRINT_KEY.decode(): spec.row_fingerprint,
                    },
                }
            )
        except Exception as error:
            failure = _operation_failed("restamp", "lancedb_restamp_failed", error)
        if failure is not None:
            raise failure
        written = self.inspect_collection(spec.physical_name)
        if (
            written is None
            or written.descriptor != spec.descriptor
            or written.config_fingerprint != spec.config_fingerprint
        ):
            raise RetrievalError("LanceDB collection re-stamp was not observable")

    def upsert(
        self,
        collection: str,
        rows: Sequence[IndexedRow],
        *,
        id_field: str,
        mutation_digest: str,
    ) -> MutationReceipt:
        if not rows:
            return MutationReceipt(mutation_digest, True, ())
        failure: RetrievalError | None = None
        try:
            table = self._open_owned_table(collection)
            payload = pa.Table.from_pylist([dict(row.values) for row in rows], schema=table.schema)
            (
                table.merge_insert(id_field)
                .when_matched_update_all()
                .when_not_matched_insert_all()
                .execute(payload)
            )
            if table.count_rows(_id_filter(id_field, [row.record_id for row in rows])) != len(
                rows
            ):
                raise RetrievalError("LanceDB upsert acknowledgement was incomplete")
        except RetrievalError:
            raise
        except Exception as error:
            failure = _operation_failed("upsert", "lancedb_upsert_failed", error)
        if failure is not None:
            raise failure
        return MutationReceipt(
            mutation_digest,
            True,
            tuple(MutationOutcome("applied") for _ in rows),
        )

    def append(
        self,
        collection: str,
        rows: Sequence[IndexedRow],
        *,
        id_field: str,
        mutation_digest: str,
    ) -> MutationReceipt:
        if not rows:
            return MutationReceipt(mutation_digest, True, ())
        failure: RetrievalError | None = None
        try:
            table = self._open_owned_table(collection)
            expected_count = table.count_rows() + len(rows)
            payload = pa.Table.from_pylist([dict(row.values) for row in rows], schema=table.schema)
            table.add(payload)
            if table.count_rows() != expected_count:
                raise RetrievalError("LanceDB append acknowledgement was incomplete")
        except RetrievalError:
            raise
        except Exception as error:
            failure = _operation_failed("append", "lancedb_append_failed", error)
        if failure is not None:
            raise failure
        return MutationReceipt(
            mutation_digest, True, tuple(MutationOutcome("applied") for _ in rows)
        )

    def delete(
        self,
        collection: str,
        record_ids: Sequence[str],
        *,
        id_field: str,
        mutation_digest: str,
    ) -> MutationReceipt:
        if not record_ids:
            return MutationReceipt(mutation_digest, True, ())
        quoted = ", ".join(_sql_string(value) for value in record_ids)
        failure: RetrievalError | None = None
        try:
            table = self._open_owned_table(collection)
            if not _COLLECTION_RE.fullmatch(id_field):
                raise RetrievalError("LanceDB ID field is invalid")
            table.delete(f"{id_field} IN ({quoted})")
            if table.count_rows(_id_filter(id_field, record_ids)) != 0:
                raise RetrievalError("LanceDB delete acknowledgement was incomplete")
        except RetrievalError:
            raise
        except Exception as error:
            failure = _operation_failed("delete", "lancedb_delete_failed", error)
        if failure is not None:
            raise failure
        return MutationReceipt(
            mutation_digest,
            True,
            tuple(MutationOutcome("deleted") for _ in record_ids),
        )

    def _build_index(
        self, step: str, build: Callable[..., Any], *, existing: bool
    ) -> None:
        """Run one index build, retrying native failures with backoff (#491).

        Only native errors are retried; a `RetrievalError` is a deliberate
        refusal. Every retry passes `replace=True`: a first attempt that failed
        after LanceDB had already committed the index would otherwise be
        refused as a duplicate. The retry warning carries only the step and the
        native exception's type (issue #490); the native text goes to DEBUG.
        """
        attempts = self._config.index_build_attempts
        delay = self._config.index_build_retry_seconds
        for attempt in range(1, attempts + 1):
            try:
                build(replace=existing or attempt > 1)
                return
            except RetrievalError:
                raise
            except Exception as error:
                if attempt == attempts:
                    if attempts == 1:
                        raise
                    raise _IndexBuildExhausted(attempts, error) from None
                log.warning(
                    "LanceDB index build on %s failed [%s]; retrying in %.1fs "
                    "(attempt %d of %d)",
                    step,
                    type(error).__name__,
                    delay,
                    attempt,
                    attempts,
                )
                log.debug("LanceDB index build retry cause", exc_info=error)
            _sleep(delay)
            delay *= 2

    def ensure_indexes(self, spec: CollectionSpec) -> CollectionMetadata:
        failure: RetrievalError | None = None
        # Tracks which index was being built when a native error surfaced.
        # This is the last step of a multi-hour publish and covers three index
        # kinds; a failure that cannot say which one it was on is undiagnosable
        # after the fact (issue #490).
        step = "collection open"
        try:
            table = self._open_owned_table(spec.physical_name)
            if table.count_rows() == 0:
                metadata = self.inspect_collection(spec.physical_name)
                if metadata is None:
                    raise RetrievalError("LanceDB collection disappeared")
                return metadata
            index_module = import_optional_dependency(
                "lancedb.index", extra="lancedb", feature="LanceDB retrieval"
            )
            step = "index listing"
            indexes = list(table.list_indices())
            for field in spec.scalar_index_fields:
                current = next(
                    (
                        index
                        for index in indexes
                        if index.columns == [field] and index.index_type == "BTree"
                    ),
                    None,
                )
                if current is None or current.num_unindexed_rows:
                    step = f"BTree index for '{field}'"
                    self._build_index(
                        step,
                        partial(
                            table.create_index,
                            field,
                            config=index_module.BTree(),
                            wait_timeout=timedelta(seconds=self._config.timeout_seconds),
                        ),
                        existing=current is not None,
                    )
            for field in spec.full_text_fields:
                current = next(
                    (
                        index
                        for index in indexes
                        if index.columns == [field] and index.index_type == "FTS"
                    ),
                    None,
                )
                if current is None or current.num_unindexed_rows:
                    step = f"FTS index for '{field}'"
                    self._build_index(
                        step,
                        partial(
                            table.create_index,
                            field,
                            config=index_module.FTS(),
                            wait_timeout=timedelta(seconds=self._config.timeout_seconds),
                        ),
                        existing=current is not None,
                    )
            if spec.vector_field is not None:
                current = next(
                    (
                        index
                        for index in indexes
                        if index.columns == [spec.vector_field]
                        and index.index_type in _VECTOR_INDEX_TYPES.values()
                    ),
                    None,
                )
                if spec.vector_search == "approximate":
                    chosen = spec.vector_index
                    if chosen not in _VECTOR_INDEX_TYPES:
                        # The spec is built from a validated config, which
                        # always names a type under `approximate`; reaching
                        # here is a caller bug, not an operator error.
                        raise RetrievalError(
                            "LanceDB approximate search needs a declared vector "
                            f"index type, got {chosen!r} (code=lancedb_index_type_missing)"
                        )
                    wanted = _VECTOR_INDEX_TYPES[chosen]
                    if (
                        current is None
                        or current.index_type != wanted
                        or current.num_unindexed_rows
                    ):
                        # Only where a build is about to happen: a collection
                        # that shrank below the floor after its PQ index was
                        # trained still has a valid index and nothing to train.
                        if chosen == "ivf_pq" and table.count_rows() < _PQ_MINIMUM_ROWS:
                            raise RetrievalError(
                                "LanceDB cannot train an ivf_pq index on fewer than "
                                f"{_PQ_MINIMUM_ROWS} rows; declare index: ivf_hnsw_sq "
                                "or ivf_hnsw_flat until the collection is larger "
                                "(code=lancedb_pq_corpus_too_small)"
                            )
                        metric = (
                            "l2"
                            if spec.distance_metric == "euclidean"
                            else spec.distance_metric
                        )
                        config_class = getattr(index_module, _VECTOR_INDEX_CONFIGS[chosen])
                        step = f"{wanted} vector index for '{spec.vector_field}'"
                        self._build_index(
                            step,
                            partial(
                                table.create_index,
                                spec.vector_field,
                                config=config_class(distance_type=metric),
                                # Inert on a native table: the build is
                                # synchronous and this only bounds LanceDB
                                # Cloud's async indexing (probed on 0.34, #474
                                # review). Kept so a remote store gets a bound
                                # rather than none.
                                wait_timeout=timedelta(
                                    seconds=self._config.timeout_seconds
                                ),
                            ),
                            existing=current is not None,
                        )
                elif current is not None:
                    # `exact` is implemented by the *absence* of an ANN index --
                    # LanceDB uses one whenever it exists. Leaving a stale index
                    # behind after a switch back would silently keep serving
                    # approximate results under a config that promises exact
                    # ones (issue #461).
                    step = f"stale {current.index_type} vector index drop for '{spec.vector_field}'"
                    table.drop_index(current.name)
        except RetrievalError:
            # A deliberate refusal keeps its own code, as in `upsert`/`delete`.
            raise
        except _IndexBuildExhausted as exhausted:
            failure = _operation_failed(
                "index creation",
                "lancedb_index_failed",
                exhausted.error,
                step=f"{step} after {exhausted.attempts} attempts",
            )
        except Exception as error:
            failure = _operation_failed(
                "index creation", "lancedb_index_failed", error, step=step
            )
        if failure is not None:
            raise failure
        metadata = self.inspect_collection(spec.physical_name)
        if metadata is None:
            raise RetrievalError("LanceDB collection disappeared during index creation")
        return metadata

    def vector_search(
        self,
        collection: str,
        vector: Sequence[float],
        *,
        vector_field: str,
        limit: int,
        columns: Sequence[str] | None = None,
        predicates: Sequence[RetrievalPredicate] = (),
    ) -> pa.Table:
        failure: RetrievalError | None = None
        try:
            table = self._open_owned_table(collection)
            _validate_query_projection(columns)
            _validate_query_limit(limit)
            field = table.schema.field(vector_field)
            if not pa.types.is_fixed_size_list(field.type):
                raise RetrievalError("LanceDB vector search field is invalid")
            if (
                len(vector) != field.type.list_size
                or any(isinstance(item, bool) for item in vector)
                or any(not isfinite(float(item)) for item in vector)
            ):
                raise RetrievalError("LanceDB vector query is invalid")
            query = table.search(list(vector), vector_column_name=vector_field)
            where = _compile_predicates(predicates)
            if where is not None:
                query = query.where(where, prefilter=True)
            if columns is not None:
                query = query.select(_include_score_column(columns, "_distance"))
            result = query.limit(limit).to_arrow()
            if not isinstance(result, pa.Table):
                raise RetrievalError("LanceDB vector search result is invalid")
            return result
        except RetrievalError:
            raise
        except Exception as error:
            failure = _operation_failed(
                "vector search", "lancedb_vector_search_failed", error
            )
        raise failure

    def text_search(
        self,
        collection: str,
        query: str,
        *,
        text_field: str,
        limit: int,
        columns: Sequence[str] | None = None,
        predicates: Sequence[RetrievalPredicate] = (),
    ) -> pa.Table:
        failure: RetrievalError | None = None
        try:
            table = self._open_owned_table(collection)
            _validate_query_projection(columns)
            _validate_query_limit(limit)
            if not query or len(query.encode()) > 32_768:
                raise RetrievalError("LanceDB text query is invalid")
            if not _COLLECTION_RE.fullmatch(text_field):
                raise RetrievalError("LanceDB text search field is invalid")
            builder = table.search(query, query_type="fts", fts_columns=[text_field])
            where = _compile_predicates(predicates)
            if where is not None:
                builder = builder.where(where, prefilter=True)
            if columns is not None:
                builder = builder.select(_include_score_column(columns, "_score"))
            result = builder.limit(limit).to_arrow()
            if not isinstance(result, pa.Table):
                raise RetrievalError("LanceDB text search result is invalid")
            return result
        except RetrievalError:
            raise
        except Exception as error:
            failure = _operation_failed("text search", "lancedb_text_search_failed", error)
        raise failure


def _operation_failed(
    operation: str,
    code: str,
    error: Exception,
    *,
    step: str | None = None,
) -> RetrievalError:
    """Build the artifact-safe failure for a native LanceDB error.

    The message names the operation, the step it was on, and the native
    exception's type: enough to tell a transient object-store error from a
    resource limit or a configuration problem (issue #490). The native text is
    never copied. LanceDB quotes object-store URIs and response bodies
    verbatim, and this message reaches run_results.json and the CLI. The cause
    chain carries the type alone, with no traceback, and the full exception
    goes only to the DEBUG log, which `--verbose` never enables.

    Callers raise the result *outside* their except block so the native
    exception is not retained as `__context__` either.
    """
    on = f" on {step}" if step else ""
    log.debug("LanceDB operation %r failed%s", operation, on, exc_info=error)
    failure = RetrievalError(
        f"LanceDB operation '{operation}' failed{on} "
        f"[{type(error).__name__}] (code={code})"
    )
    failure.__cause__ = sanitized_retrieval_cause(error)
    return failure


def _with_descriptor(schema: pa.Schema, spec: CollectionSpec) -> pa.Schema:
    """Attach the semantic descriptor to the collection's id field."""
    index = schema.get_field_index(spec.id_field)
    if index < 0:
        raise RetrievalError("LanceDB collection schema is missing its id field")
    field = schema.field(index)
    metadata = dict(field.metadata or {})
    metadata[_DESCRIPTOR_KEY] = spec.descriptor.encode()
    metadata[_FINGERPRINT_KEY] = spec.config_fingerprint.encode()
    metadata[_ROW_FINGERPRINT_KEY] = spec.row_fingerprint.encode()
    return schema.set(index, field.with_metadata(metadata))


def _read_field_value(schema: pa.Schema, key: bytes) -> str | None:
    """A stel stamp held as field metadata, or None if absent."""
    for field in schema:
        value = (field.metadata or {}).get(key)
        if value:
            return value.decode()
    return None


def _read_fingerprint(schema: pa.Schema, metadata: dict[bytes, bytes]) -> str | None:
    """The collection's configuration digest.

    Field metadata wins: a re-stamped collection carries its current digest
    there, while the schema-level key is frozen at whatever it held when the
    table was created and cannot be rewritten.
    """
    field_value = _read_field_value(schema, _FINGERPRINT_KEY)
    if field_value:
        return field_value
    legacy = metadata.get(_CONFIG_KEY)
    return legacy.decode() if legacy else None


def _identifier_piece(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_]", "_", value)
    if not normalized or normalized[0].isdigit():
        normalized = f"_{normalized}"
    return normalized[:128]


def _canonical_cloud_uri(uri: str) -> str:
    """Canonicalize a cloud URI so equivalent spellings map to one physical
    target: lowercase the scheme, fold provider aliases (s3a→s3, gcs→gs,
    abfss→abfs), and drop trailing slashes. The bucket/prefix is left
    case-sensitive — only the scheme is case-insensitive."""
    scheme, separator, rest = uri.partition("://")
    if not separator:
        return uri
    canonical_scheme = scheme.lower()
    canonical_scheme = _SCHEME_ALIASES.get(canonical_scheme, canonical_scheme)
    return f"{canonical_scheme}://{rest.rstrip('/')}"


def _sql_string(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise RetrievalError("LanceDB record IDs must be non-empty strings")
    return "'" + value.replace("'", "''") + "'"


def _id_filter(id_field: str, record_ids: Sequence[str]) -> str:
    if not _COLLECTION_RE.fullmatch(id_field):
        raise RetrievalError("LanceDB ID field is invalid")
    return f"{id_field} IN ({', '.join(_sql_string(value) for value in record_ids)})"


def _validate_query_limit(limit: int) -> None:
    if isinstance(limit, bool) or not 1 <= limit <= 1000:
        raise RetrievalError("LanceDB query limit must be between 1 and 1000")


def _validate_query_projection(columns: Sequence[str] | None) -> None:
    if columns is not None and (
        not columns or any(not _COLLECTION_RE.fullmatch(column) for column in columns)
    ):
        raise RetrievalError("LanceDB query projection is invalid")


def _include_score_column(columns: Sequence[str], score_column: str) -> list[str]:
    projection = list(columns)
    if score_column not in projection:
        projection.append(score_column)
    return projection


def _compile_predicates(predicates: Sequence[RetrievalPredicate]) -> str | None:
    if not predicates:
        return None
    operators = {
        RetrievalPredicateOperator.EQUAL: "=",
        RetrievalPredicateOperator.NOT_EQUAL: "!=",
        RetrievalPredicateOperator.LESS_THAN: "<",
        RetrievalPredicateOperator.LESS_THAN_OR_EQUAL: "<=",
        RetrievalPredicateOperator.GREATER_THAN: ">",
        RetrievalPredicateOperator.GREATER_THAN_OR_EQUAL: ">=",
    }
    clauses: list[str] = []
    for predicate in predicates:
        if not _COLLECTION_RE.fullmatch(predicate.field):
            raise RetrievalError("Retrieval predicate field is invalid")
        field = predicate.field
        if predicate.operator == RetrievalPredicateOperator.IN:
            assert isinstance(predicate.value, tuple)
            values = ", ".join(_sql_literal(value) for value in predicate.value)
            clauses.append(f"{field} IN ({values})")
        elif predicate.operator == RetrievalPredicateOperator.ARRAY_CONTAINS_ANY:
            assert isinstance(predicate.value, tuple)
            values = ", ".join(_sql_literal(value) for value in predicate.value)
            clauses.append(f"array_has_any({field}, [{values}])")
        else:
            assert not isinstance(predicate.value, tuple)
            clauses.append(
                f"{field} {operators[predicate.operator]} {_sql_literal(predicate.value)}"
            )
    return " AND ".join(clauses)


def _sql_literal(value: Any) -> str:
    if isinstance(value, str):
        return _sql_string(value)
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    # Temporal values need a *typed* literal, not a quoted string (issue
    # #337). A bare `'2020-01-01'` is Utf8 to the query engine, which will not
    # compare it to a date32/timestamp column — the filter is rejected at
    # query time, after config validation and index build have both passed, so
    # the failure lands on the querying agent rather than the author.
    # `datetime` is a `date` subclass, so it is matched first.
    if isinstance(value, datetime):
        return f"TIMESTAMP {_sql_string(value.isoformat())}"
    if isinstance(value, date):
        return f"DATE {_sql_string(value.isoformat())}"
    if isinstance(value, int | float):
        return str(value)
    raise RetrievalError("Retrieval predicate contains an unsupported value")
