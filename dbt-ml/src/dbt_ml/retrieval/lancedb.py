from __future__ import annotations

import os
import re
import sys
from collections.abc import Sequence
from contextlib import AbstractContextManager
from datetime import date, datetime, timedelta
from math import isfinite
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, Self

import pyarrow as pa
from pydantic import ConfigDict, Field, field_validator

from ..credentials import CredentialReference
from ..hashing import canonical_fingerprint
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
)
from .registry import register

_COLLECTION_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")

# Arrow schema metadata written into every collection dbt-ml creates. These
# names live in existing LanceDB collections, so they are data, not code.
# `_OWNER_VALUE` in particular is an adoption gate, not a label: a collection
# whose owner metadata does not match exactly is refused as external, so
# changing this value makes every collection already published unreadable.
_OWNER_KEY = b"dbt_ml.owner"
_OWNER_VALUE = b"dbt-ml"
_CONTRACT_KEY = b"dbt_ml.record_contract"
_CONFIG_KEY = b"dbt_ml.config_fingerprint"
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


@register
class LanceDBStore(RetrievalStore):
    def __init__(
        self,
        config: RetrievalStoreConfig,
        *,
        project_name: str,
        target_name: str,
        alias: str,
    ) -> None:
        if not isinstance(config, LanceDBConfig):
            raise RetrievalError("LanceDB store received incompatible configuration")
        super().__init__(
            config,
            project_name=project_name,
            target_name=target_name,
            alias=alias,
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
                    RetrievalFeature.SINGLE_HOST_PUBLISHER_LOCK,
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
        try:
            self._db = lancedb.connect(
                self._config.connect_target(), **connect_kwargs
            )
        except Exception:
            raise RetrievalError(
                "LanceDB operation 'connect' failed (code=lancedb_connect_failed)"
            ) from None
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
                "LanceDB collection is not owned by dbt-ml "
                "(code=lancedb_external_collection)"
            )
        return table

    def publisher_fence(self, collection: str) -> AbstractContextManager[None]:
        """OS-enforced exclusive publisher lock, valid on one host only.

        For a local store the lock file lives next to the LanceDB data, so every
        dbt-ml publisher on the host contends on the same inode/handle. A
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
        return _PublisherLock(lock_dir / f"{collection}.dbt-ml-publisher.lock")

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
        base = Path(override) if override is not None else _default_host_lock_base()
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
        )

    def physical_collection(self, logical_name: str) -> str:
        values = {
            "project": _identifier_piece(self.project_name),
            "target": _identifier_piece(self.target_name),
            "collection": _identifier_piece(logical_name),
        }
        physical = self._config.collection_template.format(**values)
        if not _COLLECTION_RE.fullmatch(physical):
            raise RetrievalError("Resolved LanceDB collection name is invalid")
        return physical

    def inspect_collection(self, name: str) -> CollectionMetadata | None:
        db = self._connection()
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
            config = metadata.get(_CONFIG_KEY)
            return CollectionMetadata(
                physical_name=name,
                config_fingerprint=config.decode() if config else None,
                physical_generation=generation,
                row_count=int(table.count_rows()),
                schema=schema,
            )
        except RetrievalError:
            raise
        except Exception:
            raise RetrievalError(
                "LanceDB operation 'inspect' failed (code=lancedb_inspect_failed)"
            ) from None

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
        schema = spec.arrow_schema.with_metadata(metadata)
        try:
            db.create_table(spec.physical_name, schema=schema)
        except Exception:
            raise RetrievalError(
                "LanceDB operation 'create collection' failed (code=lancedb_create_failed)"
            ) from None
        created = self.inspect_collection(spec.physical_name)
        if created is None:
            raise RetrievalError("LanceDB collection creation was not observable")
        return created

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
        except Exception:
            raise RetrievalError(
                "LanceDB operation 'upsert' failed (code=lancedb_upsert_failed)"
            ) from None
        return MutationReceipt(
            mutation_digest,
            True,
            tuple(MutationOutcome("applied") for _ in rows),
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
        try:
            table = self._open_owned_table(collection)
            if not _COLLECTION_RE.fullmatch(id_field):
                raise RetrievalError("LanceDB ID field is invalid")
            table.delete(f"{id_field} IN ({quoted})")
            if table.count_rows(_id_filter(id_field, record_ids)) != 0:
                raise RetrievalError("LanceDB delete acknowledgement was incomplete")
        except RetrievalError:
            raise
        except Exception:
            raise RetrievalError(
                "LanceDB operation 'delete' failed (code=lancedb_delete_failed)"
            ) from None
        return MutationReceipt(
            mutation_digest,
            True,
            tuple(MutationOutcome("deleted") for _ in record_ids),
        )

    def ensure_indexes(self, spec: CollectionSpec) -> CollectionMetadata:
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
                    table.create_index(
                        field,
                        config=index_module.BTree(),
                        replace=current is not None,
                        wait_timeout=timedelta(seconds=self._config.timeout_seconds),
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
                    table.create_index(
                        field,
                        config=index_module.FTS(),
                        replace=current is not None,
                        wait_timeout=timedelta(seconds=self._config.timeout_seconds),
                    )
            if spec.vector_field is not None and spec.vector_search == "approximate":
                current = next(
                    (
                        index
                        for index in indexes
                        if index.columns == [spec.vector_field] and "Hnsw" in index.index_type
                    ),
                    None,
                )
                if current is None or current.num_unindexed_rows:
                    metric = "l2" if spec.distance_metric == "euclidean" else spec.distance_metric
                    table.create_index(
                        spec.vector_field,
                        config=index_module.HnswFlat(distance_type=metric),
                        replace=current is not None,
                        wait_timeout=timedelta(seconds=self._config.timeout_seconds),
                    )
        except Exception:
            raise RetrievalError(
                "LanceDB operation 'index creation' failed (code=lancedb_index_failed)"
            ) from None
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
        except Exception:
            raise RetrievalError(
                "LanceDB operation 'vector search' failed (code=lancedb_vector_search_failed)"
            ) from None

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
        except Exception:
            raise RetrievalError(
                "LanceDB operation 'text search' failed (code=lancedb_text_search_failed)"
            ) from None


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


def _default_host_lock_base() -> Path:
    """Fixed per-machine base for cloud-store publisher locks, independent of
    TMPDIR/TEMP so every publisher on the host resolves to the same directory
    (a `tempfile.gettempdir()` base would vary per process/container and let
    two publishers' locks silently miss each other). Publishers in isolated
    mount namespaces still need an explicit `publisher_lock_dir` on a shared
    volume."""
    if sys.platform == "win32":
        base = os.environ.get("PROGRAMDATA") or "C:\\ProgramData"
        return Path(base) / "dbt-ml" / "lancedb-locks"
    return Path("/var/tmp/dbt-ml-lancedb-locks")


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
    if isinstance(value, datetime | date):
        return _sql_string(value.isoformat())
    if isinstance(value, int | float):
        return str(value)
    raise RetrievalError("Retrieval predicate contains an unsupported value")


class _PublisherLock(AbstractContextManager[None]):
    """Non-blocking OS file lock excluding concurrent publisher processes."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._handle: Any | None = None

    def __enter__(self) -> None:
        handle = self._path.open("a+b")
        try:
            if sys.platform == "win32":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            handle.close()
            raise RetrievalError(
                "Another publisher holds the LanceDB collection lock "
                "(code=lancedb_publisher_lock_held); terminate it before "
                "recovering the serving scope"
            ) from None
        self._handle = handle
        return None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        handle = self._handle
        self._handle = None
        if handle is None:
            return
        try:
            if sys.platform == "win32":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
