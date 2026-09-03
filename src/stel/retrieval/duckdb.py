"""DuckDB-native retrieval store using the `vss` and `fts` extensions (#371).

Every other `RetrievalStore` target is a second system beside the warehouse.
When the warehouse adapter is already `duckdb`, that is an extra moving part
for no reason: DuckDB's `vss` extension serves HNSW-indexed vector search and
`fts` serves BM25 text search, in the same file as the canonical rows.

Three things about this store are worth knowing before reading the code,
because each is a place where DuckDB does not do what a remote vector store
does:

**HNSW persistence is experimental and opt-in.** DuckDB refuses to create an
HNSW index in a file-backed database unless
`hnsw_enable_experimental_persistence` is set, because that index is not
covered by the WAL: a crash can leave it inconsistent with the table. Rather
than set that flag silently, this store treats an approximate index as
something the operator asks for. Vector search is *correct* either way —
without the index it is an exact scan — so the cost of declining is latency,
not wrong answers, which is the right way round.

**Ownership lives in the table comment.** DuckDB tables carry no Arrow schema
metadata, so the stamp LanceDB writes into the schema goes into `COMMENT ON
TABLE` here, read back through `duckdb_tables()`. It is catalog-native, it
survives reopen and rename, and it is dropped with the table it describes.

**Hybrid is not native.** DuckDB has no operator that blends `vss` and `fts`
ranking, so this store deliberately does not advertise
`SERVER_SIDE_HYBRID_RRF`. Core composes hybrid from the two legs, which the
retrieval contract explicitly allows.
"""
from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from datetime import date, datetime
from math import isfinite
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, Self

import pyarrow as pa
from pydantic import ConfigDict, Field, field_validator

from ..hashing import canonical_fingerprint
from ..optional_dependencies import optional_dependency_version
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
    reject_generation_shaped_collection_name,
    validate_generation_token,
)
from .locks import PublisherLock
from .registry import register

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
_IMPLEMENTATION_IDENTITY_PREFIX = "stel-retrieval-v1"

# The ownership stamp, written as JSON into the table comment. `_OWNER_VALUE`
# is an adoption gate rather than a label: a table whose comment does not carry
# exactly this owner is refused as external, so changing the value makes every
# published collection unreadable.
_OWNER_KEY = "owner"
_OWNER_VALUE = "stel"
_CONTRACT_KEY = "contract"
_CONTRACT_VALUE = "1"
_CONFIG_KEY = "config_fingerprint"
_LEGACY_CONFIG_KEY = "legacy_config_fingerprint"
_DESCRIPTOR_KEY = "descriptor"

_DISTANCE_FUNCTIONS = {
    "cosine": "array_cosine_distance",
    "euclidean": "array_distance",
    "dot": "array_negative_inner_product",
}

# One wording for the two places that refuse this: compile-time preflight and
# the index build itself. They must agree, because an operator who saw one and
# then hit the other would reasonably think they were different problems.
_HNSW_PERSISTENCE_REFUSAL = (
    "DuckDB cannot build an approximate (HNSW) vector index on a "
    "file-backed database unless hnsw_experimental_persistence is "
    "enabled (code=duckdb_hnsw_persistence_disabled). That index "
    "is not covered by the WAL, so a crash can leave it "
    "inconsistent with the table. Either set "
    "hnsw_experimental_persistence: true on the store, accepting "
    "that risk, or declare vector search as 'exact' -- exact "
    "search needs no index and returns the same rows, only slower."
)


class DuckDBConfig(RetrievalStoreConfig):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    type: Literal["duckdb"] = "duckdb"
    # Path to the DuckDB database file. May be the same file the warehouse
    # adapter uses -- that is the case this store exists for. This store always
    # opens its own connection and closes only that connection, so a shared
    # file is not closed out from under the warehouse adapter.
    path: str
    collection_template: str = "{project}__{target}__{collection}"
    timeout_seconds: float = Field(default=30.0, gt=0, le=3600)
    minimum_consistency: Literal["strong"] = "strong"
    # Opt-in to DuckDB's experimental HNSW persistence. Off by default: the
    # index is not WAL-covered, so a crash can desynchronize it from the table.
    # With it off, a file-backed collection serves vector queries by exact scan
    # instead -- slower, never wrong.
    hnsw_experimental_persistence: bool = False
    hnsw_ef_construction: int = Field(default=128, ge=8, le=4096)
    hnsw_ef_search: int = Field(default=64, ge=8, le=4096)
    hnsw_m: int = Field(default=16, ge=4, le=256)
    # `fts` analyzer options, mapped onto PRAGMA create_fts_index arguments.
    fts_stemmer: str = "porter"
    fts_stopwords: str = "english"
    fts_ignore: str = r"(\.|[^a-z])+"
    fts_strip_accents: bool = True
    fts_lower: bool = True
    publisher_lock_dir: str | None = None

    @field_validator("path")
    @classmethod
    def _validate_path(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("path must be a non-empty DuckDB database path")
        if value.strip() == ":memory:":
            # An in-memory store cannot be published to and then read back by
            # another process, so it would fail as a serving target in a way
            # that only shows up at query time.
            raise ValueError(
                "path must be a database file; ':memory:' cannot be served to "
                "a separate reader process"
            )
        return value

    @field_validator("collection_template")
    @classmethod
    def _validate_template(cls, value: str) -> str:
        # Rendering it is the check. A subset test accepts an unknown
        # placeholder and defers the failure to `physical_collection`, where
        # it surfaces as a raw KeyError from deep in a publish rather than as
        # a configuration error at the profile boundary.
        if not value or any(separator in value for separator in ("/", "\\", "..")):
            raise ValueError("collection_template must be a safe collection name template")
        try:
            rendered = value.format(
                project="project", target="target", collection="collection"
            )
        except (KeyError, IndexError, ValueError):
            raise ValueError(
                "collection_template may use only {project}, {target}, and {collection}"
            ) from None
        if not _IDENTIFIER_RE.fullmatch(rendered):
            raise ValueError("collection_template renders an invalid collection name")
        return value

    def local_data_path(self) -> Path:
        return Path(self.path).expanduser()

    def identity_key(self) -> str:
        return str(self.local_data_path().resolve())

    def absolutize(self, project_dir: Path) -> DuckDBConfig:
        candidate = Path(self.path).expanduser()
        if candidate.is_absolute():
            return self
        return self.model_copy(update={"path": str(project_dir / candidate)})


@register
class DuckDBStore(RetrievalStore):
    def __init__(
        self,
        config: RetrievalStoreConfig,
        *,
        project_name: str,
        target_name: str,
        alias: str,
    ) -> None:
        if not isinstance(config, DuckDBConfig):
            raise RetrievalError("DuckDB store received incompatible configuration")
        super().__init__(
            config,
            project_name=project_name,
            target_name=target_name,
            alias=alias,
        )
        self._config = config
        self._conn: Any | None = None

    @classmethod
    def store_type(cls) -> str:
        return "duckdb"

    @classmethod
    def config_model(cls) -> type[RetrievalStoreConfig]:
        return DuckDBConfig

    @classmethod
    def implementation_identity(cls) -> str:
        version = optional_dependency_version("duckdb")
        return f"{_IMPLEMENTATION_IDENTITY_PREFIX}:duckdb-{version}"

    @classmethod
    def capabilities(cls) -> RetrievalCapabilities:
        return RetrievalCapabilities(
            features=frozenset(
                {
                    # Exact search is an ORDER BY over the distance function
                    # and needs no index. Approximate needs an HNSW index,
                    # which on a file-backed database needs the experimental
                    # persistence opt-in -- declared here because the store
                    # *type* can do it, and refused at index-build time with an
                    # actionable message when the opt-in is absent.
                    RetrievalFeature.EXACT_VECTOR_SEARCH,
                    RetrievalFeature.APPROXIMATE_VECTOR_SEARCH,
                    RetrievalFeature.METADATA_FILTERING,
                    # `list_has_any` expresses set overlap against a LIST
                    # column (issue #397).
                    RetrievalFeature.ARRAY_CONTAINMENT_FILTERS,
                    RetrievalFeature.FULL_TEXT_SEARCH,
                    RetrievalFeature.KEYED_UPSERT,
                    RetrievalFeature.KEYED_DELETE,
                    RetrievalFeature.INDEX_READINESS,
                    # DuckDB commits to disk before returning, and DDL plus DML
                    # share one real transaction -- so a batch is genuinely
                    # atomic here, unlike a remote store batching over HTTP.
                    RetrievalFeature.DURABLE_WRITE_ACK,
                    RetrievalFeature.ATOMIC_BATCH_MUTATION,
                    RetrievalFeature.ONLINE_SCHEMA_EVOLUTION,
                    RetrievalFeature.SINGLE_HOST_PUBLISHER_LOCK,
                    RetrievalFeature.PRIVATE_GENERATION_BUILD,
                }
            ),
            distance_metrics=frozenset({"cosine", "euclidean", "dot"}),
            consistency_modes=frozenset({"strong"}),
            max_batch_size=100_000,
            max_id_bytes=None,
            max_dimensions=None,
        )

    # ── connection lifecycle ────────────────────────────────────────────────

    def __enter__(self) -> Self:
        import duckdb

        path = self._config.local_data_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            conn = duckdb.connect(str(path))
        except duckdb.IOException:
            # DuckDB is single-writer per file across processes, which matters
            # far more here than for a remote store: a concurrent publisher is
            # an ordinary operational condition, not a configuration mistake.
            # Classified separately so an operator is not sent to check their
            # profile when the answer is "something else has the file open".
            # Only the exception *type* is inspected -- DuckDB's message
            # carries the database path, which must not reach logs.
            raise RetrievalError(
                "DuckDB database is already open by another process "
                "(code=duckdb_database_locked). DuckDB allows one writer per "
                "file; wait for the other publisher to finish, or give the "
                "retrieval store its own path."
            ) from None
        except Exception:
            raise RetrievalError(
                "DuckDB operation 'connect' failed (code=duckdb_connect_failed)"
            ) from None
        try:
            self._load_extensions(conn)
        except Exception:
            conn.close()
            raise
        self._conn = conn
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        conn = self._conn
        self._conn = None
        if conn is not None:
            # Closes this connection only. When the warehouse adapter has the
            # same file open, DuckDB keeps the underlying instance alive for
            # it; this must never pull the file out from under the warehouse.
            conn.close()

    def _load_extensions(self, conn: Any) -> None:
        for extension in ("vss", "fts"):
            try:
                conn.execute(f"INSTALL {extension}")
                conn.execute(f"LOAD {extension}")
            except Exception:
                raise RetrievalError(
                    f"DuckDB extension '{extension}' could not be loaded "
                    f"(code=duckdb_extension_unavailable). This store needs "
                    "both 'vss' and 'fts'; an offline host must have them "
                    "pre-installed."
                ) from None
        if self._config.hnsw_experimental_persistence:
            conn.execute("SET hnsw_enable_experimental_persistence = true")
        # Applied per connection, not per query: an option accepted and never
        # read would leave an operator tuning recall against a value that has
        # no effect.
        conn.execute(f"SET hnsw_ef_search = {int(self._config.hnsw_ef_search)}")

    def _connection(self) -> Any:
        conn = self._conn
        if conn is None:
            raise RetrievalError("DuckDB store is not connected")
        return conn

    # ── identity and naming ─────────────────────────────────────────────────

    def safe_descriptor(self) -> SafeRetrievalTarget:
        identity = canonical_fingerprint(
            {"store_type": self.store_type(), "path": self._config.identity_key()},
            domain="dbt-ml-safe-retrieval-target",
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
            physical = f"{physical}__g{validate_generation_token(generation)}"
        if not _IDENTIFIER_RE.fullmatch(physical):
            raise RetrievalError("Resolved DuckDB collection name is invalid")
        return physical

    def publisher_fence(self, collection: str) -> AbstractContextManager[None]:
        """OS-enforced exclusive publisher lock, valid on one host only.

        The lock lives beside the database file so every stel publisher on the
        host contends on the same inode. It cannot fence a publisher on another
        machine sharing the file over a network filesystem -- and DuckDB's own
        single-writer model does not help there either, since it would surface
        as a lock error rather than as correct exclusion.
        """
        if not _IDENTIFIER_RE.fullmatch(collection):
            raise RetrievalError("DuckDB collection name is invalid")
        lock_dir = self._lock_dir()
        lock_dir.mkdir(parents=True, exist_ok=True)
        return PublisherLock(
            lock_dir / f"{collection}.stel-publisher.lock",
            store_type=self.store_type(),
        )

    def _lock_dir(self) -> Path:
        """Where this store's publisher locks live.

        Resolved purely from configuration, never from what happens to exist
        on disk. An earlier version fell back to a host-wide directory when
        the database's parent was missing -- but entering the store creates
        that parent, so the first publisher and the second could resolve
        different directories, take locks on different inodes, and both
        believe they held the fence. A lock path that moves when a directory
        appears is not a lock.

        A DuckDB database is always a local file, so the lock belongs beside
        it: every publisher on the host contends on the same inode.
        `publisher_fence` creates the directory before locking.
        """
        override = self._config.publisher_lock_dir
        if override is None:
            return self._config.local_data_path().parent
        return Path(override) / self.safe_descriptor().safe_target_identity[:32]

    # ── catalog ─────────────────────────────────────────────────────────────

    def list_collections(self) -> tuple[str, ...]:
        conn = self._connection()
        rows = self._query(
            conn,
            "SELECT table_name, comment FROM duckdb_tables() WHERE schema_name = 'main'",
            operation="list",
        )
        owned = [
            str(name)
            for name, comment in rows
            if _stamp(comment).get(_OWNER_KEY) == _OWNER_VALUE
        ]
        return tuple(sorted(owned))

    def inspect_collection(self, name: str) -> CollectionMetadata | None:
        conn = self._connection()
        rows = self._query(
            conn,
            "SELECT comment FROM duckdb_tables() "
            "WHERE schema_name = 'main' AND table_name = ?",
            parameters=[name],
            operation="inspect",
        )
        if not rows:
            return None
        stamp = _stamp(rows[0][0])
        if stamp.get(_OWNER_KEY) != _OWNER_VALUE:
            raise RetrievalError(
                f"DuckDB table '{name}' exists but is not stel-owned "
                "(code=duckdb_unowned_collection); refusing to adopt it"
            )
        count_rows = self._query(
            conn,
            f"SELECT count(*) FROM {_quote_identifier(name)}",
            operation="inspect",
        )
        row_count = int(count_rows[0][0]) if count_rows else 0
        schema = self._arrow_schema(conn, name)
        generation = canonical_fingerprint(
            {"name": name, "rows": row_count, "columns": sorted(schema.names)},
            domain="dbt-ml-duckdb-generation",
        )
        return CollectionMetadata(
            physical_name=name,
            config_fingerprint=stamp.get(_CONFIG_KEY),
            descriptor=stamp.get(_DESCRIPTOR_KEY),
            physical_generation=generation,
            row_count=row_count,
            schema=schema,
        )

    def drop_collection(self, name: str) -> bool:
        conn = self._connection()
        if self.inspect_collection(name) is None:
            return False
        self._execute(
            conn, f"DROP TABLE IF EXISTS {_quote_identifier(name)}", operation="drop"
        )
        return True

    # ── collection lifecycle ────────────────────────────────────────────────

    def create_collection(self, spec: CollectionSpec) -> CollectionMetadata:
        conn = self._connection()
        columns = ", ".join(
            f"{_quote_identifier(field.name)} {_duckdb_type(field, spec)}"
            for field in spec.arrow_schema
        )
        primary = _quote_identifier(spec.id_field)
        self._execute(
            conn,
            f"CREATE TABLE {_quote_identifier(spec.physical_name)} "
            f"({columns}, PRIMARY KEY ({primary}))",
            operation="create collection",
        )
        self._stamp_collection(conn, spec.physical_name, spec)
        created = self.inspect_collection(spec.physical_name)
        if created is None:
            raise RetrievalError("DuckDB collection creation was not observable")
        return created

    def evolve_collection(self, spec: CollectionSpec, added: Sequence[str]) -> None:
        conn = self._connection()
        existing = set(self._arrow_schema(conn, spec.physical_name).names)
        missing = [name for name in added if name not in existing]
        fields = {field.name: field for field in spec.arrow_schema}
        for name in missing:
            field = fields.get(name)
            if field is None:
                raise RetrievalError(
                    f"DuckDB cannot add column '{name}': it is not in the target schema"
                )
            self._execute(
                conn,
                f"ALTER TABLE {_quote_identifier(spec.physical_name)} "
                f"ADD COLUMN {_quote_identifier(name)} {_duckdb_type(field, spec)}",
                operation="evolve collection",
            )
        self._stamp_collection(conn, spec.physical_name, spec)

    def restamp_collection(self, spec: CollectionSpec) -> None:
        self._stamp_collection(self._connection(), spec.physical_name, spec)

    def _stamp_collection(
        self, conn: Any, name: str, spec: CollectionSpec
    ) -> None:
        stamp = {
            _OWNER_KEY: _OWNER_VALUE,
            _CONTRACT_KEY: _CONTRACT_VALUE,
            _CONFIG_KEY: spec.config_fingerprint,
            _LEGACY_CONFIG_KEY: spec.legacy_config_fingerprint,
            _DESCRIPTOR_KEY: spec.descriptor,
        }
        payload = json.dumps(stamp, sort_keys=True, separators=(",", ":"))
        self._execute(
            conn,
            f"COMMENT ON TABLE {_quote_identifier(name)} IS {_sql_string(payload)}",
            operation="stamp collection",
        )

    # ── mutation ────────────────────────────────────────────────────────────

    def upsert(
        self,
        collection: str,
        rows: Sequence[IndexedRow],
        *,
        id_field: str,
        mutation_digest: str,
    ) -> MutationReceipt:
        if not rows:
            return MutationReceipt(mutation_digest, atomic=True, outcomes=())
        conn = self._connection()
        schema = self._arrow_schema(conn, collection)
        table = _rows_to_arrow(rows, schema)
        quoted = _quote_identifier(collection)
        columns = ", ".join(_quote_identifier(name) for name in schema.names)
        updates = ", ".join(
            f"{_quote_identifier(name)} = EXCLUDED.{_quote_identifier(name)}"
            for name in schema.names
            if name != id_field
        )
        primary = _quote_identifier(id_field)
        # One statement inside one transaction: DuckDB gives a real atomic
        # batch here, so a partial upsert is not a state the caller can observe.
        statement = (
            f"INSERT INTO {quoted} ({columns}) SELECT {columns} FROM arrow_rows "
            f"ON CONFLICT ({primary}) DO UPDATE SET {updates}"
        )
        self._run_in_transaction(conn, statement, table, operation="upsert")
        return MutationReceipt(
            mutation_digest,
            atomic=True,
            outcomes=tuple(MutationOutcome("applied") for _ in rows),
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
            return MutationReceipt(mutation_digest, atomic=True, outcomes=())
        conn = self._connection()
        quoted = _quote_identifier(collection)
        placeholders = ", ".join("?" for _ in record_ids)
        self._execute(
            conn,
            f"DELETE FROM {quoted} WHERE {_quote_identifier(id_field)} "
            f"IN ({placeholders})",
            parameters=list(record_ids),
            operation="delete",
        )
        return MutationReceipt(
            mutation_digest,
            atomic=True,
            outcomes=tuple(MutationOutcome("deleted") for _ in record_ids),
        )

    def _run_in_transaction(
        self, conn: Any, statement: str, table: pa.Table, *, operation: str
    ) -> None:
        try:
            conn.register("arrow_rows", table)
            try:
                conn.execute("BEGIN")
                conn.execute(statement)
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
            finally:
                conn.unregister("arrow_rows")
        except RetrievalError:
            raise
        except Exception:
            raise RetrievalError(_failure(operation)) from None

    # ── indexes ─────────────────────────────────────────────────────────────

    def ensure_indexes(self, spec: CollectionSpec) -> CollectionMetadata:
        conn = self._connection()
        if spec.vector_field is not None:
            if spec.vector_search == "approximate":
                self._ensure_hnsw_index(conn, spec)
            else:
                self._drop_hnsw_index(conn, spec)
        if spec.full_text_fields:
            self._ensure_fts_index(conn, spec)
        metadata = self.inspect_collection(spec.physical_name)
        if metadata is None:
            raise RetrievalError("DuckDB collection disappeared while indexing")
        return metadata

    def _drop_hnsw_index(self, conn: Any, spec: CollectionSpec) -> None:
        """Remove an ANN index left by a previous `approximate` publish.

        `exact` is implemented by the absence of the index -- the planner uses
        one whenever it exists -- so a switch back has to take it away, or the
        collection keeps answering approximately under a config that promises
        otherwise (issue #461).
        """
        self._execute(
            conn,
            "DROP INDEX IF EXISTS "
            f"{_quote_identifier(_index_name(spec.physical_name, 'hnsw'))}",
            operation="ensure indexes",
        )

    def index_config_refusal(
        self, *, vector_search: str | None, vector_index: str | None
    ) -> str | None:
        """Approximate search needs an opt-in this store's config may not carry,
        and this store builds exactly one kind of ANN index.

        Reported here as well as from `ensure_indexes` so the compiler can
        refuse the combination before a publish begins. Since a vector-search
        change became compatible (issue #461), discovering it at index time
        would mean an in-place evolution had already republished the whole
        collection and cleared the serving pointer.
        """
        if vector_search != "approximate":
            return None
        if vector_index not in (None, "ivf_hnsw_flat"):
            return (
                "DuckDB's vss extension builds only an HNSW index over raw vectors; "
                f"search.vector.index {vector_index!r} needs the LanceDB store "
                "(code=duckdb_vector_index_unsupported)"
            )
        if self._config.hnsw_experimental_persistence:
            return None
        return _HNSW_PERSISTENCE_REFUSAL

    def _ensure_hnsw_index(self, conn: Any, spec: CollectionSpec) -> None:
        if not self._config.hnsw_experimental_persistence:
            raise RetrievalError(_HNSW_PERSISTENCE_REFUSAL)
        metric = spec.distance_metric or "cosine"
        if metric not in _DISTANCE_FUNCTIONS:
            raise RetrievalError(f"DuckDB does not support distance metric '{metric}'")
        index_name = _index_name(spec.physical_name, "hnsw")
        # Dropped and recreated rather than maintained: DuckDB's HNSW index
        # does not compact on delete, so an incrementally-churned index grows
        # without bound. Rebuilding at publish is the documented tradeoff --
        # publish cost is bounded and predictable, query cost stays flat.
        self._execute(
            conn, f"DROP INDEX IF EXISTS {_quote_identifier(index_name)}",
            operation="ensure indexes",
        )
        assert spec.vector_field is not None
        self._execute(
            conn,
            f"CREATE INDEX {_quote_identifier(index_name)} ON "
            f"{_quote_identifier(spec.physical_name)} USING HNSW "
            f"({_quote_identifier(spec.vector_field)}) WITH ("
            f"metric = {_sql_string(metric)}, "
            f"ef_construction = {self._config.hnsw_ef_construction}, "
            f"M = {self._config.hnsw_m})",
            operation="ensure indexes",
        )

    def _ensure_fts_index(self, conn: Any, spec: CollectionSpec) -> None:
        fields = ", ".join(_sql_string(name) for name in spec.full_text_fields)
        config = self._config
        # PRAGMA create_fts_index has no IF NOT EXISTS; overwrite is how it is
        # made idempotent, and it is also the rebuild after bulk mutation --
        # the BM25 index is a snapshot of the table at build time, not a live
        # view of it.
        self._execute(
            conn,
            f"PRAGMA create_fts_index("
            f"{_sql_string(spec.physical_name)}, {_sql_string(spec.id_field)}, "
            f"{fields}, "
            f"stemmer = {_sql_string(config.fts_stemmer)}, "
            f"stopwords = {_sql_string(config.fts_stopwords)}, "
            f"ignore = {_sql_string(config.fts_ignore)}, "
            f"strip_accents = {int(config.fts_strip_accents)}, "
            f"lower = {int(config.fts_lower)}, "
            f"overwrite = 1)",
            operation="ensure indexes",
        )

    # ── queries ─────────────────────────────────────────────────────────────

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
        _validate_query_limit(limit)
        _validate_query_projection(columns)
        if not _IDENTIFIER_RE.fullmatch(vector_field):
            raise RetrievalError("DuckDB vector field name is invalid")
        conn = self._connection()
        metric = self._collection_metric(conn, collection)
        function = _DISTANCE_FUNCTIONS[metric]
        literal = _vector_literal(vector)
        projection = _projection(columns, "_distance")
        where = _compile_predicates(predicates)
        clause = f" WHERE {where}" if where else ""
        return self._query_arrow(
            conn,
            f"SELECT {projection}, "
            f"{function}({_quote_identifier(vector_field)}, {literal}) AS _distance "
            f"FROM {_quote_identifier(collection)}{clause} "
            f"ORDER BY _distance LIMIT {limit}",
            operation="vector search",
        )

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
        _validate_query_limit(limit)
        _validate_query_projection(columns)
        if not query or not query.strip():
            raise RetrievalError("DuckDB text search requires a query string")
        if not _IDENTIFIER_RE.fullmatch(text_field):
            raise RetrievalError("DuckDB text field name is invalid")
        conn = self._connection()
        schema = self._arrow_schema(conn, collection)
        id_field = self._id_field(conn, collection, schema)
        projection = _projection(columns, "_score")
        where = _compile_predicates(predicates)
        clause = f" AND {where}" if where else ""
        index_schema = _quote_identifier(f"fts_main_{collection}")
        # `fields :=` scopes BM25 to the requested column. Core calls this
        # once per declared full-text field and fuses each as its own RRF leg,
        # so scoring the whole combined index every time would enter the same
        # ranking under several labels and let RRF count one match repeatedly.
        # (Note the `:=` -- create_fts_index takes `=` for its options, this
        # takes `:=`.)
        #
        # The BM25 score is NULL for a non-matching row, so the NOT NULL test
        # is the match filter, not an optimization.
        return self._query_arrow(
            conn,
            f"SELECT {projection}, _score FROM (SELECT *, {index_schema}.match_bm25("
            f"{_quote_identifier(id_field)}, ?, fields := {_sql_string(text_field)}"
            f") AS _score "
            f"FROM {_quote_identifier(collection)}) "
            f"WHERE _score IS NOT NULL{clause} "
            f"ORDER BY _score DESC LIMIT {limit}",
            parameters=[query],
            operation="text search",
        )

    # ── internals ───────────────────────────────────────────────────────────

    def _id_field(self, conn: Any, collection: str, schema: pa.Schema) -> str:
        rows = self._query(
            conn,
            "SELECT constraint_column_names FROM duckdb_constraints() "
            "WHERE table_name = ? AND constraint_type = 'PRIMARY KEY'",
            parameters=[collection],
            operation="inspect",
        )
        if rows and rows[0][0]:
            return str(rows[0][0][0])
        raise RetrievalError(
            f"DuckDB collection '{collection}' has no primary key "
            "(code=duckdb_missing_primary_key)"
        )

    def _collection_metric(self, conn: Any, collection: str) -> str:
        rows = self._query(
            conn,
            "SELECT comment FROM duckdb_tables() "
            "WHERE schema_name = 'main' AND table_name = ?",
            parameters=[collection],
            operation="inspect",
        )
        if not rows:
            raise RetrievalError(f"DuckDB collection '{collection}' does not exist")
        descriptor = _stamp(rows[0][0]).get(_DESCRIPTOR_KEY)
        if descriptor:
            parsed = json.loads(descriptor)
            metric = parsed.get("distance_metric") if isinstance(parsed, dict) else None
            if isinstance(metric, str) and metric in _DISTANCE_FUNCTIONS:
                return metric
        return "cosine"

    def _arrow_schema(self, conn: Any, collection: str) -> pa.Schema:
        try:
            result = conn.execute(
                f"SELECT * FROM {_quote_identifier(collection)} LIMIT 0"
            )
            return result.to_arrow_table().schema
        except Exception:
            raise RetrievalError(
                "DuckDB operation 'schema' failed (code=duckdb_schema_failed)"
            ) from None

    def _query(
        self,
        conn: Any,
        statement: str,
        *,
        parameters: Sequence[Any] | None = None,
        operation: str,
    ) -> list[tuple[Any, ...]]:
        try:
            cursor = conn.execute(statement, list(parameters or ()))
            return list(cursor.fetchall())
        except Exception:
            raise RetrievalError(_failure(operation)) from None

    def _query_arrow(
        self,
        conn: Any,
        statement: str,
        *,
        parameters: Sequence[Any] | None = None,
        operation: str,
    ) -> pa.Table:
        try:
            return conn.execute(statement, list(parameters or ())).to_arrow_table()
        except Exception:
            raise RetrievalError(_failure(operation)) from None

    def _execute(
        self,
        conn: Any,
        statement: str,
        *,
        parameters: Sequence[Any] | None = None,
        operation: str,
    ) -> None:
        try:
            conn.execute(statement, list(parameters or ()))
        except Exception:
            raise RetrievalError(_failure(operation)) from None


def _failure(operation: str) -> str:
    """Safe error text for a failed DuckDB call.

    The native exception is deliberately dropped rather than wrapped: DuckDB
    error text can echo the failing statement, and a statement carries filter
    values -- which for a governed query are the caller's policy values.
    """
    code = operation.replace(" ", "_")
    return f"DuckDB operation '{operation}' failed (code=duckdb_{code}_failed)"


def _stamp(comment: Any) -> Mapping[str, str]:
    """Parse the ownership stamp out of a table comment.

    A table with no comment, or one holding anything other than a JSON object,
    is simply not stel-owned. This must not raise: `list_collections` walks
    every table in the schema, including tables that have nothing to do with
    stel and never will.
    """
    if not isinstance(comment, str) or not comment:
        return {}
    try:
        parsed = json.loads(comment)
    except json.JSONDecodeError:
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {str(k): str(v) for k, v in parsed.items() if isinstance(v, str)}


def _identifier_piece(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", value)
    if not cleaned:
        raise RetrievalError("DuckDB collection name piece is empty")
    return cleaned


def _quote_identifier(name: str) -> str:
    """Quote an identifier that has already been proven safe.

    The regex is the actual defense, not the quoting: every identifier reaching
    SQL here comes from configuration or a collection spec, and anything that
    is not a plain identifier is refused rather than escaped. Values never take
    this path -- they are parameters or go through `_sql_literal`.
    """
    if not _IDENTIFIER_RE.fullmatch(name):
        raise RetrievalError(f"DuckDB identifier '{name}' is invalid")
    return '"' + name + '"'


def _index_name(collection: str, kind: str) -> str:
    return f"{collection}__{kind}"


def _sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _vector_literal(vector: Sequence[float]) -> str:
    values = list(vector)
    if not values:
        raise RetrievalError("DuckDB vector search requires a non-empty vector")
    for item in values:
        if isinstance(item, bool) or not isinstance(item, int | float):
            raise RetrievalError("DuckDB vector search requires numeric values")
        if not isfinite(float(item)):
            raise RetrievalError("DuckDB vector search requires finite values")
    joined = ", ".join(repr(float(item)) for item in values)
    return f"[{joined}]::FLOAT[{len(values)}]"


def _projection(columns: Sequence[str] | None, score_column: str) -> str:
    if columns is None:
        return "*"
    names = [name for name in columns if name != score_column]
    if not names:
        return "*"
    return ", ".join(_quote_identifier(name) for name in names)


def _validate_query_limit(limit: int) -> None:
    if limit < 1:
        raise RetrievalError("DuckDB query limit must be positive")


def _validate_query_projection(columns: Sequence[str] | None) -> None:
    if columns is None:
        return
    for name in columns:
        if not _IDENTIFIER_RE.fullmatch(name):
            raise RetrievalError("DuckDB query projection column is invalid")


def _duckdb_type(field: pa.Field, spec: CollectionSpec) -> str:
    if field.name == spec.vector_field and spec.vector_dimensions:
        return f"FLOAT[{spec.vector_dimensions}]"
    return _arrow_to_duckdb(field.type)


def _arrow_to_duckdb(arrow_type: pa.DataType) -> str:
    if pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type):
        return "VARCHAR"
    if pa.types.is_boolean(arrow_type):
        return "BOOLEAN"
    if pa.types.is_integer(arrow_type):
        return "BIGINT"
    if pa.types.is_floating(arrow_type):
        return "DOUBLE"
    if pa.types.is_date(arrow_type):
        return "DATE"
    if pa.types.is_timestamp(arrow_type):
        return "TIMESTAMP"
    if pa.types.is_fixed_size_list(arrow_type):
        inner = _arrow_to_duckdb(arrow_type.value_type)
        return f"{inner}[{arrow_type.list_size}]"
    if pa.types.is_list(arrow_type) or pa.types.is_large_list(arrow_type):
        return f"{_arrow_to_duckdb(arrow_type.value_type)}[]"
    raise RetrievalError(f"DuckDB has no mapping for Arrow type '{arrow_type}'")


def _rows_to_arrow(rows: Sequence[IndexedRow], schema: pa.Schema) -> pa.Table:
    columns: dict[str, list[Any]] = {name: [] for name in schema.names}
    for row in rows:
        for name in schema.names:
            columns[name].append(row.values.get(name))
    return pa.table(columns, schema=schema)


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
        field = _quote_identifier(predicate.field)
        if predicate.operator == RetrievalPredicateOperator.IN:
            assert isinstance(predicate.value, tuple)
            values = ", ".join(_sql_literal(item) for item in predicate.value)
            clauses.append(f"{field} IN ({values})")
        elif predicate.operator == RetrievalPredicateOperator.ARRAY_CONTAINS_ANY:
            assert isinstance(predicate.value, tuple)
            values = ", ".join(_sql_literal(item) for item in predicate.value)
            clauses.append(f"list_has_any({field}, [{values}])")
        else:
            assert not isinstance(predicate.value, tuple)
            symbol = operators[predicate.operator]
            clauses.append(f"{field} {symbol} {_sql_literal(predicate.value)}")
    return " AND ".join(clauses)


def _sql_literal(value: Any) -> str:
    if isinstance(value, str):
        return _sql_string(value)
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    # Temporal values need a typed literal, not a quoted string: a bare
    # '2020-01-01' is VARCHAR to the engine and compares as text.
    if isinstance(value, datetime):
        return "TIMESTAMP " + _sql_string(value.isoformat(sep=" "))
    if isinstance(value, date):
        return "DATE " + _sql_string(value.isoformat())
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not isfinite(value):
            raise RetrievalError("DuckDB predicate value is not finite")
        return repr(value)
    raise RetrievalError("DuckDB predicate value type is unsupported")
