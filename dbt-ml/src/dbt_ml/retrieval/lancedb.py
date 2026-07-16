from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import date, datetime, timedelta
from math import isfinite
from pathlib import Path
from typing import Any, Literal, Self, cast

import pyarrow as pa
from pydantic import ConfigDict, Field, field_validator

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
_OWNER_KEY = b"dbt_ml.owner"
_CONTRACT_KEY = b"dbt_ml.record_contract"
_CONFIG_KEY = b"dbt_ml.config_fingerprint"


class LanceDBConfig(RetrievalStoreConfig):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    type: Literal["lancedb"] = "lancedb"
    path: Path
    collection_template: str = "{project}__{target}__{collection}"
    timeout_seconds: float = Field(default=30.0, gt=0, le=3600)
    minimum_consistency: Literal["strong"] = "strong"

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

    def absolutize(self, project_dir: Path) -> LanceDBConfig:
        path = self.path if self.path.is_absolute() else project_dir / self.path
        return self.model_copy(update={"path": path.resolve()})


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
        return f"dbt_ml.retrieval.lancedb:v1:lancedb-{version}"

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
        self._config.path.mkdir(parents=True, exist_ok=True)
        try:
            self._db = lancedb.connect(str(self._config.path))
        except Exception:
            raise RetrievalError(
                "LanceDB operation 'connect' failed (code=lancedb_connect_failed)"
            ) from None
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self._db = None

    def _connection(self) -> Any:
        if self._db is None:
            raise RetrievalError("LanceDB store is not open")
        return self._db

    def _open_owned_table(self, name: str) -> Any:
        if not _COLLECTION_RE.fullmatch(name):
            raise RetrievalError("LanceDB collection name is invalid")
        table = self._connection().open_table(name)
        if (table.schema.metadata or {}).get(_OWNER_KEY) != b"dbt-ml":
            raise RetrievalError(
                "LanceDB collection is not owned by dbt-ml "
                "(code=lancedb_external_collection)"
            )
        return table

    def safe_descriptor(self) -> SafeRetrievalTarget:
        identity = canonical_fingerprint(
            {
                "store_type": self.store_type(),
                "path": self._config.path.as_posix(),
                "alias": self.alias,
            },
            domain="dbt-ml-safe-retrieval-target",
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
                _OWNER_KEY: b"dbt-ml",
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
                query = query.select(list(columns))
            return cast(pa.Table, query.limit(limit).to_arrow())
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
                builder = builder.select(list(columns))
            return cast(pa.Table, builder.limit(limit).to_arrow())
        except Exception:
            raise RetrievalError(
                "LanceDB operation 'text search' failed (code=lancedb_text_search_failed)"
            ) from None


def _identifier_piece(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_]", "_", value)
    if not normalized or normalized[0].isdigit():
        normalized = f"_{normalized}"
    return normalized[:128]


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
