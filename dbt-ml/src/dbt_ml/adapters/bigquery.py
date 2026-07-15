"""BigQuery warehouse adapter (issue #83).

Materializes dbt-ml tables into a BigQuery dataset; incremental state lives
in the same dataset (`dbt_ml_state`), so a project can run against BigQuery
with no DuckDB involvement in materialization or state.

The google-cloud-bigquery dependency is an optional extra — this module
imports lazily so the adapter registers without it and fails with an
install hint only when actually used:

    pip install 'dbt-ml[bigquery]'

DataFrames travel as Parquet load jobs (polars → parquet bytes → load), so
column matching is by name and never positional. Queries use positional `?`
parameters, converted to BigQuery query parameters here.
"""
from __future__ import annotations

import io
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal, cast
from uuid import uuid4

import polars as pl
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from ..config.profile import WarehouseConfig
from .base import (
    AdapterError,
    StateRecord,
    StateScope,
    StateValue,
    WarehouseAdapter,
    WarehouseCapability,
    validate_incremental_keys,
    validate_state_keys,
    validate_state_records,
)
from .registry import register

_INSTALL_HINT = (
    "BigQuery support requires google-cloud-bigquery. "
    "Install it with: pip install 'dbt-ml[bigquery]'"
)


_STATE_TABLE = "dbt_ml_state"
_STATE_MIGRATION_PREFIX = "dbt_ml_staging__state_migration_v2__"
_STATE_V1_COLUMNS = (
    ("model_name", "STRING", "REQUIRED"),
    ("document_id", "STRING", "REQUIRED"),
    ("content_hash", "STRING", "REQUIRED"),
    ("code_version", "STRING", "REQUIRED"),
    ("last_run_at", "TIMESTAMP", "REQUIRED"),
)
_STATE_V2_COLUMNS = (
    ("model_name", "STRING", "REQUIRED"),
    ("state_scope", "STRING", "REQUIRED"),
    ("target_identity", "STRING", "REQUIRED"),
    ("record_key", "STRING", "REQUIRED"),
    ("input_fingerprint", "STRING", "REQUIRED"),
    ("code_version", "STRING", "REQUIRED"),
    ("last_run_at", "TIMESTAMP", "REQUIRED"),
)


def _bigquery() -> Any:
    try:
        from google.cloud import bigquery
    except ImportError as e:
        raise AdapterError(_INSTALL_HINT) from e
    return bigquery


def _not_found_error() -> type[Exception]:
    try:
        from google.api_core.exceptions import NotFound
    except ImportError as e:
        raise AdapterError(_INSTALL_HINT) from e
    return NotFound


# dbt-bigquery's default scopes, kept identical so profiles port over unchanged.
DEFAULT_SCOPES = (
    "https://www.googleapis.com/auth/bigquery",
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/drive",
)

AuthMethod = Literal["oauth", "service-account", "service-account-json", "oauth-secrets"]


class BigQueryWarehouseConfig(WarehouseConfig):
    """Field names mirror dbt-bigquery's profile so existing dbt profiles
    port over. `method:` may be omitted — it is inferred from which
    credential fields are present (keyfile → service-account, keyfile_json →
    service-account-json, token/refresh_token → oauth-secrets, else oauth/ADC).

    profiles.yml:

    warehouse:
      type: bigquery
      project: my-gcp-project
      dataset: dbt_ml            # `schema:` works too
      location: US               # optional BigQuery region
      keyfile: ./sa.json         # service-account file auth; omit for ADC
    """

    type: Literal["bigquery"] = "bigquery"
    project: str
    schema_name: str = Field(
        default="dbt_ml",
        validation_alias=AliasChoices("dataset", "schema", "schema_name"),
        serialization_alias="schema",
    )
    location: str | None = None

    # ─── auth (dbt-bigquery parity) ───────────────────────────────────────
    method: AuthMethod | None = None
    keyfile: Path | None = None
    keyfile_json: dict[str, Any] | str | None = None  # dict, JSON, or base64 JSON
    token: str | None = None
    refresh_token: str | None = None
    client_id: str | None = None
    client_secret: str | None = None
    token_uri: str | None = None
    impersonate_service_account: str | None = None
    scopes: list[str] = Field(default_factory=lambda: list(DEFAULT_SCOPES))

    # ─── execution / billing (dbt-bigquery parity) ────────────────────────
    execution_project: str | None = None
    quota_project: str | None = None
    priority: Literal["interactive", "batch"] | None = None
    maximum_bytes_billed: int | None = None
    job_retries: int = 1
    job_retry_deadline_seconds: float | None = None
    job_creation_timeout_seconds: float | None = None
    job_execution_timeout_seconds: float | None = None

    @model_validator(mode="after")
    def _resolve_auth_method(self) -> BigQueryWarehouseConfig:
        if self.keyfile is not None and self.keyfile_json is not None:
            raise ValueError("set either `keyfile` or `keyfile_json`, not both")
        inferred: AuthMethod
        if self.keyfile is not None:
            inferred = "service-account"
        elif self.keyfile_json is not None:
            inferred = "service-account-json"
        elif self.token or self.refresh_token:
            inferred = "oauth-secrets"
        else:
            inferred = "oauth"
        if self.method is None:
            self.method = inferred
        elif self.method != inferred and inferred != "oauth":
            raise ValueError(
                f"method '{self.method}' conflicts with the credential fields "
                f"provided (which imply '{inferred}')"
            )

        if self.method == "service-account" and self.keyfile is None:
            raise ValueError("method 'service-account' requires `keyfile`")
        if self.method == "service-account-json" and self.keyfile_json is None:
            raise ValueError("method 'service-account-json' requires `keyfile_json`")
        if self.method == "oauth-secrets":
            has_refresh_set = all(
                (self.refresh_token, self.client_id, self.client_secret, self.token_uri)
            )
            if not self.token and not has_refresh_set:
                raise ValueError(
                    "method 'oauth-secrets' requires `token`, or all of "
                    "`refresh_token`, `client_id`, `client_secret`, `token_uri`"
                )
        return self

    def absolutize(self, project_dir: Path) -> BigQueryWarehouseConfig:
        if self.keyfile is None:
            return self
        return self.model_copy(
            update={"keyfile": (project_dir / self.keyfile).resolve()}
        )

    def storage_location(self) -> str:
        return f"{self.project}.{self.schema_name}"

    def catalog_name(self) -> str:
        return self.project


class BigQueryPartitionRange(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start: int
    end: int
    interval: int = Field(gt=0)


class BigQueryPartitionBy(BaseModel):
    """Mirrors dbt-bigquery's `partition_by` resource config: a time column
    (timestamp/date/datetime + granularity), an integer-range column
    (int64 + range), or ingestion time when `field` is omitted."""

    model_config = ConfigDict(extra="forbid")

    field: str | None = None
    data_type: Literal["timestamp", "date", "datetime", "int64"] = "date"
    granularity: Literal["hour", "day", "month", "year"] = "day"
    range: BigQueryPartitionRange | None = None

    @model_validator(mode="after")
    def _validate_shape(self) -> BigQueryPartitionBy:
        if self.data_type == "int64":
            if self.field is None:
                raise ValueError("int64 partitioning requires `field`")
            if self.range is None:
                raise ValueError(
                    "int64 partitioning requires `range` (start/end/interval)"
                )
        elif self.range is not None:
            raise ValueError("`range` applies only to `data_type: int64`")
        if self.data_type == "date" and self.granularity == "hour":
            raise ValueError("date columns cannot partition by hour granularity")
        return self


_LABEL_KEY_RE = re.compile(r"^[a-z][a-z0-9_-]{0,62}$")
_LABEL_VALUE_RE = re.compile(r"^[a-z0-9_-]{0,63}$")


class BigQueryWarehouseOptions(BaseModel):
    """Model-level `warehouse_options` for BigQuery (issue #91). Layout
    (partitioning/clustering) applies when a target table is (re)created; an
    existing incremental table keeps its layout until --full-refresh rebuilds
    it. Table options (labels, expirations, require_partition_filter,
    kms_key_name) are set on every (re)create; `labels` are also attached to
    the load and query jobs the adapter runs for the model."""

    model_config = ConfigDict(extra="forbid")

    partition_by: BigQueryPartitionBy | None = None
    cluster_by: list[str] = Field(default_factory=list, max_length=4)
    require_partition_filter: bool | None = None
    partition_expiration_days: float | None = Field(default=None, gt=0)
    hours_to_expiration: int | None = Field(default=None, gt=0)
    labels: dict[str, str] = Field(default_factory=dict)
    kms_key_name: str | None = None
    # merge (default) upserts by key. insert_overwrite replaces every
    # partition present in the incoming batch — dbt-bigquery semantics,
    # for pipelines that reprocess whole partitions at a time.
    incremental_strategy: Literal["merge", "insert_overwrite"] = "merge"

    @field_validator("cluster_by", mode="before")
    @classmethod
    def _single_column_ok(cls, value: Any) -> Any:
        return [value] if isinstance(value, str) else value

    @field_validator("labels")
    @classmethod
    def _validate_labels(cls, labels: dict[str, str]) -> dict[str, str]:
        for key, value in labels.items():
            if not _LABEL_KEY_RE.match(key):
                raise ValueError(
                    f"label key {key!r} must be 1-63 chars of lowercase "
                    "letters, digits, _ or -, starting with a letter"
                )
            if not _LABEL_VALUE_RE.match(value):
                raise ValueError(
                    f"label value {value!r} must be 0-63 chars of lowercase "
                    "letters, digits, _ or -"
                )
        return labels

    @model_validator(mode="after")
    def _validate_partition_dependencies(self) -> BigQueryWarehouseOptions:
        if self.partition_by is None:
            for name in ("require_partition_filter", "partition_expiration_days"):
                if getattr(self, name) is not None:
                    raise ValueError(f"`{name}` requires `partition_by`")
        if self.incremental_strategy == "insert_overwrite":
            if self.partition_by is None or self.partition_by.field is None:
                raise ValueError(
                    "incremental_strategy: insert_overwrite requires "
                    "`partition_by` with a `field`"
                )
            if self.partition_by.data_type == "int64":
                raise ValueError(
                    "incremental_strategy: insert_overwrite supports time "
                    "partitioning only (timestamp/date/datetime)"
                )
        return self


def parse_keyfile_json(value: dict[str, Any] | str) -> dict[str, Any]:
    """`keyfile_json` accepts a YAML mapping, a JSON string, or base64-encoded
    JSON (matching dbt-bigquery, where CI pipelines inject it via env vars)."""
    import base64
    import json

    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        try:
            parsed = json.loads(base64.b64decode(value, validate=True))
        except Exception as e:
            raise AdapterError(
                "keyfile_json must be a mapping, a JSON string, or base64-encoded JSON"
            ) from e
    if not isinstance(parsed, dict):
        raise AdapterError("keyfile_json must decode to a JSON object")
    return parsed


@dataclass(frozen=True)
class SchemaChangePlan:
    columns_to_load: list[str]
    allow_field_addition: bool


def plan_schema_change(
    existing_cols: list[str],
    staging_cols: list[str],
    on_schema_change: str,
    table: str,
) -> SchemaChangePlan:
    """Pure decision logic for incremental loads against an existing table."""
    new = [c for c in staging_cols if c not in existing_cols]
    removed = [c for c in existing_cols if c not in staging_cols]
    if not new and not removed:
        return SchemaChangePlan(list(staging_cols), False)

    if on_schema_change == "fail":
        drift = "; ".join(
            part
            for part in (
                f"new columns {new}" if new else "",
                f"removed columns {removed}" if removed else "",
            )
            if part
        )
        raise AdapterError(
            f"Schema change on incremental table '{table}': {drift}. "
            "Run with --full-refresh to rebuild it, or set "
            "`on_schema_change: append_new_columns` (or `ignore`) on the model."
        )
    if on_schema_change == "append_new_columns":
        return SchemaChangePlan(list(staging_cols), bool(new))
    if on_schema_change == "ignore":
        return SchemaChangePlan([c for c in staging_cols if c in existing_cols], False)
    raise AdapterError(
        f"Unknown on_schema_change policy '{on_schema_change}'. "
        "Allowed: fail, ignore, append_new_columns."
    )


def _bq_param_type(value: Any) -> str:
    # bool first: bool is a subclass of int
    if isinstance(value, bool):
        return "BOOL"
    if isinstance(value, int):
        return "INT64"
    if isinstance(value, float):
        return "FLOAT64"
    if isinstance(value, datetime):
        return "TIMESTAMP"
    if isinstance(value, date):
        return "DATE"
    return "STRING"


def to_query_parameters(params: list[Any]) -> list[Any]:
    """Positional `?` params → BigQuery query parameters. Lists become
    ARRAY parameters (for `IN UNNEST(?)`), scalars are type-inferred."""
    bigquery = _bigquery()
    out: list[Any] = []
    for value in params:
        if isinstance(value, list | tuple):
            elem_type = _bq_param_type(value[0]) if value else "STRING"
            out.append(bigquery.ArrayQueryParameter(None, elem_type, list(value)))
        else:
            out.append(bigquery.ScalarQueryParameter(None, _bq_param_type(value), value))
    return out


@register
class BigQueryAdapter(WarehouseAdapter):
    def __init__(
        self, config: WarehouseConfig, *, project_dir: Path | None = None
    ) -> None:
        super().__init__(config, project_dir=project_dir)
        self._client: Any = None

    @classmethod
    def adapter_type(cls) -> str:
        return "bigquery"

    @classmethod
    def config_model(cls) -> type[WarehouseConfig]:
        return BigQueryWarehouseConfig

    @classmethod
    def capabilities(cls) -> frozenset[WarehouseCapability]:
        return frozenset(WarehouseCapability) - {
            WarehouseCapability.ATOMIC_FULL_REPLACE,
            WarehouseCapability.TRANSACTIONS,
        }

    @classmethod
    def warehouse_options_model(cls) -> type[BaseModel] | None:
        return BigQueryWarehouseOptions

    # ─── lifecycle ────────────────────────────────────────────────────────

    @property
    def _cfg(self) -> BigQueryWarehouseConfig:
        config = self.config
        assert isinstance(config, BigQueryWarehouseConfig)
        return config

    @property
    def client(self) -> Any:
        if self._client is None:
            raise AdapterError("Adapter must be used as a context manager")
        return self._client

    def _credentials(self) -> Any:
        """Build google credentials per the configured auth method, then apply
        impersonation and quota-project wrapping — mirroring dbt-bigquery."""
        cfg = self._cfg
        scopes = tuple(cfg.scopes)
        try:
            if cfg.method == "service-account":
                from google.oauth2 import service_account

                creds = service_account.Credentials.from_service_account_file(  # type: ignore[no-untyped-call]
                    str(cfg.keyfile), scopes=scopes
                )
            elif cfg.method == "service-account-json":
                from google.oauth2 import service_account

                assert cfg.keyfile_json is not None
                creds = service_account.Credentials.from_service_account_info(  # type: ignore[no-untyped-call]
                    parse_keyfile_json(cfg.keyfile_json), scopes=scopes
                )
            elif cfg.method == "oauth-secrets":
                from google.oauth2.credentials import Credentials as UserCredentials

                creds = UserCredentials(  # type: ignore[no-untyped-call]
                    token=cfg.token,
                    refresh_token=cfg.refresh_token,
                    client_id=cfg.client_id,
                    client_secret=cfg.client_secret,
                    token_uri=cfg.token_uri,
                    scopes=scopes,
                )
            else:  # oauth: gcloud Application Default Credentials
                import google.auth

                creds, _ = google.auth.default(scopes=scopes)

            if cfg.impersonate_service_account:
                from google.auth import impersonated_credentials

                creds = impersonated_credentials.Credentials(  # type: ignore[no-untyped-call]
                    source_credentials=creds,
                    target_principal=cfg.impersonate_service_account,
                    target_scopes=list(scopes),
                )
            if cfg.quota_project:
                creds = creds.with_quota_project(cfg.quota_project)
        except ImportError as e:
            raise AdapterError(_INSTALL_HINT) from e
        return creds

    def _default_job_config(self) -> Any | None:
        """Client-level QueryJobConfig defaults (priority, cost cap); BigQuery
        merges these into every query job unless overridden per call."""
        cfg = self._cfg
        if cfg.priority is None and cfg.maximum_bytes_billed is None:
            return None
        bigquery = _bigquery()
        job_config = bigquery.QueryJobConfig()
        if cfg.priority is not None:
            job_config.priority = cfg.priority.upper()
        if cfg.maximum_bytes_billed is not None:
            job_config.maximum_bytes_billed = cfg.maximum_bytes_billed
        return job_config

    def _make_client(self) -> Any:
        bigquery = _bigquery()
        cfg = self._cfg
        # execution_project bills the queries; data still lives in `project`
        # (all table refs are fully qualified with the data project).
        return bigquery.Client(
            project=cfg.execution_project or cfg.project,
            credentials=self._credentials(),
            location=cfg.location,
            default_query_job_config=self._default_job_config(),
        )

    def _connect(self) -> None:
        self._client = self._make_client()

    def _close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def _ensure_schema(self) -> None:
        bigquery = _bigquery()
        dataset = bigquery.Dataset(f"{self._cfg.project}.{self.schema}")
        if self._cfg.location:
            dataset.location = self._cfg.location
        self.client.create_dataset(dataset, exists_ok=True)

    def _ensure_state_table(self) -> None:
        columns = self._state_columns(_STATE_TABLE)
        if columns is None:
            self.execute(self._create_state_table_sql(_STATE_TABLE))
            return
        if columns == _STATE_V2_COLUMNS:
            return
        if columns == _STATE_V1_COLUMNS:
            self._migrate_v1_state()
            return

        shape = ", ".join(name for name, _type, _mode in columns)
        raise AdapterError(
            "Unsupported dbt_ml_state schema; expected the legacy v1 or current "
            f"v2 shape, found columns: {shape or '(none)'}. Back up the table and "
            "run --full-refresh after resolving the state schema."
        )

    def _create_state_table_sql(
        self,
        table: str,
        *,
        if_not_exists: bool = True,
        select: str | None = None,
    ) -> str:
        action = "CREATE TABLE IF NOT EXISTS" if if_not_exists else "CREATE TABLE"
        suffix = f"\n            AS {select}" if select is not None else ""
        return f"""
            {action} {self.table_ref(table)} (
                model_name STRING NOT NULL,
                state_scope STRING NOT NULL,
                target_identity STRING NOT NULL,
                record_key STRING NOT NULL,
                input_fingerprint STRING NOT NULL,
                code_version STRING NOT NULL,
                last_run_at TIMESTAMP NOT NULL
            ){suffix}
        """

    def _state_columns(self, table: str) -> tuple[tuple[str, str, str], ...] | None:
        try:
            bq_table = self.client.get_table(self._table_id(table))
        except _not_found_error():
            return None
        return tuple(
            (
                str(field.name),
                str(field.field_type).upper(),
                str(field.mode).upper(),
            )
            for field in bq_table.schema
        )

    def _migrate_v1_state(self) -> None:
        migration_table = f"{_STATE_MIGRATION_PREFIX}{uuid4().hex}"
        migration_ref = self.table_ref(migration_table)
        source_select = (
            "SELECT model_name AS model_name, "
            "'materialization' AS state_scope, "
            "'warehouse-v1' AS target_identity, "
            "document_id AS record_key, "
            "content_hash AS input_fingerprint, "
            "code_version AS code_version, "
            f"last_run_at AS last_run_at FROM {self._state_ref}"
        )
        duplicate_count = self.scalar(
            "SELECT COUNT(*) FROM ("
            "SELECT model_name, document_id "
            f"FROM {self._state_ref} "
            "GROUP BY model_name, document_id HAVING COUNT(*) > 1)"
        )
        if int(duplicate_count or 0):
            raise AdapterError(
                "Cannot migrate BigQuery dbt_ml_state because legacy "
                "(model_name, document_id) keys contain duplicates. Back up and "
                "deduplicate the state table before retrying."
            )
        migration_created = False
        try:
            self.execute(
                self._create_state_table_sql(
                    migration_table,
                    if_not_exists=False,
                    select=source_select,
                )
            )
            migration_created = True
            counts = self.rows(
                "SELECT "
                f"(SELECT COUNT(*) FROM {self._state_ref}), "
                f"(SELECT COUNT(*) FROM {migration_ref})"
            )
            if len(counts) != 1 or int(counts[0][0]) != int(counts[0][1]):
                source_count = int(counts[0][0]) if counts else 0
                migrated_count = int(counts[0][1]) if counts else 0
                raise AdapterError(
                    "BigQuery state migration row-count verification failed: "
                    f"expected {source_count}, migrated {migrated_count}"
                )
            self.execute(f"CREATE OR REPLACE TABLE {self._state_ref} COPY {migration_ref}")
        except BaseException as error:
            if migration_created:
                try:
                    self.client.delete_table(
                        self._table_id(migration_table), not_found_ok=True
                    )
                except Exception as cleanup_error:
                    error.add_note(
                        "Failed to clean BigQuery state migration table: "
                        f"{cleanup_error}"
                    )
            raise
        else:
            self.client.delete_table(
                self._table_id(migration_table), not_found_ok=True
            )

    # ─── identity ────────────────────────────────────────────────────────

    @property
    def catalog(self) -> str:
        return self._cfg.project

    @property
    def schema_ref(self) -> str:
        return f"{self.quote_ident(self.catalog)}.{self.quote_ident(self.schema)}"

    def quote_ident(self, name: str) -> str:
        """BigQuery quotes identifiers with backticks; embedded backticks and
        backslashes are backslash-escaped."""
        return "`" + name.replace("\\", "\\\\").replace("`", "\\`") + "`"

    @property
    def _state_ref(self) -> str:
        return self.table_ref(_STATE_TABLE)

    def _table_id(self, table: str) -> str:
        """Unquoted `project.dataset.table` id for client API calls."""
        return f"{self._cfg.project}.{self.schema}.{table}"

    # ─── querying ────────────────────────────────────────────────────────

    def _run_query(
        self,
        sql: str,
        params: list[Any] | None = None,
        *,
        job_labels: dict[str, str] | None = None,
    ) -> Any:
        bigquery = _bigquery()
        cfg = self._cfg
        kwargs: dict[str, Any] = {}
        if params or job_labels:
            job_config = bigquery.QueryJobConfig()
            if params:
                job_config.query_parameters = to_query_parameters(params)
            if job_labels:
                job_config.labels = dict(job_labels)
            kwargs["job_config"] = job_config
        if cfg.job_creation_timeout_seconds is not None:
            kwargs["timeout"] = cfg.job_creation_timeout_seconds
        if cfg.job_retries == 0:
            kwargs["job_retry"] = None
        elif cfg.job_retry_deadline_seconds is not None:
            from google.cloud.bigquery.retry import DEFAULT_JOB_RETRY

            kwargs["job_retry"] = DEFAULT_JOB_RETRY.with_deadline(
                cfg.job_retry_deadline_seconds
            )
        job = self.client.query(sql, **kwargs)
        job.result(timeout=cfg.job_execution_timeout_seconds)
        return job

    def execute(self, sql: str, params: list[Any] | None = None) -> Any:
        return self._run_query(sql, params).result()

    def scalar(self, sql: str, params: list[Any] | None = None) -> Any:
        for row in self._run_query(sql, params).result():
            return row[0]
        return None

    def rows(self, sql: str, params: list[Any] | None = None) -> list[tuple[Any, ...]]:
        return [tuple(row.values()) for row in self._run_query(sql, params).result()]

    def query_df(self, sql: str, params: list[Any] | None = None) -> pl.DataFrame:
        arrow = self._run_query(sql, params).to_arrow()
        return cast(pl.DataFrame, pl.from_arrow(arrow))

    def list_tables(self) -> list[str]:
        names = sorted(
            t.table_id
            for t in self.client.list_tables(f"{self._cfg.project}.{self.schema}")
        )
        return [
            n
            for n in names
            if n != "dbt_ml_state"
            and not n.startswith(("dbt_ml_test_failures__", "dbt_ml_staging__"))
        ]

    # ─── materialization ─────────────────────────────────────────────────

    @staticmethod
    def _layout(options: BaseModel | None) -> BigQueryWarehouseOptions | None:
        if options is None:
            return None
        assert isinstance(options, BigQueryWarehouseOptions)
        return options

    def _apply_layout_to_load(
        self, job_config: Any, options: BaseModel | None
    ) -> None:
        """Partitioning/clustering/encryption on a load job creating the
        target table; labels ride along as job labels."""
        layout = self._layout(options)
        if layout is None:
            return
        bigquery = _bigquery()
        if layout.kms_key_name:
            job_config.destination_encryption_configuration = (
                bigquery.EncryptionConfiguration(kms_key_name=layout.kms_key_name)
            )
        if layout.labels:
            job_config.labels = dict(layout.labels)
        if layout.cluster_by:
            job_config.clustering_fields = list(layout.cluster_by)
        pb = layout.partition_by
        if pb is None:
            return
        if pb.data_type == "int64":
            assert pb.range is not None
            job_config.range_partitioning = bigquery.RangePartitioning(
                field=pb.field,
                range_=bigquery.PartitionRange(
                    start=pb.range.start, end=pb.range.end, interval=pb.range.interval
                ),
            )
        else:
            job_config.time_partitioning = bigquery.TimePartitioning(
                type_=pb.granularity.upper(), field=pb.field
            )

    def _partition_expression(self, pb: BigQueryPartitionBy) -> str:
        if pb.data_type == "int64":
            assert pb.field is not None and pb.range is not None
            return (
                f"RANGE_BUCKET({self.quote_ident(pb.field)}, "
                f"GENERATE_ARRAY({pb.range.start}, {pb.range.end}, "
                f"{pb.range.interval}))"
            )
        granularity = pb.granularity.upper()
        if pb.field is None:
            if granularity == "DAY":
                return "_PARTITIONDATE"
            return f"TIMESTAMP_TRUNC(_PARTITIONTIME, {granularity})"
        return self._partition_scalar(pb, self.quote_ident(pb.field))

    @staticmethod
    def _partition_scalar(pb: BigQueryPartitionBy, ref: str) -> str:
        """The partition identity of `ref` (a column reference) for a
        time-partitioned table."""
        granularity = pb.granularity.upper()
        if pb.data_type == "date":
            return ref if granularity == "DAY" else f"DATE_TRUNC({ref}, {granularity})"
        trunc = "TIMESTAMP_TRUNC" if pb.data_type == "timestamp" else "DATETIME_TRUNC"
        return f"{trunc}({ref}, {granularity})"

    def _insert_overwrite_script(
        self,
        table: str,
        staging: str,
        columns: list[str],
        pb: BigQueryPartitionBy,
    ) -> str:
        """dbt-bigquery's dynamic insert_overwrite: collect the partitions
        present in the batch, then one MERGE that drops those partitions and
        inserts the batch. Rows with a NULL partition value insert without
        clearing the NULL partition."""
        assert pb.field is not None
        array_type = {
            "date": "DATE",
            "timestamp": "TIMESTAMP",
            "datetime": "DATETIME",
        }[pb.data_type]
        field = self.quote_ident(pb.field)
        source_expr = self._partition_scalar(pb, field)
        target_expr = self._partition_scalar(pb, f"target.{field}")
        insert_columns = ", ".join(self.quote_ident(c) for c in columns)
        insert_values = ", ".join(f"source.{self.quote_ident(c)}" for c in columns)
        return (
            f"DECLARE dbt_ml_partitions ARRAY<{array_type}>;\n"
            f"SET dbt_ml_partitions = ARRAY(SELECT DISTINCT {source_expr} "
            f"FROM {self.table_ref(staging)} WHERE {field} IS NOT NULL);\n"
            f"MERGE {self.table_ref(table)} AS target "
            f"USING {self.table_ref(staging)} AS source ON FALSE "
            f"WHEN NOT MATCHED BY SOURCE AND {target_expr} "
            "IN UNNEST(dbt_ml_partitions) THEN DELETE "
            f"WHEN NOT MATCHED THEN INSERT ({insert_columns}) "
            f"VALUES ({insert_values});"
        )

    @staticmethod
    def _sql_string(value: str) -> str:
        return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"

    def _table_option_entries(
        self, layout: BigQueryWarehouseOptions, *, include_kms: bool
    ) -> list[str]:
        """`key = value` entries for OPTIONS(...) in CREATE and ALTER DDL.
        kms_key_name is create-only: on load-created tables encryption is
        already set via the job config, and re-keying via ALTER is out of
        scope for a materialization run."""
        entries: list[str] = []
        if layout.require_partition_filter is not None:
            entries.append(
                "require_partition_filter = "
                + ("TRUE" if layout.require_partition_filter else "FALSE")
            )
        if layout.partition_expiration_days is not None:
            entries.append(
                f"partition_expiration_days = {layout.partition_expiration_days}"
            )
        if layout.hours_to_expiration is not None:
            entries.append(
                "expiration_timestamp = TIMESTAMP_ADD(CURRENT_TIMESTAMP(), "
                f"INTERVAL {layout.hours_to_expiration} HOUR)"
            )
        if layout.labels:
            pairs = ", ".join(
                f"({self._sql_string(k)}, {self._sql_string(v)})"
                for k, v in sorted(layout.labels.items())
            )
            entries.append(f"labels = [{pairs}]")
        if include_kms and layout.kms_key_name:
            entries.append(f"kms_key_name = {self._sql_string(layout.kms_key_name)}")
        return entries

    def _ddl_layout_clauses(self, options: BaseModel | None) -> str:
        """` PARTITION BY … CLUSTER BY … OPTIONS (…)` for CREATE TABLE DDL,
        or ''."""
        layout = self._layout(options)
        if layout is None:
            return ""
        clauses: list[str] = []
        if layout.partition_by is not None:
            clauses.append(
                f"PARTITION BY {self._partition_expression(layout.partition_by)}"
            )
        if layout.cluster_by:
            clauses.append(
                "CLUSTER BY "
                + ", ".join(self.quote_ident(c) for c in layout.cluster_by)
            )
        option_entries = self._table_option_entries(layout, include_kms=True)
        if option_entries:
            clauses.append(f"OPTIONS ({', '.join(option_entries)})")
        return (" " + " ".join(clauses)) if clauses else ""

    def _apply_post_create_options(
        self, table: str, options: BaseModel | None
    ) -> None:
        """Table options a load job cannot express, set right after a load
        creates the table."""
        layout = self._layout(options)
        if layout is None:
            return
        entries = self._table_option_entries(layout, include_kms=False)
        if not entries:
            return
        self._run_query(
            f"ALTER TABLE {self.table_ref(table)} SET OPTIONS ({', '.join(entries)})",
            job_labels=layout.labels,
        )

    def _load_parquet(self, table: str, df: pl.DataFrame, job_config: Any) -> None:
        buffer = io.BytesIO()
        df.write_parquet(buffer)
        buffer.seek(0)
        job = self.client.load_table_from_file(
            buffer, self._table_id(table), job_config=job_config
        )
        job.result()

    def materialize_full(
        self, table: str, df: pl.DataFrame, *, options: BaseModel | None = None
    ) -> int:
        if df.width == 0:
            self.drop_table(table)
            return 0
        bigquery = _bigquery()
        job_config = bigquery.LoadJobConfig(
            source_format=bigquery.SourceFormat.PARQUET,
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        )
        layout = self._layout(options)
        if layout is None:
            self._load_parquet(table, df, job_config)
            return df.height

        # A load job cannot change an existing table's partitioning or
        # clustering spec, and dropping the target up front would make a bad
        # layout declaration destructive. Build the replacement in a staging
        # table — the load validates partition/cluster columns against the
        # data — then swap. The target only drops after the replacement is
        # fully created and configured.
        self._apply_layout_to_load(job_config, options)
        staging = f"dbt_ml_staging__{table}__{uuid4().hex[:12]}"
        target_dropped = False
        try:
            self._load_parquet(staging, df, job_config)
            self._apply_post_create_options(staging, options)
            self.drop_table(table)
            target_dropped = True
            self._rename_table(staging, table, job_labels=layout.labels or None)
        except BaseException as error:
            if target_dropped:
                error.add_note(
                    f"Target '{table}' was dropped but the swap failed; the "
                    f"replacement data is preserved in '{staging}'."
                )
            else:
                try:
                    self.drop_table(staging)
                except Exception as cleanup_error:
                    error.add_note(f"Failed to clean staging table: {cleanup_error}")
            raise
        return df.height

    def _rename_table(
        self, current: str, new_name: str, *, job_labels: dict[str, str] | None = None
    ) -> None:
        self._run_query(
            f"ALTER TABLE {self.table_ref(current)} "
            f"RENAME TO {self.quote_ident(new_name)}",
            job_labels=job_labels,
        )

    def materialize_incremental(
        self,
        table: str,
        df: pl.DataFrame,
        *,
        key_col: str,
        on_schema_change: str = "fail",
        options: BaseModel | None = None,
    ) -> int:
        if df.height == 0:
            return 0
        validate_incremental_keys(df, key_col)
        bigquery = _bigquery()
        load_df = df
        allow_field_addition = False

        existing = self._table_columns(table)
        if existing is None:
            job_config = bigquery.LoadJobConfig(
                source_format=bigquery.SourceFormat.PARQUET,
                write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
            )
            self._apply_layout_to_load(job_config, options)
            self._load_parquet(table, load_df, job_config)
            self._apply_post_create_options(table, options)
            return df.height

        if key_col not in existing:
            raise AdapterError(
                f"Incremental target '{table}' is missing key column '{key_col}'"
            )
        plan = plan_schema_change(existing, list(df.columns), on_schema_change, table)
        allow_field_addition = plan.allow_field_addition
        if plan.columns_to_load != list(df.columns):
            load_df = df.select(plan.columns_to_load)

        layout = self._layout(options)
        job_labels = layout.labels if layout is not None else None
        insert_overwrite = (
            layout is not None and layout.incremental_strategy == "insert_overwrite"
        )
        if insert_overwrite:
            assert layout is not None and layout.partition_by is not None
            if layout.partition_by.field not in load_df.columns:
                raise AdapterError(
                    f"insert_overwrite on '{table}' needs partition column "
                    f"'{layout.partition_by.field}' in the incremental batch"
                )

        if allow_field_addition:
            schema_config = bigquery.LoadJobConfig(
                source_format=bigquery.SourceFormat.PARQUET,
                write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
                schema_update_options=[bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION],
            )
            self._load_parquet(table, load_df.head(0), schema_config)

        staging = f"dbt_ml_staging__{table}__{uuid4().hex[:12]}"
        staging_config = bigquery.LoadJobConfig(
            source_format=bigquery.SourceFormat.PARQUET,
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        )
        try:
            self._load_parquet(staging, load_df, staging_config)
            if insert_overwrite:
                assert layout is not None and layout.partition_by is not None
                self._run_query(
                    self._insert_overwrite_script(
                        table, staging, list(load_df.columns), layout.partition_by
                    ),
                    job_labels=job_labels,
                )
            else:
                final_columns = [*existing]
                final_columns.extend(
                    c for c in load_df.columns if c not in final_columns
                )
                assignments = ", ".join(
                    f"target.{self.quote_ident(column)} = "
                    f"source.{self.quote_ident(column)}"
                    if column in load_df.columns
                    else f"target.{self.quote_ident(column)} = NULL"
                    for column in final_columns
                )
                insert_columns = ", ".join(
                    self.quote_ident(column) for column in load_df.columns
                )
                insert_values = ", ".join(
                    f"source.{self.quote_ident(column)}" for column in load_df.columns
                )
                self._run_query(
                    f"MERGE {self.table_ref(table)} AS target "
                    f"USING {self.table_ref(staging)} AS source "
                    f"ON target.{self.quote_ident(key_col)} = "
                    f"source.{self.quote_ident(key_col)} "
                    f"WHEN MATCHED THEN UPDATE SET {assignments} "
                    f"WHEN NOT MATCHED THEN INSERT ({insert_columns}) "
                    f"VALUES ({insert_values})",
                    job_labels=job_labels,
                )
        except BaseException as error:
            try:
                self.drop_table(staging)
            except Exception as cleanup_error:
                error.add_note(f"Failed to clean staging table: {cleanup_error}")
            raise
        else:
            self.drop_table(staging)
        return df.height

    def materialize_full_chunks(
        self,
        table: str,
        chunks: Iterable[pl.DataFrame],
        *,
        options: BaseModel | None = None,
    ) -> int:
        bigquery = _bigquery()
        layout = self._layout(options)
        job_labels = layout.labels if layout is not None else None
        staging = f"dbt_ml_staging__{table}__{uuid4().hex[:12]}"
        replacement: str | None = None
        target_dropped = False
        total = 0
        first = True
        try:
            for df in chunks:
                job_config = bigquery.LoadJobConfig(
                    source_format=bigquery.SourceFormat.PARQUET,
                    write_disposition=(
                        bigquery.WriteDisposition.WRITE_TRUNCATE
                        if first
                        else bigquery.WriteDisposition.WRITE_APPEND
                    ),
                )
                if job_labels:
                    job_config.labels = dict(job_labels)
                if not first:
                    # Union intra-run schema drift: new columns are added,
                    # columns missing from a chunk load as NULL.
                    job_config.schema_update_options = [
                        bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION
                    ]
                self._load_parquet(staging, df, job_config)
                first = False
                total += df.height
            if first:
                return self.materialize_full(table, pl.DataFrame())
            layout_clauses = self._ddl_layout_clauses(options)
            if layout_clauses:
                # CREATE OR REPLACE cannot change an existing table's
                # partitioning spec, and dropping the target before the CTAS
                # is validated would make a bad layout declaration
                # destructive. Materialize the replacement (validating the
                # declared layout against real columns), then swap.
                replacement = f"dbt_ml_staging__{table}__{uuid4().hex[:12]}"
                self._run_query(
                    f"CREATE TABLE {self.table_ref(replacement)}{layout_clauses} "
                    f"AS SELECT * FROM {self.table_ref(staging)}",
                    job_labels=job_labels,
                )
                self.drop_table(table)
                target_dropped = True
                self._rename_table(replacement, table, job_labels=job_labels)
            else:
                self._run_query(
                    f"CREATE OR REPLACE TABLE {self.table_ref(table)} "
                    f"AS SELECT * FROM {self.table_ref(staging)}",
                    job_labels=job_labels,
                )
        except BaseException as error:
            if target_dropped:
                error.add_note(
                    f"Target '{table}' was dropped but the swap failed; the "
                    f"replacement data is preserved in '{replacement}'."
                )
                cleanup = [staging]
            else:
                cleanup = [staging, replacement] if replacement else [staging]
            for name in cleanup:
                try:
                    self.drop_table(name)
                except Exception as cleanup_error:
                    error.add_note(f"Failed to clean staging table: {cleanup_error}")
            raise
        else:
            self.drop_table(staging)
        return total

    def _table_columns(self, table: str) -> list[str] | None:
        try:
            bq_table = self.client.get_table(self._table_id(table))
        except _not_found_error():
            return None
        return [f.name for f in bq_table.schema]

    def delete_rows(self, table: str, *, key_col: str, keys: list[str]) -> int:
        if not keys or self._table_columns(table) is None:
            return 0
        job = self._run_query(
            f"DELETE FROM {self.table_ref(table)} "
            f"WHERE {self.quote_ident(key_col)} IN UNNEST(?)",
            [list(keys)],
        )
        return int(job.num_dml_affected_rows or 0)

    def delete_rows_and_state(
        self,
        table: str,
        *,
        key_col: str,
        keys: Sequence[str],
        state_scope: StateScope,
        state_record_keys: Sequence[str] | None = None,
    ) -> int:
        target_keys = list(keys)
        scoped_keys = target_keys if state_record_keys is None else list(state_record_keys)
        validate_state_keys(target_keys)
        validate_state_keys(scoped_keys)
        target_exists = bool(target_keys) and self._table_columns(table) is not None
        if not target_exists and not scoped_keys:
            return 0

        statements = ["DECLARE deleted_count INT64 DEFAULT 0;", "BEGIN TRANSACTION;"]
        params: list[Any] = []
        if target_exists:
            statements.extend(
                [
                    f"DELETE FROM {self.table_ref(table)} "
                    f"WHERE {self.quote_ident(key_col)} IN UNNEST(?);",
                    "SET deleted_count = @@row_count;",
                ]
            )
            params.append(target_keys)
        if scoped_keys:
            statements.append(
                f"DELETE FROM {self._state_ref} "
                "WHERE model_name = ? AND state_scope = ? AND target_identity = ? "
                "AND record_key IN UNNEST(?);"
            )
            params.extend(
                [
                    state_scope.model_name,
                    state_scope.stage,
                    state_scope.target_identity,
                    scoped_keys,
                ]
            )
        statements.extend(["COMMIT TRANSACTION;", "SELECT deleted_count;"])
        deleted = self.scalar("\n".join(statements), params)
        return int(deleted or 0)

    def drop_table(self, table: str) -> None:
        self.client.delete_table(self._table_id(table), not_found_ok=True)

    def _reset_storage_for_test(self) -> str:
        """Drop the isolated test dataset used by live adapter integration tests."""
        cfg = self._cfg
        dataset_id = f"{cfg.project}.{cfg.schema_name}"
        client = self._client or self._make_client()
        try:
            client.delete_dataset(dataset_id, delete_contents=True, not_found_ok=True)
        finally:
            if self._client is None:
                client.close()
        return f"dropped BigQuery dataset {dataset_id}"

    # ─── state CRUD ──────────────────────────────────────────────────────

    def fetch_state(self, scope: StateScope) -> dict[str, StateValue]:
        result = self.rows(
            f"SELECT record_key, input_fingerprint, code_version FROM {self._state_ref} "
            "WHERE model_name = ? AND state_scope = ? AND target_identity = ?",
            [scope.model_name, scope.stage, scope.target_identity],
        )
        state: dict[str, StateValue] = {}
        for row in result:
            record_key = str(row[0])
            if record_key in state:
                raise AdapterError(
                    "BigQuery state contains duplicate record keys in the requested "
                    "scope; repair the state table before retrying."
                )
            state[record_key] = StateValue(str(row[1]), str(row[2]))
        return state

    def upsert_state(self, scope: StateScope, records: Sequence[StateRecord]) -> None:
        validate_state_records(records)
        if not records:
            return
        self._merge_state(scope, records, replace=False)

    def replace_state(self, scope: StateScope, records: Sequence[StateRecord]) -> None:
        validate_state_records(records)
        self._merge_state(scope, records, replace=True)

    def _merge_state(
        self,
        scope: StateScope,
        records: Sequence[StateRecord],
        *,
        replace: bool,
    ) -> None:
        record_keys = [record.record_key for record in records]
        fingerprints = [record.input_fingerprint for record in records]
        versions = [record.code_version for record in records]
        replace_clause = (
            """
            WHEN NOT MATCHED BY SOURCE
                AND target.model_name = ?
                AND target.state_scope = ?
                AND target.target_identity = ?
            THEN DELETE
            """
            if replace
            else ""
        )
        params: list[Any] = [
            scope.model_name,
            scope.stage,
            scope.target_identity,
            record_keys,
            fingerprints,
            versions,
        ]
        if replace:
            params.extend([scope.model_name, scope.stage, scope.target_identity])
        self._run_query(
            f"""
            MERGE {self._state_ref} AS target
            USING (
                SELECT
                    ? AS model_name,
                    ? AS state_scope,
                    ? AS target_identity,
                    ids[OFFSET(o)] AS record_key,
                    fs[OFFSET(o)] AS input_fingerprint,
                    vs[OFFSET(o)] AS code_version
                FROM (SELECT ? AS ids, ? AS fs, ? AS vs),
                    UNNEST(GENERATE_ARRAY(0, ARRAY_LENGTH(ids) - 1)) AS o
            ) AS source
            ON target.model_name = source.model_name
                AND target.state_scope = source.state_scope
                AND target.target_identity = source.target_identity
                AND target.record_key = source.record_key
            WHEN MATCHED THEN UPDATE SET
                input_fingerprint = source.input_fingerprint,
                code_version = source.code_version,
                last_run_at = CURRENT_TIMESTAMP()
            WHEN NOT MATCHED THEN INSERT
                (model_name, state_scope, target_identity, record_key,
                 input_fingerprint, code_version, last_run_at)
                VALUES (source.model_name, source.state_scope, source.target_identity,
                        source.record_key, source.input_fingerprint,
                        source.code_version, CURRENT_TIMESTAMP())
            {replace_clause}
            """,
            params,
        )

    def clear_state(self, scope: StateScope) -> None:
        self._run_query(
            f"DELETE FROM {self._state_ref} "
            "WHERE model_name = ? AND state_scope = ? AND target_identity = ?",
            [scope.model_name, scope.stage, scope.target_identity],
        )

    def delete_state(self, scope: StateScope, record_keys: Sequence[str]) -> None:
        validate_state_keys(record_keys)
        if not record_keys:
            return
        self._run_query(
            f"DELETE FROM {self._state_ref} "
            "WHERE model_name = ? AND state_scope = ? AND target_identity = ? "
            "AND record_key IN UNNEST(?)",
            [
                scope.model_name,
                scope.stage,
                scope.target_identity,
                list(record_keys),
            ],
        )
