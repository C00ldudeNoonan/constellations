"""BigQuery warehouse adapter (issue #83).

Materializes stel tables into a BigQuery dataset; incremental state lives
in the same dataset (`dbt_ml_state`), so a project can run against BigQuery
with no DuckDB involvement in materialization or state.

The google-cloud-bigquery dependency is an optional extra — this module
imports lazily so the adapter registers without it and fails with an
install hint only when actually used:

    pip install 'stel[bigquery]'

DataFrames travel as Parquet load jobs (polars → parquet bytes → load), so
column matching is by name and never positional. Queries use positional `?`
parameters, converted to BigQuery query parameters here.
"""
from __future__ import annotations

import io
import logging
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal, cast
from urllib.parse import quote
from uuid import uuid4

import polars as pl
import pyarrow as pa
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    GetCoreSchemaHandler,
    GetJsonSchemaHandler,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import CoreSchema, PydanticCustomError, core_schema

from ..config.identifiers import DEFAULT_SCHEMA_NAME
from ..config.profile import WarehouseConfig
from ..credentials import (
    CredentialFreeUrl,
    CredentialReference,
    CredentialResolutionError,
)
from ..hashing import canonical_fingerprint
from ..progress import get_reporter
from ..sql_models import build_key_check_sql
from .base import (
    INTERNAL_TABLE_PREFIXES,
    SERVING_LEDGER_TABLE,
    STAGING_TABLE_PREFIX,
    AdapterError,
    ReadPredicate,
    ReadPredicateOperator,
    SqlMaterializationResult,
    SqlRelationColumn,
    SqlRelationSchema,
    StaleStateFenceError,
    StatePage,
    StatePageReader,
    StatePageRecord,
    StatePageRequest,
    StateRecord,
    StateScope,
    StateScopeFence,
    StateValue,
    TableReadRequest,
    TableReadSnapshot,
    WarehouseAdapter,
    WarehouseCapability,
    change_predicate,
    decode_state_cursor,
    encode_state_cursor,
    plan_schema_change,
    sanitized_adapter_cause,
    staging_table_name,
    validate_incremental_keys,
    validate_state_keys,
    validate_state_records,
    validate_update_when_changed_columns,
)
from .base import STATE_TABLE as _STATE_TABLE
from .registry import register

log = logging.getLogger(__name__)

_INSTALL_HINT = (
    "BigQuery support requires google-cloud-bigquery. "
    "Install it with: pip install 'stel[bigquery]'"
)


def _log_publication(
    operation: str,
    table_ref: str,
    job: Any,
    *,
    key: str | None = None,
) -> None:
    """Emit safe, structured telemetry for one incremental publication job
    (issue #292). The BigQuery job id, bytes processed, and DML-affected row
    count let an operator match stel's own jobs against BigQuery job history
    and INFORMATION_SCHEMA, distinguishing many tiny stel flushes from an
    overlapping external orchestrator run. Only job-level statistics and the
    output relation are surfaced — never SQL text or row values.

    Sent to both verbose channels: the `stel` INFO log (the active channel on
    non-TTY / captured orchestrator runs) and the progress reporter (the active
    channel on an interactive TTY, where the log handler is removed to protect
    the progress bar). `_enable_verbose_output` only ever enables one of them,
    so a verbose run sees the line exactly once regardless of which."""
    message = (
        f"published {operation}: table={table_ref} "
        f"job_id={getattr(job, 'job_id', None)} "
        f"rows_affected={getattr(job, 'num_dml_affected_rows', None)} "
        f"bytes_processed={getattr(job, 'total_bytes_processed', None)}"
        + (f" key={key}" if key else "")
    )
    log.info("%s", message)
    get_reporter().publication(message)


_STATE_MIGRATION_PREFIX = f"{STAGING_TABLE_PREFIX}state_migration_v2__"
# Above this many records, `_merge_state` stages the set via a Parquet load and
# one MERGE instead of passing it inline as array query parameters — whose single
# request grows with the record count and fails at scale (issue #256). Below it,
# the inline MERGE is kept: fewer round trips, no staging table.
_STATE_MERGE_INLINE_MAX = 5000
# Rows per Parquet load when staging a large state set (bounds peak memory).
_STATE_MERGE_LOAD_BATCH = 20000
# Keys per request for the array-parameter delete/fetch paths, so a large key
# set is split across bounded requests instead of one that fails at scale (#260).
_KEY_REQUEST_BATCH = 5000


def _chunked(items: Sequence[Any], size: int) -> Iterator[list[Any]]:
    for start in range(0, len(items), size):
        yield list(items[start : start + size])
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
_ENV_REFERENCE_JSON_PATTERN = (
    r"^\{\{[ \t]*env_var\([ \t]*(['\"])[A-Za-z_][A-Za-z0-9_]*"
    r"\1[ \t]*\)[ \t]*\}\}$"
)


class _ProtectedBigQueryInput:
    __slots__ = ("value",)

    def __init__(self, value: Any) -> None:
        self.value = value

    def __repr__(self) -> str:
        return "<redacted>"

    __str__ = __repr__

    def take(self) -> Any:
        value = self.value
        self.value = None
        return value


class BigQueryWarehouseConfig(WarehouseConfig):
    """Non-secret fields mirror dbt-bigquery's profile. Secret-bearing fields
    accept only an exact ``env_var()`` reference, which remains opaque until
    native credential construction. `method:` may be omitted and is inferred
    from the selected auth fields.

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
        default=DEFAULT_SCHEMA_NAME,
        validation_alias=AliasChoices("dataset", "schema", "schema_name"),
        serialization_alias="schema",
    )
    location: str | None = None

    # ─── auth (dbt-bigquery parity) ───────────────────────────────────────
    method: AuthMethod | None = None
    keyfile: Path | CredentialReference | None = Field(
        default=None,
        repr=False,
        exclude=True,
    )
    keyfile_json: CredentialReference | None = Field(
        default=None,
        repr=False,
        exclude=True,
    )
    token: CredentialReference | None = Field(
        default=None,
        repr=False,
        exclude=True,
    )
    refresh_token: CredentialReference | None = Field(
        default=None,
        repr=False,
        exclude=True,
    )
    client_id: str | None = None
    client_secret: CredentialReference | None = Field(
        default=None,
        repr=False,
        exclude=True,
    )
    token_uri: CredentialReference | CredentialFreeUrl | None = Field(
        default=None,
        repr=False,
        exclude=True,
    )
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
    # Target-scoped physical-layout policy. The adapter validates and merges
    # this mapping before source discovery; model-level warehouse_options win.
    warehouse_defaults: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: Any,
        handler: GetCoreSchemaHandler,
    ) -> CoreSchema:
        model_schema = handler(source_type)
        return core_schema.chain_schema(
            [
                core_schema.no_info_plain_validator_function(
                    _ProtectedBigQueryInput,
                    json_schema_input_schema=model_schema,
                ),
                core_schema.no_info_plain_validator_function(
                    cls._prepare_model_input,
                    json_schema_input_schema=model_schema,
                ),
                model_schema,
            ]
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls,
        schema: CoreSchema,
        handler: GetJsonSchemaHandler,
    ) -> JsonSchemaValue:
        json_schema = handler(schema)
        resolved = handler.resolve_ref_schema(json_schema)
        properties = resolved.get("properties", {})
        for field_name in (
            "client_secret",
            "keyfile",
            "keyfile_json",
            "refresh_token",
            "token",
            "token_uri",
        ):
            property_schema = properties.get(field_name, {})
            for candidate in property_schema.get("anyOf", ()):
                if candidate.get("format") == "password":
                    candidate["pattern"] = _ENV_REFERENCE_JSON_PATTERN
                    candidate["description"] = (
                        "Exact {{ env_var('NAME') }} credential reference"
                    )
        return json_schema

    @classmethod
    def _prepare_model_input(cls, wrapped: _ProtectedBigQueryInput) -> Any:
        raw = wrapped.take()
        if isinstance(raw, cls):
            return raw
        if not isinstance(raw, Mapping):
            raise PydanticCustomError(
                "bigquery_config",
                "BigQuery warehouse config must be a mapping",
            )
        try:
            return cls.prepare_profile_input(raw)
        except (TypeError, ValueError) as error:
            raise PydanticCustomError(
                "protected_bigquery_config",
                "{message}",
                {"message": str(error)},
            ) from None

    @classmethod
    def prepare_profile_input(cls, raw: Mapping[str, Any]) -> dict[str, Any]:
        prepared = dict(raw)
        if "warehouse_defaults" in prepared:
            prepared["warehouse_defaults"] = _prepare_warehouse_defaults_input(
                prepared["warehouse_defaults"]
            )
        for field_name in (
            "keyfile_json",
            "token",
            "refresh_token",
            "client_secret",
        ):
            value = prepared.get(field_name)
            if value is None or isinstance(value, CredentialReference):
                continue
            try:
                prepared[field_name] = (
                    CredentialReference.from_env_var_expression(value)
                )
            except (TypeError, ValueError):
                raise ValueError(
                    f"`{field_name}` must be an exact "
                    "{{ env_var('NAME') }} reference with no default; move "
                    "literal credentials to an environment variable"
                ) from None

        for field_name in ("keyfile", "token_uri"):
            value = prepared.get(field_name)
            if value is None or isinstance(value, CredentialReference):
                continue
            if isinstance(value, str) and (
                "{{" in value or "env_var(" in value
            ):
                try:
                    prepared[field_name] = (
                        CredentialReference.from_env_var_expression(value)
                    )
                except (TypeError, ValueError):
                    raise ValueError(
                        f"`{field_name}` environment configuration must be an "
                        "exact {{ env_var('NAME') }} reference with no default"
                    ) from None

        token_uri = prepared.get("token_uri")
        if isinstance(token_uri, str):
            prepared["token_uri"] = _validate_token_uri(token_uri)
        return prepared

    @model_validator(mode="after")
    def _resolve_auth_method(self) -> BigQueryWarehouseConfig:
        if self.keyfile is not None and self.keyfile_json is not None:
            raise ValueError("set either `keyfile` or `keyfile_json`, not both")

        oauth_values = (
            self.token,
            self.refresh_token,
            self.client_id,
            self.client_secret,
            self.token_uri,
        )
        has_oauth_values = any(value is not None for value in oauth_values)
        if (self.keyfile is not None or self.keyfile_json is not None) and (
            has_oauth_values
        ):
            raise ValueError(
                "service-account and OAuth credential fields cannot be combined"
            )

        inferred: AuthMethod
        if self.keyfile is not None:
            inferred = "service-account"
        elif self.keyfile_json is not None:
            inferred = "service-account-json"
        elif has_oauth_values:
            inferred = "oauth-secrets"
        else:
            inferred = "oauth"
        if self.method is None:
            self.method = inferred

        if self.method == "service-account" and self.keyfile is None:
            raise ValueError("method 'service-account' requires `keyfile`")
        if self.method == "service-account-json" and self.keyfile_json is None:
            raise ValueError("method 'service-account-json' requires `keyfile_json`")
        if self.method == "oauth-secrets":
            refresh_values = (
                self.refresh_token,
                self.client_id,
                self.client_secret,
                self.token_uri,
            )
            has_refresh_set = all(value is not None for value in refresh_values)
            has_partial_refresh_set = any(
                value is not None for value in refresh_values
            ) and not has_refresh_set
            if has_partial_refresh_set or (
                self.token is None and not has_refresh_set
            ):
                raise ValueError(
                    "method 'oauth-secrets' requires `token`, or all of "
                    "`refresh_token`, `client_id`, `client_secret`, `token_uri`"
                )

        if self.method != inferred:
            raise ValueError(
                f"method '{self.method}' conflicts with the credential fields "
                f"provided (which imply '{inferred}')"
            )
        return self

    @model_validator(mode="after")
    def _validate_warehouse_defaults(self) -> BigQueryWarehouseConfig:
        try:
            _bigquery_default_options(
                self.warehouse_defaults,
                target_name="target",
                schema_name=self.schema_name,
                model_name="model",
            )
        except ValueError as error:
            raise ValueError(str(error)) from None
        return self

    def absolutize(self, project_dir: Path) -> BigQueryWarehouseConfig:
        if self.keyfile is None or isinstance(
            self.keyfile, CredentialReference
        ):
            return self
        return self.model_copy(
            update={"keyfile": (project_dir / self.keyfile).resolve()}
        )

    def storage_location(self) -> str:
        return f"{self.project}.{self.schema_name}"

    def catalog_name(self) -> str:
        return self.project


def _validate_token_uri(value: str) -> CredentialFreeUrl:
    return CredentialFreeUrl(value)


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
# BigLake Cloud Resource connection identifier (issue #163). Accepts the
# `project.region.name` short form and the `projects/…/locations/…/connections/…`
# resource path; the character class deliberately excludes backticks, quotes,
# whitespace, and statement punctuation so a connection value cannot break out of
# the backtick-quoted identifier in `WITH CONNECTION`.
_CONNECTION_RE = re.compile(r"^[A-Za-z0-9._/-]+$")

# Polars integer dtypes that map to BigQuery INT64 for explicit Iceberg column
# DDL (issue #163). Unsigned 64-bit values above 2^63-1 overflow INT64; BigQuery
# has no unsigned integer type, matching how the Parquet load path coerces them.
_BQ_INT_DTYPES = (
    pl.Int8,
    pl.Int16,
    pl.Int32,
    pl.Int64,
    pl.UInt8,
    pl.UInt16,
    pl.UInt32,
    pl.UInt64,
)


class BigQueryWarehouseOptions(BaseModel):
    """Model-level `warehouse_options` for BigQuery (issue #91). Layout
    (partitioning/clustering) applies when a target table is (re)created; an
    existing incremental table keeps its layout until --full-refresh rebuilds
    it. Table options (labels, expirations, require_partition_filter,
    kms_key_name) are set on every (re)create; `labels` are also attached to
    the load and query jobs the adapter runs for the model."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

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
    # BigLake managed Apache Iceberg tables (issue #163). When set, the target is
    # created with explicit column DDL and a `WITH CONNECTION … OPTIONS(…)` clause
    # rather than via load-job schema autodetect; data is stored as Iceberg in the
    # `storage_uri` bucket. `connection` is a Cloud Resource connection name
    # (`project.region.name` or `DEFAULT`).
    table_format: Literal["iceberg"] | None = None
    storage_uri: str | None = None
    connection: str | None = None

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

    @model_validator(mode="after")
    def _validate_iceberg(self) -> BigQueryWarehouseOptions:
        """Interaction matrix for BigLake Iceberg tables (issue #163). Managed
        Iceberg tables need a Cloud Storage location and a Cloud Resource
        connection, support time partitioning only, and — for now — cannot pair
        with customer-managed encryption (unconfirmed on managed Iceberg)."""
        if self.table_format is None:
            if self.storage_uri is not None or self.connection is not None:
                raise ValueError(
                    "`storage_uri` and `connection` require `table_format: iceberg`"
                )
            return self
        if self.storage_uri is None or self.connection is None:
            missing = ", ".join(
                f"`{name}`"
                for name in ("storage_uri", "connection")
                if getattr(self, name) is None
            )
            raise ValueError(f"`table_format: iceberg` requires {missing}")
        if not self.storage_uri.startswith("gs://") or len(self.storage_uri) <= len(
            "gs://"
        ):
            raise ValueError(
                "`storage_uri` must be a gs:// Cloud Storage URI, e.g. "
                "gs://my-bucket/my-table"
            )
        if not self.connection.strip():
            raise ValueError("`connection` must not be empty")
        if not _CONNECTION_RE.match(self.connection):
            raise ValueError(
                "`connection` must be a Cloud Resource connection identifier "
                "(project.region.name, a projects/…/locations/…/connections/… "
                "path, or DEFAULT) — letters, digits, `.`, `_`, `-`, `/` only"
            )
        if self.kms_key_name is not None:
            raise ValueError(
                "`kms_key_name` is not supported with `table_format: iceberg` "
                "(customer-managed encryption on managed Iceberg tables is not "
                "yet supported by stel)"
            )
        if self.partition_by is not None and self.partition_by.data_type == "int64":
            raise ValueError(
                "`table_format: iceberg` supports time partitioning only "
                "(timestamp/date/datetime), not int64 range partitioning"
            )
        return self


_BIGQUERY_WAREHOUSE_DEFAULT_KEYS = frozenset(
    {*BigQueryWarehouseOptions.model_fields, "external_volume"}
)
_GCS_EXTERNAL_VOLUME_RE = re.compile(r"^gs://[^/\s?#]+(?:/[^\s?#]*)?$")


def _prepare_warehouse_defaults_input(value: Any) -> dict[str, Any]:
    """Reject unsafe/unknown default keys before generic env interpolation."""
    if not isinstance(value, Mapping):
        raise ValueError("`warehouse_defaults` must be a mapping")
    raw = dict(value)
    unknown = sorted(
        str(key) for key in raw if key not in _BIGQUERY_WAREHOUSE_DEFAULT_KEYS
    )
    if unknown:
        raise ValueError(
            "unknown BigQuery warehouse_defaults key(s): " + ", ".join(unknown)
        )
    if "storage_uri" in raw:
        raise ValueError(
            "`warehouse_defaults.storage_uri` is unsafe because every model would "
            "share one location; set `external_volume` so stel derives a unique "
            "target/dataset/model path"
        )
    return raw


def _warehouse_defaults_validation_message(error: ValidationError) -> str:
    details: list[str] = []
    for item in error.errors(
        include_url=False,
        include_context=False,
        include_input=False,
    ):
        location = ".".join(str(part) for part in item["loc"])
        prefix = f"{location}: " if location else ""
        details.append(f"{prefix}{item['msg']}")
    return "; ".join(details)


def _bigquery_default_options(
    defaults: Mapping[str, Any],
    *,
    target_name: str,
    schema_name: str,
    model_name: str,
) -> dict[str, Any]:
    raw = _prepare_warehouse_defaults_input(defaults)
    if not raw:
        return {}
    external_volume = raw.pop("external_volume", None)
    table_format = raw.get("table_format")
    if table_format == "iceberg" and external_volume is None:
        raise ValueError(
            "`warehouse_defaults.table_format: iceberg` requires `external_volume`"
        )
    if external_volume is not None:
        if table_format != "iceberg":
            raise ValueError(
                "`warehouse_defaults.external_volume` requires "
                "`table_format: iceberg`"
            )
        if not isinstance(external_volume, str) or not _GCS_EXTERNAL_VOLUME_RE.match(
            external_volume
        ):
            raise ValueError(
                "`warehouse_defaults.external_volume` must be a gs:// Cloud "
                "Storage URI"
            )
        segments = (target_name, schema_name, model_name)
        raw["storage_uri"] = "/".join(
            [
                external_volume.rstrip("/"),
                *(quote(segment, safe="") for segment in segments),
            ]
        )
    try:
        BigQueryWarehouseOptions.model_validate(raw)
    except ValidationError as error:
        message = _warehouse_defaults_validation_message(error)
        raise ValueError(f"invalid BigQuery warehouse_defaults: {message}") from None
    return raw


def parse_keyfile_json(value: dict[str, Any] | str) -> dict[str, Any]:
    """Decode a resolved service-account mapping, JSON, or base64 JSON value."""
    import base64
    import json

    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        del value
        raise AdapterError(
            "keyfile_json must be a mapping, a JSON string, or base64-encoded JSON"
        )
    parsed: Any = None
    failure: AdapterError | None = None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        try:
            parsed = json.loads(base64.b64decode(value, validate=True))
        except Exception:
            failure = AdapterError(
                "keyfile_json must be a mapping, a JSON string, or base64-encoded JSON"
            )
    if failure is not None:
        value = ""
        parsed = None
        raise failure
    if not isinstance(parsed, dict):
        value = ""
        parsed = None
        failure = AdapterError("keyfile_json must decode to a JSON object")
        raise failure
    return parsed


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


_READ_BINARY_OPERATORS = {
    ReadPredicateOperator.EQUAL: "=",
    ReadPredicateOperator.NOT_EQUAL: "!=",
    ReadPredicateOperator.LESS_THAN: "<",
    ReadPredicateOperator.LESS_THAN_OR_EQUAL: "<=",
    ReadPredicateOperator.GREATER_THAN: ">",
    ReadPredicateOperator.GREATER_THAN_OR_EQUAL: ">=",
}


def _bigquery_read_predicates(
    adapter: BigQueryAdapter, predicates: Sequence[ReadPredicate]
) -> tuple[str, list[Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    for predicate in predicates:
        column = adapter.quote_ident(predicate.column)
        if predicate.operator in _READ_BINARY_OPERATORS:
            clauses.append(f"{column} {_READ_BINARY_OPERATORS[predicate.operator]} ?")
            params.append(predicate.value)
        elif predicate.operator is ReadPredicateOperator.IN:
            clauses.append(f"{column} IN UNNEST(?)")
            params.append(list(cast(tuple[Any, ...], predicate.value)))
        elif predicate.operator is ReadPredicateOperator.NOT_IN:
            clauses.append(f"{column} NOT IN UNNEST(?)")
            params.append(list(cast(tuple[Any, ...], predicate.value)))
        elif predicate.operator is ReadPredicateOperator.IS_NULL:
            clauses.append(f"{column} IS NULL")
        else:
            assert predicate.operator is ReadPredicateOperator.IS_NOT_NULL
            clauses.append(f"{column} IS NOT NULL")
    return (" WHERE " + " AND ".join(clauses) if clauses else "", params)


def _bigquery_table_generation(table: Any) -> str:
    etag = getattr(table, "etag", None)
    modified = getattr(table, "modified", None)
    if etag is None and modified is None:
        raise AdapterError("BigQuery table does not expose a stable generation")
    return canonical_fingerprint(
        {
            "etag": str(etag) if etag is not None else None,
            "modified": modified,
            "num_rows": getattr(table, "num_rows", None),
        },
        domain="dbt-ml-bigquery-table-generation",
        version=1,
    )


def _empty_record_batch(schema: pa.Schema) -> pa.RecordBatch:
    return pa.RecordBatch.from_arrays(
        [pa.array([], type=field.type) for field in schema],
        schema=schema,
    )


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
        return frozenset(
            {
                # Full replacement is one atomic truncating load or CREATE OR
                # REPLACE whenever the target's partitioning spec is unchanged;
                # only a partitioning-spec migration (which BigQuery cannot
                # replace atomically) falls back to a staged drop-and-rename.
                WarehouseCapability.ATOMIC_FULL_REPLACE,
                WarehouseCapability.ATOMIC_KEYED_UPSERT,
                WarehouseCapability.ATOMIC_PARENT_CHILD_REPLACE,
                WarehouseCapability.ATOMIC_STATE_SCOPE_REPLACE,
                WarehouseCapability.CHUNKED_WRITES,
                WarehouseCapability.ICEBERG_TABLE_FORMAT,
                WarehouseCapability.PAGED_STATE_RECONCILIATION,
                WarehouseCapability.SCHEMA_EVOLUTION,
                WarehouseCapability.SQL_QUERIES,
                WarehouseCapability.SQL_MODEL_MATERIALIZATION,
                WarehouseCapability.SQL_INCREMENTAL_MATERIALIZATION,
                WarehouseCapability.SQL_SCHEMA_TESTS,
                WarehouseCapability.STREAMING_TABULAR_READS,
                WarehouseCapability.TABULAR_PREDICATE_PUSHDOWN,
                WarehouseCapability.TABULAR_READS,
                WarehouseCapability.TYPED_EMPTY_RELATIONS,
            }
        )

    @classmethod
    def warehouse_options_model(cls) -> type[BaseModel] | None:
        return BigQueryWarehouseOptions

    def warehouse_option_defaults(self, *, model_name: str) -> dict[str, Any]:
        try:
            return _bigquery_default_options(
                self._cfg.warehouse_defaults,
                target_name=self._cfg.target_name or "default",
                schema_name=self.schema,
                model_name=model_name,
            )
        except ValueError as error:
            raise AdapterError(str(error)) from None

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
        failure: AdapterError | None = None
        failure_cause: AdapterError | None = None
        try:
            return self._build_credentials()
        except ImportError:
            failure = AdapterError(_INSTALL_HINT)
        except CredentialResolutionError as error:
            failure = AdapterError(str(error))
        except AdapterError as error:
            failure = AdapterError(str(error))
        except Exception as error:
            failure = AdapterError("BigQuery credential construction failed")
            failure_cause = sanitized_adapter_cause(error)
        if failure is not None:
            if failure_cause is not None:
                raise failure from failure_cause
            raise failure
        raise AssertionError("unreachable credential construction state")

    def _build_credentials(self) -> Any:
        cfg = self._cfg
        scopes = tuple(cfg.scopes)
        if cfg.method == "service-account":
            from google.oauth2 import service_account

            keyfile = cfg.keyfile
            if isinstance(keyfile, CredentialReference):
                keyfile_value = keyfile.resolve().reveal()
                keyfile_path = Path(keyfile_value)
                if self.project_dir is not None and not keyfile_path.is_absolute():
                    keyfile_path = (self.project_dir / keyfile_path).resolve()
            else:
                assert isinstance(keyfile, Path)
                keyfile_path = keyfile
            creds = service_account.Credentials.from_service_account_file(  # type: ignore[no-untyped-call]
                str(keyfile_path), scopes=scopes
            )
        elif cfg.method == "service-account-json":
            from google.oauth2 import service_account

            assert cfg.keyfile_json is not None
            keyfile_json = cfg.keyfile_json.resolve().reveal()
            creds = service_account.Credentials.from_service_account_info(  # type: ignore[no-untyped-call]
                parse_keyfile_json(keyfile_json), scopes=scopes
            )
        elif cfg.method == "oauth-secrets":
            from google.oauth2.credentials import Credentials as UserCredentials

            token = (
                cfg.token.resolve().reveal()
                if cfg.token is not None
                else None
            )
            refresh_token = (
                cfg.refresh_token.resolve().reveal()
                if cfg.refresh_token is not None
                else None
            )
            client_secret = (
                cfg.client_secret.resolve().reveal()
                if cfg.client_secret is not None
                else None
            )
            token_uri = cfg.token_uri
            resolved_token_uri: str | None
            if isinstance(token_uri, CredentialReference):
                resolved_token_uri = token_uri.resolve().reveal()
            elif token_uri is not None:
                resolved_token_uri = str(token_uri)
            else:
                resolved_token_uri = None
            if resolved_token_uri is not None:
                resolved_token_uri = str(_validate_token_uri(resolved_token_uri))
            creds = UserCredentials(  # type: ignore[no-untyped-call]
                token=token,
                refresh_token=refresh_token,
                client_id=cfg.client_id,
                client_secret=client_secret,
                token_uri=resolved_token_uri,
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

    def _start_query(
        self,
        sql: str,
        params: list[Any] | None = None,
        *,
        job_labels: dict[str, str] | None = None,
        use_query_cache: bool | None = None,
    ) -> Any:
        bigquery = _bigquery()
        cfg = self._cfg
        kwargs: dict[str, Any] = {}
        if params or job_labels or use_query_cache is not None:
            job_config = bigquery.QueryJobConfig()
            if params:
                job_config.query_parameters = to_query_parameters(params)
            if job_labels:
                job_config.labels = dict(job_labels)
            if use_query_cache is not None:
                job_config.use_query_cache = use_query_cache
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
        return self.client.query(sql, **kwargs)

    def _run_query(
        self,
        sql: str,
        params: list[Any] | None = None,
        *,
        job_labels: dict[str, str] | None = None,
    ) -> Any:
        cfg = self._cfg
        job = self._start_query(sql, params, job_labels=job_labels)
        job.result(timeout=cfg.job_execution_timeout_seconds)
        return job

    def _open_table_snapshot(self, request: TableReadRequest) -> TableReadSnapshot:
        arrow_batches: Iterator[pa.RecordBatch] | None = None
        params: list[Any] = []
        rows: Any = None
        job: Any = None
        fully_consumed = False
        failure: AdapterError | None = None
        failure_cause: AdapterError | None = None
        try:
            table_id = self._table_id(request.table)
            initial_table = self.client.get_table(table_id)
            initial_generation = _bigquery_table_generation(initial_table)
            available_names = tuple(str(field.name) for field in initial_table.schema)
            available_columns = frozenset(available_names)
            missing = sorted(request.referenced_columns - available_columns)
            if missing:
                raise AdapterError(
                    "Table snapshot references missing column(s): "
                    + ", ".join(missing)
                )

            where_sql, params = _bigquery_read_predicates(self, request.predicates)
            projection = (
                "*"
                if request.columns is None
                else ", ".join(self.quote_ident(column) for column in request.columns)
            )
            output_names = request.columns or available_names
            hidden_null: str | None = None
            hidden_duplicate: str | None = None
            if request.key_column is None:
                sql = (
                    f"SELECT {projection} FROM "
                    f"{self.table_ref(request.table)}{where_sql}"
                )
            else:
                suffix = uuid4().hex
                hidden_key = f"stel_read_key_{suffix}"
                hidden_null = f"stel_read_nulls_{suffix}"
                hidden_duplicate = f"stel_read_duplicates_{suffix}"
                key = self.quote_ident(request.key_column)
                sql = (
                    "WITH stel_read_source AS ("
                    f"SELECT {projection}, {key} AS {self.quote_ident(hidden_key)} "
                    f"FROM {self.table_ref(request.table)}{where_sql}"
                    ") SELECT * EXCEPT("
                    f"{self.quote_ident(hidden_key)}), "
                    f"COUNTIF({self.quote_ident(hidden_key)} IS NULL) OVER() AS "
                    f"{self.quote_ident(hidden_null)}, "
                    f"COUNT({self.quote_ident(hidden_key)}) OVER() - "
                    f"COUNT(DISTINCT {self.quote_ident(hidden_key)}) OVER() AS "
                    f"{self.quote_ident(hidden_duplicate)} FROM stel_read_source"
                )

            job = self._start_query(sql, params, use_query_cache=False)
            schema_probe = job.to_arrow(
                create_bqstorage_client=False,
                max_results=1,
                timeout=self._cfg.job_execution_timeout_seconds,
            )
            query_schema = schema_probe.schema
            del schema_probe
            current_generation = _bigquery_table_generation(
                self.client.get_table(table_id)
            )
            if current_generation != initial_generation:
                raise AdapterError("BigQuery table changed while opening its snapshot")
            output_indices = [
                query_schema.get_field_index(name) for name in output_names
            ]
            if any(index < 0 for index in output_indices):
                raise AdapterError(
                    "BigQuery snapshot returned an unexpected projected schema"
                )
            output_schema = pa.schema(
                [query_schema.field(index) for index in output_indices]
            )
            rows = job.result(
                page_size=request.batch_size,
                timeout=self._cfg.job_execution_timeout_seconds,
            )
            arrow_batches = rows.to_arrow_iterable(
                bqstorage_client=None,
                max_queue_size=1,
                timeout=self._cfg.job_execution_timeout_seconds,
            )

            def batches() -> Iterator[pa.RecordBatch]:
                nonlocal fully_consumed
                validated_key_domain = request.key_column is None
                batch: pa.RecordBatch | None = None
                projected: pa.RecordBatch | None = None
                batch_failure: AdapterError | None = None
                batch_failure_cause: AdapterError | None = None
                try:
                    assert arrow_batches is not None
                    for batch in arrow_batches:
                        if not validated_key_domain:
                            assert hidden_null is not None
                            assert hidden_duplicate is not None
                            null_index = batch.schema.get_field_index(hidden_null)
                            duplicate_index = batch.schema.get_field_index(
                                hidden_duplicate
                            )
                            if null_index < 0 or duplicate_index < 0:
                                batch = _empty_record_batch(batch.schema)
                                raise AdapterError(
                                    "BigQuery snapshot omitted key-domain "
                                    "validation fields"
                                )
                            null_count = int(batch.column(null_index)[0].as_py())
                            duplicate_count = int(
                                batch.column(duplicate_index)[0].as_py()
                            )
                            if null_count or duplicate_count:
                                batch = _empty_record_batch(batch.schema)
                                raise AdapterError(
                                    "Table snapshot key domain is invalid: "
                                    f"{null_count} NULL and {duplicate_count} "
                                    "duplicate value(s)"
                                )
                            validated_key_domain = True
                        projected = batch.select(output_indices)
                        for offset in range(0, len(projected), request.batch_size):
                            yield projected.slice(offset, request.batch_size)
                        projected = None
                        batch = None
                    fully_consumed = True
                except AdapterError:
                    raise
                except Exception as error:
                    batch = None
                    projected = None
                    batch_failure = AdapterError(
                        "BigQuery table snapshot batch read failed"
                    )
                    batch_failure_cause = sanitized_adapter_cause(error)
                finally:
                    close_batches = getattr(arrow_batches, "close", None)
                    if callable(close_batches):
                        close_batches()
                if batch_failure is not None:
                    assert batch_failure_cause is not None
                    raise batch_failure from batch_failure_cause

            def validate_unchanged() -> None:
                validation_failure: AdapterError | None = None
                validation_failure_cause: AdapterError | None = None
                try:
                    generation = _bigquery_table_generation(
                        self.client.get_table(table_id)
                    )
                except AdapterError:
                    raise
                except Exception as error:
                    validation_failure = AdapterError(
                        "BigQuery table snapshot generation could not be validated"
                    )
                    validation_failure_cause = sanitized_adapter_cause(error)
                if validation_failure is not None:
                    assert validation_failure_cause is not None
                    raise validation_failure from validation_failure_cause
                if generation != initial_generation:
                    raise AdapterError("BigQuery table changed during its snapshot read")

            def close() -> None:
                if fully_consumed or job is None:
                    return
                try:
                    job.cancel()
                except Exception:
                    pass

            job_id = str(getattr(job, "job_id", "unavailable"))
            fingerprint = canonical_fingerprint(
                {
                    "adapter": self.adapter_type(),
                    "generation": initial_generation,
                    "job_id": job_id,
                    "request": request._fingerprint_payload(),
                },
                domain="dbt-ml-warehouse-table-snapshot",
                version=1,
            )
            generation_fingerprint = canonical_fingerprint(
                {
                    "generation": initial_generation,
                    "request": request._fingerprint_payload(),
                },
                domain="dbt-ml-warehouse-table-generation",
                version=1,
            )
            return TableReadSnapshot(
                schema=output_schema,
                fingerprint=fingerprint,
                batches=batches(),
                validate_unchanged=validate_unchanged,
                close=close,
                generation_fingerprint=generation_fingerprint,
            )
        except AdapterError:
            params.clear()
            rows = None
            arrow_batches = None
            if job is not None:
                try:
                    job.cancel()
                except Exception:
                    pass
            job = None
            raise
        except Exception as error:
            params.clear()
            rows = None
            arrow_batches = None
            if job is not None:
                try:
                    job.cancel()
                except Exception:
                    pass
            job = None
            failure = AdapterError("BigQuery table snapshot could not be opened")
            failure_cause = sanitized_adapter_cause(error)
        if failure is not None:
            assert failure_cause is not None
            raise failure from failure_cause
        raise AssertionError("unreachable BigQuery table snapshot state")

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
            if n != _STATE_TABLE and not n.startswith(INTERNAL_TABLE_PREFIXES)
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
            f"DECLARE stel_partitions ARRAY<{array_type}>;\n"
            f"SET stel_partitions = ARRAY(SELECT DISTINCT {source_expr} "
            f"FROM {self.table_ref(staging)} WHERE {field} IS NOT NULL);\n"
            f"MERGE {self.table_ref(table)} AS target "
            f"USING {self.table_ref(staging)} AS source ON FALSE "
            f"WHEN NOT MATCHED BY SOURCE AND {target_expr} "
            "IN UNNEST(stel_partitions) THEN DELETE "
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

    # ─── Iceberg (BigLake managed) tables, issue #163 ─────────────────────
    #
    # Managed Iceberg tables cannot be created by load-job autodetect and do not
    # support CREATE OR REPLACE, so they need explicit column DDL. These helpers
    # build the column list and the `WITH CONNECTION … OPTIONS(…)` suffix.

    def _bq_column_type(self, dtype: pl.DataType) -> str:
        """The BigQuery column type for a polars dtype, for explicit Iceberg DDL.
        Raises ValueError for dtypes BigQuery Iceberg cannot represent."""
        if isinstance(dtype, pl.Struct):
            fields = ", ".join(
                f"{self.quote_ident(field.name)} "
                f"{self._bq_column_type(cast('pl.DataType', field.dtype))}"
                for field in dtype.fields
            )
            return f"STRUCT<{fields}>"
        if isinstance(dtype, pl.List | pl.Array):
            inner = cast("pl.DataType", dtype.inner)
            if isinstance(inner, pl.List | pl.Array):
                raise ValueError(
                    "nested arrays are not supported in a BigQuery Iceberg column"
                )
            return f"ARRAY<{self._bq_column_type(inner)}>"
        if isinstance(dtype, pl.Datetime):
            return "TIMESTAMP" if dtype.time_zone is not None else "DATETIME"
        if isinstance(dtype, pl.Decimal):
            precision = dtype.precision if dtype.precision is not None else 38
            if precision > 38 or dtype.scale > 9:
                raise ValueError(
                    f"decimal(precision={precision}, scale={dtype.scale}) exceeds "
                    "BigQuery NUMERIC; BIGNUMERIC is not supported on Iceberg tables"
                )
            return "NUMERIC"
        if isinstance(dtype, pl.Enum):
            return "STRING"
        if dtype == pl.Boolean:
            return "BOOL"
        if dtype in _BQ_INT_DTYPES:
            return "INT64"
        if dtype in (pl.Float32, pl.Float64):
            return "FLOAT64"
        if dtype in (pl.String, pl.Categorical):
            return "STRING"
        if dtype == pl.Date:
            return "DATE"
        if dtype == pl.Time:
            return "TIME"
        if dtype == pl.Binary:
            return "BYTES"
        raise ValueError(
            f"polars dtype {dtype} has no BigQuery Iceberg column type "
            "(unsupported: Duration/INTERVAL, JSON, Object, or an all-null column)"
        )

    def _iceberg_column_ddl(self, df: pl.DataFrame) -> str:
        columns: list[str] = []
        for name, dtype in df.schema.items():
            try:
                bq_type = self._bq_column_type(dtype)
            except ValueError as error:
                raise AdapterError(
                    f"Cannot create Iceberg table column '{name}': {error}"
                ) from None
            columns.append(f"{self.quote_ident(name)} {bq_type}")
        return ", ".join(columns)

    @staticmethod
    def _connection_clause(connection: str) -> str:
        if connection.strip().upper() == "DEFAULT":
            return "WITH CONNECTION DEFAULT"
        # Config validation already restricts `connection` to a safe grammar
        # (`_CONNECTION_RE`); escaping backticks is defense-in-depth so the value
        # can never break out of the quoted identifier.
        escaped = connection.replace("\\", "\\\\").replace("`", "\\`")
        return f"WITH CONNECTION `{escaped}`"

    def _iceberg_ddl_clauses(self, layout: BigQueryWarehouseOptions) -> str:
        """`[PARTITION BY …] [CLUSTER BY …] WITH CONNECTION … OPTIONS(…)` for an
        Iceberg CREATE TABLE. kms_key_name is excluded (rejected by config)."""
        assert layout.storage_uri is not None and layout.connection is not None
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
        clauses.append(self._connection_clause(layout.connection))
        option_entries = [
            "file_format = 'PARQUET'",
            "table_format = 'ICEBERG'",
            f"storage_uri = {self._sql_string(layout.storage_uri)}",
            *self._table_option_entries(layout, include_kms=False),
        ]
        clauses.append(f"OPTIONS ({', '.join(option_entries)})")
        return " " + " ".join(clauses)

    def _iceberg_create_sql(
        self, table: str, df: pl.DataFrame, layout: BigQueryWarehouseOptions
    ) -> str:
        """The Iceberg `CREATE TABLE` statement. Building it validates every
        column dtype — `_iceberg_column_ddl` raises on an unsupported type — so a
        caller that must drop the target first should build this *before* the
        drop, to avoid destroying the last good table on a bad schema."""
        return (
            f"CREATE TABLE {self.table_ref(table)} "
            f"({self._iceberg_column_ddl(df)})"
            f"{self._iceberg_ddl_clauses(layout)}"
        )

    def _create_iceberg_table(
        self, table: str, df: pl.DataFrame, layout: BigQueryWarehouseOptions
    ) -> None:
        self._run_query(
            self._iceberg_create_sql(table, df, layout),
            job_labels=layout.labels or None,
        )

    def _iceberg_append_load(
        self, table: str, df: pl.DataFrame, layout: BigQueryWarehouseOptions
    ) -> None:
        """Append `df` to an existing Iceberg table. Iceberg load jobs are
        append-only, so callers create/truncate the target first."""
        bigquery = _bigquery()
        job_config = bigquery.LoadJobConfig(
            source_format=bigquery.SourceFormat.PARQUET,
            write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
        )
        if layout.labels:
            job_config.labels = dict(layout.labels)
        self._load_parquet(table, df, job_config)

    def _iceberg_full_materialize(
        self, table: str, df: pl.DataFrame, layout: BigQueryWarehouseOptions
    ) -> int:
        """Full replacement for a managed Iceberg target. Iceberg supports
        neither CREATE OR REPLACE nor a truncating load, so this drops the
        target, recreates it with explicit column DDL, then appends. The
        drop→create→append window is intentionally not atomic — Iceberg
        full-refresh is gated by the ICEBERG_TABLE_FORMAT capability rather than
        ATOMIC_FULL_REPLACE, and a failed append leaves an empty table that the
        next run repopulates."""
        create_sql = self._iceberg_create_sql(table, df, layout)  # validates schema
        self.drop_table(table)
        self._run_query(create_sql, job_labels=layout.labels or None)
        self._iceberg_append_load(table, df, layout)
        return df.height

    def _iceberg_add_columns(
        self, table: str, df: pl.DataFrame, columns: list[str]
    ) -> None:
        """Evolve an Iceberg target's schema via `ALTER TABLE … ADD COLUMN`,
        the DDL equivalent of the load-job field addition used for standard
        tables (`on_schema_change: append_new_columns`)."""
        if not columns:
            return
        schema = df.schema
        try:
            additions = ", ".join(
                f"ADD COLUMN {self.quote_ident(name)} "
                f"{self._bq_column_type(schema[name])}"
                for name in columns
            )
        except ValueError as error:
            raise AdapterError(f"Cannot add Iceberg column: {error}") from None
        self._run_query(f"ALTER TABLE {self.table_ref(table)} {additions}")

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
        # Polars writes a List column as a Parquet LIST logical type. Without
        # list inference, BigQuery's Parquet loader represents that group as a
        # nested RECORD (`{ list: RECORD REPEATED }`) instead of ARRAY<T>, which
        # breaks the embed→search vector contract (a vector column must read
        # back as a numeric Arrow list). Enabling it on every load makes all
        # List-typed columns materialize as native ARRAY<T> (issue #226).
        bigquery = _bigquery()
        parquet_options = job_config.parquet_options or bigquery.ParquetOptions()
        parquet_options.enable_list_inference = True
        job_config.parquet_options = parquet_options
        buffer = io.BytesIO()
        df.write_parquet(buffer)
        buffer.seek(0)
        job = self.client.load_table_from_file(
            buffer, self._table_id(table), job_config=job_config
        )
        job.result()

    def materialize_sql_full(
        self,
        table: str,
        select_sql: str,
        *,
        options: BaseModel | None = None,
    ) -> SqlMaterializationResult:
        layout = self._layout(options)
        if layout is not None and layout.table_format == "iceberg":
            return self._materialize_sql_full_iceberg(table, select_sql, layout)
        # Pure DDL: a single CREATE OR REPLACE ... AS SELECT is atomic and,
        # unlike a load job, may also change the partition/cluster spec — so no
        # staging swap is needed. A failed statement leaves the target intact.
        try:
            job = self._run_query(
                f"CREATE OR REPLACE TABLE {self.table_ref(table)}"
                f"{self._ddl_layout_clauses(options)} AS {select_sql}",
                job_labels=(layout.labels or None) if layout is not None else None,
            )
        except Exception as e:
            # Raw warehouse text can echo SQL fragments or row values into
            # run_results.json; surface only the safe error class and preserve
            # the cause via `from e` for local tracebacks (#262).
            raise AdapterError(
                f"SQL model materialization for '{table}' failed "
                f"[{type(e).__name__}]"
            ) from e
        num_rows = int(self.client.get_table(self._table_id(table)).num_rows or 0)
        job_metadata: dict[str, Any] = {}
        job_id = getattr(job, "job_id", None)
        if job_id:
            job_metadata["job_id"] = job_id
        total_bytes = getattr(job, "total_bytes_processed", None)
        if total_bytes is not None:
            job_metadata["total_bytes_processed"] = total_bytes
        return SqlMaterializationResult(
            relation=self.table_ref(table),
            rows_written=num_rows,
            job_metadata=job_metadata,
        )

    def _materialize_sql_full_iceberg(
        self, table: str, select_sql: str, layout: BigQueryWarehouseOptions
    ) -> SqlMaterializationResult:
        """Full SQL materialization into a managed Iceberg target (issue #290).
        Iceberg supports neither CREATE OR REPLACE nor a truncating load, so the
        query is staged once into a standard table, its schema drives an explicit
        Iceberg CREATE TABLE, and the rows are INSERT…SELECTed across. Staging
        once (rather than running select_sql inside INSERT) keeps a
        nondeterministic query from producing a different rowset than the one the
        schema was derived from. The drop→create→insert window is intentionally
        not atomic — same tradeoff as the DataFrame Iceberg full path, gated by
        ICEBERG_TABLE_FORMAT rather than ATOMIC_FULL_REPLACE."""
        job_labels = layout.labels or None
        staging = staging_table_name(table)
        staging_ref = self.table_ref(staging)
        try:
            try:
                self._run_query(f"CREATE TABLE {staging_ref} AS {select_sql}")
            except Exception as e:
                raise AdapterError(
                    f"SQL model materialization for '{table}' failed "
                    f"[{type(e).__name__}]"
                ) from e
            schema_df = self.query_df(f"SELECT * FROM {staging_ref} LIMIT 0")
            # Build (and dtype-validate) the CREATE before dropping the target, so
            # an unsupported Iceberg column type never destroys the last good table.
            create_sql = self._iceberg_create_sql(table, schema_df, layout)
            self.drop_table(table)
            self._run_query(create_sql, job_labels=job_labels)
            job = self._run_query(
                f"INSERT INTO {self.table_ref(table)} SELECT * FROM {staging_ref}",
                job_labels=job_labels,
            )
        finally:
            self.drop_table(staging)
        num_rows = int(self.client.get_table(self._table_id(table)).num_rows or 0)
        job_metadata: dict[str, Any] = {}
        job_id = getattr(job, "job_id", None)
        if job_id:
            job_metadata["job_id"] = job_id
        total_bytes = getattr(job, "total_bytes_processed", None)
        if total_bytes is not None:
            job_metadata["total_bytes_processed"] = total_bytes
        return SqlMaterializationResult(
            relation=self.table_ref(table),
            rows_written=num_rows,
            job_metadata=job_metadata,
        )

    def dry_run_sql(self, select_sql: str) -> SqlRelationSchema:
        bigquery = _bigquery()
        job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
        try:
            job = self.client.query(select_sql, job_config=job_config)
        except Exception as e:
            raise AdapterError(f"SQL dry-run failed [{type(e).__name__}]") from e
        columns = tuple(
            SqlRelationColumn(name=str(f.name), data_type=str(f.field_type))
            for f in (job.schema or [])
        )
        return SqlRelationSchema(columns=columns)

    def relation_exists(self, table: str) -> bool:
        return self._table_columns(table) is not None

    def materialize_sql_incremental(
        self,
        table: str,
        select_sql: str,
        *,
        unique_key: str,
        on_schema_change: str = "fail",
        options: BaseModel | None = None,
    ) -> SqlMaterializationResult:
        key = self.quote_ident(unique_key)
        # Fail fast on a storage-format mismatch before staging anything (issue
        # #289, now reachable for SQL models via #290): the MERGE writes in place
        # regardless of the declared table_format, so an existing standard target
        # under an iceberg config (or the reverse) would silently keep its format.
        # --full-refresh routes through materialize_sql_full and rebuilds it.
        existing_table = self._existing_table(table)
        if existing_table is not None:
            layout = self._layout(options)
            self._check_incremental_format(
                table,
                existing_table,
                declared_iceberg=layout is not None
                and layout.table_format == "iceberg",
            )
        # Materialize select_sql exactly once into a staging table, then
        # validate and merge that SAME rowset. Re-executing select_sql for the
        # key check and again inside the MERGE would let a nondeterministic
        # query — or an upstream table that changed between the two jobs —
        # merge a different, unvalidated rowset than the one that passed
        # validation (issue #142 review).
        staging = staging_table_name(table)
        staging_ref = self.table_ref(staging)
        try:
            try:
                self._run_query(f"CREATE TABLE {staging_ref} AS {select_sql}")
            except Exception as e:
                raise AdapterError(
                    f"Incremental SQL model staging for '{table}' failed "
                    f"[{type(e).__name__}]"
                ) from e

            check_rows = list(
                self._run_query(
                    build_key_check_sql(f"SELECT * FROM {staging_ref}", key)
                ).result()
            )
            null_count, duplicate_count = (
                tuple(check_rows[0].values()) if check_rows else (0, 0)
            )
            null_count = int(null_count or 0)
            duplicate_count = int(duplicate_count or 0)
            if null_count or duplicate_count:
                raise AdapterError(
                    f"Incremental SQL model '{table}' unique_key '{unique_key}' "
                    f"has {null_count} null and {duplicate_count} duplicate "
                    "value(s) in the query result."
                )

            staging_schema = self._relation_schema(staging)
            source_cols = [c.name for c in staging_schema.columns]
            if unique_key not in source_cols:
                raise AdapterError(
                    f"Incremental SQL model '{table}' query does not select its "
                    f"unique_key column '{unique_key}'"
                )
            target_cols = self._table_columns(table) or []
            if unique_key not in target_cols:
                # The key changed (or was never in this target). Appending it
                # as "just another new column" would leave every existing row
                # with a NULL key, so this is fatal under every
                # on_schema_change policy, not only `fail`.
                raise AdapterError(
                    f"Incremental SQL model '{table}' unique_key '{unique_key}' "
                    "does not exist in the current target table (the key may "
                    "have changed). Run with --full-refresh to rebuild the "
                    "target under the new key."
                )
            insert_cols = self._reconcile_sql_schema(
                table, staging_schema, target_cols, on_schema_change
            )

            other_cols = [c for c in insert_cols if c != unique_key]
            update_set = ", ".join(
                f"T.{self.quote_ident(c)} = S.{self.quote_ident(c)}"
                for c in other_cols
            )
            insert_col_list = ", ".join(self.quote_ident(c) for c in insert_cols)
            insert_val_list = ", ".join(
                f"S.{self.quote_ident(c)}" for c in insert_cols
            )
            # A single MERGE is one atomic DML statement — it either fully
            # commits or fails, so a failed merge never leaves the target
            # partially updated. USING references the staged table directly
            # (not the original select_sql), so the merged rowset is exactly
            # what was validated above.
            merge_sql = (
                f"MERGE {self.table_ref(table)} AS T "
                f"USING {staging_ref} AS S "
                f"ON T.{key} = S.{key} "
                + (
                    f"WHEN MATCHED THEN UPDATE SET {update_set} "
                    if update_set
                    else ""
                )
                + f"WHEN NOT MATCHED THEN INSERT ({insert_col_list}) "
                f"VALUES ({insert_val_list})"
            )
            try:
                job = self._run_query(merge_sql)
            except Exception as e:
                raise AdapterError(
                    f"Incremental SQL model materialization for '{table}' "
                    f"failed [{type(e).__name__}]"
                ) from e
            _log_publication(
                "incremental sql merge", self.table_ref(table), job, key=unique_key
            )
            affected = int(job.num_dml_affected_rows or 0)
            job_metadata: dict[str, Any] = {}
            job_id = getattr(job, "job_id", None)
            if job_id:
                job_metadata["job_id"] = job_id
            return SqlMaterializationResult(
                relation=self.table_ref(table),
                rows_written=affected,
                job_metadata=job_metadata,
            )
        finally:
            self.drop_table(staging)

    def _relation_schema(self, table: str) -> SqlRelationSchema:
        """Full column name + type schema of an existing table (unlike
        `_table_columns`, which discards types)."""
        bq_table = self.client.get_table(self._table_id(table))
        columns = tuple(
            SqlRelationColumn(name=str(f.name), data_type=str(f.field_type))
            for f in bq_table.schema
        )
        return SqlRelationSchema(columns=columns)

    def _reconcile_sql_schema(
        self,
        table: str,
        source_schema: SqlRelationSchema,
        target_cols: list[str],
        on_schema_change: str,
    ) -> list[str]:
        """Compares the compiled query's schema against the existing target's
        columns and applies the on_schema_change policy — the BigQuery analogue
        of DuckDB's `_reconcile_sql_schema`."""
        source_cols = [c.name for c in source_schema.columns]
        plan = plan_schema_change(target_cols, source_cols, on_schema_change, table)
        if plan.columns_to_add:
            types_by_name = {c.name: c.data_type for c in source_schema.columns}
            for col in plan.columns_to_add:
                self._run_query(
                    f"ALTER TABLE {self.table_ref(table)} "
                    f"ADD COLUMN {self.quote_ident(col)} {types_by_name[col]}"
                )
        return plan.columns_to_load

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
        if layout is not None and layout.table_format == "iceberg":
            return self._iceberg_full_materialize(table, df, layout)
        if layout is None:
            self._load_parquet(table, df, job_config)
            return df.height

        # A load job cannot change an existing table's partitioning or
        # clustering spec, and dropping the target up front would make a bad
        # layout declaration destructive. Build the replacement in a staging
        # table — the load validates partition/cluster columns against the
        # data — then swap.
        self._apply_layout_to_load(job_config, options)
        staging = staging_table_name(table)
        target_dropped = False
        try:
            self._load_parquet(staging, df, job_config)
            if self._partition_spec_matches(table, layout):
                # One CREATE OR REPLACE swaps the replacement in atomically;
                # BigQuery permits it whenever the partitioning spec is
                # unchanged (clustering and table options may differ), and a
                # failed statement leaves the target untouched.
                self._run_query(
                    f"CREATE OR REPLACE TABLE {self.table_ref(table)}"
                    f"{self._ddl_layout_clauses(options)} "
                    f"AS SELECT * FROM {self.table_ref(staging)}",
                    job_labels=layout.labels or None,
                )
                self.drop_table(staging)
                return df.height
            # A partitioning-spec migration cannot replace atomically:
            # configure the replacement, then drop and rename. The target
            # only drops after the replacement is fully created and
            # configured, so a bad layout never destroys the last good table.
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

    def _partition_spec_matches(
        self, table: str, layout: BigQueryWarehouseOptions | None
    ) -> bool:
        """Whether `table`'s partitioning spec already matches the declared
        layout (a missing target trivially matches), making CREATE OR REPLACE
        a valid — and atomic — full replacement. BigQuery rejects replacing a
        table under a different partitioning spec; clustering and table
        options are free to change."""
        try:
            existing = self.client.get_table(self._table_id(table))
        except _not_found_error():
            return True
        time_part = getattr(existing, "time_partitioning", None)
        range_part = getattr(existing, "range_partitioning", None)
        pb = layout.partition_by if layout is not None else None
        if pb is None:
            return time_part is None and range_part is None
        if pb.data_type == "int64":
            assert pb.range is not None
            if range_part is None:
                return False
            existing_range = range_part.range_
            return bool(
                range_part.field == pb.field
                and existing_range.start == pb.range.start
                and existing_range.end == pb.range.end
                and existing_range.interval == pb.range.interval
            )
        return bool(
            time_part is not None
            and time_part.type_ == pb.granularity.upper()
            and getattr(time_part, "field", None) == pb.field
        )

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
        update_when_changed: Sequence[str] = (),
    ) -> int:
        if df.height == 0:
            return 0
        validate_incremental_keys(df, key_col)
        bigquery = _bigquery()
        load_df = df
        allow_field_addition = False

        layout = self._layout(options)
        is_iceberg = layout is not None and layout.table_format == "iceberg"

        existing_table = self._existing_table(table)
        if existing_table is None:
            if is_iceberg:
                assert layout is not None
                self._create_iceberg_table(table, load_df, layout)
                self._iceberg_append_load(table, load_df, layout)
                return df.height
            job_config = bigquery.LoadJobConfig(
                source_format=bigquery.SourceFormat.PARQUET,
                write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
            )
            self._apply_layout_to_load(job_config, options)
            self._load_parquet(table, load_df, job_config)
            self._apply_post_create_options(table, options)
            return df.height

        # Fail fast on a storage-format mismatch (issue #289). An existing target
        # is written through the standard MERGE/load path regardless of the
        # declared `table_format`, so a config that declares Iceberg against a
        # standard target (or vice versa) would silently leave the format
        # unchanged and report success forever. Only --full-refresh (which routes
        # through materialize_full → _iceberg_full_materialize) can change the
        # stored format, so surface that instead of a silent no-op.
        self._check_incremental_format(table, existing_table, declared_iceberg=is_iceberg)

        existing = [f.name for f in existing_table.schema]
        if key_col not in existing:
            raise AdapterError(
                f"Incremental target '{table}' is missing key column '{key_col}'"
            )
        plan = plan_schema_change(existing, list(df.columns), on_schema_change, table)
        allow_field_addition = plan.allow_field_addition
        if plan.columns_to_load != list(df.columns):
            load_df = df.select(plan.columns_to_load)

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
            if is_iceberg:
                # Iceberg load jobs are append-only and cannot evolve the target
                # schema, so add the new columns with DDL instead.
                self._iceberg_add_columns(
                    table, load_df, [c for c in load_df.columns if c not in existing]
                )
            else:
                schema_config = bigquery.LoadJobConfig(
                    source_format=bigquery.SourceFormat.PARQUET,
                    write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
                    schema_update_options=[
                        bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION
                    ],
                )
                self._load_parquet(table, load_df.head(0), schema_config)

        staging = staging_table_name(table)
        staging_config = bigquery.LoadJobConfig(
            source_format=bigquery.SourceFormat.PARQUET,
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        )
        try:
            self._load_parquet(staging, load_df, staging_config)
            if insert_overwrite:
                assert layout is not None and layout.partition_by is not None
                job = self._run_query(
                    self._insert_overwrite_script(
                        table, staging, list(load_df.columns), layout.partition_by
                    ),
                    job_labels=job_labels,
                )
                _log_publication(
                    "incremental insert_overwrite",
                    self.table_ref(table),
                    job,
                    key=layout.partition_by.field,
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
                # Change-detection fingerprint (issue #281): a matched row is
                # updated only when a listed column differs, so re-publishing an
                # unchanged row does not rewrite its (possibly large) payload
                # columns and BigQuery bills far fewer bytes for that MERGE.
                when_matched = "WHEN MATCHED THEN UPDATE SET"
                if update_when_changed:
                    validate_update_when_changed_columns(
                        update_when_changed, load_df.columns, existing, table
                    )
                    changed = change_predicate(
                        update_when_changed, self.quote_ident
                    )
                    when_matched = f"WHEN MATCHED AND ({changed}) THEN UPDATE SET"
                job = self._run_query(
                    f"MERGE {self.table_ref(table)} AS target "
                    f"USING {self.table_ref(staging)} AS source "
                    f"ON target.{self.quote_ident(key_col)} = "
                    f"source.{self.quote_ident(key_col)} "
                    f"{when_matched} {assignments} "
                    f"WHEN NOT MATCHED THEN INSERT ({insert_columns}) "
                    f"VALUES ({insert_values})",
                    job_labels=job_labels,
                )
                _log_publication(
                    "incremental merge", self.table_ref(table), job, key=key_col
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
        staging = staging_table_name(table)
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
            if layout is not None and layout.table_format == "iceberg":
                # Iceberg cannot CREATE OR REPLACE or be renamed into; build the
                # explicit-schema target from the staged rows and INSERT them.
                # Non-atomic like the single-frame Iceberg full path.
                schema_df = self.query_df(
                    f"SELECT * FROM {self.table_ref(staging)} LIMIT 0"
                )
                create_sql = self._iceberg_create_sql(table, schema_df, layout)
                self.drop_table(table)
                self._run_query(create_sql, job_labels=job_labels)
                self._run_query(
                    f"INSERT INTO {self.table_ref(table)} "
                    f"SELECT * FROM {self.table_ref(staging)}",
                    job_labels=job_labels,
                )
                self.drop_table(staging)
                return total
            layout_clauses = self._ddl_layout_clauses(options)
            if not layout_clauses or self._partition_spec_matches(table, layout):
                # One CREATE OR REPLACE swaps the staged rows in atomically;
                # BigQuery permits it whenever the partitioning spec is
                # unchanged (clustering and table options may differ). The
                # statement validates the declared layout against real columns
                # and leaves the target untouched if it fails.
                self._run_query(
                    f"CREATE OR REPLACE TABLE {self.table_ref(table)}"
                    f"{layout_clauses} "
                    f"AS SELECT * FROM {self.table_ref(staging)}",
                    job_labels=job_labels,
                )
            else:
                # A partitioning-spec migration cannot replace atomically:
                # materialize the replacement (validating the declared layout
                # against real columns), then drop and rename. The target only
                # drops after the replacement fully exists, so a bad layout
                # never destroys the last good table.
                replacement = staging_table_name(table)
                self._run_query(
                    f"CREATE TABLE {self.table_ref(replacement)}{layout_clauses} "
                    f"AS SELECT * FROM {self.table_ref(staging)}",
                    job_labels=job_labels,
                )
                self.drop_table(table)
                target_dropped = True
                self._rename_table(replacement, table, job_labels=job_labels)
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
        bq_table = self._existing_table(table)
        return None if bq_table is None else [f.name for f in bq_table.schema]

    def _existing_table(self, table: str) -> Any | None:
        """The target's BigQuery Table object, or None if it does not exist."""
        try:
            return self.client.get_table(self._table_id(table))
        except _not_found_error():
            return None

    @staticmethod
    def _table_is_iceberg(bq_table: Any) -> bool:
        """Whether an existing target is actually stored as a managed Iceberg /
        BigLake table. Derived from the target's real metadata, not the declared
        config — the two can disagree (issue #289)."""
        if getattr(bq_table, "biglake_configuration", None) is not None:
            return True
        properties = getattr(bq_table, "_properties", None)
        return isinstance(properties, dict) and (
            properties.get("biglakeConfiguration") is not None
        )

    def _check_incremental_format(
        self, table: str, existing_table: Any, *, declared_iceberg: bool
    ) -> None:
        """Raise when the declared storage format disagrees with the existing
        target's actual format (issue #289). Incremental publication cannot
        change a table's format in place; --full-refresh rebuilds it."""
        target_is_iceberg = self._table_is_iceberg(existing_table)
        if target_is_iceberg == declared_iceberg:
            return
        if declared_iceberg:
            raise AdapterError(
                f"Incremental target '{table}' exists as a standard table but the "
                "config declares `table_format: iceberg`. Re-run with "
                "--full-refresh to rebuild it as Iceberg."
            )
        raise AdapterError(
            f"Incremental target '{table}' exists as an Iceberg table but the "
            "config does not declare `table_format: iceberg`. Re-run with "
            "--full-refresh to rebuild it as a standard table, or add "
            "`table_format: iceberg` to keep it as Iceberg."
        )

    def delete_rows(self, table: str, *, key_col: str, keys: list[str]) -> int:
        if not keys or self._table_columns(table) is None:
            return 0
        # Batch the key set so the request never carries an unbounded array (#260).
        affected = 0
        for chunk in _chunked(keys, _KEY_REQUEST_BATCH):
            job = self._run_query(
                f"DELETE FROM {self.table_ref(table)} "
                f"WHERE {self.quote_ident(key_col)} IN UNNEST(?)",
                [chunk],
            )
            affected += int(job.num_dml_affected_rows or 0)
        return affected

    def delete_rows_and_state(
        self,
        table: str,
        *,
        key_col: str,
        keys: Sequence[Any],
        state_scope: StateScope,
        state_record_keys: Sequence[str] | None = None,
    ) -> int:
        target_keys = list(keys)
        scoped_keys = target_keys if state_record_keys is None else list(state_record_keys)
        validate_state_keys(scoped_keys)
        target_exists = bool(target_keys) and self._table_columns(table) is not None
        if not target_exists and not scoped_keys:
            return 0
        # One atomic transaction deletes the target rows and their state together
        # (issue #229). Small key sets ride inline as array params; large sets are
        # staged into temp tables the transaction references, so the request stays
        # bounded without splitting the atomic delete into non-atomic chunks (#260).
        if max(len(target_keys), len(scoped_keys)) <= _STATE_MERGE_INLINE_MAX:
            return self._delete_rows_and_state_inline(
                table, key_col, target_keys, scoped_keys, state_scope, target_exists
            )
        return self._delete_rows_and_state_staged(
            table,
            key_col,
            target_keys,
            scoped_keys,
            state_scope,
            target_exists,
            reuse_state_from_target=state_record_keys is None,
        )

    def _delete_rows_and_state_inline(
        self,
        table: str,
        key_col: str,
        target_keys: list[Any],
        scoped_keys: list[str],
        state_scope: StateScope,
        target_exists: bool,
    ) -> int:
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

    def _delete_rows_and_state_staged(
        self,
        table: str,
        key_col: str,
        target_keys: list[Any],
        scoped_keys: list[str],
        state_scope: StateScope,
        target_exists: bool,
        *,
        reuse_state_from_target: bool,
    ) -> int:
        bigquery = _bigquery()
        append = bigquery.LoadJobConfig(
            source_format=bigquery.SourceFormat.PARQUET,
            write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
        )
        target_staging: str | None = None
        state_staging: str | None = None
        try:
            statements = [
                "DECLARE deleted_count INT64 DEFAULT 0;",
                "BEGIN TRANSACTION;",
            ]
            params: list[Any] = []
            if target_exists:
                target_staging = staging_table_name("delete_target")
                self._load_keys(target_staging, target_keys, append)
                statements.extend(
                    [
                        f"DELETE FROM {self.table_ref(table)} "
                        f"WHERE {self.quote_ident(key_col)} IN "
                        f"(SELECT k FROM {self.table_ref(target_staging)});",
                        "SET deleted_count = @@row_count;",
                    ]
                )
            if scoped_keys:
                if reuse_state_from_target and target_staging is not None:
                    # scoped_keys IS target_keys — reuse the staged set, no reload.
                    state_source = target_staging
                else:
                    state_staging = staging_table_name("delete_state")
                    self._load_keys(state_staging, scoped_keys, append)
                    state_source = state_staging
                statements.append(
                    f"DELETE FROM {self._state_ref} "
                    "WHERE model_name = ? AND state_scope = ? AND target_identity = ? "
                    f"AND record_key IN (SELECT k FROM {self.table_ref(state_source)});"
                )
                params.extend(
                    [
                        state_scope.model_name,
                        state_scope.stage,
                        state_scope.target_identity,
                    ]
                )
            statements.extend(["COMMIT TRANSACTION;", "SELECT deleted_count;"])
            deleted = self.scalar("\n".join(statements), params or None)
            return int(deleted or 0)
        finally:
            for staging in (target_staging, state_staging):
                if staging is not None:
                    self.client.delete_table(self._table_id(staging), not_found_ok=True)

    def _load_keys(self, table: str, keys: Sequence[Any], job_config: Any) -> None:
        """Stream a key set into a single-column (`k`) staging table via bounded
        Parquet loads, so a large IN-list can reference the table instead of
        shipping every key as an array query parameter. The column dtype is
        inferred from the keys — never coerced to string — so the staged
        subquery still matches a native (e.g. INT64) key column (#260 review)."""
        for chunk in _chunked(keys, _STATE_MERGE_LOAD_BATCH):
            self._load_parquet(table, pl.DataFrame({"k": chunk}), job_config)

    def _load_state_records(
        self, table: str, records: Sequence[StateRecord], job_config: Any
    ) -> None:
        """Stream state records into a staging table (record_key/input_fingerprint/
        code_version) via bounded Parquet loads, so a large state MERGE can read
        the table instead of shipping parallel array query parameters."""
        for chunk in _chunked(records, _STATE_MERGE_LOAD_BATCH):
            df = pl.DataFrame(
                {
                    "record_key": [r.record_key for r in chunk],
                    "input_fingerprint": [r.input_fingerprint for r in chunk],
                    "code_version": [r.code_version for r in chunk],
                },
                schema={
                    "record_key": pl.String,
                    "input_fingerprint": pl.String,
                    "code_version": pl.String,
                },
            )
            self._load_parquet(table, df, job_config)

    def replace_children(
        self,
        table: str,
        *,
        parent_key: str,
        parent_ids: Sequence[Any],
        child_key: str,
        new_rows: pl.DataFrame,
        state_scope: StateScope,
        state_records: Sequence[StateRecord],
        on_schema_change: str = "fail",
        options: BaseModel | None = None,
    ) -> int:
        validate_state_records(state_records)
        if new_rows.height > 0:
            validate_incremental_keys(new_rows, child_key)
        bigquery = _bigquery()

        existing = self._table_columns(table)

        # First run: create an empty table from the schema so the transactional
        # MERGE path below can run for both children and state in one script,
        # eliminating the gap where a failed state merge leaves committed children
        # without state (issue #229 P1).
        if existing is None:
            if new_rows.width > 0:
                create_config = bigquery.LoadJobConfig(
                    source_format=bigquery.SourceFormat.PARQUET,
                    write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
                )
                self._apply_layout_to_load(create_config, options)
                self._load_parquet(table, new_rows.head(0), create_config)
                self._apply_post_create_options(table, options)
                existing = self._table_columns(table)
            else:
                # No schema available; only state needs advancing (no children to be
                # inconsistent with when the table does not exist yet).
                if state_records:
                    self._merge_state(state_scope, state_records, replace=False)
                return new_rows.height

        # Existing table: build one atomic multi-statement script.
        assert existing is not None
        load_df = new_rows
        if new_rows.height > 0:
            plan = plan_schema_change(
                existing, list(new_rows.columns), on_schema_change, table
            )
            if plan.columns_to_load != list(new_rows.columns):
                load_df = new_rows.select(plan.columns_to_load)
            if plan.allow_field_addition:
                schema_config = bigquery.LoadJobConfig(
                    source_format=bigquery.SourceFormat.PARQUET,
                    write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
                    schema_update_options=[
                        bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION
                    ],
                )
                self._load_parquet(table, load_df.head(0), schema_config)

        # Stage the child rows, and any large parent-id / state-record set, into
        # temp tables the transaction references — so the request never carries an
        # unbounded array while the delete + child MERGE + state MERGE stay one
        # atomic script (#260). Every staging table name is chosen before its load
        # and loaded inside the try, so a mid-load failure still cleans them up.
        append_config = bigquery.LoadJobConfig(
            source_format=bigquery.SourceFormat.PARQUET,
            write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
        )
        staging: str | None = (
            staging_table_name(table)
            if load_df.height > 0
            else None
        )
        parent_staging: str | None = (
            staging_table_name("replace_parents")
            if parent_ids and len(parent_ids) > _STATE_MERGE_INLINE_MAX
            else None
        )
        state_staging: str | None = (
            staging_table_name("replace_state")
            if state_records and len(state_records) > _STATE_MERGE_INLINE_MAX
            else None
        )
        staging_tables = [
            t for t in (staging, parent_staging, state_staging) if t is not None
        ]

        try:
            if staging is not None:
                staging_config = bigquery.LoadJobConfig(
                    source_format=bigquery.SourceFormat.PARQUET,
                    write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
                )
                self._load_parquet(staging, load_df, staging_config)
            if parent_staging is not None:
                self._load_keys(parent_staging, list(parent_ids), append_config)
            if state_staging is not None:
                self._load_state_records(state_staging, state_records, append_config)

            statements: list[str] = ["BEGIN TRANSACTION;"]
            params: list[Any] = []

            if parent_ids:
                if parent_staging is not None:
                    statements.append(
                        f"DELETE FROM {self.table_ref(table)} "
                        f"WHERE {self.quote_ident(parent_key)} IN "
                        f"(SELECT k FROM {self.table_ref(parent_staging)});"
                    )
                else:
                    statements.append(
                        f"DELETE FROM {self.table_ref(table)} "
                        f"WHERE {self.quote_ident(parent_key)} IN UNNEST(?);"
                    )
                    params.append(list(parent_ids))

            if staging is not None:
                final_columns = list(existing)
                final_columns.extend(
                    c for c in load_df.columns if c not in final_columns
                )
                key = self.quote_ident(child_key)
                assignments = ", ".join(
                    f"target.{self.quote_ident(col)} = source.{self.quote_ident(col)}"
                    if col in load_df.columns
                    else f"target.{self.quote_ident(col)} = NULL"
                    for col in final_columns
                )
                insert_cols_str = ", ".join(
                    self.quote_ident(c) for c in load_df.columns
                )
                insert_vals_str = ", ".join(
                    f"source.{self.quote_ident(c)}" for c in load_df.columns
                )
                statements.append(
                    f"MERGE {self.table_ref(table)} AS target "
                    f"USING {self.table_ref(staging)} AS source "
                    f"ON target.{key} = source.{key} "
                    f"WHEN MATCHED THEN UPDATE SET {assignments} "
                    f"WHEN NOT MATCHED THEN INSERT ({insert_cols_str}) "
                    f"VALUES ({insert_vals_str});"
                )

            if state_records:
                if state_staging is not None:
                    source_select = (
                        "    SELECT ? AS model_name, ? AS state_scope,\n"
                        "        ? AS target_identity,\n"
                        "        record_key, input_fingerprint, code_version\n"
                        f"    FROM {self.table_ref(state_staging)}\n"
                    )
                else:
                    source_select = (
                        "    SELECT\n"
                        "        ? AS model_name,\n"
                        "        ? AS state_scope,\n"
                        "        ? AS target_identity,\n"
                        "        ids[OFFSET(o)] AS record_key,\n"
                        "        fs[OFFSET(o)] AS input_fingerprint,\n"
                        "        vs[OFFSET(o)] AS code_version\n"
                        "    FROM (SELECT ? AS ids, ? AS fs, ? AS vs),\n"
                        "        UNNEST(GENERATE_ARRAY(0, ARRAY_LENGTH(ids) - 1)) AS o\n"
                    )
                statements.append(
                    f"MERGE {self._state_ref} AS target\n"
                    "USING (\n"
                    f"{source_select}"
                    ") AS source\n"
                    "ON target.model_name = source.model_name\n"
                    "    AND target.state_scope = source.state_scope\n"
                    "    AND target.target_identity = source.target_identity\n"
                    "    AND target.record_key = source.record_key\n"
                    "WHEN MATCHED THEN UPDATE SET\n"
                    "    input_fingerprint = source.input_fingerprint,\n"
                    "    code_version = source.code_version,\n"
                    "    last_run_at = CURRENT_TIMESTAMP()\n"
                    "WHEN NOT MATCHED THEN INSERT\n"
                    "    (model_name, state_scope, target_identity, record_key,\n"
                    "     input_fingerprint, code_version, last_run_at)\n"
                    "    VALUES (source.model_name, source.state_scope,\n"
                    "            source.target_identity, source.record_key,\n"
                    "            source.input_fingerprint, source.code_version,\n"
                    "            CURRENT_TIMESTAMP());"
                )
                params.extend(
                    [
                        state_scope.model_name,
                        state_scope.stage,
                        state_scope.target_identity,
                    ]
                )
                if state_staging is None:
                    params.extend(
                        [
                            [r.record_key for r in state_records],
                            [r.input_fingerprint for r in state_records],
                            [r.code_version for r in state_records],
                        ]
                    )

            statements.append("COMMIT TRANSACTION;")
            if len(statements) > 2:
                self._run_query("\n".join(statements), params or None)
        except BaseException as error:
            for staging_table in staging_tables:
                try:
                    self.drop_table(staging_table)
                except Exception as cleanup_error:
                    error.add_note(
                        f"Failed to clean staging table {staging_table}: {cleanup_error}"
                    )
            raise
        else:
            for staging_table in staging_tables:
                self.drop_table(staging_table)
        return new_rows.height

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
        # A single inline MERGE passes the whole record set as array query
        # parameters, so its one request grows with the record count and fails at
        # scale (SSLEOFError at ~75k, issue #256). Small sets stay inline (cheap,
        # no staging); large sets are staged via bounded Parquet loads and merged
        # from the staging table — still one atomic MERGE.
        if len(records) <= _STATE_MERGE_INLINE_MAX:
            self._merge_state_inline(scope, records, replace=replace)
        else:
            self._merge_state_staged(scope, records, replace=replace)

    def _state_replace_clause(self, replace: bool) -> str:
        return (
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

    def _merge_state_inline(
        self,
        scope: StateScope,
        records: Sequence[StateRecord],
        *,
        replace: bool,
    ) -> None:
        record_keys = [record.record_key for record in records]
        fingerprints = [record.input_fingerprint for record in records]
        versions = [record.code_version for record in records]
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
            {self._state_replace_clause(replace)}
            """,
            params,
        )

    def _merge_state_staged(
        self,
        scope: StateScope,
        records: Sequence[StateRecord],
        *,
        replace: bool,
    ) -> None:
        bigquery = _bigquery()
        staging = staging_table_name("state_merge")
        append_config = bigquery.LoadJobConfig(
            source_format=bigquery.SourceFormat.PARQUET,
            write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
        )
        try:
            for start in range(0, len(records), _STATE_MERGE_LOAD_BATCH):
                chunk = records[start : start + _STATE_MERGE_LOAD_BATCH]
                df = pl.DataFrame(
                    {
                        "record_key": [r.record_key for r in chunk],
                        "input_fingerprint": [r.input_fingerprint for r in chunk],
                        "code_version": [r.code_version for r in chunk],
                    },
                    schema={
                        "record_key": pl.String,
                        "input_fingerprint": pl.String,
                        "code_version": pl.String,
                    },
                )
                self._load_parquet(staging, df, append_config)
            params: list[Any] = [scope.model_name, scope.stage, scope.target_identity]
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
                        record_key, input_fingerprint, code_version
                    FROM {self.table_ref(staging)}
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
                    VALUES (source.model_name, source.state_scope,
                            source.target_identity, source.record_key,
                            source.input_fingerprint, source.code_version,
                            CURRENT_TIMESTAMP())
                {self._state_replace_clause(replace)}
                """,
                params,
            )
        finally:
            self.client.delete_table(self._table_id(staging), not_found_ok=True)

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
        # Batch the key set so the request never carries an unbounded array (#260).
        for chunk in _chunked(record_keys, _KEY_REQUEST_BATCH):
            self._run_query(
                f"DELETE FROM {self._state_ref} "
                "WHERE model_name = ? AND state_scope = ? AND target_identity = ? "
                "AND record_key IN UNNEST(?)",
                [
                    scope.model_name,
                    scope.stage,
                    scope.target_identity,
                    chunk,
                ],
            )

    # ─── paged state reconciliation (issue #153) ─────────────────────────

    def _fetch_state_subset(
        self, scope: StateScope, record_keys: Sequence[str]
    ) -> dict[str, StateValue]:
        # Dedupe so a repeated input key cannot be mistaken for a duplicate state
        # row, then batch so the request never carries an unbounded array (#260).
        unique_keys = list(dict.fromkeys(record_keys))
        state: dict[str, StateValue] = {}
        for chunk in _chunked(unique_keys, _KEY_REQUEST_BATCH):
            result = self.rows(
                "SELECT record_key, input_fingerprint, code_version "
                f"FROM {self._state_ref} "
                "WHERE model_name = ? AND state_scope = ? AND target_identity = ? "
                "AND record_key IN UNNEST(?)",
                [
                    scope.model_name,
                    scope.stage,
                    scope.target_identity,
                    chunk,
                ],
            )
            for row in result:
                record_key = str(row[0])
                if record_key in state:
                    raise AdapterError(
                        "BigQuery state contains duplicate record keys in the "
                        "requested scope; repair the state table before retrying."
                    )
                state[record_key] = StateValue(str(row[1]), str(row[2]))
        return state

    def _open_state_page_reader(self, request: StatePageRequest) -> StatePageReader:
        # BigQuery has no cross-statement transactions here; pinning every
        # page (and the absence probe) to one server-side timestamp via
        # `FOR SYSTEM_TIME AS OF` gives the same immutable-snapshot guarantee:
        # interleaved mutations, including the caller's own state deletions,
        # cannot skip or repeat records between pages.
        snapshot_at = self.scalar("SELECT CURRENT_TIMESTAMP()")
        if not isinstance(snapshot_at, datetime):
            raise AdapterError(
                "BigQuery did not return a snapshot timestamp for state paging"
            )
        absence_sql = ""
        probe = request.absent_from
        if probe is not None:
            try:
                probe_table = self.client.get_table(self._table_id(probe.table))
            except Exception:
                raise AdapterError(
                    "State absence probe relation "
                    f"'{probe.table}.{probe.key_column}' is unavailable"
                ) from None
            if probe.key_column not in {field.name for field in probe_table.schema}:
                raise AdapterError(
                    "State absence probe relation "
                    f"'{probe.table}.{probe.key_column}' is unavailable"
                )
            absence_sql = (
                " AND NOT EXISTS (SELECT 1 FROM "
                f"{self.table_ref(probe.table)} AS probe FOR SYSTEM_TIME AS OF ? "
                f"WHERE probe.{self.quote_ident(probe.key_column)} = "
                "state.record_key)"
            )
        nonce = uuid4().hex
        scope = request.scope

        def fetch(cursor_value: str | None) -> StatePage:
            last_key = decode_state_cursor(cursor_value, nonce)
            key_sql = " AND state.record_key > ?" if last_key is not None else ""
            params: list[Any] = [
                snapshot_at,
                scope.model_name,
                scope.stage,
                scope.target_identity,
            ]
            if last_key is not None:
                params.append(last_key)
            if probe is not None:
                params.append(snapshot_at)
            try:
                rows = self.rows(
                    "SELECT state.record_key, state.input_fingerprint, "
                    "state.code_version, state.last_run_at "
                    f"FROM {self._state_ref} AS state FOR SYSTEM_TIME AS OF ? "
                    "WHERE state.model_name = ? AND state.state_scope = ? "
                    f"AND state.target_identity = ?{key_sql}{absence_sql} "
                    "ORDER BY state.record_key LIMIT ?",
                    [*params, request.page_size],
                )
            except AdapterError:
                raise
            except Exception as exc:
                # Preserve the cause for the exception chain, but keep the
                # artifact-visible message generic: raw warehouse exception
                # text may carry SQL or response details and must not reach
                # run_results.json or the CLI (AGENTS.md).
                raise AdapterError("BigQuery state page read failed") from exc
            records = tuple(
                StatePageRecord(
                    str(row[0]),
                    str(row[1]),
                    str(row[2]),
                    row[3] if isinstance(row[3], datetime) else None,
                )
                for row in rows
            )
            next_cursor = (
                encode_state_cursor(nonce, records[-1].record_key)
                if len(records) == request.page_size
                else None
            )
            return StatePage(records=records, next_cursor=next_cursor)

        return StatePageReader(
            page_size=request.page_size, fetch=fetch, close=lambda: None
        )

    def _replace_state_scope(
        self,
        scope: StateScope,
        record_batches: Iterator[Sequence[StateRecord]],
        fence: StateScopeFence | None,
    ) -> int:
        bigquery = _bigquery()
        staging = staging_table_name("state_replace")
        append_config = bigquery.LoadJobConfig(
            source_format=bigquery.SourceFormat.PARQUET,
            write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
        )
        total = 0
        try:
            for batch in record_batches:
                df = pl.DataFrame(
                    {
                        "record_key": [r.record_key for r in batch],
                        "input_fingerprint": [r.input_fingerprint for r in batch],
                        "code_version": [r.code_version for r in batch],
                    },
                    schema={
                        "record_key": pl.String,
                        "input_fingerprint": pl.String,
                        "code_version": pl.String,
                    },
                )
                self._load_parquet(staging, df, append_config)
                total += len(batch)
            if total == 0:
                self._run_query(
                    f"CREATE TABLE {self.table_ref(staging)} (record_key STRING, "
                    "input_fingerprint STRING, code_version STRING)"
                )
            self._run_state_replace_merge(scope, staging, fence)
        finally:
            self.client.delete_table(self._table_id(staging), not_found_ok=True)
        return total

    def _run_state_replace_merge(
        self, scope: StateScope, staging: str, fence: StateScopeFence | None
    ) -> None:
        """One fence-gated MERGE makes the staged snapshot the scope's state.

        BigQuery DML is atomic per statement but merge search conditions
        cannot hold subqueries, so the fence ride along in the source query:
        every source row's join key evaluates a gate that ERROR()s the whole
        statement on a stale fence or duplicated staging keys. A sentinel
        source row keeps the gate evaluated even when staging is empty, so a
        pure delete-all replacement is fenced too."""
        staging_ref = self.table_ref(staging)
        params: list[Any] = []
        if fence is None:
            fence_check = "TRUE"
        else:
            fence_check = (
                "(SELECT COUNT(*) FROM "
                f"{self.table_ref(SERVING_LEDGER_TABLE)} "
                "WHERE model_name = ? AND stage = ? AND target_identity = ? "
                "AND publication_id = ? AND fencing_token = ?) = 1"
            )
            params.extend(
                [
                    scope.model_name,
                    scope.stage,
                    scope.target_identity,
                    fence.publication_id,
                    fence.fencing_token,
                ]
            )
        params.extend([scope.model_name, scope.stage, scope.target_identity])
        params.extend([scope.model_name, scope.stage, scope.target_identity])
        params.extend([scope.model_name, scope.stage, scope.target_identity])
        sql = f"""
            MERGE {self._state_ref} AS target
            USING (
                SELECT
                    IF(gate.ok, source_rows.record_key, NULL) AS record_key,
                    source_rows.input_fingerprint,
                    source_rows.code_version,
                    source_rows.is_sentinel
                FROM (
                    SELECT record_key, input_fingerprint, code_version,
                        0 AS is_sentinel
                    FROM {staging_ref}
                    UNION ALL
                    SELECT CAST(NULL AS STRING), CAST(NULL AS STRING),
                        CAST(NULL AS STRING), 1
                ) AS source_rows
                CROSS JOIN (
                    SELECT IF(
                        {fence_check},
                        TRUE,
                        ERROR('stel-stale-state-fence: publication reassigned')
                    ) AND IF(
                        (SELECT COUNT(*) - COUNT(DISTINCT record_key)
                         FROM {staging_ref}) = 0,
                        TRUE,
                        ERROR('stel-state-replace-invalid: duplicate keys')
                    ) AS ok
                ) AS gate
            ) AS source
            ON target.model_name = ? AND target.state_scope = ?
                AND target.target_identity = ?
                AND target.record_key = source.record_key
                AND source.is_sentinel = 0
            WHEN MATCHED THEN UPDATE SET
                input_fingerprint = source.input_fingerprint,
                code_version = source.code_version,
                last_run_at = CURRENT_TIMESTAMP()
            WHEN NOT MATCHED AND source.is_sentinel = 0 THEN INSERT
                (model_name, state_scope, target_identity, record_key,
                 input_fingerprint, code_version, last_run_at)
                VALUES (?, ?, ?, source.record_key, source.input_fingerprint,
                        source.code_version, CURRENT_TIMESTAMP())
            WHEN NOT MATCHED BY SOURCE
                AND target.model_name = ?
                AND target.state_scope = ?
                AND target.target_identity = ?
            THEN DELETE
            """
        try:
            self._run_query(sql, params)
        except Exception as error:
            message = str(error)
            if "stel-stale-state-fence" in message:
                raise StaleStateFenceError(
                    "Serving publication authority was reassigned; state scope "
                    "replacement aborted without mutation"
                ) from None
            if "stel-state-replace-invalid" in message:
                raise AdapterError(
                    "State scope replacement contains duplicate record keys "
                    "across batches"
                ) from None
            if (
                fence is not None
                and SERVING_LEDGER_TABLE in message
                and "Not found" in message
            ):
                raise StaleStateFenceError(
                    "Fenced state replacement requires a serving ledger, and "
                    "this dataset has none"
                ) from None
            raise
