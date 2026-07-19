from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    GetCoreSchemaHandler,
    GetJsonSchemaHandler,
)
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import CoreSchema, PydanticCustomError, core_schema

from ..budget import LLMBudgetConfig
from ..credentials import (
    CredentialReference,
    CredentialReferenceError,
    ProtectedCredential,
)

DEFAULT_LLM_API_KEY_ENV = "ANTHROPIC_API_KEY"
DEFAULT_LLM_PROVIDER = "anthropic"


def resolve_llm_credential(
    options: Mapping[str, Any],
) -> ProtectedCredential | None:
    """Resolve the selected provider's protected runtime credential."""
    from ..providers import (
        ProviderError,
        ProviderRequestError,
        get_inference_provider,
        sanitized_provider_error,
    )

    provider_name = options.get("provider", DEFAULT_LLM_PROVIDER)
    configured = options.get("api_key_env")
    options = {}
    if not isinstance(provider_name, str):
        provider_name = None
        configured = None
        raise ValueError("llm provider must be a registered provider name")
    failure: Exception | None = None
    reference: CredentialReference | None = None
    if configured is not None:
        try:
            reference = (
                configured
                if isinstance(configured, CredentialReference)
                else CredentialReference.from_env_name(configured)
            )
        except (TypeError, CredentialReferenceError):
            failure = ValueError(
                "llm api_key_env must be a valid environment-variable name"
            )
    configured = None
    if failure is not None:
        raise failure

    provider = get_inference_provider(provider_name)
    if not provider.requires_credentials:
        return None
    if reference is None:
        default_reference = provider.default_credential_env
        if default_reference is None:
            failure = ValueError(
                f"llm api_key_env must be configured for provider '{provider_name}'"
            )
        else:
            try:
                reference = CredentialReference.from_env_name(default_reference)
            except (TypeError, CredentialReferenceError):
                failure = ValueError(
                    "llm api_key_env must be a valid environment-variable name"
                )
        default_reference = None
    if failure is None and reference is not None:
        try:
            return provider.resolve_credential(reference)
        except ProviderError as error:
            failure = sanitized_provider_error(
                provider.name(), "credential resolution", error
            )
        except Exception as error:
            failure = ProviderRequestError(
                provider.name(),
                "credential resolution",
                code=type(error).__name__,
            )
    if failure is not None:
        raise failure
    raise AssertionError("llm credential resolution did not complete")


class WarehouseConfig(BaseModel):
    """Base for adapter-specific warehouse configs, discriminated by `type:`.

    Each adapter owns a subclass declaring its connection fields (DuckDB: a
    file path; BigQuery: project/dataset; ...) and registers it via
    `WarehouseAdapter.config_model()`. Raw profile blocks are validated
    against the subclass at resolve time, so unknown fields and missing
    credentials fail with the adapter named.
    """

    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
        hide_input_in_errors=True,
    )

    type: str = "duckdb"
    schema_name: str = Field(default="dbt_ml", alias="schema")

    @classmethod
    def prepare_profile_input(cls, raw: Mapping[str, Any]) -> dict[str, Any]:
        """Protect adapter-owned values before generic profile interpolation."""
        return dict(raw)

    def absolutize(self, project_dir: Path) -> WarehouseConfig:
        """Resolve project-relative paths. Default: nothing to resolve."""
        return self

    def storage_location(self) -> str:
        """Human-readable storage location (file path, DSN, dataset, ...)."""
        return ""

    def catalog_name(self) -> str:
        """Catalog/database name for SQL references and emitted dbt sources
        (DuckDB: database file stem; BigQuery: GCP project)."""
        return ""

    def local_path(self) -> Path | None:
        """Filesystem location backing this warehouse, if file-backed
        (DuckDB); None for remote warehouses."""
        return None


class PricingConfig(BaseModel):
    """User-supplied token prices (USD per million tokens) for cost estimates
    in run results. No prices ship with dbt-ml — they drift; you own them."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    input_usd_per_mtok: float
    output_usd_per_mtok: float
    cache_read_usd_per_mtok: float | None = None
    cache_write_usd_per_mtok: float | None = None


class LLMConfig(BaseModel):
    """Defaults for the LLM extraction backend and LLM-using transforms."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    provider: str = Field(
        default=DEFAULT_LLM_PROVIDER,
        min_length=1,
        pattern=r"^[a-z0-9][a-z0-9_-]*$",
    )
    model: str | None = Field(default=None, min_length=1)
    api_key_env: CredentialReference | None = Field(
        default=None,
        repr=False,
        exclude=True,
    )
    cache_path: Path | None = None
    system_prompt: str | None = None
    pricing: PricingConfig | None = None
    # Run-level execution caps shared by every model in one invocation;
    # operator-owned policy, like the credential reference.
    budget: LLMBudgetConfig | None = None


class _RedactedWarehouseInput:
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


class ProtectedWarehouseConfig(dict[str, Any]):
    """Raw adapter mapping whose credential fields are protected on entry."""

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: Any,
        handler: GetCoreSchemaHandler,
    ) -> CoreSchema:
        del source_type, handler
        return core_schema.chain_schema(
            [
                core_schema.no_info_plain_validator_function(
                    _RedactedWarehouseInput
                ),
                core_schema.no_info_plain_validator_function(cls._validate),
            ],
            serialization=core_schema.plain_serializer_function_ser_schema(
                dict,
                return_schema=core_schema.dict_schema(
                    keys_schema=core_schema.str_schema(),
                    values_schema=core_schema.any_schema(),
                ),
            ),
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls,
        schema: CoreSchema,
        handler: GetJsonSchemaHandler,
    ) -> JsonSchemaValue:
        del schema, handler
        return {"type": "object", "additionalProperties": True}

    @classmethod
    def _validate(cls, wrapped: _RedactedWarehouseInput) -> ProtectedWarehouseConfig:
        raw = wrapped.take()
        if not isinstance(raw, Mapping):
            raise PydanticCustomError(
                "warehouse_config",
                "warehouse config must be a mapping",
            )
        from ..adapters import AdapterError, prepare_warehouse_profile_input

        try:
            prepared = prepare_warehouse_profile_input(raw)
        except AdapterError as error:
            raise PydanticCustomError(
                "warehouse_config",
                str(error),
            ) from None
        for field_name in (
            "client_secret",
            "keyfile",
            "keyfile_json",
            "refresh_token",
            "token",
            "token_uri",
        ):
            value = prepared.get(field_name)
            if isinstance(value, str) and (
                "{{" in value or "env_var(" in value
            ):
                raise PydanticCustomError(
                    "warehouse_config",
                    f"credential field `{field_name}` was not protected by a "
                    "registered adapter",
                )
        return cls(prepared)


class RetrievalProfileConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    default: str
    allow_public_indexes: bool = False
    stores: dict[str, dict[str, Any]]


class TargetConfig(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
        hide_input_in_errors=True,
    )

    # Raw mapping: validated against the adapter's config model at resolve
    # time, once `type:` is known (the adapter registry owns that lookup).
    warehouse: ProtectedWarehouseConfig
    llm: LLMConfig | None = None
    retrieval: RetrievalProfileConfig | None = None
    # Operator-owned source roots for this target. Keys are source names from
    # project YAML; values replace SourceConfig.path after target resolution.
    source_paths: dict[str, str] = Field(default_factory=dict, alias="source-paths")


class ProfileConfig(BaseModel):
    """A named profile: one or more targets (e.g. dev/prod), plus default target."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    target: str = "dev"
    outputs: dict[str, TargetConfig]
