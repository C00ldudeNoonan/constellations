from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    GetCoreSchemaHandler,
    PrivateAttr,
    field_serializer,
    field_validator,
    model_validator,
)
from pydantic_core import CoreSchema, PydanticCustomError, core_schema

from ..credentials import CredentialReference, CredentialReferenceError
from .identifiers import validate_node_name
from .yaml_diagnostics import ConfigPath, YamlProvenance

_STRICT_CONFIG = ConfigDict(extra="forbid")

INTERNAL_LINEAGE_FIELDS = frozenset(
    {
        "document_id",
        "source_path",
        "source_uri",
        "source_metadata",
        "content_hash",
        "code_version",
        "backend_name",
        "backend_version",
        "extracted_at",
    }
)
EMBED_METADATA_FIELDS = frozenset(
    {
        "embedding_provider",
        "embedding_model",
        "embedding_dimensions",
        "embedding_provider_implementation",
        "embedding_input_hash",
        "embedding_config_hash",
        "embedded_at",
    }
)


class _RedactedConfigInput:
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


def _protect_extraction_mapping(raw: Mapping[str, Any]) -> dict[str, Any]:
    prepared = dict(raw)
    if prepared.get("backend") != "llm":
        return prepared
    options = prepared.get("options")
    if not isinstance(options, Mapping) or "api_key_env" not in options:
        return prepared
    protected_options = dict(options)
    value = protected_options["api_key_env"]
    if not isinstance(value, CredentialReference):
        try:
            value = CredentialReference.from_env_name(value)
        except (TypeError, CredentialReferenceError):
            raise PydanticCustomError(
                "credential_reference",
                "llm api_key_env is operator-owned configuration and must be "
                "set in profiles.yml",
            ) from None
    protected_options["api_key_env"] = value
    prepared["options"] = protected_options
    return prepared

def _configured_extraction_field_names(options: Mapping[str, Any]) -> set[str]:
    names: set[str] = set()
    fields = options.get("fields")
    if isinstance(fields, list):
        for field in fields:
            if isinstance(field, str):
                names.add(field)
            elif isinstance(field, dict) and isinstance(field.get("name"), str):
                names.add(field["name"])

    frontmatter_fields = options.get("frontmatter_fields")
    if isinstance(frontmatter_fields, list):
        names.update(field for field in frontmatter_fields if isinstance(field, str))

    for option in ("text_field", "body_field"):
        value = options.get(option)
        if isinstance(value, str):
            names.add(value)

    selectors = options.get("selectors")
    if isinstance(selectors, dict):
        names.update(name for name in selectors if isinstance(name, str))
    return names


def validate_extraction_field_names(options: Mapping[str, Any]) -> None:
    conflicts = sorted(
        name
        for name in _configured_extraction_field_names(options)
        if name.casefold() in INTERNAL_LINEAGE_FIELDS
    )
    if conflicts:
        raise ValueError(
            "extraction fields collide with reserved dbt-ml lineage columns: "
            f"{', '.join(conflicts)}"
        )


class ExtractionConfig(BaseModel):
    model_config = _STRICT_CONFIG

    backend: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)
    # Rows flush to the warehouse every N documents (issue #77) — bounds
    # memory and gives incremental runs per-flush crash recovery. Excluded
    # from code_version: it changes execution, never output content.
    flush_every: int = Field(default=5000, gt=0)

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
                    _RedactedConfigInput,
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
    def _prepare_model_input(cls, wrapped: _RedactedConfigInput) -> Any:
        raw = wrapped.take()
        if isinstance(raw, cls):
            return raw
        if not isinstance(raw, Mapping):
            raise PydanticCustomError(
                "extraction_config",
                "extraction config must be a mapping",
            )
        return _protect_extraction_mapping(raw)

    @field_validator("options")
    @classmethod
    def _validate_options(cls, options: dict[str, Any]) -> dict[str, Any]:
        validate_extraction_field_names(options)
        return options


class TransformConfig(BaseModel):
    model_config = _STRICT_CONFIG

    type: str
    module: str | None = None
    uses_llm: bool = False
    options: dict[str, Any] = Field(default_factory=dict)


class ChunkConfig(BaseModel):
    """Split an upstream document's text into one row per chunk (issue #86).

    strategy:      "recursive" (character splitter on a separator hierarchy,
                   pure-python) or "tokens" (tiktoken-based).
    text_field:    upstream column holding the text to split (default "text").
    chunk_size:    target chunk size — characters for recursive, tokens for
                   tokens.
    chunk_overlap: overlap carried between adjacent chunks, same unit.
    encoding:      tiktoken encoding name for the tokens strategy.
    """

    model_config = _STRICT_CONFIG

    strategy: Literal["recursive", "tokens"] = "recursive"
    text_field: str = "text"
    chunk_size: int = 1000
    chunk_overlap: int = 100
    encoding: str = "cl100k_base"
    options: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_sizes(self) -> ChunkConfig:
        if self.chunk_size <= 0:
            raise ValueError("chunk.chunk_size must be positive")
        if self.chunk_overlap < 0:
            raise ValueError("chunk.chunk_overlap must be non-negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk.chunk_overlap must be smaller than chunk_size")
        return self


class EmbedConfig(BaseModel):
    model_config = _STRICT_CONFIG

    provider: str = Field(
        min_length=1,
        pattern=r"^[a-z0-9][a-z0-9_-]*$",
    )
    model: str = Field(min_length=1)
    text_field: str = Field(default="text", min_length=1)
    id_field: str = Field(default="chunk_id", min_length=1)
    vector_field: str = Field(default="embedding", min_length=1)
    dimensions: int = Field(gt=0, le=65_536)
    batch_size: int = Field(default=128, gt=0, le=10_000)
    max_retries: int = Field(default=4, ge=0)

    @model_validator(mode="after")
    def _validate_fields(self) -> EmbedConfig:
        fields = (self.id_field, self.text_field, self.vector_field)
        if len(set(fields)) != len(fields):
            raise ValueError(
                "embed.id_field, text_field, and vector_field must be distinct"
            )
        conflicts = sorted(set(fields) & EMBED_METADATA_FIELDS)
        if conflicts:
            raise ValueError(
                "embed fields are reserved for embedding metadata: "
                f"{', '.join(conflicts)}"
            )
        return self


class MLArtifactConfig(BaseModel):
    model_config = _STRICT_CONFIG

    path: Path | None = None
    include_metrics: bool = True
    # Opt-in for an artifact location outside the project directory
    # (issue #65). Excluded from code_version.
    external: bool = False

    @field_serializer("path")
    def _serialize_path(self, path: Path | None) -> str | None:
        return path.as_posix() if path is not None else None


class MLConfig(BaseModel):
    model_config = _STRICT_CONFIG

    task: Literal[
        "features",
        "classifier",
        "regressor",
        "cluster",
        "topic_model",
        "nlp",
    ]
    mode: Literal["fit_transform", "fit", "predict", "load_pretrained"] = "fit_transform"
    provider: str | None = None
    text_field: str | None = None
    label_field: str | None = None
    artifact: MLArtifactConfig = Field(default_factory=MLArtifactConfig)
    metrics: list[str] = Field(default_factory=list)
    options: dict[str, Any] = Field(default_factory=dict)


class FieldConfig(BaseModel):
    model_config = _STRICT_CONFIG

    name: str
    description: str | None = None
    data_type: Literal[
        "string", "integer", "float", "boolean", "date", "timestamp", "json"
    ] | None = Field(
        default=None,
        validation_alias=AliasChoices("data_type", "data-type", "type", "dtype"),
    )

    @field_validator("data_type", mode="before")
    @classmethod
    def _normalize_data_type(cls, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        normalized = value.strip().lower()
        aliases = {
            "str": "string",
            "text": "string",
            "varchar": "string",
            "int": "integer",
            "int64": "integer",
            "bigint": "integer",
            "number": "float",
            "double": "float",
            "float64": "float",
            "decimal": "float",
            "bool": "boolean",
            "datetime": "timestamp",
            "datetime64": "timestamp",
            "object": "json",
            "array": "json",
            "struct": "json",
        }
        return aliases.get(normalized, normalized)


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    _yaml_provenance: YamlProvenance | None = PrivateAttr(default=None)

    name: str
    description: str | None = None
    source: str | None = None
    depends_on: list[str] | None = None
    extraction: ExtractionConfig | None = None
    transform: TransformConfig | None = None
    ml: MLConfig | None = None
    chunk: ChunkConfig | None = None
    embed: EmbedConfig | None = None
    fields: list[FieldConfig] = Field(default_factory=list)
    materialization: Literal["full", "incremental"] = "full"
    on_schema_change: Literal["fail", "ignore", "append_new_columns"] = "fail"
    # Adapter-specific physical-layout knobs (issue #91), opaque to core:
    # the active adapter validates its own keys (e.g. BigQuery partition_by /
    # cluster_by); adapters that support none ignore the block so one project
    # can target DuckDB in dev and BigQuery in prod. Excluded from
    # code_version: layout never changes row content. Changing partitioning
    # on an existing table requires --full-refresh to rebuild it.
    warehouse_options: dict[str, Any] = Field(default_factory=dict)
    tests: list[Any] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)

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
                    _RedactedConfigInput,
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
    def _prepare_model_input(cls, wrapped: _RedactedConfigInput) -> Any:
        raw = wrapped.take()
        if isinstance(raw, cls):
            return raw
        if not isinstance(raw, Mapping):
            raise PydanticCustomError(
                "model_config",
                "model config must be a mapping",
            )
        prepared = dict(raw)
        extraction = prepared.get("extraction")
        if isinstance(extraction, Mapping):
            prepared["extraction"] = _protect_extraction_mapping(extraction)
        return prepared

    @field_validator("name")
    @classmethod
    def _validate_name(cls, v: str) -> str:
        return validate_node_name(v, kind="Model", reserve_internal=True)

    @model_validator(mode="after")
    def _validate_single_kind(self) -> ModelConfig:
        kinds = [
            label
            for label, block in (
                ("extraction", self.extraction),
                ("transform", self.transform),
                ("ml", self.ml),
                ("chunk", self.chunk),
                ("embed", self.embed),
            )
            if block is not None
        ]
        if len(kinds) > 1:
            raise ValueError(
                f"Model '{self.name}' declares multiple kind blocks "
                f"({', '.join(kinds)}); exactly one of "
                "extraction/transform/ml/chunk/embed is allowed"
            )
        return self

    @property
    def kind_block_count(self) -> int:
        return sum(
            b is not None
            for b in (
                self.extraction,
                self.transform,
                self.ml,
                self.chunk,
                self.embed,
            )
        )

    @property
    def yaml_provenance(self) -> YamlProvenance | None:
        return self._yaml_provenance

    def format_yaml_diagnostic(
        self,
        message: str,
        *,
        relative_path: ConfigPath = (),
    ) -> str:
        if self._yaml_provenance is None:
            return message
        return self._yaml_provenance.format_message(
            message,
            relative_path=relative_path,
        )


def protect_model_llm_credential_option(model: ModelConfig) -> None:
    extraction = model.extraction
    if extraction is None or "api_key_env" not in extraction.options:
        return
    value = extraction.options["api_key_env"]
    if isinstance(value, CredentialReference):
        return
    try:
        extraction.options["api_key_env"] = CredentialReference.from_env_name(
            value
        )
    except (TypeError, CredentialReferenceError):
        extraction.options["api_key_env"] = None


class ModelFile(BaseModel):
    model_config = _STRICT_CONFIG

    version: Literal[2] = 2
    models: list[ModelConfig]

    @model_validator(mode="after")
    def _validate_models_have_kind(self) -> ModelFile:
        # Bare ModelConfig (no kind block) is allowed programmatically (DAG
        # fixtures, docs tooling); models loaded from project YAML must
        # declare what they run.
        missing = [m.name for m in self.models if m.kind_block_count == 0]
        if missing:
            raise ValueError(
                f"Models missing an extraction/transform/ml/chunk/embed block: "
                f"{', '.join(sorted(missing))}"
            )
        return self
