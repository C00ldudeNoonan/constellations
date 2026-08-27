from __future__ import annotations

import re
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

from ..agent_context import AgentContextGrain
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
# Columns a native `llm:` map model generates for lineage/provenance (issue
# #144). Reserved like the extraction lineage columns: user `fields` and the
# configured id/input columns must not collide with them.
LLM_METADATA_FIELDS = frozenset(
    {
        "llm_provider",
        "llm_model",
        "llm_provider_implementation",
        "llm_input_hash",
        "llm_config_hash",
        # Which prompt produced the row, in a form a human reads and a query
        # groups by — `llm_config_hash` records that something changed but
        # cannot say what (issue #303).
        "prompt_name",
        "prompt_version",
        "generated_at",
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
            "extraction fields collide with reserved stel lineage columns: "
            f"{', '.join(conflicts)}"
        )


class PostExtractConfig(BaseModel):
    """Project-local field derivation applied before an extracted row is staged."""

    model_config = _STRICT_CONFIG

    module: str
    options: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _expand_module_shorthand(cls, value: Any) -> Any:
        if isinstance(value, str):
            return {"module": value}
        return value


class ExtractionConfig(BaseModel):
    model_config = _STRICT_CONFIG

    backend: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)
    post_extract: PostExtractConfig | None = None
    # Rows flush to the warehouse every N documents (issue #77) — bounds
    # memory and gives incremental runs per-flush crash recovery. Excluded
    # from code_version: it changes execution, never output content.
    flush_every: int = Field(default=5000, gt=0)
    # Incremental publication coalesces this many flushes into one upsert (issue
    # #293), so a run of many small flushes shares one warehouse MERGE instead of
    # one per flush. `1` (default) publishes every flush, matching prior behavior.
    # Higher values cut MERGE count (and BigQuery bytes billed) at the cost of
    # ~publish_every× peak memory and coarser crash recovery — a partial buffer is
    # discarded and re-extracted on the next run. Excluded from code_version like
    # flush_every: it changes execution cadence, never output content.
    publish_every: int = Field(default=1, gt=0)

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
    path: str | None = None
    uses_llm: bool = False
    options: dict[str, Any] = Field(default_factory=dict)
    # Incremental transforms invoke and publish in batches of this many changed
    # parents (issue #379), committing each batch the way extraction commits
    # each flush. A failed run then re-pays one batch instead of the corpus.
    #
    # The default is high enough that a run with fewer changed parents than
    # this is a single batch — identical behavior and one warehouse MERGE, so
    # ordinary projects see no change. Large runs get checkpointing, at one
    # MERGE per batch. Excluded from code_version like extraction's
    # flush_every: it changes execution cadence, never output content.
    commit_every: int = Field(default=1000, gt=0)

    @model_validator(mode="after")
    def _validate_implementation(self) -> TransformConfig:
        # Exactly one implementation surface per type: python uses `module`,
        # sql uses a `.sql` `path`. The compiler validates the file itself; this
        # guards the config shape so a mismatched pair fails fast at load time.
        if self.type == "python":
            if self.path is not None:
                raise ValueError("`transform.type: python` does not accept a `path`")
        elif self.type == "sql":
            if self.module is not None:
                raise ValueError("`transform.type: sql` does not accept a `module`")
            if self.uses_llm:
                raise ValueError("`transform.type: sql` does not accept `uses_llm`")
        return self


class AgentContextConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    contract: Literal["agent_context/v1"] = "agent_context/v1"
    grain: AgentContextGrain


# Columns a `chunk:` model produces itself. Declared here rather than in
# execution so config validation can reject a heading column that would
# overwrite one (issue #343); `execution/chunk.py` imports it as the single
# definition.
CHUNK_GENERATED_FIELDS = frozenset(
    {
        "chunk_id",
        "document_id",
        "chunk_index",
        "chunk_count",
        "text",
        "chunk_strategy",
        "code_version",
        "chunked_at",
    }
)


class HeadingConfig(BaseModel):
    r"""Detect section headings while splitting, and attribute chunks to them.

    The splitter knows the document's full text and every boundary position,
    so it can say exactly which heading a chunk falls under (issue #332). A
    downstream transform can only re-derive that from chunk fragments, and
    misses the edge cases — a heading in a chunk's tail, a chunk starting
    mid-heading — that offsets settle outright.

    `pattern` is matched line-anchored (`re.MULTILINE`) against the source
    text. A capture group names the section; without one the whole match is
    used — so `^(Item\s+\d{1,2}[A-C]?)[.:]` yields `Item 1A` while
    `^Item\s+\d{1,2}[A-C]?[.:]` yields `Item 1A.`.
    """

    model_config = _STRICT_CONFIG

    pattern: str = Field(min_length=1)
    column: str = "section"

    @field_validator("pattern")
    @classmethod
    def _validate_pattern(cls, v: str) -> str:
        try:
            compiled = re.compile(v, re.MULTILINE)
        except re.error as error:
            raise ValueError(f"chunk.headings.pattern is not a valid regex: {error}") from None
        if compiled.groups > 1:
            raise ValueError(
                "chunk.headings.pattern must have at most one capture group; "
                "the group names the section, and more than one is ambiguous"
            )
        return v

    @field_validator("column")
    @classmethod
    def _validate_column(cls, v: str) -> str:
        return validate_node_name(v, kind="Heading column")


class ChunkConfig(BaseModel):
    """Split an upstream document's text into one row per chunk (issue #86).

    strategy:      "recursive" (character splitter on a separator hierarchy,
                   pure-python) or "tokens" (tiktoken-based).
    text_field:    upstream column holding the text to split (default "text").
    chunk_size:    target chunk size — characters for recursive, tokens for
                   tokens.
    chunk_overlap: overlap carried between adjacent chunks, same unit.
    encoding:      tiktoken encoding name for the tokens strategy.
    in_text_metadata:
                   upstream columns to render into the chunk text itself, as a
                   small block ahead of it (issue #308). Additive: the columns
                   are still carried onto every chunk row, because SQL reads
                   columns and the embedding model reads only the text, and a
                   rendering aimed at one reader must never remove the copy the
                   other depends on. The block counts against `chunk_size`, so
                   a chunk never exceeds the size the embedder was configured
                   for.
    """

    model_config = _STRICT_CONFIG

    strategy: Literal["recursive", "tokens"] = "recursive"
    text_field: str = "text"
    chunk_size: int = 1000
    chunk_overlap: int = 100
    encoding: str = "cl100k_base"
    in_text_metadata: list[str] = Field(default_factory=list)
    # Attribute each chunk to the heading it falls under (issue #332).
    headings: HeadingConfig | None = None
    options: dict[str, Any] = Field(default_factory=dict)

    @field_validator("in_text_metadata")
    @classmethod
    def _validate_in_text_metadata(cls, v: list[str]) -> list[str]:
        seen: set[str] = set()
        for name in v:
            if not name.strip():
                raise ValueError("chunk.in_text_metadata entries must not be empty")
            if name in seen:
                raise ValueError(
                    f"chunk.in_text_metadata lists '{name}' twice; the block is "
                    "rendered in declared order, so a repeat is a typo"
                )
            seen.add(name)
        return v

    @model_validator(mode="after")
    def _validate_sizes(self) -> ChunkConfig:
        if self.chunk_size <= 0:
            raise ValueError("chunk.chunk_size must be positive")
        if self.chunk_overlap < 0:
            raise ValueError("chunk.chunk_overlap must be non-negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk.chunk_overlap must be smaller than chunk_size")
        if self.headings is not None and self.strategy != "recursive":
            raise ValueError(
                "chunk.headings requires `strategy: recursive`: attribution "
                "works from the source character offsets the recursive "
                "splitter produces, which the token splitter does not have"
            )
        if self.headings is not None and self.headings.column == self.text_field:
            raise ValueError(
                f"chunk.headings.column must not be the text field "
                f"'{self.text_field}'"
            )
        if self.headings is not None and self.headings.column in CHUNK_GENERATED_FIELDS:
            # The upstream-column guard cannot catch these: an extraction
            # model has no `chunk_id`, so `column: chunk_id` passed validation
            # and then overwrote every generated chunk id with a section name
            # — duplicate identifiers on a full materialization, and failed
            # key validation on an incremental one (Codex review, #343).
            raise ValueError(
                f"chunk.headings.column '{self.headings.column}' is a column "
                "the chunk model generates; naming it would overwrite that "
                f"value. Generated columns: {', '.join(sorted(CHUNK_GENERATED_FIELDS))}"
            )
        if self.text_field in self.in_text_metadata:
            raise ValueError(
                f"chunk.in_text_metadata must not name the text field "
                f"'{self.text_field}'; the block is prepended to that text"
            )
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
    # Embedded rows publish to the warehouse every N rows, advancing state for
    # exactly those rows (issue #401). Embeds were the last all-or-nothing
    # stage and the worst one to leave that way: their re-run cost is metered
    # provider spend, not CPU, so an end-of-run failure threw away every paid
    # call. Peak memory is now one flush rather than the corpus. Excluded from
    # code_version like extraction's: it changes execution cadence, never
    # output content -- including it would re-embed every existing corpus at
    # provider prices on upgrade.
    flush_every: int = Field(default=5000, gt=0)

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


class PromptRef(BaseModel):
    """A named, versioned prompt file: `prompts/<name>/<version>.md` (#303).

    Both parts become path segments, so they are charset-validated here rather
    than sanitized at read time. There is deliberately no `latest` — a moving
    reference would make two runs of the same committed project resolve to
    different text, which is the mutable-prompt problem versions exist to fix.
    """

    model_config = _STRICT_CONFIG

    name: str = Field(min_length=1)
    version: str = Field(min_length=1)

    @field_validator("name")
    @classmethod
    def _validate_name(cls, v: str) -> str:
        from ..prompts import validate_prompt_segment

        return validate_prompt_segment(v, label="name")

    @field_validator("version")
    @classmethod
    def _validate_version(cls, v: str) -> str:
        from ..prompts import validate_prompt_segment

        return validate_prompt_segment(v, label="version")


class LLMTransformConfig(BaseModel):
    """Map an LLM prompt over an upstream warehouse relation (issue #144).

    The model's `fields:` list is the structured output schema; the prompt is an
    inline instruction. One provider call per unprocessed input row produces one
    validated object (`output_cardinality: one`) or a list of objects
    (`output_cardinality: many`, fanned out to one row each with a deterministic
    id). Credentials stay operator-owned in profiles/env — the `llm:` block never
    carries an api key.
    """

    model_config = _STRICT_CONFIG

    mode: Literal["map"] = "map"
    input_field: str = Field(min_length=1)
    id_field: str = Field(default="id", min_length=1)
    output_cardinality: Literal["one", "many"] = "one"
    # Inline text, or a reference to a versioned prompt file (issue #303).
    # Inline stays supported: it is right for quick projects and examples.
    prompt: str | PromptRef = Field(union_mode="left_to_right")
    provider: str = Field(default="default", min_length=1)
    model: str = Field(default="default", min_length=1)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, gt=0, le=1_000_000)
    max_concurrent: int = Field(default=8, gt=0, le=1_000)
    max_retries: int = Field(default=4, ge=0)
    # For output_cardinality: many — the fan-out row's stable primary key
    # (f"{id_value}__{ordinal}") and its position within the parent's output.
    row_id_field: str = Field(default="llm_row_id", min_length=1)
    ordinal_field: str = Field(default="ordinal", min_length=1)

    @field_validator("prompt")
    @classmethod
    def _validate_inline_prompt(cls, v: str | PromptRef) -> str | PromptRef:
        # The union dropped `min_length`, and an empty instruction is a typo
        # rather than a prompt.
        if isinstance(v, str) and not v.strip():
            raise ValueError("llm.prompt must not be empty")
        return v

    @model_validator(mode="after")
    def _validate_reserved_fields(self) -> LLMTransformConfig:
        configured = {
            "input_field": self.input_field,
            "id_field": self.id_field,
            "row_id_field": self.row_id_field,
            "ordinal_field": self.ordinal_field,
        }
        conflicts = sorted(
            f"{label}={value}"
            for label, value in configured.items()
            if value.casefold() in LLM_METADATA_FIELDS
        )
        if conflicts:
            raise ValueError(
                "llm fields are reserved for generation metadata: "
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


class SearchEmbeddingIdentityConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    provider_contract_version: int = Field(gt=0)
    provider_implementation: str = Field(min_length=1)
    semantic_config_fingerprint: str = Field(min_length=1)
    dimensions: int = Field(gt=0)


class SearchVectorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    field: str
    dimensions: int = Field(gt=0)
    metric: Literal["cosine", "euclidean", "dot"] = "cosine"
    search: Literal["exact", "approximate"] = "exact"
    embedding: Literal["inherit"] | SearchEmbeddingIdentityConfig

    @model_validator(mode="after")
    def _validate_embedding_dimensions(self) -> SearchVectorConfig:
        if (
            isinstance(self.embedding, SearchEmbeddingIdentityConfig)
            and self.embedding.dimensions != self.dimensions
        ):
            raise ValueError(
                "search.vector embedding identity dimensions must match vector dimensions"
            )
        return self


class SearchFullTextConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    fields: tuple[str, ...]

    @field_validator("fields")
    @classmethod
    def _validate_fields(cls, fields: tuple[str, ...]) -> tuple[str, ...]:
        if not fields:
            raise ValueError("search.full_text.fields must not be empty")
        if len(fields) != len(set(fields)):
            raise ValueError("search.full_text.fields must not contain duplicates")
        return fields


class SearchAttributeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    data_type: Literal[
        "string",
        "integer",
        "float",
        "boolean",
        "date",
        "timestamp",
        "array[string]",
    ]
    nullable: bool = False
    filter_role: Literal["none", "user", "policy", "user_and_policy"] = "none"
    sortable: bool = False
    returned: bool = False


class SearchQueryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    modes: frozenset[Literal["vector", "text", "hybrid", "filter"]]
    consistency: Literal["strong"] = "strong"

    @field_validator("modes")
    @classmethod
    def _validate_modes(
        cls, modes: frozenset[str]
    ) -> frozenset[str]:
        if not modes:
            raise ValueError("search.query.modes must not be empty")
        return modes


# `ref('name')` or a bare name — the same grammar `dag.parse_ref` accepts.
# Duplicated here because config is imported by dag and cannot import it back;
# the eval tests assert the two stay in agreement.
_REF_EXPRESSION = re.compile(r"^\s*ref\(\s*['\"]([^'\"]+)['\"]\s*\)\s*$")


def _parse_ref_expression(value: str) -> str:
    match = _REF_EXPRESSION.match(value)
    if match:
        return match.group(1)
    return value.strip()


class EvalConfig(BaseModel):
    """Score a model's predictions against labelled ground truth (issue #309).

    Reads two already-materialized relations and emits metric rows, so it costs
    no inference and can run in CI on every change. `golden` answers "identical
    or not"; this answers "which labels moved, and by how much", which is the
    question a prompt or model change actually raises.
    """

    model_config = _STRICT_CONFIG

    # Only single-label classification today. Multi-label and regression are
    # different metric families, so they get their own `kind:` rather than
    # overloading this one.
    kind: Literal["classification"] = "classification"
    predictions: str
    predicted_field: str
    expected: str
    expected_field: str
    key: str
    # Overrides the taxonomy taken from the predicted field's `enum` (#304).
    # Only needed when the predictions model does not declare one.
    labels: list[str] = Field(default_factory=list)

    @field_validator("predicted_field", "expected_field", "key")
    @classmethod
    def _validate_column(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("eval column names must not be empty")
        return v

    @field_validator("labels")
    @classmethod
    def _validate_labels(cls, v: list[str]) -> list[str]:
        seen: set[str] = set()
        for label in v:
            if not label.strip():
                raise ValueError("eval labels must not be empty")
            if label in seen:
                raise ValueError(f"eval labels list '{label}' twice")
            seen.add(label)
        return v


class SearchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    access: Literal["governed", "public"] = "public"
    store: str | None = None
    collection: str | None = None
    id_field: str
    document_id_field: str | None = "document_id"
    chunk_id_field: str | None = None
    text_fields: tuple[str, ...]
    return_text_fields: tuple[str, ...] = ()
    vector: SearchVectorConfig | None = None
    full_text: SearchFullTextConfig | None = None
    attributes: tuple[SearchAttributeConfig, ...] = ()
    display_fields: tuple[str, ...] = ()
    query: SearchQueryConfig
    on_index_change: Literal["fail", "rebuild", "online"] = "fail"
    batch_size: int = Field(default=1000, ge=1, le=100_000)
    index_options: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_contract(self) -> SearchConfig:
        if not self.text_fields:
            raise ValueError("search.text_fields must not be empty")
        declared_fields = {
            self.id_field,
            self.document_id_field,
            self.chunk_id_field,
            *self.text_fields,
            self.vector.field if self.vector is not None else None,
            *(attribute.name for attribute in self.attributes),
            *self.display_fields,
        }
        reserved = {"_distance", "_score", "_relevance_score"}
        if declared_fields & reserved:
            raise ValueError(
                "search fields cannot use reserved retrieval score column names"
            )
        for label, values in (
            ("text_fields", self.text_fields),
            ("return_text_fields", self.return_text_fields),
            ("display_fields", self.display_fields),
        ):
            if any(not value for value in values):
                raise ValueError(f"search.{label} must contain non-empty field names")
            if len(values) != len(set(values)):
                raise ValueError(f"search.{label} must not contain duplicates")
        if not set(self.return_text_fields).issubset(self.text_fields):
            raise ValueError("search.return_text_fields must be a subset of text_fields")
        if self.full_text is not None and not set(self.full_text.fields).issubset(
            self.text_fields
        ):
            raise ValueError("search.full_text.fields must be a subset of text_fields")
        if {"vector", "hybrid"} & self.query.modes and self.vector is None:
            raise ValueError("vector and hybrid query modes require search.vector")
        if {"text", "hybrid"} & self.query.modes and self.full_text is None:
            raise ValueError("text and hybrid query modes require search.full_text")
        attribute_names = [attribute.name for attribute in self.attributes]
        if len(attribute_names) != len(set(attribute_names)):
            raise ValueError("search.attributes must not contain duplicate names")
        if self.access == "public" and any(
            attribute.filter_role in {"policy", "user_and_policy"}
            for attribute in self.attributes
        ):
            raise ValueError("public search resources cannot declare policy attributes")
        if self.access == "governed" and not any(
            attribute.filter_role in {"policy", "user_and_policy"}
            for attribute in self.attributes
        ):
            raise ValueError("governed search resources require a policy attribute")
        return self

    def projected_fields(self) -> tuple[str, ...]:
        fields: list[str] = [self.id_field]
        for field in (
            self.document_id_field,
            self.chunk_id_field,
            *self.text_fields,
            self.vector.field if self.vector is not None else None,
            *(attribute.name for attribute in self.attributes),
            *self.display_fields,
        ):
            if field is not None and field not in fields:
                fields.append(field)
        return tuple(fields)


_RETRIEVAL_THRESHOLD_KEY = re.compile(
    r"^(recall|precision|hit_rate|mrr|ndcg)_at_(\d+)$"
)


class RetrievalThresholdConfig(BaseModel):
    """Minimum acceptable value for one `<metric>_at_<k>` aggregate (issue #137).

    `severity: error` fails `stel eval` (exit 1); `severity: warn` reports
    below-threshold without failing, mirroring schema-test severity."""

    model_config = _STRICT_CONFIG

    min: float = Field(ge=0.0, le=1.0)
    severity: Literal["error", "warn"] = "error"


class RetrievalTestConfig(BaseModel):
    """A golden-set retrieval evaluation attached to a `search:` model
    (issue #137). `golden_set` is a `ref('model')` expression naming an
    ordinary stel model whose materialized rows carry the golden-query
    contract (query_id, query_text/query_vector, relevant_ids, ...) — see
    docs/architecture/semantic-retrieval.md."""

    model_config = _STRICT_CONFIG

    name: str
    golden_set: str
    mode: Literal["vector", "text", "hybrid"] | None = None
    at: tuple[int, ...] = (10,)
    thresholds: dict[str, RetrievalThresholdConfig] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_contract(self) -> RetrievalTestConfig:
        if not self.golden_set.strip():
            raise ValueError("retrieval_tests.golden_set must not be empty")
        if not self.at:
            raise ValueError("retrieval_tests.at must not be empty")
        if any(isinstance(k, bool) or k <= 0 for k in self.at):
            raise ValueError("retrieval_tests.at must contain positive integers")
        if len(self.at) != len(set(self.at)):
            raise ValueError("retrieval_tests.at must not contain duplicate cutoffs")
        for key in self.thresholds:
            match = _RETRIEVAL_THRESHOLD_KEY.match(key)
            if not match:
                raise ValueError(
                    f"Unknown retrieval threshold '{key}'; expected "
                    "'<metric>_at_<k>' where metric is one of recall, "
                    "precision, hit_rate, mrr, ndcg"
                )
            cutoff = int(match.group(2))
            if cutoff not in self.at:
                raise ValueError(
                    f"Threshold '{key}' references cutoff {cutoff}, which is "
                    f"not declared in `at`: {list(self.at)}"
                )
        return self


class FieldConfig(BaseModel):
    model_config = _STRICT_CONFIG

    name: str
    description: str | None = None
    data_type: Literal[
        "string", "integer", "float", "boolean", "date", "timestamp", "json", "enum"
    ] | None = Field(
        default=None,
        validation_alias=AliasChoices("data_type", "data-type", "type", "dtype"),
    )
    # The closed set a `type: enum` field may take (issue #304). Declared once
    # here and derived everywhere else: the provider's output schema, the
    # implicit accepted_values check, and — where a provider's structured
    # output cannot carry an enum — the rendered prompt. A label list written
    # out in several places is a list that drifts.
    values: list[str] = Field(default_factory=list)

    @field_validator("values")
    @classmethod
    def _validate_values(cls, v: list[str]) -> list[str]:
        seen: set[str] = set()
        for value in v:
            if not value.strip():
                raise ValueError("field values must not be empty")
            if value in seen:
                raise ValueError(f"field values list '{value}' twice")
            seen.add(value)
        return v

    @model_validator(mode="after")
    def _validate_enum(self) -> FieldConfig:
        if self.data_type == "enum" and not self.values:
            raise ValueError(
                f"Field '{self.name}' is `type: enum` but declares no `values:`; "
                "an enum with no closed set constrains nothing"
            )
        if self.values and self.data_type != "enum":
            raise ValueError(
                f"Field '{self.name}' declares `values:` but is "
                f"`type: {self.data_type or 'string'}`; use `type: enum`"
            )
        return self

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
    llm: LLMTransformConfig | None = None
    search: SearchConfig | None = None
    eval: EvalConfig | None = None
    # Golden-set retrieval evaluations over this search index (issue #137);
    # only meaningful — and only allowed — when `search` is set.
    retrieval_tests: list[RetrievalTestConfig] = Field(default_factory=list)
    agent_context: AgentContextConfig | None = None
    fields: list[FieldConfig] = Field(default_factory=list)
    materialization: Literal["full", "incremental"] = "full"
    on_schema_change: Literal["fail", "ignore", "append_new_columns"] = "fail"
    # Change-detection columns for incremental publication (issue #281): when a
    # matched row's listed columns are all NULL-safe-equal between the batch and
    # the target, the row is left untouched instead of rewriting every column
    # (including large payloads). A declared fingerprint — e.g. [content_hash,
    # code_version] for extraction — keeps document semantics out of the generic
    # adapter. Empty (default) preserves the always-overwrite behavior. Only
    # valid with `materialization: incremental`. Excluded from code_version: it
    # changes publication, not row content.
    update_when_changed: list[str] = Field(default_factory=list)
    # Required for `materialization: incremental` SQL transforms (issue #142);
    # other incremental model kinds key on a fixed identity column instead
    # (document_id, chunk_id, ...) and must leave this unset.
    unique_key: str | None = None
    # Adapter-specific physical-layout knobs (issue #91), opaque to core:
    # the active adapter validates its own keys (e.g. BigQuery partition_by /
    # cluster_by); adapters that support none ignore the block so one project
    # can target DuckDB in dev and BigQuery in prod. Excluded from
    # code_version: layout never changes row content. Changing partitioning
    # on an existing table requires --full-refresh to rebuild it.
    warehouse_options: dict[str, Any] = Field(default_factory=dict)
    tests: list[Any] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    # Matrix expansion (issue #57): expanded by loader._expand_for_each; absent
    # on concrete models. Keys are axis names; values are lists of axis values.
    for_each: dict[str, list[Any]] | None = None

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

    def transform_commit_every(self) -> int:
        """Changed parents per commit batch for an incremental transform."""
        return self.transform.commit_every if self.transform is not None else 1000

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
                ("llm", self.llm),
                ("search", self.search),
                ("eval", self.eval),
            )
            if block is not None
        ]
        if len(kinds) > 1:
            raise ValueError(
                f"Model '{self.name}' declares multiple kind blocks "
                f"({', '.join(kinds)}); exactly one of "
                "extraction/transform/ml/chunk/embed/llm/search/eval is allowed"
            )
        if self.search is not None and "materialization" not in self.model_fields_set:
            raise ValueError(
                f"Search resource '{self.name}' must explicitly declare materialization"
            )
        if self.agent_context is not None and self.search is not None:
            raise ValueError(
                f"Search resource '{self.name}' cannot declare agent_context; "
                "declare it on the upstream warehouse model"
            )
        if self.agent_context is not None and self.transform is None:
            raise ValueError(
                f"Model '{self.name}' can declare agent_context only on a "
                "warehouse transform model. The built-in extraction/chunk "
                "primitives cannot derive the contract's trusted policy and "
                "bitemporal fields, so this is an intentional v1 boundary, not "
                "a missing kind — see docs/architecture/agent-context-v1.md. "
                "Wrap the pipeline in a custom `transform: {type: python}` "
                "model to make it MCP-discoverable."
            )
        if self.llm is not None:
            self._validate_llm_fields()
        return self

    @model_validator(mode="after")
    def _validate_update_when_changed(self) -> ModelConfig:
        columns = self.update_when_changed
        if not columns:
            return self
        if self.materialization != "incremental":
            raise ValueError(
                f"Model '{self.name}': update_when_changed requires "
                "`materialization: incremental`"
            )
        ident_re = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
        seen: set[str] = set()
        for column in columns:
            if not isinstance(column, str) or not ident_re.match(column):
                raise ValueError(
                    f"Model '{self.name}': update_when_changed column "
                    f"{column!r} must be a valid identifier"
                )
            if column in seen:
                raise ValueError(
                    f"Model '{self.name}': update_when_changed lists duplicate "
                    f"column '{column}'"
                )
            seen.add(column)
        return self

    @model_validator(mode="after")
    def _validate_for_each(self) -> ModelConfig:
        fe = self.for_each
        if fe is None:
            return self
        if not fe:
            raise ValueError(
                f"Model '{self.name}': for_each must declare at least one axis"
            )
        ident_re = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
        for axis_name, values in fe.items():
            if not ident_re.match(axis_name):
                raise ValueError(
                    f"Model '{self.name}': for_each axis name '{axis_name}' must be a "
                    "valid identifier (letters, digits, underscores; start with letter or _)"
                )
            if not values:
                raise ValueError(
                    f"Model '{self.name}': for_each axis '{axis_name}' must have at least "
                    "one value"
                )
        return self

    def _validate_llm_fields(self) -> None:
        assert self.llm is not None
        if not self.fields:
            raise ValueError(
                f"llm model '{self.name}' requires `fields:` to define its "
                "structured output schema"
            )
        field_names = [field.name for field in self.fields]
        duplicates = sorted({n for n in field_names if field_names.count(n) > 1})
        if duplicates:
            raise ValueError(
                f"llm model '{self.name}' declares duplicate output fields: "
                f"{', '.join(duplicates)}"
            )
        reserved = {n for n in field_names if n.casefold() in LLM_METADATA_FIELDS}
        generated = {self.llm.id_field, self.llm.row_id_field, self.llm.ordinal_field}
        reserved |= {n for n in field_names if n in generated}
        if reserved:
            raise ValueError(
                f"llm model '{self.name}' output fields collide with generated "
                f"columns: {', '.join(sorted(reserved))}"
            )

    @model_validator(mode="after")
    def _derive_eval_edges(self) -> ModelConfig:
        # An eval's inputs are its two scored relations, derived here rather
        # than in a compiler pass so every ProjectDAG construction path — the
        # runner, `stel ls`, manifest and run_results generation — sees the
        # same edges (Codex review, #328). Declaring `depends_on:` directly is
        # rejected so the edges keep one source of truth.
        if self.eval is None:
            return self
        if self.depends_on:
            raise ValueError(
                f"Eval model '{self.name}' must not declare `depends_on:`; its "
                "inputs are `predictions:` and `expected:`"
            )
        refs: list[str] = []
        for label, expression in (
            ("predictions", self.eval.predictions),
            ("expected", self.eval.expected),
        ):
            name = _parse_ref_expression(expression)
            if not name:
                raise ValueError(
                    f"Eval model '{self.name}' `{label}:` must name a model"
                )
            refs.append(name)
        # Deduplicate: scoring a relation against itself is degenerate but the
        # DAG must not carry the same edge twice.
        self.depends_on = [f"ref('{name}')" for name in dict.fromkeys(refs)]
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
                self.llm,
                self.search,
                self.eval,
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
                f"Models missing an extraction/transform/ml/chunk/embed/llm block "
                f"or search block: "
                f"{', '.join(sorted(missing))}"
            )
        return self
