from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import (
    AfterValidator,
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)

from ..budget import LLMBudgetConfig
from ..credentials import CredentialReference
from ..optional_dependencies import import_optional_dependency

_STRICT_OPTIONS = ConfigDict(extra="forbid", strict=True, populate_by_name=True)
_NonEmptyString = Annotated[str, StringConstraints(min_length=1, pattern=r"\S")]
_ProviderName = Annotated[
    str,
    StringConstraints(min_length=1, pattern=r"^[a-z0-9][a-z0-9_-]*$"),
]
_JsonSchemaType = Literal[
    "array",
    "boolean",
    "integer",
    "null",
    "number",
    "object",
    "string",
]

_LLM_INTEGER_BOUNDS: dict[str, tuple[int, int]] = {
    "max_tokens": (1, 65_536),
    "max_retries": (0, 20),
    "max_concurrent": (1, 100),
}
_LLM_NUMBER_BOUNDS: dict[str, tuple[float, float]] = {
    "temperature": (0.0, 1.0),
    "batch_poll_seconds": (0.1, 3600.0),
}


def validate_llm_numeric_options(options: Mapping[str, Any]) -> None:
    """Validate reusable LLM helper arguments outside a backend option mapping."""
    for name, (minimum, maximum) in _LLM_INTEGER_BOUNDS.items():
        if name not in options:
            continue
        value = options[name]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                f"llm option '{name}' must be an integer between "
                f"{minimum} and {maximum}"
            )
        if not minimum <= value <= maximum:
            raise ValueError(
                f"llm option '{name}' must be between {minimum} and {maximum}; "
                f"got {value}"
            )

    for number_name, (number_minimum, number_maximum) in _LLM_NUMBER_BOUNDS.items():
        if number_name not in options:
            continue
        number_value = options[number_name]
        if isinstance(number_value, bool) or not isinstance(number_value, int | float):
            raise ValueError(
                f"llm option '{number_name}' must be a number between "
                f"{number_minimum} and {number_maximum}"
            )
        numeric = float(number_value)
        if (
            not math.isfinite(numeric)
            or not number_minimum <= numeric <= number_maximum
        ):
            raise ValueError(
                f"llm option '{number_name}' must be between "
                f"{number_minimum} and {number_maximum}; got {number_value}"
            )


def _validate_css_selector(selector: str) -> str:
    soupsieve = import_optional_dependency(
        "soupsieve", extra="html", feature="HTML selector validation"
    )

    try:
        soupsieve.compile(selector)
    except soupsieve.SelectorSyntaxError:
        raise ValueError("must be a valid CSS selector") from None
    return selector


_CssSelector = Annotated[_NonEmptyString, AfterValidator(_validate_css_selector)]


def _available_html_parsers() -> tuple[str, ...]:
    builder_registry = import_optional_dependency(
        "bs4.builder", extra="html", feature="HTML parser validation"
    ).builder_registry

    candidates = ("html.parser", "lxml")
    return tuple(
        parser for parser in candidates if builder_registry.lookup(parser) is not None
    )


def _ensure_unique_output_fields(names: list[str]) -> None:
    folded = [name.casefold() for name in names]
    if len(folded) != len(set(folded)):
        raise ValueError(
            "configured output fields must be unique (case-insensitive)"
        )


class BackendOptionsError(ValueError):
    """A backend option mapping does not satisfy its executable contract."""

    def __init__(
        self,
        message: str,
        *,
        path: tuple[str | int, ...] = (),
    ) -> None:
        super().__init__(message)
        self.path = path


class _BackendOptions(BaseModel):
    model_config = _STRICT_OPTIONS


class _PassthroughBackendOptions(BaseModel):
    model_config = ConfigDict(extra="allow")


class JsonBackendOptions(_BackendOptions):
    fields: list[_NonEmptyString] | None = None


class MarkdownBackendOptions(_BackendOptions):
    frontmatter_fields: list[_NonEmptyString] | None = None
    include_body: bool = True
    compute_word_count: bool = False

    @model_validator(mode="after")
    def _validate_output_fields(self) -> MarkdownBackendOptions:
        # None and [] both select dynamic frontmatter. Its keys are unknowable
        # at compile time; at runtime backend-owned body/word_count fields win.
        if not self.frontmatter_fields:
            return self
        names = list(self.frontmatter_fields)
        if self.include_body:
            names.append("body")
        if self.compute_word_count:
            names.append("word_count")
        _ensure_unique_output_fields(names)
        return self


class PdfBackendOptions(_BackendOptions):
    text_field: _NonEmptyString = "text"
    include_text: bool = True
    include_page_count: bool = True
    include_metadata: bool = False
    include_pages: bool = False
    page_separator: str = "\n\n"

    @model_validator(mode="after")
    def _validate_output_fields(self) -> PdfBackendOptions:
        names: list[str] = []
        if self.include_text:
            names.append(self.text_field)
        if self.include_page_count:
            names.append("page_count")
        if self.include_metadata:
            names.append("pdf_metadata")
        if self.include_pages:
            names.append("pages")
        _ensure_unique_output_fields(names)
        return self


class HtmlBackendOptions(_BackendOptions):
    text_field: _NonEmptyString = "text"
    include_text: bool = True
    include_structure: bool = False
    heading_selectors: list[_CssSelector] | None = None
    styled_headings: bool = False
    selectors: dict[_NonEmptyString, _CssSelector] | None = None
    include_meta: bool = False
    include_opengraph: bool = False
    include_links: bool = False
    parser: _NonEmptyString = "html.parser"

    @field_validator("parser")
    @classmethod
    def _validate_parser(cls, parser: str) -> str:
        if parser == "html.parser":
            return parser
        available = _available_html_parsers()
        if parser not in available:
            supported = ", ".join(available)
            raise ValueError(f"parser must be one of the available parsers: {supported}")
        return parser

    @model_validator(mode="after")
    def _validate_output_fields(self) -> HtmlBackendOptions:
        names: list[str] = []
        if self.include_structure or self.include_text:
            names.append(self.text_field)
        if self.include_structure:
            names.extend(("sections", "tables"))
        if self.selectors:
            names.extend(self.selectors)
        if self.include_meta:
            names.append("meta")
        if self.include_opengraph:
            names.append("og")
        if self.include_links:
            names.append("links")
        _ensure_unique_output_fields(names)
        return self


class EmailBackendOptions(_BackendOptions):
    include_body: bool = True
    body_field: _NonEmptyString = "body"
    include_html: bool = False
    include_headers: bool = False

    @model_validator(mode="after")
    def _validate_output_fields(self) -> EmailBackendOptions:
        names = ["from", "to", "cc", "subject", "date", "message_id"]
        if self.include_body:
            names.append(self.body_field)
        if self.include_html:
            names.append("html_body")
        if self.include_headers:
            names.append("headers")
        _ensure_unique_output_fields(names)
        return self


class LLMFieldSpec(_BackendOptions):
    name: _NonEmptyString
    type: _JsonSchemaType = Field(
        default="string",
        validation_alias=AliasChoices("type", "data_type", "data-type", "dtype"),
    )
    description: str | None = None
    # The backend forwards an array's item schema to providers unchanged. Keep
    # extension keywords available while validating the part dbt-ml consumes.
    items: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _validate_items(self) -> LLMFieldSpec:
        if self.items is not None and self.type != "array":
            raise ValueError("items is only valid when type is 'array'")
        if self.items is not None:
            item_type = self.items.get("type", "string")
            allowed = (
                "array",
                "boolean",
                "integer",
                "null",
                "number",
                "object",
                "string",
            )
            if not isinstance(item_type, str) or item_type not in allowed:
                raise ValueError("items.type must be a valid JSON Schema type")
        return self


class LLMBackendOptions(_BackendOptions):
    fields: list[LLMFieldSpec] = Field(min_length=1)
    provider: _ProviderName = "anthropic"
    model: _NonEmptyString | None = None
    system_prompt: str = (
        "You extract structured fields from documents. "
        "Return structured fields that match the requested output schema. "
        "If a field is genuinely missing from the document, use null."
    )
    cache_path: str | Path | None = None
    temperature: float = Field(default=0.0, ge=0.0, le=1.0, allow_inf_nan=False)
    max_tokens: int = Field(default=2048, ge=1, le=65_536)
    max_retries: int = Field(default=4, ge=0, le=20)
    max_concurrent: int = Field(default=4, ge=1, le=100)
    api_key_env: CredentialReference | None = Field(
        default=None,
        repr=False,
        exclude=True,
    )
    batch: bool = False
    batch_poll_seconds: float = Field(
        default=30.0, ge=0.1, le=3600.0, allow_inf_nan=False
    )
    # Deterministic native-batch partition size; the effective size is
    # min(batch_size, provider.max_batch_requests).
    batch_size: int = Field(default=1000, ge=1, le=100_000)
    batch_poll_max_seconds: float = Field(
        default=300.0, ge=0.1, le=3600.0, allow_inf_nan=False
    )
    batch_timeout_seconds: float = Field(
        default=86_400.0, gt=0.0, le=604_800.0, allow_inf_nan=False
    )
    # Explicit partial-success policy (issue #149): "fail" publishes nothing
    # from a partition containing a failed item; "publish_successful" keeps
    # per-document errors and publishes (and state-advances) only successes.
    on_partial_batch: Literal["fail", "publish_successful"] = "fail"
    budget: LLMBudgetConfig | None = None

    @field_validator("fields")
    @classmethod
    def _unique_fields(cls, fields: list[LLMFieldSpec]) -> list[LLMFieldSpec]:
        names = [field.name.casefold() for field in fields]
        if len(names) != len(set(names)):
            raise ValueError("LLM field names must be unique (case-insensitive)")
        return fields

    @model_validator(mode="after")
    def _validate_poll_bounds(self) -> LLMBackendOptions:
        if self.batch_poll_max_seconds < self.batch_poll_seconds:
            raise ValueError(
                "batch_poll_max_seconds must be >= batch_poll_seconds"
            )
        return self


@dataclass(frozen=True)
class BackendOptionContract:
    options_model: type[BaseModel]
    native_batch: bool = False
    requires_credentials: bool = False


_OPTION_CONTRACTS: dict[str, BackendOptionContract] = {}


def register_backend_option_contract(
    backend: str,
    options_model: type[BaseModel] | None = None,
    *,
    native_batch: bool = False,
    requires_credentials: bool = False,
) -> BackendOptionContract:
    """Register the option schema and execution capabilities for a backend.

    Custom backends using bare ``@register`` receive a compatibility-preserving
    pass-through contract. Supplying a Pydantic model to ``@register`` or this
    function enables strict compile/runtime validation.
    """
    contract = BackendOptionContract(
        options_model=options_model or _PassthroughBackendOptions,
        native_batch=native_batch,
        requires_credentials=requires_credentials,
    )
    _OPTION_CONTRACTS[backend] = contract
    return contract


def get_backend_option_contract(backend: str) -> BackendOptionContract:
    try:
        return _OPTION_CONTRACTS[backend]
    except KeyError as e:
        raise BackendOptionsError(
            f"Backend '{backend}' has no registered option contract"
        ) from e


def list_backend_option_contracts() -> list[str]:
    return sorted(_OPTION_CONTRACTS)


def validate_backend_options(
    backend: str, options: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate and canonicalize options without including values in errors.

    The compiler can call this on model-owned options before I/O. The runner
    calls it again after trusted profile defaults are merged. `exclude_unset`
    leaves backend defaults in one place: each backend's runtime implementation.
    """
    contract = get_backend_option_contract(backend)
    failure: BackendOptionsError | None = None
    try:
        parsed = contract.options_model.model_validate(dict(options))
    except ValidationError as e:
        validation_errors = e.errors(
            include_url=False,
            include_context=False,
            include_input=False,
        )
        details: list[str] = []
        for error in validation_errors:
            location = ".".join(str(part) for part in error["loc"])
            prefix = f"options.{location}: " if location else "options: "
            details.append(f"{prefix}{error['msg']}")
        joined = "; ".join(details)
        first_location = (
            tuple(validation_errors[0]["loc"]) if validation_errors else ()
        )
        failure = BackendOptionsError(
            f"Invalid options for extraction backend '{backend}': {joined}",
            path=("options", *first_location),
        )
    if failure is not None:
        options = {}
        raise failure
    validated = parsed.model_dump(
        mode="python",
        by_alias=True,
        exclude_none=True,
        exclude_unset=True,
    )
    if isinstance(parsed, LLMBackendOptions) and parsed.api_key_env is not None:
        validated["api_key_env"] = parsed.api_key_env
    return validated
