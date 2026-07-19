from __future__ import annotations

from typing import Any, Self
from urllib.parse import SplitResult, urlsplit, urlunsplit

from pydantic import GetCoreSchemaHandler, GetJsonSchemaHandler
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import CoreSchema, PydanticCustomError, core_schema


class EndpointUrlError(ValueError):
    pass


class OpenAICompatibleBaseUrl(str):
    """Normalized HTTP(S) base URL without credentials or request parameters."""

    def __new__(cls, value: str) -> Self:
        normalized = _normalize_base_url(value)
        return super().__new__(cls, normalized)

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: Any,
        handler: GetCoreSchemaHandler,
    ) -> CoreSchema:
        del source_type, handler
        return core_schema.no_info_plain_validator_function(cls._validate)

    @classmethod
    def __get_pydantic_json_schema__(
        cls,
        schema: CoreSchema,
        handler: GetJsonSchemaHandler,
    ) -> JsonSchemaValue:
        del schema, handler
        return {"type": "string", "format": "uri"}

    @classmethod
    def _validate(cls, value: object) -> Self:
        try:
            return cls(value)  # type: ignore[arg-type]
        except EndpointUrlError as error:
            raise PydanticCustomError("endpoint_url", str(error)) from None


def _normalize_base_url(value: str) -> str:
    if not isinstance(value, str) or not value or any(
        character.isspace() or ord(character) < 32 for character in value
    ):
        raise EndpointUrlError("base_url must be a valid HTTP(S) URL")
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError:
        raise EndpointUrlError("base_url must be a valid HTTP(S) URL") from None
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"} or parsed.hostname is None:
        raise EndpointUrlError("base_url must be a valid HTTP(S) URL")
    if parsed.username is not None or parsed.password is not None:
        raise EndpointUrlError("base_url must not contain URL user information")
    if parsed.query or parsed.fragment:
        raise EndpointUrlError("base_url must not contain a query or fragment")

    hostname = parsed.hostname.lower()
    if ":" in hostname:
        hostname = f"[{hostname}]"
    default_port = (scheme == "http" and port == 80) or (
        scheme == "https" and port == 443
    )
    netloc = hostname if port is None or default_port else f"{hostname}:{port}"
    normalized = SplitResult(
        scheme,
        netloc,
        parsed.path.rstrip("/"),
        "",
        "",
    )
    return urlunsplit(normalized)
