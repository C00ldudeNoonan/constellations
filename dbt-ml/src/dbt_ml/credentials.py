from __future__ import annotations

import os
import re
from typing import Any, ClassVar, NoReturn, Self, SupportsIndex
from urllib.parse import urlsplit

from pydantic import GetCoreSchemaHandler, GetJsonSchemaHandler
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import CoreSchema, PydanticCustomError, SchemaSerializer, core_schema

_ENV_NAME_PATTERN = r"[A-Za-z_][A-Za-z0-9_]*"
_ENV_NAME_RE = re.compile(rf"\A{_ENV_NAME_PATTERN}\Z")
_ENV_VAR_EXPRESSION_RE = re.compile(
    r"\A\{\{[ \t]*env_var\([ \t]*(?P<quote>['\"])"
    rf"(?P<name>{_ENV_NAME_PATTERN})(?P=quote)[ \t]*\)[ \t]*\}}\}}\Z"
)
_REDACTED = "<redacted>"


class CredentialReferenceError(ValueError):
    pass


class CredentialResolutionError(RuntimeError):
    pass


class CredentialUrlError(ValueError):
    pass


class _RedactedInput:
    __slots__ = ("value",)

    def __init__(self, value: Any) -> None:
        self.value = value

    def __repr__(self) -> str:
        return _REDACTED

    __str__ = __repr__

    def take(self) -> Any:
        value = self.value
        self.value = None
        return value


def _redacted_serializer(_value: object) -> str:
    return _REDACTED


_ANY_SERIALIZER = SchemaSerializer(
    core_schema.any_schema(
        serialization=core_schema.plain_serializer_function_ser_schema(
            _redacted_serializer,
            return_schema=core_schema.str_schema(),
        )
    )
)


class _OpaqueCredential:
    __slots__ = ("_value",)

    __pydantic_serializer__: ClassVar[SchemaSerializer] = _ANY_SERIALIZER

    def __init__(self, value: str) -> None:
        self._value = value

    def __repr__(self) -> str:
        return f"{type(self).__name__}({_REDACTED})"

    def __str__(self) -> str:
        return _REDACTED

    def __eq__(self, other: object) -> bool:
        return type(self) is type(other)

    def __hash__(self) -> int:
        return hash(type(self))

    def __reduce__(self) -> NoReturn:
        raise TypeError("credential objects cannot be serialized")

    def __reduce_ex__(self, protocol: SupportsIndex) -> NoReturn:
        del protocol
        raise TypeError("credential objects cannot be serialized")

    @classmethod
    def _pydantic_schema(cls) -> CoreSchema:
        return core_schema.chain_schema(
            [
                core_schema.no_info_plain_validator_function(_RedactedInput),
                core_schema.no_info_plain_validator_function(cls._from_redacted_input),
            ],
            serialization=core_schema.plain_serializer_function_ser_schema(
                _redacted_serializer,
                return_schema=core_schema.str_schema(),
            ),
        )

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: Any,
        handler: GetCoreSchemaHandler,
    ) -> CoreSchema:
        del source_type, handler
        return cls._pydantic_schema()

    @classmethod
    def __get_pydantic_json_schema__(
        cls,
        schema: CoreSchema,
        handler: GetJsonSchemaHandler,
    ) -> JsonSchemaValue:
        del schema, handler
        return {
            "type": "string",
            "format": "password",
            "writeOnly": True,
        }

    @classmethod
    def _from_redacted_input(cls, wrapped: _RedactedInput) -> Self:
        raise NotImplementedError


class ProtectedCredential(_OpaqueCredential):
    def __init__(self, value: str) -> None:
        if not isinstance(value, str) or not value:
            del value
            raise ValueError("credential value must be a non-empty string")
        super().__init__(value)

    def reveal(self) -> str:
        return self._value

    @classmethod
    def _from_redacted_input(cls, wrapped: _RedactedInput) -> Self:
        value = wrapped.take()
        if isinstance(value, cls):
            return value
        if not isinstance(value, str) or not value:
            raise PydanticCustomError(
                "protected_credential",
                "credential value must be a non-empty string",
            )
        return cls(value)


class CredentialReference(_OpaqueCredential):
    def __init__(self, env_name: str) -> None:
        if not isinstance(env_name, str) or not _ENV_NAME_RE.fullmatch(env_name):
            del env_name
            raise CredentialReferenceError(
                "credential reference must be a valid environment-variable name"
            )
        super().__init__(env_name)

    @classmethod
    def from_env_name(cls, env_name: str) -> Self:
        if not isinstance(env_name, str) or not _ENV_NAME_RE.fullmatch(env_name):
            del env_name
            raise CredentialReferenceError(
                "credential reference must be a valid environment-variable name"
            )
        return cls(env_name)

    @classmethod
    def from_env_var_expression(cls, expression: str) -> Self:
        if not isinstance(expression, str):
            del expression
            raise CredentialReferenceError(
                "credential reference must be exactly {{ env_var('NAME') }} "
                "with no default or surrounding text"
            )
        match = _ENV_VAR_EXPRESSION_RE.fullmatch(expression)
        if match is None:
            del expression, match
            raise CredentialReferenceError(
                "credential reference must be exactly {{ env_var('NAME') }} "
                "with no default or surrounding text"
            )
        return cls(match.group("name"))

    def resolve(self) -> ProtectedCredential:
        value = os.environ.get(self._value)
        if not value:
            raise CredentialResolutionError(
                "configured credential environment variable is not set or is empty"
            )
        return ProtectedCredential(value)

    def is_available(self) -> bool:
        return bool(os.environ.get(self._value))

    @classmethod
    def _from_redacted_input(cls, wrapped: _RedactedInput) -> Self:
        value = wrapped.take()
        if isinstance(value, cls):
            return value
        if not isinstance(value, str) or not _ENV_NAME_RE.fullmatch(value):
            raise PydanticCustomError(
                "credential_reference",
                "credential reference must be a valid environment-variable name",
            )
        return cls(value)

    @classmethod
    def __get_pydantic_json_schema__(
        cls,
        schema: CoreSchema,
        handler: GetJsonSchemaHandler,
    ) -> JsonSchemaValue:
        json_schema = super().__get_pydantic_json_schema__(schema, handler)
        json_schema["pattern"] = rf"^{_ENV_NAME_PATTERN}$"
        return json_schema


class CredentialFreeUrl(str):
    """A URL that cannot carry username/password user information."""

    def __new__(cls, value: str) -> Self:
        if not isinstance(value, str):
            del value
            raise CredentialUrlError(
                "credential endpoint must be a URL without user information"
            )
        parsed = None
        failure: str | None = None
        try:
            parsed = urlsplit(value)
        except ValueError:
            failure = "credential endpoint must be a URL without user information"
        if parsed is not None:
            try:
                if not parsed.scheme or parsed.hostname is None:
                    failure = (
                        "credential endpoint must be a URL without user information"
                    )
                elif parsed.username is not None or parsed.password is not None:
                    failure = (
                        "credential endpoint must not contain URL user information"
                    )
            except ValueError:
                failure = (
                    "credential endpoint must be a URL without user information"
                )
        if failure is not None:
            del parsed, value
            raise CredentialUrlError(failure)
        assert parsed is not None
        return super().__new__(cls, value)

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: Any,
        handler: GetCoreSchemaHandler,
    ) -> CoreSchema:
        del source_type, handler
        return core_schema.chain_schema(
            [
                core_schema.no_info_plain_validator_function(_RedactedInput),
                core_schema.no_info_plain_validator_function(
                    cls._from_redacted_input
                ),
            ],
            serialization=core_schema.to_string_ser_schema(),
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls,
        schema: CoreSchema,
        handler: GetJsonSchemaHandler,
    ) -> JsonSchemaValue:
        del schema, handler
        return {"type": "string", "format": "uri"}

    @classmethod
    def _from_redacted_input(cls, wrapped: _RedactedInput) -> Self:
        value = wrapped.take()
        try:
            return cls(value)
        except CredentialUrlError as error:
            raise PydanticCustomError("credential_url", str(error)) from None
