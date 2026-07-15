from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Mapping
from datetime import UTC, date, datetime, time
from decimal import Decimal
from enum import Enum
from pathlib import PurePath
from typing import Any

from pydantic import BaseModel
from pydantic_core import PydanticSerializationError, to_jsonable_python

HASH_DIGEST_SIZE = 16

_FINGERPRINT_PREFIX = b"dbt-ml-canonical-fingerprint"


def canonical_json(value: Any) -> str:
    """Return a deterministic, type-preserving JSON representation."""
    return _encoded_json(_canonical_value(value))


def canonical_bytes(value: Any) -> bytes:
    return canonical_json(value).encode("utf-8")


def canonical_fingerprint(
    value: Any,
    *,
    domain: str,
    version: int = 1,
    digest_size: int = HASH_DIGEST_SIZE,
) -> str:
    """Hash a canonical value in an explicit semantic domain."""
    if not domain:
        raise ValueError("Fingerprint domain must not be empty")
    if version < 1:
        raise ValueError("Fingerprint version must be positive")
    if not 1 <= digest_size <= 64:
        raise ValueError("Fingerprint digest_size must be between 1 and 64")

    digest = hashlib.blake2b(digest_size=digest_size)
    digest.update(_FINGERPRINT_PREFIX)
    _update_frame(digest, domain.encode("utf-8"))
    _update_frame(digest, str(version).encode("ascii"))
    _update_frame(digest, canonical_bytes(value))
    return digest.hexdigest()


def _update_frame(digest: Any, value: bytes) -> None:
    # Length framing keeps adjacent inputs unambiguous without constraining
    # domains or canonical values to a particular character set.
    digest.update(len(value).to_bytes(8, "big"))
    digest.update(value)


def _encoded_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _canonical_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return ["enum", _type_identity(value), _canonical_value(value.value)]
    if isinstance(value, BaseModel):
        return [
            "pydantic_model",
            _type_identity(value),
            _canonical_value(value.model_dump(mode="python", by_alias=True)),
        ]
    if isinstance(value, PurePath):
        return ["path", _type_identity(value), value.as_posix()]
    if type(value) is datetime:
        canonical = value.astimezone(UTC) if value.utcoffset() is not None else value
        return ["datetime", canonical.isoformat(), canonical.fold]
    if type(value) is date:
        return ["date", value.isoformat()]
    if type(value) is time:
        return ["time", value.isoformat(), value.fold]
    if type(value) is Decimal:
        # Match the historical pydantic-core conversion explicitly so Decimal
        # support cannot drift with a dependency upgrade.
        return ["special", _type_identity(value), _canonical_value(str(value))]
    if isinstance(value, Mapping):
        entries = [[_canonical_value(key), _canonical_value(item)] for key, item in value.items()]
        entries.sort(key=_encoded_json)
        return ["mapping", _type_identity(value), entries]
    if isinstance(value, list):
        return [
            "sequence",
            _type_identity(value),
            [_canonical_value(item) for item in value],
        ]
    if isinstance(value, tuple):
        return [
            "sequence",
            _type_identity(value),
            [_canonical_value(item) for item in value],
        ]
    if isinstance(value, set | frozenset):
        items = [_canonical_value(item) for item in value]
        items.sort(key=_encoded_json)
        return ["set", _type_identity(value), items]
    if value is None:
        return ["none"]
    if type(value) is bool:
        return ["bool", value]
    if type(value) is int:
        return ["int", str(value)]
    if type(value) is float:
        return ["float", struct.pack(">d", value).hex()]
    if type(value) is complex:
        return [
            "complex",
            struct.pack(">d", value.real).hex(),
            struct.pack(">d", value.imag).hex(),
        ]
    if type(value) is str:
        return ["string", value]
    if type(value) is bytes:
        return ["bytes", value.hex()]
    if type(value) is bytearray:
        return ["bytearray", bytes(value).hex()]
    if type(value) is memoryview:
        return ["memoryview", value.tobytes().hex()]
    try:
        converted = to_jsonable_python(value)
    except PydanticSerializationError as e:
        raise TypeError(f"Value of type {type(value).__name__} is not fingerprintable") from e
    if converted is value:
        raise TypeError(f"Value of type {type(value).__name__} is not fingerprintable")
    return ["special", _type_identity(value), _canonical_value(converted)]


def _type_identity(value: Any) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"
