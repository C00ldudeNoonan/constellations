from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Mapping
from datetime import date, datetime, time
from enum import Enum
from pathlib import Path, PurePath
from typing import Any

from pydantic import BaseModel
from pydantic_core import PydanticSerializationError, to_jsonable_python

from .backends import get_backend, validate_backend_options
from .config.model import (
    ChunkConfig,
    ExtractionConfig,
    FieldConfig,
    MLConfig,
    ModelConfig,
    TransformConfig,
)
from .config.project import ProjectConfig
from .dag import parse_ref
from .paths import resolve_within_project
from .profile import ResolvedProfile, resolve_llm_options

_HASH_CHUNK_SIZE = 1024 * 1024
_NON_SEMANTIC_EXTRACTION_OPTIONS = frozenset(
    {
        "api_key_env",
        "batch",
        "batch_poll_seconds",
        "cache_path",
        "max_concurrent",
        "max_retries",
    }
)


def compute_content_hash(path: Path) -> str:
    return _hash_file(path)


def compute_document_id(scope: str, relative_path: str) -> str:
    return hashlib.blake2b(f"{scope}:{relative_path}".encode(), digest_size=8).hexdigest()


def compute_code_version(
    *,
    extraction: ExtractionConfig | None,
    transform: TransformConfig | None,
    ml: MLConfig | None = None,
    chunk: ChunkConfig | None = None,
    depends_on: list[str] | None = None,
    fields: list[FieldConfig] | None = None,
    effective_extraction: Mapping[str, Any] | None = None,
    project_dir: Path,
) -> str:
    payload: dict[str, Any] = {
        # flush_every shapes execution (memory/flush cadence), never output
        # content — including it would invalidate every model's incremental
        # state on upgrade. ModelConfig.warehouse_options stays out for the
        # same reason (issue #91): partitioning/clustering shape physical
        # layout, not row content, and applying a layout change needs
        # --full-refresh regardless.
        "extraction": (
            dict(effective_extraction)
            if effective_extraction is not None
            else extraction.model_dump(exclude={"flush_every"})
            if extraction
            else None
        ),
        "transform": transform.model_dump() if transform else None,
        # artifact.external is boundary policy, not code identity (see
        # flush_every above).
        "ml": ml.model_dump(mode="json", exclude={"artifact": {"external"}})
        if ml
        else None,
        "chunk": chunk.model_dump() if chunk else None,
        "depends_on": depends_on or None,
        "fields": [
            {"name": field.name, "data_type": field.data_type} for field in fields
        ]
        if fields
        else None,
    }
    if transform and transform.module:
        module_file = resolve_module_file(transform.module, project_dir)
        if module_file.exists():
            payload["transform_code_hash"] = _hash_file(module_file)
        else:
            payload["transform_code_hash"] = "missing"

    canonical = _canonical_json(payload)
    return hashlib.blake2b(canonical.encode(), digest_size=8).hexdigest()


def compute_model_code_version(
    model: ModelConfig,
    project: ProjectConfig,
    project_dir: Path,
    *,
    resolved: ResolvedProfile | None = None,
) -> str:
    effective_extraction: dict[str, Any] | None = None
    if model.extraction is not None:
        backend_name = model.extraction.backend or project.extraction.default_backend
        options = model.extraction.options
        if backend_name == "llm" and resolved is not None:
            options = resolve_llm_options(options, resolved)
        canonical_options = validate_backend_options(backend_name, options)
        semantic_options = (
            {
                key: value
                for key, value in canonical_options.items()
                if key not in _NON_SEMANTIC_EXTRACTION_OPTIONS
            }
            if backend_name == "llm"
            else canonical_options
        )
        backend = get_backend(backend_name)
        effective_extraction = {
            "backend": backend_name,
            "backend_version": backend.version(),
            "backend_implementation": backend.implementation_identity(),
            "options": semantic_options,
        }

    return compute_code_version(
        extraction=model.extraction,
        transform=model.transform,
        ml=model.ml,
        chunk=model.chunk,
        depends_on=(
            [parse_ref(dependency) for dependency in model.depends_on]
            if model.chunk is not None and model.depends_on
            else None
        ),
        fields=model.fields,
        effective_extraction=effective_extraction,
        project_dir=project_dir,
    )


def resolve_module_file(module: str, project_dir: Path) -> Path:
    """Resolve a dotted module path (e.g. 'transforms.summarize') to a .py file
    relative to the project directory."""
    parts = module.split(".")
    relative_path = Path(*parts).with_suffix(".py")
    return resolve_within_project(
        relative_path,
        project_dir,
        surface=f"Python module '{module}'",
    )


def _hash_file(path: Path) -> str:
    h = hashlib.blake2b(digest_size=8)
    with path.open("rb") as f:
        while chunk := f.read(_HASH_CHUNK_SIZE):
            h.update(chunk)
    return h.hexdigest()


def _canonical_json(value: Any) -> str:
    return _encoded_json(_json_safe(value))


def _encoded_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return ["enum", _type_identity(value), _json_safe(value.value)]
    if isinstance(value, BaseModel):
        return [
            "pydantic_model",
            _type_identity(value),
            _json_safe(value.model_dump(mode="python", by_alias=True)),
        ]
    if isinstance(value, PurePath):
        return ["path", _type_identity(value), value.as_posix()]
    if type(value) is datetime:
        return ["datetime", value.isoformat(), value.fold]
    if type(value) is date:
        return ["date", value.isoformat()]
    if type(value) is time:
        return ["time", value.isoformat(), value.fold]
    if isinstance(value, Mapping):
        entries = [
            [_json_safe(key), _json_safe(item)] for key, item in value.items()
        ]
        entries.sort(key=_encoded_json)
        return ["mapping", _type_identity(value), entries]
    if isinstance(value, list):
        return ["sequence", _type_identity(value), [_json_safe(item) for item in value]]
    if isinstance(value, tuple):
        return ["sequence", _type_identity(value), [_json_safe(item) for item in value]]
    if isinstance(value, set | frozenset):
        items = [_json_safe(item) for item in value]
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
        raise TypeError(
            f"Option value of type {type(value).__name__} is not fingerprintable"
        ) from e
    if converted is value:
        raise TypeError(
            f"Option value of type {type(value).__name__} is not fingerprintable"
        )
    return ["special", _type_identity(value), _json_safe(converted)]


def _type_identity(value: Any) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"
