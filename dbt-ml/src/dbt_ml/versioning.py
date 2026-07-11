from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .config.model import (
    ChunkConfig,
    ExtractionConfig,
    FieldConfig,
    MLConfig,
    TransformConfig,
)
from .paths import resolve_within_project

_HASH_CHUNK_SIZE = 1024 * 1024


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
    project_dir: Path,
) -> str:
    payload: dict[str, Any] = {
        # flush_every shapes execution (memory/flush cadence), never output
        # content — including it would invalidate every model's incremental
        # state on upgrade.
        "extraction": extraction.model_dump(exclude={"flush_every"})
        if extraction
        else None,
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

    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.blake2b(canonical.encode(), digest_size=8).hexdigest()


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
