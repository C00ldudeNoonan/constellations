"""JSON projection for warehouse snapshots without local payload files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .base import BaseBackend, ExtractionResult
from .options import JsonBackendOptions
from .registry import register


@register(options_model=JsonBackendOptions)
class JsonMemoryBackend(BaseBackend):
    # The executor must not fall back to fetch(Path) for this backend.
    memory_only = True

    def name(self) -> str:
        return "json_memory"

    def supported_formats(self) -> list[str]:
        return [".json"]

    def extract(self, path: Path, options: dict[str, Any]) -> ExtractionResult:
        raise RuntimeError("json_memory requires an in-memory warehouse source")

    def extract_bytes(self, payload: bytes, options: dict[str, Any]) -> ExtractionResult:
        options = self.parse_options(options)
        data = json.loads(payload)
        if not isinstance(data, dict):
            raise ValueError("json_memory expects a JSON object")
        fields = options.get("fields")
        if not fields:
            return ExtractionResult(fields=dict(data))
        return ExtractionResult(
            fields={key: data.get(key) for key in fields},
            warnings=[f"Field '{key}' missing in row.json" for key in fields if key not in data],
        )
