from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .base import BaseBackend, ExtractionResult
from .options import JsonBackendOptions
from .registry import register


@register(options_model=JsonBackendOptions)
class JsonBackend(BaseBackend):
    def name(self) -> str:
        return "json"

    def supported_formats(self) -> list[str]:
        return [".json"]

    def extract(self, path: Path, options: dict[str, Any]) -> ExtractionResult:
        options = self.parse_options(options)
        with path.open() as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError(
                f"json backend expects each document to be a JSON object; "
                f"got {type(data).__name__} from {path}"
            )

        fields_to_project = options.get("fields")
        warnings: list[str] = []
        if fields_to_project:
            extracted: dict[str, Any] = {}
            for key in fields_to_project:
                if key not in data:
                    warnings.append(f"Field '{key}' missing in {path.name}")
                extracted[key] = data.get(key)
        else:
            extracted = dict(data)

        return ExtractionResult(fields=extracted, warnings=warnings)
