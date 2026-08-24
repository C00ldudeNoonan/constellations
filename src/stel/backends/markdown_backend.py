from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .base import BaseBackend, ExtractionResult
from .options import MarkdownBackendOptions
from .registry import register


@register(options_model=MarkdownBackendOptions)
class MarkdownBackend(BaseBackend):
    """Reads `.md` files. YAML frontmatter (between `---` fences) becomes fields;
    the rest is the body. Options:

        frontmatter_fields: [a, b, ...]   # optional projection
        include_body: bool (default true)
        compute_word_count: bool (default false)

    With no frontmatter projection, backend-owned `body` and `word_count`
    fields overwrite same-named dynamic frontmatter keys.
    """

    def name(self) -> str:
        return "markdown"

    def supported_formats(self) -> list[str]:
        return [".md", ".markdown"]

    def extract(self, path: Path, options: dict[str, Any]) -> ExtractionResult:
        options = self.parse_options(options)
        content = path.read_text(encoding="utf-8")
        warnings: list[str] = []
        frontmatter, body = self._split_frontmatter(content, warnings)

        fields: dict[str, Any] = {}
        wanted = options.get("frontmatter_fields")
        if wanted:
            for key in wanted:
                if key not in frontmatter:
                    warnings.append(f"Frontmatter key '{key}' missing in {path.name}")
                fields[key] = frontmatter.get(key)
        else:
            fields.update(frontmatter)

        # Dynamic frontmatter is unknowable during compile. Derived fields
        # intentionally win so their documented meaning remains deterministic.
        if options.get("include_body", True):
            fields["body"] = body
        if options.get("compute_word_count", False):
            fields["word_count"] = len(body.split())

        return ExtractionResult(fields=fields, warnings=warnings)

    @staticmethod
    def _split_frontmatter(
        content: str, warnings: list[str]
    ) -> tuple[dict[str, Any], str]:
        lines = content.split("\n")
        if not lines or lines[0].strip() != "---":
            return {}, content

        end_idx: int | None = None
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                end_idx = i
                break
        if end_idx is None:
            warnings.append("Frontmatter fence opened but not closed")
            return {}, content

        try:
            fm = yaml.safe_load("\n".join(lines[1:end_idx])) or {}
        except yaml.YAMLError as e:
            warnings.append(f"Invalid frontmatter YAML: {e}")
            return {}, "\n".join(lines[end_idx + 1 :]).lstrip("\n")

        if not isinstance(fm, dict):
            warnings.append("Frontmatter must parse as a YAML mapping")
            fm = {}

        body = "\n".join(lines[end_idx + 1 :]).lstrip("\n")
        return fm, body
