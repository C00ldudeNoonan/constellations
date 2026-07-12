from __future__ import annotations

from pathlib import Path
from typing import Any

from pypdf import PdfReader

from .base import BaseBackend, ExtractionResult
from .options import PdfBackendOptions
from .registry import register


@register(options_model=PdfBackendOptions)
class PdfBackend(BaseBackend):
    """Read .pdf files via pypdf; extract text per page.

    Options:
        text_field:           Name of the text field in the row (default "text").
        include_text:         Whether to emit the full text (default True).
        include_page_count:   Emit page_count field (default True).
        include_metadata:     Emit pdf_metadata dict (title, author, etc.) (default False).
        page_separator:       Joiner between pages (default "\\n\\n").
    """

    def name(self) -> str:
        return "pdf"

    def supported_formats(self) -> list[str]:
        return [".pdf"]

    def version(self) -> str:
        import pypdf

        return f"pypdf/{pypdf.__version__}"

    def extract(self, path: Path, options: dict[str, Any]) -> ExtractionResult:
        options = self.parse_options(options)
        text_field = options.get("text_field", "text")
        include_text = options.get("include_text", True)
        include_page_count = options.get("include_page_count", True)
        include_metadata = options.get("include_metadata", False)
        include_pages = options.get("include_pages", False)
        page_separator = options.get("page_separator", "\n\n")

        warnings: list[str] = []
        reader = PdfReader(str(path))
        pages: list[str] = []
        for i, page in enumerate(reader.pages):
            try:
                text = page.extract_text() or ""
            except Exception as e:
                warnings.append(f"page {i}: extraction failed: {e}")
                text = ""
            pages.append(text)

        fields: dict[str, Any] = {}
        if include_text:
            full_text = page_separator.join(pages)
            fields[text_field] = full_text
            if not full_text.strip():
                warnings.append(
                    f"{path.name}: no text extracted — the PDF may be scanned "
                    "or image-only. Consider OCR (e.g. ocrmypdf) before dbt-ml run."
                )
        if include_pages:
            # Char offsets into the joined text, so downstream parsing
            # (e.g. transcript speaker turns) can attribute any match to a
            # page (#85).
            page_spans: list[dict[str, int]] = []
            offset = 0
            for i, text in enumerate(pages):
                page_spans.append(
                    {"page": i + 1, "char_start": offset, "char_end": offset + len(text)}
                )
                offset += len(text) + len(page_separator)
            fields["pages"] = page_spans
        if include_page_count:
            fields["page_count"] = len(pages)
        if include_metadata:
            md: dict[Any, Any] = dict(reader.metadata or {})
            fields["pdf_metadata"] = {str(k): str(v) for k, v in md.items()}

        return ExtractionResult(fields=fields, warnings=warnings)
