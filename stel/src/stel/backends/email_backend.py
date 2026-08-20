from __future__ import annotations

from email import message_from_bytes
from email.message import Message
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

from .base import BaseBackend, ExtractionResult
from .options import EmailBackendOptions
from .registry import register


@register(options_model=EmailBackendOptions)
class EmailBackend(BaseBackend):
    """Read .eml files via stdlib `email`. No external dependencies.

    Options:
        include_body:      Emit the plaintext body (default True).
        body_field:        Field name for the body (default "body").
        include_html:      Emit the HTML alternative if present (default False).
        include_headers:   Emit a `headers` dict of all headers (default False).
    """

    def name(self) -> str:
        return "email"

    def supported_formats(self) -> list[str]:
        return [".eml"]

    def extract(self, path: Path, options: dict[str, Any]) -> ExtractionResult:
        options = self.parse_options(options)
        msg: Message = message_from_bytes(path.read_bytes())
        warnings: list[str] = []
        fields: dict[str, Any] = {
            "from": msg.get("From"),
            "to": msg.get("To"),
            "cc": msg.get("Cc"),
            "subject": msg.get("Subject"),
            "date": _parse_date(msg.get("Date")),
            "message_id": msg.get("Message-ID"),
        }

        text_body, html_body = _walk_parts(msg)
        if options.get("include_body", True):
            body_field = options.get("body_field", "body")
            if text_body is None and html_body is not None:
                warnings.append(f"{path.name}: no text/plain part, fell back to text/html")
                fields[body_field] = html_body
            else:
                fields[body_field] = text_body or ""
                if not (text_body or html_body):
                    warnings.append(f"{path.name}: no text or html body found")

        if options.get("include_html", False):
            fields["html_body"] = html_body

        if options.get("include_headers", False):
            fields["headers"] = {k: v for k, v in msg.items()}

        return ExtractionResult(fields=fields, warnings=warnings)


def _parse_date(raw: str | None) -> str | None:
    if not raw:
        return None
    try:
        return parsedate_to_datetime(raw).isoformat()
    except (TypeError, ValueError):
        return raw


def _walk_parts(msg: Message) -> tuple[str | None, str | None]:
    """Return (text_body, html_body) — preferring multipart/alternative parts."""
    text_body: str | None = None
    html_body: str | None = None
    for part in msg.walk():
        ctype = part.get_content_type()
        if part.is_multipart():
            continue
        try:
            payload = part.get_payload(decode=True)
        except Exception:
            continue
        if payload is None:
            continue
        if isinstance(payload, str):
            decoded = payload
        elif isinstance(payload, bytes):
            try:
                decoded = payload.decode(part.get_content_charset() or "utf-8", errors="replace")
            except (LookupError, ValueError):
                decoded = payload.decode("utf-8", errors="replace")
        else:
            continue
        if ctype == "text/plain" and text_body is None:
            text_body = decoded
        elif ctype == "text/html" and html_body is None:
            html_body = decoded
    return text_body, html_body
