from __future__ import annotations

from email.message import EmailMessage
from email.utils import format_datetime
from pathlib import Path

from stel.backends import get_backend
from stel.synth import generate_support_emails


def _build(
    tmp_path: Path,
    *,
    from_addr: str = "Alice <alice@example.com>",
    to: str = "support@example.com",
    subject: str = "hi",
    body: str = "hello there",
    html: str | None = None,
    date: str | None = None,
) -> Path:
    msg = EmailMessage()
    msg["From"] = from_addr
    msg["To"] = to
    msg["Subject"] = subject
    if date:
        msg["Date"] = date
    if html is None:
        msg.set_content(body)
    else:
        msg.set_content(body)
        msg.add_alternative(html, subtype="html")
    p = tmp_path / "msg.eml"
    p.write_bytes(bytes(msg))
    return p


def test_email_registered() -> None:
    backend = get_backend("email")
    assert backend.name() == "email"
    assert ".eml" in backend.supported_formats()


def test_extract_basic(tmp_path: Path) -> None:
    p = _build(tmp_path, subject="Question about billing", body="please refund")
    result = get_backend("email").extract(p, {})
    assert result.fields["from"].startswith("Alice")
    assert result.fields["subject"] == "Question about billing"
    assert "please refund" in result.fields["body"]


def test_extract_date_iso(tmp_path: Path) -> None:
    from datetime import UTC, datetime

    when = datetime(2026, 5, 21, 12, 0, tzinfo=UTC)
    p = _build(tmp_path, date=format_datetime(when))
    result = get_backend("email").extract(p, {})
    assert result.fields["date"].startswith("2026-05-21T12:00:00")


def test_include_html(tmp_path: Path) -> None:
    p = _build(tmp_path, body="plain text", html="<p>hello</p>")
    result = get_backend("email").extract(p, {"include_html": True})
    assert "plain text" in result.fields["body"]
    assert "<p>hello</p>" in result.fields["html_body"]


def test_headers_dict(tmp_path: Path) -> None:
    p = _build(tmp_path, subject="hdr")
    result = get_backend("email").extract(p, {"include_headers": True})
    assert result.fields["headers"]["Subject"] == "hdr"


def test_synth_emails_parse_back(tmp_path: Path) -> None:
    paths = generate_support_emails(3, tmp_path, seed=1)
    backend = get_backend("email")
    for p in paths:
        result = backend.extract(p, {})
        assert result.fields["from"]
        assert "support@" in result.fields["to"]
        assert result.fields["subject"]
        assert result.fields["body"].strip()
