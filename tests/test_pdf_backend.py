from __future__ import annotations

from pathlib import Path

from stel.backends import get_backend
from stel.synth import generate_invoice_pdfs


def test_pdf_registered() -> None:
    backend = get_backend("pdf")
    assert backend.name() == "pdf"
    assert ".pdf" in backend.supported_formats()


def test_pdf_extracts_invoice_fields(tmp_path: Path) -> None:
    paths = generate_invoice_pdfs(1, tmp_path, seed=1)
    backend = get_backend("pdf")
    result = backend.extract(paths[0], {})
    text = result.fields["text"]
    assert "INVOICE" in text
    assert "Invoice number:" in text
    assert "INV-00000" in text
    assert "Total due:" in text
    assert result.fields["page_count"] >= 1


def test_pdf_text_only_option(tmp_path: Path) -> None:
    paths = generate_invoice_pdfs(1, tmp_path, seed=1)
    result = get_backend("pdf").extract(
        paths[0], {"include_text": False, "include_page_count": True}
    )
    assert "text" not in result.fields
    assert "page_count" in result.fields


def test_pdf_custom_text_field(tmp_path: Path) -> None:
    paths = generate_invoice_pdfs(1, tmp_path, seed=1)
    result = get_backend("pdf").extract(paths[0], {"text_field": "body"})
    assert "body" in result.fields
    assert "text" not in result.fields


def test_pdf_synth_is_deterministic(tmp_path: Path) -> None:
    a = generate_invoice_pdfs(2, tmp_path / "a", seed=7)
    b = generate_invoice_pdfs(2, tmp_path / "b", seed=7)
    for pa, pb in zip(a, b, strict=True):
        # PDFs may have non-deterministic byte ordering (timestamps), but
        # extracted text should be identical.
        ta = get_backend("pdf").extract(pa, {}).fields["text"]
        tb = get_backend("pdf").extract(pb, {}).fields["text"]
        assert ta == tb
