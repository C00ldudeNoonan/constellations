"""The first-party PDF writer that replaced fpdf2 (ALE-629)."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from pypdf import PdfReader

from stel.synth import generate_invoice_pdfs
from stel.synth.pdf_writer import Line, render_pdf, write_pdf
from stel.synth.words import SYNTHETIC_NOW


def _read(data: bytes) -> PdfReader:
    return PdfReader(io.BytesIO(data), strict=True)


def test_pages_and_lines_round_trip_through_pypdf() -> None:
    data = render_pdf(
        [[Line("First page"), Line("second line", style="B")], [Line("Page two", style="I")]],
        SYNTHETIC_NOW,
    )
    pages = _read(data).pages
    assert len(pages) == 2
    assert pages[0].extract_text().splitlines() == ["First page", "second line"]
    assert pages[1].extract_text() == "Page two"


def test_text_special_to_pdf_strings_is_escaped() -> None:
    """Parentheses and backslashes end or escape a PDF string if left raw."""
    text = r"Net (30) \ days) (open"
    page = _read(render_pdf([[Line(text)]], SYNTHETIC_NOW)).pages[0]
    assert page.extract_text() == text


def test_latin_1_text_survives_and_other_text_degrades_to_question_mark() -> None:
    page = _read(render_pdf([[Line("Café €5 中")]], SYNTHETIC_NOW)).pages[0]
    assert page.extract_text() == "Café €5 ?"


def test_output_is_deterministic_and_carries_the_creation_date() -> None:
    pages = [[Line("same")]]
    assert render_pdf(pages, SYNTHETIC_NOW) == render_pdf(pages, SYNTHETIC_NOW)
    info = _read(render_pdf(pages, SYNTHETIC_NOW)).metadata
    assert info is not None
    assert info.creation_date == SYNTHETIC_NOW


def test_blank_lines_take_vertical_space_without_drawing() -> None:
    page = _read(render_pdf([[Line("a"), Line("", leading=20), Line("b")]], SYNTHETIC_NOW)).pages[0]
    assert page.extract_text().replace("\n", "") == "ab"
    assert "b" in page.extract_text()


def test_overflowing_a_page_is_an_error_not_silent_clipping() -> None:
    with pytest.raises(ValueError, match="overflows the page"):
        render_pdf([[Line("x") for _ in range(200)]], SYNTHETIC_NOW)


def test_a_pdf_needs_a_page() -> None:
    with pytest.raises(ValueError, match="at least one page"):
        render_pdf([], SYNTHETIC_NOW)


def test_write_pdf_writes_the_rendered_bytes(tmp_path: Path) -> None:
    out = tmp_path / "x.pdf"
    write_pdf(out, [[Line("hi")]], SYNTHETIC_NOW)
    assert out.read_bytes() == render_pdf([[Line("hi")]], SYNTHETIC_NOW)


def test_invoice_pdfs_extract_the_fields_the_example_pipeline_expects(tmp_path: Path) -> None:
    """Every field the pdf_invoice_pipeline example reads is present as text."""
    [path] = generate_invoice_pdfs(1, tmp_path, seed=1)
    reader = PdfReader(str(path), strict=True)
    assert len(reader.pages) == 1
    text = reader.pages[0].extract_text()
    for label in ("INVOICE", "From: ", "To: ", "Invoice number: INV-00000", "Issue date: ",
                  "Currency: ", "Line items:", "Total due: ", "Payment terms: Net 30 days."):
        assert label in text, label
