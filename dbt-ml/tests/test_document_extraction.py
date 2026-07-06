"""Structure-preserving extraction + the common output contract (issue #85).

SEC-shaped HTML must yield sections/tables a downstream Item parser can
slice by char offset; FOMC-shaped PDFs must yield page offsets a speaker
parser can attribute matches to; every extraction row must carry lineage
and parser identity.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

import duckdb
import pytest

from dbt_ml.backends import get_backend

FIXTURES = Path(__file__).parent / "fixtures"


# ─── SEC HTML: structure ────────────────────────────────────────────────────


def _extract_sec(**options: Any) -> dict[str, Any]:
    backend = get_backend("html")
    result = backend.extract(
        FIXTURES / "sec_10k_snippet.html", {"include_structure": True, **options}
    )
    return result.fields


def test_sections_carry_levels_offsets_and_anchors() -> None:
    fields = _extract_sec()
    headings = [(s["level"], s["heading"]) for s in fields["sections"]]
    assert headings == [
        (1, "ACME CORP"),
        (2, "Item 1. Business"),
        (2, "Item 1A. Risk Factors"),
        (2, "Item 7. Management’s Discussion and Analysis"),
    ]
    by_heading = {s["heading"]: s for s in fields["sections"]}
    assert by_heading["Item 1. Business"]["anchor"] == "item1"
    assert by_heading["Item 1A. Risk Factors"]["anchor"] == "item1a"


def test_section_offsets_slice_the_text() -> None:
    fields = _extract_sec()
    text, sections = fields["text"], fields["sections"]
    item1a = next(s for s in sections if s["heading"] == "Item 1A. Risk Factors")
    item7 = next(s for s in sections if s["heading"].startswith("Item 7"))

    body = text[item1a["char_start"] : item7["char_start"]]
    assert body.startswith("Item 1A. Risk Factors")
    assert "single customer" in body
    assert "rocket-powered" not in body  # Item 1 content stays out
    # every section starts exactly at its recorded offset
    for s in sections:
        assert text[s["char_start"] :].startswith(s["heading"])


def test_tables_extracted_as_cells_with_offsets() -> None:
    fields = _extract_sec()
    tables = fields["tables"]
    assert len(tables) == 1
    t = tables[0]
    assert t["n_rows"] == 3
    assert t["n_cols"] == 3
    assert t["cells"][0] == ["Year", "Revenue", "Net Income"]
    assert t["cells"][2] == ["2026", "$1,400", "$100"]
    # the rendered table starts at its recorded offset in the text
    assert fields["text"][t["char_start"] :].startswith("Year | Revenue | Net Income")


def test_script_and_style_excluded() -> None:
    fields = _extract_sec()
    assert "should never appear" not in fields["text"]
    assert ".hidden" not in fields["text"]


def test_structured_extraction_is_deterministic() -> None:
    assert _extract_sec() == _extract_sec()


def test_malformed_html_does_not_crash(tmp_path: Path) -> None:
    p = tmp_path / "bad.html"
    p.write_text("<html><body><h2>Item 1. <b>Business</h2><p>text<table><tr><td>x")
    backend = get_backend("html")
    fields = backend.extract(p, {"include_structure": True}).fields
    assert fields["sections"][0]["heading"].startswith("Item 1.")
    assert fields["tables"][0]["cells"] == [["x"]]


def test_empty_html_file(tmp_path: Path) -> None:
    p = tmp_path / "empty.html"
    p.write_text("")
    fields = get_backend("html").extract(p, {"include_structure": True}).fields
    assert fields["text"] == ""
    assert fields["sections"] == []
    assert fields["tables"] == []


def test_large_html_is_deterministic_and_ordered(tmp_path: Path) -> None:
    parts = ["<html><body>"]
    for i in range(500):
        parts.append(f"<h2>Item {i}</h2><p>Paragraph body {i}</p>")
    parts.append("</body></html>")
    p = tmp_path / "large.html"
    p.write_text("".join(parts))

    backend = get_backend("html")
    first = backend.extract(p, {"include_structure": True}).fields
    second = backend.extract(p, {"include_structure": True}).fields
    assert first == second
    starts = [s["char_start"] for s in first["sections"]]
    assert starts == sorted(starts)
    assert len(starts) == 500


# ─── FOMC PDF: pages ────────────────────────────────────────────────────────


@pytest.fixture
def fomc_pdf(tmp_path: Path) -> Path:
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_font("Helvetica", size=11)
    pdf.add_page()
    pdf.multi_cell(
        0, 8, "Meeting of the Federal Open Market Committee\nJanuary 28-29, 2026"
    )
    pdf.add_page()
    pdf.multi_cell(
        0, 8,
        "CHAIR POWELL. Good morning, everybody.\n"
        "MR. WILLIAMS. Thank you, Mr. Chairman.",
    )
    pdf.add_page()
    pdf.multi_cell(0, 8, "CHAIR POWELL. We are adjourned.")
    out = tmp_path / "fomc.pdf"
    pdf.output(str(out))
    return out


def test_pdf_pages_offsets_slice_the_text(fomc_pdf: Path) -> None:
    fields = get_backend("pdf").extract(fomc_pdf, {"include_pages": True}).fields
    text, pages = fields["text"], fields["pages"]
    assert [p["page"] for p in pages] == [1, 2, 3]
    assert fields["page_count"] == 3
    for span in pages:
        assert 0 <= span["char_start"] <= span["char_end"] <= len(text)
    page2 = text[pages[1]["char_start"] : pages[1]["char_end"]]
    assert "MR. WILLIAMS" in page2
    assert "adjourned" not in page2


def test_pdf_speaker_turns_attributable_to_pages(fomc_pdf: Path) -> None:
    """The downstream FOMC pattern: regex over the full text, then page
    attribution via the offset spans."""
    fields = get_backend("pdf").extract(fomc_pdf, {"include_pages": True}).fields
    text, pages = fields["text"], fields["pages"]

    def page_of(pos: int) -> int:
        return next(
            p["page"] for p in pages if p["char_start"] <= pos <= p["char_end"]
        )

    turns = [(m.group(1), page_of(m.start())) for m in
             re.finditer(r"(CHAIR POWELL|MR\. WILLIAMS)\.", text)]
    assert ("CHAIR POWELL", 2) in turns
    assert ("MR. WILLIAMS", 2) in turns
    assert ("CHAIR POWELL", 3) in turns


def test_pdf_extraction_is_deterministic(fomc_pdf: Path) -> None:
    backend = get_backend("pdf")
    opts = {"include_pages": True, "include_metadata": False}
    assert backend.extract(fomc_pdf, opts).fields == backend.extract(fomc_pdf, opts).fields


def test_corrupt_pdf_raises_cleanly(tmp_path: Path) -> None:
    from pypdf.errors import PdfReadError

    p = tmp_path / "corrupt.pdf"
    p.write_bytes(b"%PDF-1.7 this is not really a pdf")
    with pytest.raises(PdfReadError):
        get_backend("pdf").extract(p, {})


def test_empty_file_as_pdf_raises_cleanly(tmp_path: Path) -> None:
    from pypdf.errors import PdfReadError

    p = tmp_path / "empty.pdf"
    p.write_bytes(b"")
    with pytest.raises(PdfReadError):
        get_backend("pdf").extract(p, {})


# ─── parser identity ────────────────────────────────────────────────────────


def test_backend_versions_name_their_parser() -> None:
    import bs4
    import pypdf

    assert get_backend("html").version() == f"beautifulsoup4/{bs4.__version__}"
    assert get_backend("pdf").version() == f"pypdf/{pypdf.__version__}"
    assert get_backend("json").version().startswith("dbt-ml/")


# ─── the common output contract, end to end ─────────────────────────────────


def test_extraction_rows_carry_contract_columns(
    tmp_path: Path, example_project_dir: Path
) -> None:
    from dbt_ml.runner import run_project
    from dbt_ml.synth import generate_invoices

    project = tmp_path / "proj"
    shutil.copytree(
        example_project_dir,
        project,
        ignore=shutil.ignore_patterns("data", "target", "__pycache__"),
    )
    generate_invoices(3, project / "data" / "invoices", seed=1)
    run_project(project, select="raw_invoices")

    con = duckdb.connect(str(project / "target" / "dbt_ml.duckdb"), read_only=True)
    try:
        rows = con.execute(
            "SELECT document_id, source_uri, backend_name, backend_version, "
            'extracted_at, content_hash FROM "dbt_ml".dbt_ml.raw_invoices'
        ).fetchall()
    finally:
        con.close()

    assert len(rows) == 3
    for doc_id, uri, backend_name, backend_version, extracted_at, content_hash in rows:
        assert doc_id and content_hash
        assert uri.startswith("file://")
        assert backend_name == "json"
        assert backend_version.startswith("dbt-ml/")
        assert extracted_at  # ISO timestamp, same for the whole run
    assert len({r[4] for r in rows}) == 1
