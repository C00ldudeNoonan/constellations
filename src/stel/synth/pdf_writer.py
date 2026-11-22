"""A minimal text-only PDF writer for synthetic documents and test fixtures.

Left-aligned lines in the standard Helvetica family, which every PDF reader
supplies, so nothing is embedded and no third-party package is needed (ALE-629).
The output is byte-deterministic for a given input, which keeps downstream LLM
caches meaningful.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

_PAGE_WIDTH = 595.28  # A4, in points
_PAGE_HEIGHT = 841.89
_PT_PER_MM = 72 / 25.4
_MARGIN = 10 * _PT_PER_MM

_FONTS = {"": "Helvetica", "B": "Helvetica-Bold", "I": "Helvetica-Oblique"}


@dataclass(frozen=True)
class Line:
    """One line of text, `leading` millimetres tall. Empty text is a vertical gap."""

    text: str
    size: float = 11
    leading: float = 7
    style: str = ""  # "", "B" (bold) or "I" (italic)


def _escape(text: str) -> bytes:
    # WinAnsiEncoding is cp1252; anything outside it degrades to "?".
    raw = text.encode("cp1252", errors="replace")
    return raw.replace(b"\\", b"\\\\").replace(b"(", b"\\(").replace(b")", b"\\)")


def _page_stream(lines: Sequence[Line]) -> bytes:
    out = bytearray()
    y = _PAGE_HEIGHT - _MARGIN
    for line in lines:
        y -= line.leading * _PT_PER_MM
        if y < _MARGIN:
            raise ValueError("page content overflows the page; split it across pages")
        if not line.text:
            continue
        font = "F" + (line.style or "R")
        out += f"BT /{font} {line.size:g} Tf {_MARGIN:.2f} {y:.2f} Td (".encode("ascii")
        out += _escape(line.text) + b") Tj ET\n"
    return bytes(out)


def render_pdf(pages: Sequence[Sequence[Line]], created: datetime) -> bytes:
    """Render `pages` (each a sequence of lines) to PDF bytes."""
    if not pages:
        raise ValueError("a PDF needs at least one page")
    # Objects: 1 catalog, 2 page tree, 3-5 fonts, 6 info, then page/content pairs.
    page_ids = [7 + 2 * i for i in range(len(pages))]
    kids = " ".join(f"{n} 0 R" for n in page_ids)
    stamp = created.strftime("D:%Y%m%d%H%M%S") + "Z"
    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode("ascii"),
    ]
    for base_font in _FONTS.values():
        objects.append(
            f"<< /Type /Font /Subtype /Type1 /BaseFont /{base_font} "
            "/Encoding /WinAnsiEncoding >>".encode("ascii")
        )
    objects.append(f"<< /Producer (stel) /CreationDate ({stamp}) >>".encode("ascii"))
    for page_id, lines in zip(page_ids, pages, strict=True):
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {_PAGE_WIDTH} {_PAGE_HEIGHT}] "
            "/Resources << /Font << /FR 3 0 R /FB 4 0 R /FI 5 0 R >> >> "
            f"/Contents {page_id + 1} 0 R >>".encode("ascii")
        )
        stream = _page_stream(lines)
        objects.append(
            f"<< /Length {len(stream)} >>\nstream\n".encode("ascii") + stream + b"endstream"
        )

    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode("ascii") + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode("ascii")
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode("ascii")
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R /Info 6 0 R >>\n"
        f"startxref\n{xref}\n%%EOF\n"
    ).encode("ascii")
    return bytes(out)


def write_pdf(path: Path, pages: Sequence[Sequence[Line]], created: datetime) -> None:
    path.write_bytes(render_pdf(pages, created))
