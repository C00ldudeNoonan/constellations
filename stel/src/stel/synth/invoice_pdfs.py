from __future__ import annotations

import random
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from faker import Faker

from ..optional_dependencies import import_optional_dependency

if TYPE_CHECKING:
    from fpdf import FPDF

_CURRENCIES = ["USD", "EUR", "GBP", "CAD"]


def generate_invoice_pdfs(count: int, output_dir: Path, seed: int = 42) -> list[Path]:
    """Generate `count` synthetic invoice PDFs into `output_dir`.

    Deterministic for a given (count, seed) pair. Layout is plain enough that
    pypdf can extract the same text every time, which makes downstream LLM
    caching meaningful.
    """
    Faker.seed(seed)
    rng = random.Random(seed)
    fake = Faker()

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for i in range(count):
        invoice = _make_invoice(fake, rng, i)
        pdf = _render(invoice)
        path = output_dir / f"invoice_{i:05d}.pdf"
        pdf.output(str(path))
        paths.append(path)
    return paths


def _make_invoice(fake: Faker, rng: random.Random, index: int) -> dict[str, Any]:
    vendor = fake.company()
    billed_to = fake.company()
    line_items: list[dict[str, Any]] = [
        {
            "description": fake.bs().title(),
            "qty": rng.randint(1, 5),
            "unit_price": round(rng.uniform(15.0, 350.0), 2),
        }
        for _ in range(rng.randint(1, 4))
    ]
    total = round(sum(li["qty"] * li["unit_price"] for li in line_items), 2)
    return {
        "vendor": vendor,
        "billed_to": billed_to,
        "invoice_id": f"INV-{index:05d}",
        "issue_date": fake.date_between(start_date="-1y", end_date="today").isoformat(),
        "currency": rng.choice(_CURRENCIES),
        "tagline": fake.catch_phrase(),
        "line_items": line_items,
        "total": total,
    }


def _render(inv: dict[str, Any]) -> FPDF:
    pdf = _fpdf().FPDF()
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, "INVOICE", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "I", 10)
    pdf.cell(0, 7, inv["tagline"], new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    pdf.set_font("Helvetica", size=11)
    for label, key in [
        ("From", "vendor"),
        ("To", "billed_to"),
        ("Invoice number", "invoice_id"),
        ("Issue date", "issue_date"),
        ("Currency", "currency"),
    ]:
        pdf.cell(0, 7, f"{label}: {inv[key]}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Line items:", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=10)
    for li in inv["line_items"]:
        line_total = li["qty"] * li["unit_price"]
        pdf.cell(
            0,
            6,
            f"  - {li['description']}: {li['qty']} x {inv['currency']} "
            f"{li['unit_price']:.2f} = {inv['currency']} {line_total:.2f}",
            new_x="LMARGIN",
            new_y="NEXT",
        )
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(
        0,
        8,
        f"Total due: {inv['currency']} {inv['total']:.2f}",
        new_x="LMARGIN",
        new_y="NEXT",
    )

    pdf.set_font("Helvetica", size=10)
    pdf.cell(0, 6, "Payment terms: Net 30 days.", new_x="LMARGIN", new_y="NEXT")

    return cast("FPDF", pdf)


def _fpdf() -> Any:
    return import_optional_dependency(
        "fpdf", extra="pdf", feature="Synthetic PDF generation"
    )
