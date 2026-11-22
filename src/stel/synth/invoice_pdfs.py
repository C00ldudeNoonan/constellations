from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from .pdf_writer import Line, write_pdf
from .words import SYNTHETIC_NOW, SyntheticText

_CURRENCIES = ["USD", "EUR", "GBP", "CAD"]


def generate_invoice_pdfs(count: int, output_dir: Path, seed: int = 42) -> list[Path]:
    """Generate `count` synthetic invoice PDFs into `output_dir`.

    Deterministic for a given (count, seed) pair. Layout is plain enough that
    pypdf can extract the same text every time, which makes downstream LLM
    caching meaningful.
    """
    rng = random.Random(seed)
    fake = SyntheticText(rng)

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for i in range(count):
        invoice = _make_invoice(fake, rng, i)
        path = output_dir / f"invoice_{i:05d}.pdf"
        write_pdf(path, [_render(invoice)], SYNTHETIC_NOW)
        paths.append(path)
    return paths


def _make_invoice(fake: SyntheticText, rng: random.Random, index: int) -> dict[str, Any]:
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
        "issue_date": fake.recent_date(),
        "currency": rng.choice(_CURRENCIES),
        "tagline": fake.catch_phrase(),
        "line_items": line_items,
        "total": total,
    }


def _render(inv: dict[str, Any]) -> list[Line]:
    lines = [
        Line("INVOICE", size=18, leading=12, style="B"),
        Line(inv["tagline"], size=10, leading=7, style="I"),
        Line("", leading=4),
    ]
    for label, key in [
        ("From", "vendor"),
        ("To", "billed_to"),
        ("Invoice number", "invoice_id"),
        ("Issue date", "issue_date"),
        ("Currency", "currency"),
    ]:
        lines.append(Line(f"{label}: {inv[key]}", size=11, leading=7))
    lines += [Line("", leading=4), Line("Line items:", size=11, leading=7, style="B")]
    for li in inv["line_items"]:
        line_total = li["qty"] * li["unit_price"]
        lines.append(
            Line(
                f"  - {li['description']}: {li['qty']} x {inv['currency']} "
                f"{li['unit_price']:.2f} = {inv['currency']} {line_total:.2f}",
                size=10,
                leading=6,
            )
        )
    lines += [
        Line("", leading=4),
        Line(f"Total due: {inv['currency']} {inv['total']:.2f}", size=12, leading=8, style="B"),
        Line("Payment terms: Net 30 days.", size=10, leading=6),
    ]
    return lines
