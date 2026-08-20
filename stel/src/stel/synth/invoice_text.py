from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from faker import Faker

_CURRENCIES = ["USD", "EUR", "GBP", "CAD"]


def generate_invoice_texts(count: int, output_dir: Path, seed: int = 42) -> list[Path]:
    """Generate `count` synthetic free-form invoice text files into `output_dir`.

    These are plaintext paragraphs (not JSON) — meant to exercise an LLM
    extraction backend that needs to recover structured fields from prose.
    """
    Faker.seed(seed)
    rng = random.Random(seed)
    fake = Faker()

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for i in range(count):
        vendor = fake.company()
        billed_to = fake.company()
        invoice_id = f"INV-{i:05d}"
        date = fake.date_between(start_date="-1y", end_date="today").isoformat()
        currency = rng.choice(_CURRENCIES)
        line_items: list[dict[str, Any]] = [
            {
                "description": fake.bs().title(),
                "qty": rng.randint(1, 5),
                "unit_price": round(rng.uniform(15.0, 350.0), 2),
            }
            for _ in range(rng.randint(1, 4))
        ]
        total = round(
            sum(li["qty"] * li["unit_price"] for li in line_items), 2
        )

        body = (
            f"INVOICE — {fake.catch_phrase()}\n\n"
            f"From: {vendor}\n"
            f"To: {billed_to}\n"
            f"Invoice number: {invoice_id}\n"
            f"Issue date: {date}\n"
            f"Currency: {currency}\n\n"
            "Line items:\n"
        )
        for li in line_items:
            line_total = li["qty"] * li["unit_price"]
            body += (
                f"  - {li['description']}: {li['qty']} x {currency} "
                f"{li['unit_price']:.2f} = {currency} {line_total:.2f}\n"
            )
        body += (
            f"\nTotal due: {currency} {total:.2f}\n"
            "Payment terms: Net 30 days.\n"
        )

        path = output_dir / f"invoice_{i:05d}.txt"
        path.write_text(body)
        paths.append(path)
    return paths
