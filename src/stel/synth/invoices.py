from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from faker import Faker

_CURRENCIES = ["USD", "EUR", "GBP", "CAD"]


def generate_invoices(count: int, output_dir: Path, seed: int = 42) -> list[Path]:
    """Generate `count` synthetic invoice JSON files into `output_dir`.

    Deterministic for a given (count, seed) pair.
    """
    Faker.seed(seed)
    rng = random.Random(seed)
    fake = Faker()

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for i in range(count):
        invoice = _make_invoice(fake, rng, i)
        path = output_dir / f"invoice_{i:05d}.json"
        path.write_text(json.dumps(invoice, indent=2, default=str))
        paths.append(path)
    return paths


def _make_invoice(fake: Faker, rng: random.Random, index: int) -> dict[str, Any]:
    line_count = rng.randint(1, 6)
    line_items: list[dict[str, Any]] = [
        {
            "description": fake.bs().title(),
            "qty": rng.randint(1, 10),
            "unit_price": round(rng.uniform(5.0, 500.0), 2),
        }
        for _ in range(line_count)
    ]
    total = round(sum(li["qty"] * li["unit_price"] for li in line_items), 2)
    return {
        "invoice_id": f"INV-{index:05d}",
        "vendor": fake.company(),
        "issue_date": fake.date_between(start_date="-1y", end_date="today").isoformat(),
        "currency": rng.choice(_CURRENCIES),
        "line_items": line_items,
        "total": total,
    }
