from __future__ import annotations

import json
import random
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from faker import Faker

# A realistic slice of arXiv CS / stats categories.
_CATEGORIES = [
    "cs.LG", "cs.CL", "cs.CV", "cs.AI", "cs.IR",
    "cs.DB", "stat.ML", "cs.NE", "cs.DC",
]


def generate_arxiv_papers(count: int, output_dir: Path, seed: int = 42) -> list[Path]:
    """Generate `count` synthetic arXiv-style paper records as JSON files.

    Each record mimics arXiv metadata (arxiv_id, title, authors, abstract,
    primary_category, published). The title is embedded verbatim in the abstract
    so the `grounded_in` quality check passes on this (well-formed) synthetic data
    — real extraction errors are what that check is designed to catch.

    Deterministic for a given (count, seed).
    """
    Faker.seed(seed)
    rng = random.Random(seed)
    fake = Faker()

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    now = datetime.now(UTC)

    for i in range(count):
        record = _make_paper(fake, rng, i, now)
        path = output_dir / f"paper_{i:05d}.json"
        path.write_text(json.dumps(record, indent=2, default=str))
        paths.append(path)
    return paths


def _make_paper(fake: Faker, rng: random.Random, index: int, now: datetime) -> dict[str, Any]:
    # arXiv id: YYMM.NNNNN
    month_offset = rng.randint(0, 23)
    published = now - timedelta(days=month_offset * 30 + rng.randint(0, 29))
    yymm = published.strftime("%y%m")
    arxiv_id = f"{yymm}.{index % 100000:05d}"

    title = fake.sentence(nb_words=rng.randint(5, 10)).rstrip(".")
    n_authors = rng.randint(1, 8)
    authors = [fake.name() for _ in range(n_authors)]
    primary_category = rng.choice(_CATEGORIES)

    # Embed the title verbatim so grounded_in(title -> abstract) passes on clean data.
    body = " ".join(fake.paragraphs(nb=rng.randint(2, 4)))
    abstract = f"In this paper we present {title}. {body}"

    return {
        "arxiv_id": arxiv_id,
        "title": title,
        "authors": authors,
        "n_authors": n_authors,
        "primary_category": primary_category,
        "categories": sorted({primary_category, *rng.sample(_CATEGORIES, rng.randint(0, 2))}),
        "published": published.date().isoformat(),
        "abstract": abstract,
    }
