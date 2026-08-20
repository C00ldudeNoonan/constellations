from __future__ import annotations

import random
from pathlib import Path

from faker import Faker

_TAG_POOL = [
    "python",
    "data",
    "ml",
    "infra",
    "ops",
    "design",
    "career",
    "performance",
    "duckdb",
    "dbt",
]


def generate_posts(count: int, output_dir: Path, seed: int = 42) -> list[Path]:
    """Generate `count` synthetic markdown blog posts into `output_dir`.

    Deterministic for a given (count, seed) pair.
    """
    Faker.seed(seed)
    rng = random.Random(seed)
    fake = Faker()

    output_dir.mkdir(parents=True, exist_ok=True)
    authors = [fake.name() for _ in range(max(1, min(count, 8)))]

    paths: list[Path] = []
    for i in range(count):
        title = fake.sentence(nb_words=6).rstrip(".")
        author = rng.choice(authors)
        tags = sorted(rng.sample(_TAG_POOL, k=rng.randint(1, 3)))
        date = fake.date_between(start_date="-1y", end_date="today").isoformat()
        body = "\n\n".join(fake.paragraphs(nb=rng.randint(3, 6)))

        tag_list = ", ".join(tags)
        md = (
            "---\n"
            f'title: "{title}"\n'
            f'author: "{author}"\n'
            f"date: {date}\n"
            f"tags: [{tag_list}]\n"
            "---\n"
            f"\n{body}\n"
        )
        path = output_dir / f"post_{i:05d}.md"
        path.write_text(md)
        paths.append(path)
    return paths
