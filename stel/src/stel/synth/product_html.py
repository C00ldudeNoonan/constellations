from __future__ import annotations

import random
from pathlib import Path

from faker import Faker

_CATEGORIES = ["electronics", "books", "home", "sports", "tools"]


def generate_product_pages(count: int, output_dir: Path, seed: int = 42) -> list[Path]:
    """Generate `count` synthetic product-listing HTML pages."""
    Faker.seed(seed)
    rng = random.Random(seed)
    fake = Faker()

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for i in range(count):
        sku = f"SKU-{i:05d}"
        name = fake.catch_phrase()
        category = rng.choice(_CATEGORIES)
        price = round(rng.uniform(5.0, 999.99), 2)
        rating = round(rng.uniform(1.0, 5.0), 1)
        in_stock = rng.choice([True, True, True, False])
        description = " ".join(fake.paragraphs(nb=rng.randint(1, 3)))

        html = _render(
            sku=sku,
            name=name,
            category=category,
            price=price,
            rating=rating,
            in_stock=in_stock,
            description=description,
        )
        path = output_dir / f"product_{i:05d}.html"
        path.write_text(html)
        paths.append(path)
    return paths


def _render(
    *,
    sku: str,
    name: str,
    category: str,
    price: float,
    rating: float,
    in_stock: bool,
    description: str,
) -> str:
    stock_label = "In stock" if in_stock else "Out of stock"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{name} | demo shop</title>
  <meta name="description" content="{description[:140]}">
  <meta property="og:title" content="{name}">
  <meta property="og:type" content="product">
  <meta property="og:price" content="{price}">
</head>
<body>
  <header><a href="/">demo shop</a></header>
  <main>
    <h1 class="product-name">{name}</h1>
    <div class="product-sku" data-sku="{sku}">SKU: {sku}</div>
    <div class="product-category">Category: {category}</div>
    <div class="product-price">${price:.2f}</div>
    <div class="product-rating">Rating: {rating} / 5</div>
    <div class="product-stock">{stock_label}</div>
    <section class="product-description"><p>{description}</p></section>
  </main>
  <footer><a href="/about">about</a> | <a href="/contact">contact</a></footer>
</body>
</html>
"""
