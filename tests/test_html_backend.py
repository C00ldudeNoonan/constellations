from __future__ import annotations

from pathlib import Path

from stel.backends import get_backend
from stel.synth import generate_product_pages


def _write(path: Path, html: str) -> Path:
    path.write_text(html)
    return path


def test_html_registered() -> None:
    backend = get_backend("html")
    assert ".html" in backend.supported_formats()


def test_extracts_body_text(tmp_path: Path) -> None:
    doc = _write(
        tmp_path / "p.html",
        "<html><body><h1>Title</h1><p>Hello world</p></body></html>",
    )
    result = get_backend("html").extract(doc, {})
    assert "Title" in result.fields["text"]
    assert "Hello world" in result.fields["text"]


def test_selectors(tmp_path: Path) -> None:
    doc = _write(
        tmp_path / "p.html",
        "<html><body>"
        "<h1 class='name'>Widget X</h1>"
        "<span class='price'>$19.99</span>"
        "</body></html>",
    )
    result = get_backend("html").extract(
        doc,
        {
            "selectors": {"name": "h1.name", "price": "span.price"},
            "include_text": False,
        },
    )
    assert result.fields["name"] == "Widget X"
    assert result.fields["price"] == "$19.99"


def test_selector_miss_warns(tmp_path: Path) -> None:
    doc = _write(tmp_path / "p.html", "<html><body><h1>X</h1></body></html>")
    result = get_backend("html").extract(
        doc, {"selectors": {"missing": ".never"}, "include_text": False}
    )
    assert result.fields["missing"] is None
    assert any("missing" in w for w in result.warnings)


def test_meta_and_opengraph(tmp_path: Path) -> None:
    doc = _write(
        tmp_path / "p.html",
        "<html><head>"
        '<meta name="description" content="A widget">'
        '<meta property="og:title" content="Widget X">'
        '<meta property="og:price" content="19.99">'
        "</head><body>body</body></html>",
    )
    result = get_backend("html").extract(
        doc, {"include_meta": True, "include_opengraph": True}
    )
    assert result.fields["meta"]["description"] == "A widget"
    assert result.fields["og"]["title"] == "Widget X"
    assert result.fields["og"]["price"] == "19.99"


def test_synth_product_page_extractable(tmp_path: Path) -> None:
    paths = generate_product_pages(2, tmp_path, seed=1)
    backend = get_backend("html")
    for p in paths:
        result = backend.extract(
            p,
            {
                "selectors": {
                    "name": "h1.product-name",
                    "price": ".product-price",
                    "category": ".product-category",
                    "stock": ".product-stock",
                },
                "include_opengraph": True,
                "include_text": False,
            },
        )
        assert result.fields["name"]
        assert result.fields["price"].startswith("$")
        assert "Category:" in result.fields["category"]
        assert result.fields["stock"] in {"In stock", "Out of stock"}
        assert "title" in result.fields["og"]
