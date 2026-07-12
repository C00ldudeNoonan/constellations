from __future__ import annotations

from pathlib import Path

from dbt_ml.backends import get_backend


def _write(path: Path, content: str) -> Path:
    path.write_text(content)
    return path


def test_markdown_registered() -> None:
    backend = get_backend("markdown")
    assert backend.name() == "markdown"
    assert set(backend.supported_formats()) >= {".md"}


def test_markdown_extracts_frontmatter_and_body(tmp_path: Path) -> None:
    doc = _write(
        tmp_path / "post.md",
        '---\ntitle: "Hello"\nauthor: alex\ntags: [a, b]\n---\n\nHello world\n',
    )
    result = get_backend("markdown").extract(doc, {})
    assert result.fields["title"] == "Hello"
    assert result.fields["author"] == "alex"
    assert result.fields["tags"] == ["a", "b"]
    assert "Hello world" in result.fields["body"]
    assert result.warnings == []


def test_markdown_projection(tmp_path: Path) -> None:
    doc = _write(
        tmp_path / "p.md",
        '---\ntitle: "T"\nauthor: a\ntags: [x]\nextra: ignored\n---\nbody\n',
    )
    result = get_backend("markdown").extract(
        doc, {"frontmatter_fields": ["title", "author"], "include_body": False}
    )
    assert set(result.fields) == {"title", "author"}


def test_markdown_word_count(tmp_path: Path) -> None:
    doc = _write(tmp_path / "p.md", "---\ntitle: T\n---\none two three four\n")
    result = get_backend("markdown").extract(
        doc,
        {"frontmatter_fields": [], "compute_word_count": True},
    )
    assert result.fields["word_count"] == 4


def test_dynamic_frontmatter_collision_uses_backend_owned_fields(tmp_path: Path) -> None:
    doc = _write(
        tmp_path / "p.md",
        "---\nbody: frontmatter body\nword_count: 999\n---\none two three\n",
    )

    result = get_backend("markdown").extract(doc, {"compute_word_count": True})

    assert result.fields["body"] == "one two three\n"
    assert result.fields["word_count"] == 3


def test_markdown_missing_frontmatter(tmp_path: Path) -> None:
    doc = _write(tmp_path / "p.md", "Just a body, no frontmatter\n")
    result = get_backend("markdown").extract(doc, {})
    assert result.fields["body"].strip() == "Just a body, no frontmatter"


def test_markdown_unclosed_fence_warns(tmp_path: Path) -> None:
    doc = _write(tmp_path / "p.md", "---\ntitle: T\nno closing fence here\n")
    result = get_backend("markdown").extract(doc, {})
    assert any("not closed" in w for w in result.warnings)


def test_markdown_warns_on_missing_projected_key(tmp_path: Path) -> None:
    doc = _write(tmp_path / "p.md", "---\ntitle: T\n---\n\nbody\n")
    result = get_backend("markdown").extract(
        doc, {"frontmatter_fields": ["title", "author"]}
    )
    assert any("author" in w for w in result.warnings)
    assert result.fields["author"] is None
