"""Rendering landed block rows back into ordered markdown (issue #352).

The renderer is the half of "land, then render" that nobody had written: an
EL tool lands a page as flat block rows in fetch order, and these tests pin
that the rows come back as a document with reading order, real heading
levels, nesting, and explicit accounting for the imperfections landed data
actually has.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from stel.text.transforms._blocks import (
    Block,
    RenderBlocksOptions,
    build_tree,
    declared_render_dependencies,
    declared_render_incremental_contract,
    render_page,
    run_render_blocks,
)
from stel.transforms import ReferenceDep, TransformContext


def _block(
    block_id: str,
    parent: str | None,
    position: int,
    type_: str,
    text: str | None,
    *,
    checked: bool | None = None,
    language: str | None = None,
) -> Block:
    return Block(block_id, parent, position, type_, text, checked, language)


def _render(blocks: list[Block], *, title: str | None = "Page", title_heading: bool = True) -> str:
    text, _ = render_page(title, blocks, title_heading=title_heading)
    return text


# ─── ordering and structure ──────────────────────────────────────────────────


def test_siblings_render_by_position_not_by_arrival_order() -> None:
    blocks = [
        _block("c", None, 2, "paragraph", "third"),
        _block("a", None, 0, "paragraph", "first"),
        _block("b", None, 1, "paragraph", "second"),
    ]
    assert _render(blocks, title_heading=False) == "first\n\nsecond\n\nthird"


def test_equal_positions_break_ties_by_block_id_deterministically() -> None:
    blocks = [
        _block("z", None, 0, "paragraph", "zed"),
        _block("a", None, 0, "paragraph", "ay"),
    ]
    assert _render(blocks, title_heading=False) == "ay\n\nzed"
    assert _render(list(reversed(blocks)), title_heading=False) == "ay\n\nzed"


def test_title_becomes_h1_and_page_headings_nest_under_it() -> None:
    blocks = [
        _block("h1", None, 0, "heading_1", "Scope"),
        _block("h2", None, 1, "heading_2", "Limits"),
        _block("h3", None, 2, "heading_3", "Detail"),
    ]
    assert _render(blocks, title="Expense  policy") == (
        "# Expense policy\n\n## Scope\n\n### Limits\n\n#### Detail"
    )
    assert _render(blocks, title_heading=False) == "# Scope\n\n## Limits\n\n### Detail"


def test_nested_lists_indent_and_numbering_runs_over_consecutive_siblings() -> None:
    blocks = [
        _block("n1", None, 0, "numbered_list_item", "one"),
        _block("n2", None, 1, "numbered_list_item", "two"),
        _block("n2a", "n2", 0, "bulleted_list_item", "nested under two"),
        _block("n2a1", "n2a", 0, "bulleted_list_item", "deeper"),
        _block("n3", None, 2, "numbered_list_item", "three"),
        _block("p", None, 3, "paragraph", "break"),
        _block("n4", None, 4, "numbered_list_item", "restarts"),
    ]
    assert _render(blocks, title_heading=False) == (
        "1. one\n"
        "2. two\n"
        "  - nested under two\n"
        "    - deeper\n"
        "3. three\n\n"
        "break\n\n"
        "1. restarts"
    )


def test_non_list_children_of_a_list_item_are_indented_and_spaced() -> None:
    blocks = [
        _block("b", None, 0, "bulleted_list_item", "Pick a laptop"),
        _block("p", "b", 0, "paragraph", "A note under the bullet."),
        _block("c", None, 1, "bulleted_list_item", "Badge"),
    ]
    assert _render(blocks, title_heading=False) == (
        "- Pick a laptop\n\n  A note under the bullet.\n\n- Badge"
    )


def test_headings_are_never_indented_even_inside_a_list() -> None:
    blocks = [
        _block("b", None, 0, "bulleted_list_item", "item"),
        _block("h", "b", 0, "heading_2", "Inside"),
    ]
    assert _render(blocks, title_heading=False) == "- item\n\n## Inside"


def test_block_types_render_to_their_markdown_forms() -> None:
    blocks = [
        _block("t1", None, 0, "to_do", "done", checked=True),
        _block("t2", None, 1, "to_do", "open", checked=False),
        _block("q", None, 2, "quote", "line one\nline two"),
        _block("call", None, 3, "callout", "note"),
        _block("code", None, 4, "code", "print('hi')", language="python"),
        _block("code2", None, 5, "code", "raw"),
        _block("d", None, 6, "divider", None),
        _block("cp", None, 7, "child_page", "Subpage"),
        _block("cd", None, 8, "child_database", "Runbooks"),
        _block("tg", None, 9, "toggle", "Details"),
        _block("tgp", "tg", 0, "paragraph", "inside the toggle"),
    ]
    assert _render(blocks, title_heading=False) == (
        "- [x] done\n"
        "- [ ] open\n\n"
        "> line one\n> line two\n\n"
        "> note\n\n"
        "```python\nprint('hi')\n```\n\n"
        "```\nraw\n```\n\n"
        "---\n\n"
        "[[Subpage]]\n\n"
        "[[Runbooks]]\n\n"
        "Details\n\n"
        "inside the toggle"
    )


def test_containers_contribute_only_their_children() -> None:
    blocks = [
        _block("cl", None, 0, "column_list", None),
        _block("c1", "cl", 0, "column", None),
        _block("p1", "c1", 0, "paragraph", "left"),
        _block("c2", "cl", 1, "column", None),
        _block("p2", "c2", 0, "paragraph", "right"),
    ]
    assert _render(blocks, title_heading=False) == "left\n\nright"


def test_empty_and_whitespace_blocks_are_skipped_but_multiline_text_is_kept() -> None:
    blocks = [
        _block("e", None, 0, "paragraph", None),
        _block("w", None, 1, "paragraph", "   \n  "),
        _block("m", None, 2, "paragraph", "\nfirst line  \nsecond line\n\n"),
        _block("l", None, 3, "bulleted_list_item", "item line one\nitem line two"),
    ]
    assert _render(blocks, title_heading=False) == (
        "first line\nsecond line\n\n- item line one\n  item line two"
    )


def test_a_page_with_no_blocks_renders_its_title_only() -> None:
    assert _render([], title="Empty") == "# Empty"
    assert _render([], title=None) == ""


# ─── imperfect landings ──────────────────────────────────────────────────────


def test_orphans_render_at_the_end_and_are_counted_not_dropped() -> None:
    blocks = [
        _block("p", None, 0, "paragraph", "body"),
        _block("o", "missing", 0, "paragraph", "lost my parent"),
        _block("oc", "o", 0, "bulleted_list_item", "child of the orphan"),
    ]
    text, tree = render_page("T", blocks, title_heading=False)
    assert text == "body\n\nlost my parent\n\n- child of the orphan"
    assert [orphan.block_id for orphan in tree.orphans] == ["o"]


def test_a_parent_cycle_is_broken_rather_than_looping() -> None:
    blocks = [
        _block("a", "b", 0, "paragraph", "cycle a"),
        _block("b", "a", 0, "paragraph", "cycle b"),
        _block("p", None, 0, "paragraph", "body"),
    ]
    text, tree = render_page("T", blocks, title_heading=False)
    assert text == "body\n\ncycle a\n\ncycle b"
    assert [orphan.block_id for orphan in tree.orphans] == ["a"]


def test_unknown_types_render_as_paragraphs_and_are_counted() -> None:
    blocks = [
        _block("bm", None, 0, "bookmark", "https://example.com"),
        _block("img", None, 1, "image", None),
    ]
    text, tree = render_page("T", blocks, title_heading=False)
    assert text == "https://example.com"
    assert tree.unknown_count == 2


def test_build_tree_indexes_children_in_sibling_order() -> None:
    tree = build_tree(
        [
            _block("b", None, 1, "paragraph", "b"),
            _block("a", None, 0, "paragraph", "a"),
            _block("a2", "a", 1, "paragraph", "a2"),
            _block("a1", "a", 0, "paragraph", "a1"),
        ]
    )
    assert [b.block_id for b in tree.children[None]] == ["a", "b"]
    assert [b.block_id for b in tree.children["a"]] == ["a1", "a2"]
    assert tree.orphans == ()


# ─── the transform driver ────────────────────────────────────────────────────

_OPTIONS: dict[str, Any] = {"pages": "pages", "blocks": "blocks"}


def _ctx(tmp_path: Path, **options: Any) -> TransformContext:
    from stel.adapters import parse_warehouse_config

    return TransformContext(
        project_dir=tmp_path,
        profile_name="p",
        target_name="dev",
        warehouse=parse_warehouse_config(
            {"type": "duckdb", "path": str(tmp_path / "w.duckdb"), "schema": "main"}
        ),
        llm=None,
        options={**_OPTIONS, **options},
    )


def _pages(*rows: dict[str, Any]) -> pl.DataFrame:
    return pl.DataFrame(
        list(rows),
        schema={
            "document_id": pl.String,
            "page_id": pl.String,
            "title": pl.String,
            "parent_page_id": pl.String,
        },
    )


def _blocks(*rows: tuple[str, str, str | None, int, str, str | None]) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "block_id": block_id,
                "page_id": page_id,
                "parent_block_id": parent,
                "position": position,
                "type": type_,
                "text": text,
                "checked": None,
                "language": None,
            }
            for block_id, page_id, parent, position, type_, text in rows
        ],
        schema={
            "block_id": pl.String,
            "page_id": pl.String,
            "parent_block_id": pl.String,
            "position": pl.Int64,
            "type": pl.String,
            "text": pl.String,
            "checked": pl.Boolean,
            "language": pl.String,
        },
    )


def test_driver_emits_one_document_per_page_with_counts_and_included_fields(
    tmp_path: Path,
) -> None:
    pages = _pages(
        {"document_id": "d1", "page_id": "p1", "title": "One", "parent_page_id": None},
        {"document_id": "d2", "page_id": "p2", "title": "Two", "parent_page_id": "p1"},
    )
    blocks = _blocks(
        ("b2", "p1", None, 1, "paragraph", "second"),
        ("b1", "p1", None, 0, "heading_1", "Head"),
        ("b3", "p1", "gone", 0, "paragraph", "orphan"),
        ("b4", "p2", None, 0, "bookmark", "https://x"),
        ("b5", "p9", None, 0, "paragraph", "page not in pages"),
    )
    out = run_render_blocks(
        {"pages": pages, "blocks": blocks},
        _ctx(tmp_path, include_fields=["parent_page_id"]),
    )
    assert out.columns == [
        "document_id",
        "page_id",
        "title",
        "text",
        "block_count",
        "orphan_block_count",
        "unknown_block_count",
        "parent_page_id",
    ]
    rows = {row["page_id"]: row for row in out.to_dicts()}
    assert rows["p1"]["text"] == "# One\n\n## Head\n\nsecond\n\norphan"
    assert (rows["p1"]["block_count"], rows["p1"]["orphan_block_count"]) == (3, 1)
    assert rows["p2"]["text"] == "# Two\n\nhttps://x"
    assert rows["p2"]["unknown_block_count"] == 1
    assert rows["p2"]["parent_page_id"] == "p1"
    # A block for a page that is not a parent never appears anywhere.
    assert "page not in pages" not in "".join(row["text"] for row in rows.values())


def test_driver_keeps_the_schema_for_an_empty_page_frame(tmp_path: Path) -> None:
    out = run_render_blocks(
        {"pages": _pages(), "blocks": _blocks()},
        _ctx(tmp_path, include_fields=["parent_page_id"]),
    )
    assert out.is_empty()
    assert out.schema["block_count"] == pl.Int64
    assert out.schema["parent_page_id"] == pl.String


@pytest.mark.parametrize(
    ("pages", "blocks", "message"),
    [
        (
            _pages({"document_id": "d", "page_id": "p", "title": "T", "parent_page_id": None}),
            _blocks(("b", "p", None, 0, "paragraph", "x"), ("b", "p", None, 1, "paragraph", "y")),
            "duplicate value 'b'",
        ),
        (
            _pages(
                {"document_id": "d", "page_id": "p", "title": "T", "parent_page_id": None},
                {"document_id": "e", "page_id": "p", "title": "U", "parent_page_id": None},
            ),
            _blocks(),
            "duplicate value 'p'",
        ),
        (
            _pages({"document_id": "d", "page_id": "p", "title": "T", "parent_page_id": None}),
            _blocks(("b", "p", None, 0, "paragraph", "x")).drop("position"),
            "missing configured columns ['position']",
        ),
        (
            _pages({"document_id": None, "page_id": "p", "title": "T", "parent_page_id": None}),
            _blocks(),
            "column 'document_id' contains a null or empty value",
        ),
    ],
)
def test_driver_rejects_contract_violations_actionably(
    tmp_path: Path, pages: pl.DataFrame, blocks: pl.DataFrame, message: str
) -> None:
    with pytest.raises(ValueError, match=re.escape(message)):
        run_render_blocks({"pages": pages, "blocks": blocks}, _ctx(tmp_path))


def test_driver_rejects_a_non_integer_position(tmp_path: Path) -> None:
    pages = _pages({"document_id": "d", "page_id": "p", "title": "T", "parent_page_id": None})
    blocks = _blocks(("b", "p", None, 0, "paragraph", "x")).with_columns(
        pl.col("position").cast(pl.Float64)
    )
    with pytest.raises(ValueError, match="must hold integers"):
        run_render_blocks({"pages": pages, "blocks": blocks}, _ctx(tmp_path))


def test_driver_requires_exactly_the_named_dependencies(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="expects dependencies named by"):
        run_render_blocks({"pages": _pages(), "other": _blocks()}, _ctx(tmp_path))


# ─── options and contract ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"pages": "x", "blocks": "x"}, "two different upstream models"),
        ({**_OPTIONS, "include_fields": ["title"]}, "must not repeat a document-row column"),
        ({**_OPTIONS, "include_fields": ["block_count"]}, "must not repeat a document-row column"),
        ({**_OPTIONS, "include_fields": ["a", "a"]}, "must be unique"),
        ({**_OPTIONS, "checked_field": " "}, "use null to disable"),
        ({**_OPTIONS, "title_heading": "yes"}, "bool"),
        ({**_OPTIONS, "unknown": 1}, "extra"),
    ],
)
def test_options_are_strict(options: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        RenderBlocksOptions.model_validate(options)


def test_declared_dependencies_and_contract_follow_the_options() -> None:
    options = {"pages": "np", "blocks": "nb", "page_id_field": "pid"}
    assert declared_render_dependencies(options) == ("np", "nb")
    contract = declared_render_incremental_contract(options)
    assert contract.parent_source == "np"
    assert contract.parent_source_key == "pid"
    assert contract.parent_key == "pid"
    assert contract.child_key == "document_id"
    # Blocks are keyed to their page, so an edited block re-renders only its
    # page (issue #364) rather than every page.
    assert contract.reference_deps == (ReferenceDep("nb", join_key="pid"),)
    contract.validate_against(["np", "nb"])
