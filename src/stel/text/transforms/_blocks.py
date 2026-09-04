"""Render landed block rows back into ordered markdown documents (issue #352).

SaaS context — Notion, Confluence, Linear — reaches the warehouse through an
EL tool as *block rows*: one row per block, a parent pointer, a position among
siblings, a type, and the block's text. That is structurally faithful and
semantically useless: nothing in the table has reading order, heading
structure, or nesting, which is exactly what chunking and heading attribution
need. This module is the missing half. It consumes a vendor-neutral block
contract and emits one markdown document per page, so the project's only
vendor-specific code is the SQL staging model that maps the connector's tables
onto that contract.

The contract is deliberately small: a ``pages`` relation (one row per page,
with a title) and a ``blocks`` relation (one row per block, keyed to a page,
with an optional parent block, a sibling position, a type from the vocabulary
below, and text). Anything an EL tool lands can be projected onto it in SQL.
"""
from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import polars as pl
from pydantic import BaseModel, ConfigDict, StrictBool, field_validator, model_validator

from ...transforms import IncrementalContract, ReferenceDep, TransformContext

# Block types the renderer understands. A type outside this vocabulary is
# rendered as a paragraph when it carries text and counted on the page row, so
# an unmapped connector type is visible in the output instead of silently
# dropped.
HEADING_TYPES: dict[str, int] = {"heading_1": 1, "heading_2": 2, "heading_3": 3}
LIST_TYPES = frozenset({"bulleted_list_item", "numbered_list_item", "to_do"})
QUOTE_TYPES = frozenset({"quote", "callout"})
# Structural blocks with no text of their own; only their children render.
CONTAINER_TYPES = frozenset({"column_list", "column", "synced_block"})
LINK_TYPES = frozenset({"child_page", "child_database"})
KNOWN_TYPES = (
    frozenset(HEADING_TYPES)
    | LIST_TYPES
    | QUOTE_TYPES
    | CONTAINER_TYPES
    | LINK_TYPES
    | {"paragraph", "toggle", "code", "divider"}
)

_OUTPUT_COUNTS = ("block_count", "orphan_block_count", "unknown_block_count")


def _require_non_empty(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("must not be empty")
    return normalized


class RenderBlocksOptions(BaseModel):
    """Strict options for ``stel.text.transforms.render_blocks``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    pages: str
    blocks: str
    page_id_field: str = "page_id"
    document_id_field: str = "document_id"
    title_field: str = "title"
    block_id_field: str = "block_id"
    parent_block_id_field: str = "parent_block_id"
    position_field: str = "position"
    type_field: str = "type"
    text_field: str = "text"
    # Optional per-block columns. Null declares the relation lacks them.
    checked_field: str | None = "checked"
    language_field: str | None = "language"
    # Render the page title as a level-1 heading and nest the page's own
    # headings one level under it, so the document's outline sits under its
    # title and a heading-attributed chunk before the first page heading
    # belongs to the page rather than to nothing.
    title_heading: StrictBool = True
    output_text_field: str = "text"
    # Page columns carried onto the document row (parent page, database,
    # properties, timestamps). Allow-listed, like the NLP transforms.
    include_fields: tuple[str, ...] = ()

    @field_validator(
        "pages",
        "blocks",
        "page_id_field",
        "document_id_field",
        "title_field",
        "block_id_field",
        "parent_block_id_field",
        "position_field",
        "type_field",
        "text_field",
        "output_text_field",
    )
    @classmethod
    def _non_empty(cls, value: str) -> str:
        return _require_non_empty(value)

    @field_validator("checked_field", "language_field")
    @classmethod
    def _non_empty_optional(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("must not be empty; use null to disable")
        return normalized

    @field_validator("include_fields")
    @classmethod
    def _unique_include_fields(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(value.strip() for value in values)
        if any(not value for value in normalized):
            raise ValueError("include_fields entries must not be empty")
        if len(normalized) != len(set(normalized)):
            raise ValueError("include_fields entries must be unique")
        return normalized

    @model_validator(mode="after")
    def _consistent(self) -> RenderBlocksOptions:
        if self.pages == self.blocks:
            raise ValueError(
                "pages and blocks must reference two different upstream models"
            )
        reserved = {
            self.document_id_field,
            self.page_id_field,
            self.title_field,
            self.output_text_field,
            *_OUTPUT_COUNTS,
        }
        collisions = sorted(reserved & set(self.include_fields))
        if collisions:
            raise ValueError(
                f"include_fields must not repeat a document-row column: {collisions}"
            )
        return self


def parse_render_blocks_options(options: Mapping[str, Any]) -> RenderBlocksOptions:
    return RenderBlocksOptions.model_validate(dict(options))


def declared_render_dependencies(options: Mapping[str, Any]) -> tuple[str, str]:
    parsed = parse_render_blocks_options(options)
    return (parsed.pages, parsed.blocks)


def declared_render_incremental_contract(
    options: Mapping[str, Any],
) -> IncrementalContract:
    """Pages are the parents; blocks are a reference keyed to their page, so
    an edited block re-renders exactly its page and nothing else (issue #364).
    One document row per page: the page id scopes the delete, the document id
    the upsert."""
    parsed = parse_render_blocks_options(options)
    return IncrementalContract(
        parent_key=parsed.page_id_field,
        child_key=parsed.document_id_field,
        parent_source=parsed.pages,
        parent_source_key=parsed.page_id_field,
        reference_deps=(ReferenceDep(parsed.blocks, join_key=parsed.page_id_field),),
    )


# --- Tree ---------------------------------------------------------------------


@dataclass(frozen=True)
class Block:
    block_id: str
    parent_id: str | None
    position: int
    type: str
    text: str | None
    checked: bool | None
    language: str | None


@dataclass(frozen=True)
class PageTree:
    """A page's blocks indexed by parent, siblings in render order. ``orphans``
    are the roots whose parent chain never reaches the page — a parent the
    connector did not land, or a cycle — rendered after the well-formed tree
    at the top level. Landed data is imperfect, and a dropped block is worse
    than a misplaced one."""

    children: Mapping[str | None, tuple[Block, ...]]
    orphans: tuple[Block, ...]
    unknown_count: int


def _sort_key(block: Block) -> tuple[int, str]:
    # Position orders siblings; the id breaks ties so two blocks a connector
    # landed at the same position render in one order on every run.
    return (block.position, block.block_id)


def build_tree(blocks: Iterable[Block]) -> PageTree:
    by_id = {block.block_id: block for block in blocks}
    by_parent: dict[str | None, list[Block]] = defaultdict(list)
    for block in by_id.values():
        by_parent[block.parent_id].append(block)
    for siblings in by_parent.values():
        siblings.sort(key=_sort_key)

    children: dict[str | None, list[Block]] = defaultdict(list)
    visited: set[str] = set()

    def attach(root: str | None) -> None:
        # Breadth-first from `root`, accepting each block once; a cycle's
        # back-edge finds its target already visited and is dropped.
        pending: deque[str | None] = deque([root])
        while pending:
            parent = pending.popleft()
            for child in by_parent.get(parent, ()):
                if child.block_id in visited:
                    continue
                visited.add(child.block_id)
                children[parent].append(child)
                pending.append(child.block_id)

    attach(None)
    orphans: list[Block] = []
    dangling = [
        block
        for block in by_id.values()
        if block.block_id not in visited and block.parent_id not in by_id
    ]
    remaining = [block for block in by_id.values() if block.block_id not in visited]
    # Dangling parents first, then whatever a cycle left, both in sibling order.
    for block in sorted(dangling, key=_sort_key) + sorted(remaining, key=_sort_key):
        if block.block_id in visited:
            continue
        visited.add(block.block_id)
        orphans.append(block)
        attach(block.block_id)
    unknown = sum(1 for block in by_id.values() if block.type not in KNOWN_TYPES)
    return PageTree(
        children={parent: tuple(items) for parent, items in children.items()},
        orphans=tuple(orphans),
        unknown_count=unknown,
    )


# --- Markdown -----------------------------------------------------------------


@dataclass(frozen=True)
class _Rendered:
    lines: tuple[str, ...]
    is_list_item: bool


def _text_lines(text: str | None) -> list[str]:
    if text is None:
        return []
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").split("\n")]
    while lines and not lines[-1]:
        lines.pop()
    while lines and not lines[0]:
        lines.pop(0)
    return lines


def _list_marker(block: Block, ordinal: int) -> str:
    if block.type == "numbered_list_item":
        return f"{ordinal}. "
    if block.type == "to_do":
        return "- [x] " if block.checked else "- [ ] "
    return "- "


def _render_leaf(block: Block, *, heading_shift: int, ordinal: int) -> _Rendered | None:
    """One block's own lines, before its children. ``None`` for a block that
    contributes nothing: a container, or an empty paragraph."""
    lines = _text_lines(block.text)
    if block.type in HEADING_TYPES:
        if not lines:
            return None
        level = min(6, HEADING_TYPES[block.type] + heading_shift)
        return _Rendered((f"{'#' * level} {' '.join(lines)}",), False)
    if block.type in LIST_TYPES:
        marker = _list_marker(block, ordinal)
        body = lines or [""]
        continuation = " " * len(marker)
        rendered = [marker + body[0], *(continuation + line for line in body[1:])]
        return _Rendered(tuple(line.rstrip() for line in rendered), True)
    if block.type in QUOTE_TYPES:
        if not lines:
            return None
        return _Rendered(tuple(f"> {line}".rstrip() for line in lines), False)
    if block.type == "code":
        return _Rendered((f"```{block.language or ''}", *lines, "```"), False)
    if block.type == "divider":
        return _Rendered(("---",), False)
    if block.type in LINK_TYPES:
        if not lines:
            return None
        return _Rendered((f"[[{' '.join(lines)}]]",), False)
    if block.type in CONTAINER_TYPES or not lines:
        return None
    # paragraph, toggle, and any type outside the vocabulary: plain text.
    return _Rendered(tuple(lines), False)


def _indented(rendered: _Rendered, width: int) -> _Rendered:
    if width == 0:
        return rendered
    pad = " " * width
    return _Rendered(
        tuple(pad + line if line else "" for line in rendered.lines),
        rendered.is_list_item,
    )


def _render_blocks(
    tree: PageTree,
    blocks: Iterable[Block],
    *,
    heading_shift: int,
    list_depth: int,
    out: list[_Rendered],
) -> None:
    ordinal = 0
    for block in blocks:
        # Numbering runs over consecutive numbered siblings and restarts after
        # anything else, as it does on the page.
        ordinal = ordinal + 1 if block.type == "numbered_list_item" else 0
        leaf = _render_leaf(block, heading_shift=heading_shift, ordinal=max(ordinal, 1))
        if leaf is not None:
            # Headings cannot be indented in markdown; everything else nests
            # under its list ancestors.
            width = 0 if block.type in HEADING_TYPES else 2 * list_depth
            out.append(_indented(leaf, width))
        _render_blocks(
            tree,
            tree.children.get(block.block_id, ()),
            heading_shift=heading_shift,
            list_depth=list_depth + 1 if block.type in LIST_TYPES else list_depth,
            out=out,
        )


def _join(rendered: list[_Rendered]) -> str:
    parts: list[str] = []
    previous: _Rendered | None = None
    for item in rendered:
        if previous is not None:
            both_items = previous.is_list_item and item.is_list_item
            parts.append("\n" if both_items else "\n\n")
        parts.append("\n".join(item.lines))
        previous = item
    return "".join(parts)


def render_page(
    title: str | None, blocks: Iterable[Block], *, title_heading: bool
) -> tuple[str, PageTree]:
    """Markdown for one page, and the tree it was rendered from."""
    tree = build_tree(blocks)
    rendered: list[_Rendered] = []
    shift = 0
    if title_heading and title is not None and title.strip():
        rendered.append(_Rendered((f"# {' '.join(title.split())}",), False))
        shift = 1
    _render_blocks(
        tree, tree.children.get(None, ()), heading_shift=shift, list_depth=0, out=rendered
    )
    _render_blocks(tree, tree.orphans, heading_shift=shift, list_depth=0, out=rendered)
    return _join(rendered), tree


# --- Driver -------------------------------------------------------------------


def _require_columns(frame: pl.DataFrame, model: str, columns: Iterable[str]) -> None:
    missing = sorted({column for column in columns if column not in frame.columns})
    if missing:
        raise ValueError(
            f"Model '{model}' is missing configured columns {missing}; got: "
            f"{sorted(frame.columns)}"
        )


def _string(row: Mapping[str, Any], column: str, *, model: str, position: int) -> str:
    value = row[column]
    if value is None or not str(value).strip():
        raise ValueError(
            f"Model '{model}' column '{column}' contains a null or empty value "
            f"(row {position})"
        )
    return str(value)


def _optional_string(value: Any) -> str | None:
    return None if value is None else str(value)


def _position(value: Any, *, model: str, column: str, position: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"Model '{model}' column '{column}' must hold integers (row {position})"
        )
    return value


def _checked(value: Any, *, model: str, column: str | None, position: int) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ValueError(
            f"Model '{model}' column '{column}' must hold booleans or nulls "
            f"(row {position})"
        )
    return value


def _block_columns(options: RenderBlocksOptions) -> tuple[str, ...]:
    return (
        options.block_id_field,
        options.page_id_field,
        options.parent_block_id_field,
        options.position_field,
        options.type_field,
        options.text_field,
        *((options.checked_field,) if options.checked_field else ()),
        *((options.language_field,) if options.language_field else ()),
    )


def _block(row: Mapping[str, Any], options: RenderBlocksOptions, position: int) -> Block:
    model = options.blocks
    return Block(
        block_id=_string(row, options.block_id_field, model=model, position=position),
        parent_id=_optional_string(row[options.parent_block_id_field]),
        position=_position(
            row[options.position_field],
            model=model,
            column=options.position_field,
            position=position,
        ),
        type=_string(row, options.type_field, model=model, position=position),
        text=_optional_string(row[options.text_field]),
        checked=_checked(
            row[options.checked_field] if options.checked_field else None,
            model=model,
            column=options.checked_field,
            position=position,
        ),
        language=(
            _optional_string(row[options.language_field]) if options.language_field else None
        ),
    )


def _blocks_by_page(
    frame: pl.DataFrame, options: RenderBlocksOptions
) -> dict[str, list[Block]]:
    _require_columns(frame, options.blocks, _block_columns(options))
    by_page: dict[str, list[Block]] = defaultdict(list)
    seen: set[str] = set()
    for position, row in enumerate(frame.iter_rows(named=True)):
        block = _block(row, options, position)
        if block.block_id in seen:
            raise ValueError(
                f"Model '{options.blocks}' column '{options.block_id_field}' contains "
                f"duplicate value '{block.block_id}'"
            )
        seen.add(block.block_id)
        page_id = _string(row, options.page_id_field, model=options.blocks, position=position)
        by_page[page_id].append(block)
    return by_page


def _output_schema(
    pages: pl.DataFrame, options: RenderBlocksOptions
) -> dict[str, pl.DataType]:
    schema: dict[str, pl.DataType] = {
        options.document_id_field: pl.String(),
        options.page_id_field: pl.String(),
        options.title_field: pl.String(),
        options.output_text_field: pl.String(),
        "block_count": pl.Int64(),
        "orphan_block_count": pl.Int64(),
        "unknown_block_count": pl.Int64(),
    }
    for name in options.include_fields:
        schema[name] = pages.schema[name]
    return schema


def _page_row(
    row: Mapping[str, Any],
    position: int,
    blocks: list[Block],
    options: RenderBlocksOptions,
) -> dict[str, Any]:
    title = _optional_string(row[options.title_field])
    text, tree = render_page(title, blocks, title_heading=options.title_heading)
    return {
        options.document_id_field: _string(
            row, options.document_id_field, model=options.pages, position=position
        ),
        options.page_id_field: row[options.page_id_field],
        options.title_field: title,
        options.output_text_field: text,
        "block_count": len(blocks),
        "orphan_block_count": len(tree.orphans),
        "unknown_block_count": tree.unknown_count,
        **{name: row[name] for name in options.include_fields},
    }


def run_render_blocks(
    deps: dict[str, pl.DataFrame],
    ctx: TransformContext,
) -> pl.DataFrame:
    options = parse_render_blocks_options(ctx.options)
    expected = {options.pages, options.blocks}
    if set(deps) != expected:
        raise ValueError(
            "render_blocks expects dependencies named by the `pages` and `blocks` "
            f"options ({sorted(expected)}); got: {sorted(deps)}"
        )
    pages, blocks = deps[options.pages], deps[options.blocks]
    _require_columns(
        pages,
        options.pages,
        (
            options.page_id_field,
            options.document_id_field,
            options.title_field,
            *options.include_fields,
        ),
    )
    schema = _output_schema(pages, options)
    if pages.is_empty():
        return pl.DataFrame(schema=schema)
    blocks_by_page = _blocks_by_page(blocks, options)

    rows: list[dict[str, Any]] = []
    seen_pages: set[str] = set()
    for position, row in enumerate(pages.iter_rows(named=True)):
        page_id = _string(row, options.page_id_field, model=options.pages, position=position)
        if page_id in seen_pages:
            raise ValueError(
                f"Model '{options.pages}' column '{options.page_id_field}' contains "
                f"duplicate value '{page_id}'"
            )
        seen_pages.add(page_id)
        rows.append(_page_row(row, position, blocks_by_page.get(page_id, []), options))
    return pl.DataFrame(rows, schema=schema, strict=False)
