from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import bs4
from bs4 import BeautifulSoup
from bs4.element import (
    CData,
    Comment,
    Doctype,
    NavigableString,
    ProcessingInstruction,
    Tag,
)

from .base import BaseBackend, ExtractionResult
from .options import HtmlBackendOptions
from .registry import register

_HEADING_TAGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}
# block-level elements that should force a line break in the rendered text
_BLOCK_TAGS = {
    "p", "div", "section", "article", "li", "tr", "br", "hr",
    "blockquote", "pre", "ul", "ol", "dl", "dt", "dd",
}

_STYLED_CANDIDATE_TAGS = ["div", "p"]
# a styled-heading candidate must be a leaf block: containing any of these
# disqualifies it (br/hr excluded — trailing line breaks are common in
# styled headings and don't carry content)
_STYLED_DISQUALIFYING_TAGS = sorted(
    (_BLOCK_TAGS | set(_HEADING_TAGS) | {"table"}) - {"br", "hr"}
)
_MAX_STYLED_HEADING_CHARS = 150
_MAX_HEADING_LEVEL = 6

_BOLD_STYLE_RE = re.compile(r"font-weight\s*:\s*(?:bold(?:er)?|[6-9]\d\d)\b", re.I)
_FONT_SIZE_RE = re.compile(r"font-size\s*:\s*([\d.]+)\s*(pt|px)", re.I)


@register(options_model=HtmlBackendOptions)
class HtmlBackend(BaseBackend):
    """Read .html files via BeautifulSoup.

    Options:
        text_field:        Field name for the plain-text body (default "text").
        include_text:      Emit body text with tags stripped (default True).
        include_structure: Emit `sections` (heading hierarchy with char offsets
                           into the text field, plus anchor ids) and `tables`
                           (cell matrices with char offsets) as JSON — enough
                           for a downstream section parser to slice the
                           document without touching HTML again
                           (default False). Each section entry carries a
                           `source`: "tag" (semantic <h1>-<h6>), "selector",
                           or "style".
        heading_selectors: List of CSS selectors naming what counts as a
                           heading, for corpora that style headings instead
                           of using <h1>-<h6> (e.g. SEC inline-XBRL filings).
                           Selector order sets the level: matches of the
                           first selector become level 1, the second level 2,
                           and so on. Only used with include_structure.
        styled_headings:   Heuristic heading detection for styled documents
                           (default False): a leaf block element whose text
                           is short and entirely bold is treated as a heading
                           candidate; levels are ranked by font size, largest
                           first. Explicit heading tags and heading_selectors
                           matches take precedence. Only used with
                           include_structure.
        selectors:         dict of {field_name: css_selector}. First match's text
                           per selector is emitted; missing selectors yield None
                           with a warning.
        include_meta:      Emit a `meta` dict of <meta> name→content pairs.
        include_opengraph: Emit `og` dict of OpenGraph properties (og:*).
        include_links:     Emit `links` as a list of href strings.
        parser:            "html.parser" (default, stdlib) or "lxml" if installed.
    """

    def name(self) -> str:
        return "html"

    def version(self) -> str:
        return f"beautifulsoup4/{bs4.__version__}"

    def supported_formats(self) -> list[str]:
        return [".html", ".htm"]

    def extract(self, path: Path, options: dict[str, Any]) -> ExtractionResult:
        options = self.parse_options(options)
        parser = options.get("parser", "html.parser")
        soup = BeautifulSoup(path.read_text(), parser)

        warnings: list[str] = []
        fields: dict[str, Any] = {}

        include_structure = options.get("include_structure", False)
        if include_structure:
            text_field = options.get("text_field", "text")
            body = soup.body or soup
            heading_map = _build_heading_map(body, options, warnings)
            text, sections, tables = _render_structured(body, heading_map)
            fields[text_field] = text
            fields["sections"] = sections
            fields["tables"] = tables
        elif options.get("include_text", True):
            text_field = options.get("text_field", "text")
            body = soup.body or soup
            fields[text_field] = body.get_text(separator="\n", strip=True)

        selectors = options.get("selectors") or {}
        for field_name, selector in selectors.items():
            match = soup.select_one(selector)
            if match is None:
                warnings.append(
                    f"selector {selector!r} for field '{field_name}' matched nothing"
                )
                fields[field_name] = None
            else:
                fields[field_name] = match.get_text(strip=True)

        if options.get("include_meta", False):
            meta: dict[str, str] = {}
            for tag in soup.find_all("meta"):
                key = tag.get("name") or tag.get("property")
                content = tag.get("content")
                if key and content:
                    meta[str(key)] = str(content)
            fields["meta"] = meta

        if options.get("include_opengraph", False):
            og: dict[str, str] = {}
            for tag in soup.find_all("meta"):
                prop = tag.get("property")
                if prop and isinstance(prop, str) and prop.startswith("og:"):
                    og[prop[3:]] = str(tag.get("content") or "")
            fields["og"] = og

        if options.get("include_links", False):
            fields["links"] = [
                str(a.get("href"))
                for a in soup.find_all("a", href=True)
            ]

        return ExtractionResult(fields=fields, warnings=warnings)


def _build_heading_map(
    root: Tag, options: dict[str, Any], warnings: list[str]
) -> dict[int, tuple[int, str]]:
    """Map element identity -> (level, source) for non-semantic headings.

    Keyed by id() because Tag equality compares content, and distinct
    elements with identical markup must stay distinct heading entries.
    """
    heading_map: dict[int, tuple[int, str]] = {}
    selectors = options.get("heading_selectors") or []
    if isinstance(selectors, str) or not all(isinstance(s, str) for s in selectors):
        raise ValueError(
            "heading_selectors must be a list of CSS selector strings, "
            f"got {selectors!r}"
        )
    for index, selector in enumerate(selectors):
        matches = root.select(selector)
        if not matches:
            warnings.append(f"heading_selector {selector!r} matched nothing")
        level = min(index + 1, _MAX_HEADING_LEVEL)
        for match in matches:
            heading_map.setdefault(id(match), (level, "selector"))
    if options.get("styled_headings", False):
        for tag_id, level in _styled_heading_levels(root).items():
            heading_map.setdefault(tag_id, (level, "style"))
    return heading_map


def _styled_heading_levels(root: Tag) -> dict[int, int]:
    """Heuristic heading detection for documents without semantic heading
    tags (SEC inline-XBRL filings express headings as bold styled divs).

    A candidate is a leaf block whose visible text is short, contains a
    letter, and is entirely bold. Levels rank candidates by font size
    (largest = level 1); candidates without a parseable size sort below
    all sized ones.
    """
    candidates: list[tuple[Tag, float | None]] = []
    for tag in root.find_all(_STYLED_CANDIDATE_TAGS):
        if tag.find(_STYLED_DISQUALIFYING_TAGS) is not None:
            continue
        if tag.find_parent("table") is not None:
            continue
        text = " ".join(tag.get_text(" ", strip=True).split())
        if not 3 <= len(text) <= _MAX_STYLED_HEADING_CHARS:
            continue
        if not any(c.isalpha() for c in text):
            continue
        if not _all_text_bold(tag):
            continue
        candidates.append((tag, _max_font_size_pt(tag)))

    sizes = sorted({size for _, size in candidates if size is not None}, reverse=True)
    level_of_size = {
        size: min(rank + 1, _MAX_HEADING_LEVEL) for rank, size in enumerate(sizes)
    }
    unsized_level = min(len(sizes) + 1, _MAX_HEADING_LEVEL) if sizes else 1
    return {
        id(tag): level_of_size[size] if size is not None else unsized_level
        for tag, size in candidates
    }


def _all_text_bold(tag: Tag) -> bool:
    strings = [
        s
        for s in tag.find_all(string=True)
        if not isinstance(s, Comment | CData | Doctype | ProcessingInstruction)
        and str(s).strip()
    ]
    if not strings:
        return False

    def bold_within(node: NavigableString) -> bool:
        parent = node.parent
        while isinstance(parent, bs4.Tag):
            if parent.name in ("b", "strong"):
                return True
            style = parent.get("style")
            if isinstance(style, str) and _BOLD_STYLE_RE.search(style):
                return True
            if parent is tag:
                break
            parent = parent.parent
        return False

    return all(bold_within(s) for s in strings)


def _max_font_size_pt(tag: Tag) -> float | None:
    best: float | None = None
    for el in [tag, *tag.find_all(True)]:
        style = el.get("style")
        if not isinstance(style, str):
            continue
        m = _FONT_SIZE_RE.search(style)
        if not m:
            continue
        size = float(m.group(1))
        if m.group(2).lower() == "px":
            size *= 0.75
        if best is None or size > best:
            best = size
    return best


def _anchor_of(tag: Tag) -> str | None:
    anchor = tag.get("id")
    if not anchor:
        a = tag.find("a", attrs={"id": True}) or tag.find("a", attrs={"name": True})
        if isinstance(a, bs4.Tag):
            anchor = a.get("id") or a.get("name")
    return str(anchor) if anchor else None


def _render_structured(
    root: Tag,
    heading_map: dict[int, tuple[int, str]] | None = None,
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    """Single-pass render of normalized text with heading/table char offsets.

    Offsets index into the returned text, so `text[section.char_start:...]`
    slices a section without re-parsing HTML. Nested tables are flattened
    into their outer table's cell matrix.
    """
    parts: list[str] = []
    length = 0
    sections: list[dict[str, Any]] = []
    tables: list[dict[str, Any]] = []
    heading_map = heading_map or {}

    def emit_heading(child: Tag, level: int, source: str) -> None:
        newline()
        heading = " ".join(child.get_text(" ", strip=True).split())
        if not heading:
            return
        entry: dict[str, Any] = {
            "level": level,
            "heading": heading,
            "char_start": length,
            "source": source,
        }
        anchor = _anchor_of(child)
        if anchor:
            entry["anchor"] = anchor
        sections.append(entry)
        emit(heading)
        newline()

    def emit(text: str) -> None:
        nonlocal length
        if text:
            parts.append(text)
            length += len(text)

    def newline() -> None:
        if parts and not parts[-1].endswith("\n"):
            emit("\n")

    def walk(node: Tag) -> None:
        for child in node.children:
            if isinstance(child, Comment | CData | Doctype | ProcessingInstruction):
                continue
            if isinstance(child, NavigableString):
                text = " ".join(str(child).split())
                if text:
                    if parts and not parts[-1].endswith(("\n", " ")):
                        emit(" ")
                    emit(text)
                continue
            if not isinstance(child, bs4.Tag):
                continue
            name = child.name or ""
            if name in ("script", "style", "head", "template"):
                continue

            mapped = heading_map.get(id(child))
            if mapped is not None:
                emit_heading(child, level=mapped[0], source=mapped[1])
                continue

            if name in _HEADING_TAGS:
                emit_heading(child, level=_HEADING_TAGS[name], source="tag")
                continue

            if name == "table":
                newline()
                cells = [
                    [
                        " ".join(td.get_text(" ", strip=True).split())
                        for td in tr.find_all(["td", "th"])
                    ]
                    for tr in child.find_all("tr")
                ]
                cells = [row for row in cells if any(row)]
                tables.append(
                    {
                        "index": len(tables),
                        "char_start": length,
                        "n_rows": len(cells),
                        "n_cols": max((len(r) for r in cells), default=0),
                        "cells": cells,
                    }
                )
                emit("\n".join(" | ".join(row) for row in cells))
                newline()
                continue

            if name in _BLOCK_TAGS:
                newline()
                walk(child)
                newline()
            else:
                walk(child)

    walk(root)
    return "".join(parts).rstrip("\n"), sections, tables
