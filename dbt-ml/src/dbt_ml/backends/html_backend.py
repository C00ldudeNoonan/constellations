from __future__ import annotations

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
from .registry import register

_HEADING_TAGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}
# block-level elements that should force a line break in the rendered text
_BLOCK_TAGS = {
    "p", "div", "section", "article", "li", "tr", "br", "hr",
    "blockquote", "pre", "ul", "ol", "dl", "dt", "dd",
}


@register
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
                           (default False).
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
        parser = options.get("parser", "html.parser")
        soup = BeautifulSoup(path.read_text(), parser)

        warnings: list[str] = []
        fields: dict[str, Any] = {}

        include_structure = options.get("include_structure", False)
        if include_structure:
            text_field = options.get("text_field", "text")
            body = soup.body or soup
            text, sections, tables = _render_structured(body)
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


def _render_structured(
    root: Tag,
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

            if name in _HEADING_TAGS:
                newline()
                heading = " ".join(child.get_text(" ", strip=True).split())
                entry: dict[str, Any] = {
                    "level": _HEADING_TAGS[name],
                    "heading": heading,
                    "char_start": length,
                }
                anchor = child.get("id")
                if not anchor:
                    a = child.find("a", attrs={"id": True}) or child.find(
                        "a", attrs={"name": True}
                    )
                    if isinstance(a, bs4.Tag):
                        anchor = a.get("id") or a.get("name")
                if anchor:
                    entry["anchor"] = str(anchor)
                sections.append(entry)
                emit(heading)
                newline()
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
