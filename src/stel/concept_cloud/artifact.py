"""Render a self-contained concept-cloud HTML artifact from an export bundle.

This is the "core artifact" half of #255: it takes one `ConceptCloudExport` and
produces a single static HTML page (3D concept cloud over a fixed dbt DAG plane).
It reads only the bundle — no warehouse, no manifest, no credentials — so it runs
identically on the built-in placeholder and on a real export produced later by
the export job.
"""
from __future__ import annotations

import json
from html import escape
from pathlib import Path

from .schema import ConceptCloudExport

_TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates" / "concept_cloud"
_TEMPLATE = _TEMPLATE_DIR / "concept_cloud.html"
_VENDOR_LIB = _TEMPLATE_DIR / "vendor" / "3d-force-graph.min.js"
_DATA_SENTINEL = "__CONCEPT_CLOUD_DATA__"
_LIB_SENTINEL = "/*__CONCEPT_CLOUD_LIB__*/"
_TITLE_SENTINEL = "__CONCEPT_CLOUD_TITLE__"
_SUBTITLE_SENTINEL = "__CONCEPT_CLOUD_SUBTITLE__"
_DEFAULT_TITLE = "stel star map"


def render_concept_cloud(
    export: ConceptCloudExport,
    *,
    inline_lib: bool = True,
    title: str | None = None,
    subtitle: str | None = None,
) -> str:
    """Return the artifact HTML with `export` embedded as an inline JSON island.

    When `inline_lib` is true (the default) the vendored 3d-force-graph library is
    inlined too, making the page fully self-contained and offline. If the vendored
    copy is absent the page still works online via its CDN fallback.

    `title` replaces the hardcoded "stel star map" heading and `<title>`; it
    names the tool, not the map. `subtitle` replaces the viewer's own
    "· <project> · N concepts" line, which is still the default when omitted."""
    template = _TEMPLATE.read_text(encoding="utf-8")
    if _DATA_SENTINEL not in template:
        raise RuntimeError("concept-cloud template is missing its data sentinel")
    html = template.replace(_DATA_SENTINEL, _embed_json(export), 1)
    html = html.replace(_TITLE_SENTINEL, escape(title or _DEFAULT_TITLE, quote=True))
    html = html.replace(_SUBTITLE_SENTINEL, _embed_subtitle(subtitle))
    if inline_lib and _VENDOR_LIB.exists():
        html = html.replace(_LIB_SENTINEL, _VENDOR_LIB.read_text(encoding="utf-8"), 1)
    return html


def write_concept_cloud(
    export: ConceptCloudExport,
    out_path: str | Path,
    *,
    title: str | None = None,
    subtitle: str | None = None,
) -> Path:
    """Write the artifact HTML to `out_path` and return the resolved path."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        render_concept_cloud(export, title=title, subtitle=subtitle), encoding="utf-8"
    )
    return path


def _embed_json(export: ConceptCloudExport) -> str:
    """Serialize the bundle for a `<script type="application/json">` island.

    Escaping `<` to its JSON unicode form makes it impossible for bundle
    content to break out of the script tag (`</script>`, `<!--`), while
    remaining valid JSON for `JSON.parse`."""
    return export.to_json().replace("<", "\\u003c")


def _embed_subtitle(subtitle: str | None) -> str:
    """Serialize an operator-provided subtitle override as a JS literal, or
    `null` to keep the viewer's own computed default. Escaping `<` prevents
    the same script-breakout the data island guards against."""
    if subtitle is None:
        return "null"
    return json.dumps(subtitle).replace("<", "\\u003c")
