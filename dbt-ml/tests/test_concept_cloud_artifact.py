from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from dbt_ml.concept_cloud import (
    Concept,
    ConceptCloudExport,
    DagNode,
    DagPlane,
    Provenance,
    placeholder_export,
    render_concept_cloud,
    write_concept_cloud,
)


def _extract_data_island(html: str) -> dict[str, Any]:
    match = re.search(
        r'<script id="cc-data" type="application/json">(.*?)</script>', html, re.S
    )
    assert match, "artifact must embed a cc-data JSON island"
    return json.loads(match.group(1))


def test_render_embeds_the_bundle_and_replaces_the_sentinel() -> None:
    export = placeholder_export()
    html = render_concept_cloud(export)
    assert "__CONCEPT_CLOUD_DATA__" not in html
    island = _extract_data_island(html)
    assert island["project"] == "invoice_pipeline"
    assert {c["display"] for c in island["concepts"]} == {"Acme Corp", "Globex", "New York"}
    # dag edge alias survives the round trip.
    assert island["dag_plane"]["edges"][0]["from"].startswith("source.")


def test_render_does_not_hard_block_on_the_cdn() -> None:
    # The library must be injected dynamically (non-blocking), not via a static
    # <script src> that would hang the page if the CDN is slow/unreachable.
    html = render_concept_cloud(placeholder_export())
    assert '<script src="https://unpkg.com/3d-force-graph"></script>' not in html
    assert 'createElement("script")' in html
    # And it must degrade to a legible message.
    assert "Couldn't load the 3d-force-graph library" in html


def test_render_escapes_angle_brackets_to_prevent_script_breakout() -> None:
    # A concept display carrying HTML must not be able to break out of the
    # <script> island; `<` is escaped to its JSON unicode form.
    export = ConceptCloudExport(
        generated_at="2026-08-04T00:00:00Z",
        project="p",
        dag_plane=DagPlane(
            nodes=(DagNode(id="model.p.m", label="m", resource_type="model"),)
        ),
        concepts=(
            Concept(
                canonical_id="org:x",
                display="</script><script>alert(1)</script>",
                frequency=1,
                provenance=Provenance(model="m"),
            ),
        ),
    )
    html = render_concept_cloud(export)
    assert "</script><script>alert(1)" not in html
    assert "\\u003c/script>" in html
    # The escaped payload is still valid JSON and decodes back to the original.
    island = _extract_data_island(html)
    assert island["concepts"][0]["display"] == "</script><script>alert(1)</script>"


def test_write_concept_cloud_creates_the_file(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "cloud.html"
    written = write_concept_cloud(placeholder_export(), out)
    assert written == out
    assert out.read_text(encoding="utf-8").startswith("<!DOCTYPE html>")


def test_cli_concept_cloud_placeholder_writes_artifact(tmp_path: Path) -> None:
    from click.testing import CliRunner

    from dbt_ml.cli import cli

    out = tmp_path / "cloud.html"
    result = CliRunner().invoke(
        cli, ["concept-cloud", "--placeholder", "--output", str(out)]
    )
    assert result.exit_code == 0, result.output
    assert out.exists()
    assert "invoice_pipeline" in _extract_data_island(out.read_text(encoding="utf-8"))["project"]


def test_cli_concept_cloud_requires_placeholder_until_export_job() -> None:
    from click.testing import CliRunner

    from dbt_ml.cli import cli

    result = CliRunner().invoke(cli, ["concept-cloud"])
    assert result.exit_code != 0
    assert "milestone 3" in result.output
