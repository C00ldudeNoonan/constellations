from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from stel.concept_cloud import (
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


def test_render_inlines_the_vendored_library_by_default() -> None:
    # A rendered artifact must be self-contained: the 3d-force-graph library is
    # inlined so it renders offline with no network.
    html = render_concept_cloud(placeholder_export())
    assert "3d-force-graph - https://github.com/vasturiano" in html
    assert "/*__CONCEPT_CLOUD_LIB__*/" not in html


def test_render_can_skip_library_inlining() -> None:
    html = render_concept_cloud(placeholder_export(), inline_lib=False)
    assert "3d-force-graph - https://github.com/vasturiano" not in html
    # The CDN fallback loader is still present for an online viewer.
    assert 'createElement("script")' in html


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

    from stel.cli import cli

    out = tmp_path / "cloud.html"
    result = CliRunner().invoke(
        cli, ["concept-cloud", "--placeholder", "--output", str(out)]
    )
    assert result.exit_code == 0, result.output
    assert out.exists()
    assert "invoice_pipeline" in _extract_data_island(out.read_text(encoding="utf-8"))["project"]


def test_cli_concept_cloud_requires_a_source() -> None:
    from click.testing import CliRunner

    from stel.cli import cli

    result = CliRunner().invoke(cli, ["concept-cloud"])
    assert result.exit_code != 0
    assert "--linking-model" in result.output


def test_demo_export_is_valid_and_sizable() -> None:
    from stel.concept_cloud import demo_export, parse_concept_cloud_export

    export = demo_export()
    assert len(export.concepts) >= 40
    assert len(export.concept_edges) >= 30
    assert any(e.directed for e in export.concept_edges)  # has asserted relations
    assert len({c.label for c in export.concepts}) >= 6  # varied types for color
    # Round-trips through the parse gate (referential validation, versioning).
    parse_concept_cloud_export(json.loads(export.to_json()))


def test_render_pins_canvas_to_viewport() -> None:
    # Regression: the WebGL canvas initialized 0x0 when the container had not
    # laid out; the artifact must size the graph to the viewport explicitly.
    html = render_concept_cloud(placeholder_export())
    assert ".width(window.innerWidth)" in html
    assert 'addEventListener("resize"' in html


def test_cli_concept_cloud_demo_writes_sizable_artifact(tmp_path: Path) -> None:
    from click.testing import CliRunner

    from stel.cli import cli

    out = tmp_path / "demo.html"
    result = CliRunner().invoke(cli, ["concept-cloud", "--demo", "--output", str(out)])
    assert result.exit_code == 0, result.output
    island = _extract_data_island(out.read_text(encoding="utf-8"))
    assert island["project"] == "economic_data"
    assert len(island["concepts"]) >= 40


def test_render_includes_orphan_and_filter_controls() -> None:
    html = render_concept_cloud(placeholder_export())
    # Orphan highlight + filtering controls are present and wired.
    assert 'id="orphans"' in html and "highlightOrphans" in html
    assert "isOrphan" in html          # orphan = no meaningful edges
    assert 'id="minfreq"' in html and "minFreq" in html
    # The plane checkbox became the lineage-mode toggle in v2 (#345), and it
    # defaults OFF: the star map is the primary view.
    assert 'id="lineage"' in html and "lineageMode" in html
    assert 'id="lineage"> Lineage' in html and "checked" not in html.split(
        'id="lineage"'
    )[0].rsplit("<label", 1)[-1]


def test_render_includes_the_dimension_picker_and_entry_point() -> None:
    """v2 (#345): the color-by picker, dimension-aware legend, and the entry
    point that opens on the hottest concept instead of the whole hairball."""
    from stel.concept_cloud import demo_export

    html = render_concept_cloud(demo_export())
    assert 'id="colorby"' in html and "buildColorBy" in html
    assert "HEAT_COLORS" in html          # retrieval reads as temperature
    assert "entryNode" in html            # opens on something specific
    assert '"retrieval"' in html          # demo dimension serialized
    assert "stel star map" in html


def test_baked_positions_pin_concept_nodes() -> None:
    """Concepts with positions are pinned (fx/fy/fz): position IS the meaning,
    and the force simulation would erase the projection in seconds. This is a
    recorded deviation from 'seed then relax'."""
    from stel.concept_cloud import demo_export

    html = render_concept_cloud(demo_export())
    assert "node.fx = c.position.x" in html
    assert "node.fy = c.position.y + Y_OFFSET" in html


def test_render_carries_the_star_motif() -> None:
    """The constellation look (#345 follow-up): glowing star sprites tinted by
    the active dimension, a drifting faded starfield, depth fog, and sky-chart
    labels on the brightest concepts. All of it degrades to the library's
    default rendering when THREE is unavailable."""
    from stel.concept_cloud import demo_export

    html = render_concept_cloud(demo_export())
    assert "makeGlowTexture" in html and "SpriteMaterial" in html
    assert "addStarfield" in html and "drift" in html    # slightly moving stars
    assert "FogExp2" in html                              # depth cue
    assert "makeLabelSprite" in html                      # sky-chart labels
    assert 'backgroundColor("rgba(0,0,0,0)")' in html     # nebula CSS shows through
    assert "window.THREE" in html and "return undefined" in html  # graceful fallback
    # A star is a crisp sphere core with the halo behind it: a lone sprite
    # pixelates as its texture scales and reads flat.
    assert "SphereGeometry" in html and "MeshBasicMaterial" in html
    assert "__coreMaterial" in html


def test_lineage_mode_shows_beams_without_requiring_a_selection() -> None:
    """The point of lineage mode is seeing the two maps connected. Beams used
    to draw only for a selected star, so toggling the mode showed nothing
    until a blind click; now every beam shows faintly and the selected star's
    brighten."""
    from stel.concept_cloud import demo_export

    html = render_concept_cloud(demo_export())
    assert "? touching.has(linkKey(l)) : true" in html
    # Two-state beam color: bright when traced, ember otherwise.
    assert "#ffcb47" in html and "#7a6329" in html


# ── display names and descriptions (#554) ──────────────────────────────────


def _described_export(description: str) -> ConceptCloudExport:
    return ConceptCloudExport(
        generated_at="2026-08-04T00:00:00Z",
        project="p",
        dag_plane=DagPlane(
            nodes=(DagNode(id="model.p.m", label="m", resource_type="model"),)
        ),
        concepts=(
            Concept(
                canonical_id="org:pnr",
                display="Pentair",
                description=description,
                frequency=1,
                provenance=Provenance(model="m"),
            ),
        ),
    )


def test_a_concept_description_reaches_the_bundle_and_the_viewer() -> None:
    """"I need to know what the entities mean" -- the first thing a user said
    on opening a real map of tickers and acronyms (issue #554)."""
    html = render_concept_cloud(
        _described_export("Water treatment equipment maker (NYSE: PNR).")
    )
    island = _extract_data_island(html)
    assert island["concepts"][0]["description"] == (
        "Water treatment equipment maker (NYSE: PNR)."
    )
    # Both surfaces the issue asks for: the click panel and the hover tooltip.
    assert 'class="desc"' in html
    assert "n.description ?" in html


def test_the_viewer_escapes_warehouse_text_before_it_becomes_markup() -> None:
    """Descriptions are free text an operator wrote, and the panel and the
    library's tooltip both render as HTML. The JSON island is already
    breakout-proof; this is the second half of that."""
    html = render_concept_cloud(_described_export("<img src=x onerror=alert(1)>"))
    # The payload survives into the bundle verbatim (escaped only as JSON)...
    assert _extract_data_island(html)["concepts"][0]["description"] == (
        "<img src=x onerror=alert(1)>"
    )
    # ...and every place it can reach markup goes through `esc`.
    assert "const esc = s =>" in html
    assert "${esc(node.description)}" in html
    assert "${esc(truncate(n.description, 90))}" in html
    assert "${esc(node.name)}" in html
    # No unescaped interpolation of node text is left in the detail panel.
    assert "${node.name}" not in html
    assert "${node.id}" not in html


# ─── v3: the period slider (issue #553) ─────────────────────────────────────


def _timed_export() -> ConceptCloudExport:
    """A bundle with a period axis, as `--time-field` produces."""
    return ConceptCloudExport(
        generated_at="2026-01-01T00:00:00Z",
        project="timed",
        dag_plane=DagPlane(nodes=()),
        concepts=(
            Concept(
                canonical_id="FERC", display="FERC", frequency=3,
                provenance=Provenance(model="link_entities", documents=3),
                by_period={"2019": 1, "2020": 1, "2021": 1},
            ),
            Concept(
                canonical_id="COVID", display="COVID-19", frequency=2,
                provenance=Provenance(model="link_entities", documents=2),
                by_period={"2020": 2},
            ),
        ),
        periods=("2019", "2020", "2021"),
    )


def test_the_period_slider_is_wired_to_the_bundle_axis() -> None:
    """v3 (#553): a period control that steps the axis the bundle carries."""
    html = render_concept_cloud(_timed_export())

    assert 'id="period"' in html and 'id="period-ctl"' in html
    # The axis comes from the bundle, not from the concepts.
    assert "DATA.periods" in html
    # Stop 0 is "all periods", so the totals stay one drag away.
    assert 'id="period-val">all<' in html
    assert "periodFreq" in html and "periodWeight" in html


def test_the_period_control_stays_hidden_without_an_axis() -> None:
    """A bundle with no time field must not grow a slider over nothing."""
    html = render_concept_cloud(placeholder_export())
    island = _extract_data_island(html)

    assert island["periods"] == []
    # Present in the template but not shown: the control reveals itself only
    # when `DATA.periods` is non-empty.
    assert 'id="period-ctl" style="display:none"' in html


def test_the_period_never_feeds_the_force_simulation() -> None:
    """Layout is computed once over every period, so a star that grows is
    recognisably the same star (#553, and #555's first item).

    `nodeVal` drives the simulation, so it must read the *total*. The period
    changes only what a star looks like — its drawn radius — and whether it
    is shown at all.
    """
    html = render_concept_cloud(_timed_export())

    # The simulation reads the unchanging total...
    assert ".nodeVal(n => n.kind === \"dag\" ? 6 : Math.max(2, n.val))" in html
    # ...while appearance and visibility read the period.
    assert "starRadius(periodFreq(n)) / n.__baseRadius" in html
    assert "if (period !== null && freq === 0) return false;" in html
