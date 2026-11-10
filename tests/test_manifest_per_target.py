"""A compile or run for one target must not replace another target's manifest
(issue #760).

`target/manifest.json` was written for whichever target compiled last, and
`stel mcp serve --target prod` refuses a manifest compiled for another target.
So one `stel run --target dev` from a shared project directory took the prod
MCP server down, and nothing in the run output said so. Each target now also
has its own manifest, and the server reads the one for the target it serves.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from stel.manifest import (
    MANIFEST_FILENAME,
    TARGET_MANIFESTS_DIRNAME,
    target_manifest_path,
    write_manifest,
)
from stel.mcp_server.catalog import ArtifactCatalog, ArtifactCatalogError


def _write_project(tmp_path: Path) -> None:
    (tmp_path / "stel_project.yml").write_text(
        "name: context_project\nprofile: context_project\n", encoding="utf-8"
    )
    warehouse = (
        "      warehouse:\n"
        "        type: duckdb\n"
        "        path: ./target/{name}.duckdb\n"
        "        schema: context\n"
    )
    (tmp_path / "profiles.yml").write_text(
        "context_project:\n"
        "  target: dev\n"
        "  outputs:\n"
        f"    dev:\n{warehouse.format(name='dev')}"
        f"    prod:\n{warehouse.format(name='prod')}",
        encoding="utf-8",
    )
    (tmp_path / "sources").mkdir()
    (tmp_path / "sources" / "documents.yml").write_text(
        "version: 2\nsources:\n  - name: documents\n    path: data/documents\n",
        encoding="utf-8",
    )
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "context.yml").write_text(
        "version: 2\n"
        "models:\n"
        "  - name: document_chunks\n"
        "    depends_on: [ref('documents')]\n"
        "    transform:\n"
        "      type: python\n"
        "      module: transforms.context\n"
        "    agent_context:\n"
        "      contract: agent_context/v1\n"
        "      grain: document_chunks\n",
        encoding="utf-8",
    )


def test_a_dev_compile_does_not_take_the_prod_catalog_down(tmp_path: Path) -> None:
    """The reproduction: compile prod, compile dev, serve prod."""
    _write_project(tmp_path)
    write_manifest(tmp_path, target="prod")
    shared = write_manifest(tmp_path, target="dev")

    # The shared file is still last-writer-wins: it is the artifact state,
    # docs and downstream readers use, and it is not what the server reads.
    assert json.loads(shared.read_text(encoding="utf-8"))["target"]["name"] == "dev"

    catalog = ArtifactCatalog.load(tmp_path, expected_target="prod")
    assert catalog is not None
    assert ArtifactCatalog.load(tmp_path, expected_target="dev") is not None


def test_each_targets_manifest_names_its_own_target(tmp_path: Path) -> None:
    _write_project(tmp_path)
    write_manifest(tmp_path, target="prod")
    write_manifest(tmp_path, target="dev")
    target_dir = tmp_path / "target"

    for name in ("prod", "dev"):
        path = target_manifest_path(target_dir, name)
        assert json.loads(path.read_text(encoding="utf-8"))["target"]["name"] == name


def test_a_target_that_never_compiled_is_named_in_the_refusal(tmp_path: Path) -> None:
    """Falling back to the shared file would serve another target's manifest."""
    _write_project(tmp_path)
    write_manifest(tmp_path, target="dev")

    with pytest.raises(ArtifactCatalogError, match=r"stel compile --target prod"):
        ArtifactCatalog.load(tmp_path, expected_target="prod")


def test_the_shared_manifest_still_serves_a_load_with_no_target(tmp_path: Path) -> None:
    _write_project(tmp_path)
    write_manifest(tmp_path, target="dev")

    assert ArtifactCatalog.load(tmp_path) is not None


def test_a_target_name_never_reaches_the_filename_as_a_path(tmp_path: Path) -> None:
    """A profile key is arbitrary text; only a slug and a digest become a name."""
    target_dir = tmp_path / "target"

    hostile = target_manifest_path(target_dir, "../../etc/passwd")
    assert hostile.parent == target_dir / TARGET_MANIFESTS_DIRNAME
    assert hostile.name.endswith(".json")
    assert ".." not in hostile.name

    # Names that slug alike stay distinct.
    assert target_manifest_path(target_dir, "Prod") != target_manifest_path(
        target_dir, "prod"
    )
    assert target_manifest_path(target_dir, "<inline>").parent == hostile.parent
    assert MANIFEST_FILENAME != hostile.name
