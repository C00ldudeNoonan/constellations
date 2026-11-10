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
from click.testing import CliRunner

from stel.cli import cli
from stel.manifest import (
    MANIFEST_FILENAME,
    TARGET_MANIFESTS_DIRNAME,
    target_manifest_path,
    write_manifest,
)
from stel.mcp_server import server
from stel.mcp_server.catalog import ArtifactCatalog, ArtifactCatalogError
from stel.mcp_server.service import ContextService
from stel.runner import clean_project


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
        "  target: prod\n"
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
        "  - name: raw_documents\n"
        "    source: ref('documents')\n"
        "    extraction:\n"
        "      backend: json\n"
        "      options: {fields: [title]}\n"
        "  - name: document_chunks\n"
        "    depends_on: [ref('raw_documents')]\n"
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


@pytest.mark.e2e
def test_a_dev_run_preserves_the_prod_healthcheck_and_default_mcp_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ALE-621 (#760): serve the profile's prod target after a dev run."""
    _write_project(tmp_path)
    documents = tmp_path / "data" / "documents"
    documents.mkdir(parents=True)
    (documents / "one.json").write_text('{"title": "One"}', encoding="utf-8")
    (tmp_path / "transforms").mkdir()
    (tmp_path / "transforms" / "context.py").write_text(
        "from stel.agent_context import empty_agent_context_frame\n"
        "def run(deps):\n"
        "    return empty_agent_context_frame('document_chunks')\n",
        encoding="utf-8",
    )
    runner = CliRunner()
    args = ["--project-dir", str(tmp_path)]
    compiled = runner.invoke(cli, [*args, "compile", "--target", "prod"])
    assert compiled.exit_code == 0, compiled.output
    target_dir = tmp_path / "target"
    prod_manifest = target_manifest_path(target_dir, "prod")
    prod_content = prod_manifest.read_bytes()

    # A dev experiment drops the serving declaration, so its shared manifest
    # would fail catalog validation if prod serving accidentally read it.
    config = tmp_path / "models" / "context.yml"
    model_yaml, _, _ = config.read_text(encoding="utf-8").partition("    agent_context:")
    config.write_text(model_yaml, encoding="utf-8")
    ran = runner.invoke(cli, [*args, "run", "--target", "dev"])
    assert ran.exit_code == 0, ran.output
    shared = (target_dir / MANIFEST_FILENAME).read_bytes()
    assert json.loads(shared)["project"]["target"] == "dev"
    assert target_manifest_path(target_dir, "dev").read_bytes() == shared
    assert prod_manifest.read_bytes() == prod_content

    ArtifactCatalog.load(tmp_path, expected_target="prod")
    transports: list[str] = []

    def record_transport(service: ContextService, *, transport: str) -> None:
        service.close()
        transports.append(transport)

    monkeypatch.setattr(server, "_run", record_transport)
    served = runner.invoke(cli, [*args, "mcp", "serve"])
    assert served.exit_code == 0, served.output
    assert transports == ["stdio"]


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


def test_clean_removes_its_manifests_and_keeps_what_is_not_its_own(
    tmp_path: Path,
) -> None:
    """Review of #781: `target/manifests/` may already hold a warehouse.

    A `duckdb.path` under that directory is valid configuration, and the
    directory did not belong to stel before #760. Clean removes only files
    named the way `target_manifest_path` names them.
    """
    _write_project(tmp_path)
    write_manifest(tmp_path, target="prod")
    write_manifest(tmp_path, target="dev")
    manifests = tmp_path / "target" / TARGET_MANIFESTS_DIRNAME
    warehouse = manifests / "warehouse.duckdb"
    warehouse.write_bytes(b"not ours")

    clean_project(tmp_path)

    assert warehouse.read_bytes() == b"not ours"
    assert [path.name for path in manifests.iterdir()] == ["warehouse.duckdb"]


def test_clean_removes_the_manifests_directory_when_only_its_files_were_there(
    tmp_path: Path,
) -> None:
    _write_project(tmp_path)
    write_manifest(tmp_path, target="prod")

    clean_project(tmp_path)

    assert not (tmp_path / "target").exists()


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
