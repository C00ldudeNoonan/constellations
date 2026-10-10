"""The serving catalog reads the declaration the artifact was compiled with,
not the project file the server happens to start beside (issue #669).

`context_entity_links` rows are published under one `vocabularies:`
declaration; a scope that resolved them under a later edit would answer
wrongly, and silently. So the declaration travels in the manifest, and a
manifest that cannot carry it is refused rather than read as declaring none.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from stel.manifest import SERVING_MANIFEST_VERSION, write_manifest
from stel.mcp_server.catalog import ArtifactCatalog, ArtifactCatalogError

_PROJECT_YAML = (
    "name: context_project\n"
    "profile: context_project\n"
    "classes: [institution, indicator]\n"
    "vocabularies:\n"
    "  institutions:\n"
    "    terms:\n"
    "      - label: Central bank\n"
    "        class: institution\n"
    "      - label: Federal Reserve\n"
    "        class: {fed_class}\n"
    "        broader: Central bank\n"
    "        aliases: [the Fed]\n"
)


def _write_project(tmp_path: Path, *, fed_class: str) -> None:
    (tmp_path / "stel_project.yml").write_text(
        _PROJECT_YAML.format(fed_class=fed_class), encoding="utf-8"
    )
    (tmp_path / "profiles.yml").write_text(
        "context_project:\n"
        "  target: dev\n"
        "  outputs:\n"
        "    dev:\n"
        "      warehouse:\n"
        "        type: duckdb\n"
        "        path: ./target/context.duckdb\n"
        "        schema: context\n",
        encoding="utf-8",
    )
    (tmp_path / "sources").mkdir(exist_ok=True)
    (tmp_path / "sources" / "documents.yml").write_text(
        "version: 2\nsources:\n  - name: documents\n    path: data/documents\n",
        encoding="utf-8",
    )
    (tmp_path / "models").mkdir(exist_ok=True)
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


def test_the_manifest_carries_the_declaration_as_authored(tmp_path: Path) -> None:
    _write_project(tmp_path, fed_class="institution")

    manifest = json.loads(write_manifest(tmp_path).read_text(encoding="utf-8"))

    assert manifest["manifest_version"] == SERVING_MANIFEST_VERSION
    declarations = manifest["declarations"]
    assert declarations["classes"] == ["institution", "indicator"]
    terms = declarations["vocabularies"]["institutions"]["terms"]
    # Authored spelling and order survive, so the block reads like the file
    # it came from and validates back into the same model.
    assert [term["label"] for term in terms] == ["Central bank", "Federal Reserve"]
    assert terms[1] == {
        "label": "Federal Reserve",
        "description": None,
        "aliases": ["the Fed"],
        "broader": "Central bank",
        "class": "institution",
    }


def test_the_catalog_serves_the_compiled_declaration_not_the_live_file(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The mismatch path: a term moved between classes after compile.

    Before #669 the catalog read `vocabularies:` from the project file at
    load, so this edit would have moved every published Federal Reserve link
    into the `indicator` scope without anything being recompiled or
    republished. Now the compiled declaration answers, and the only trace of
    the edit is a warning that names what to do about it.
    """
    _write_project(tmp_path, fed_class="institution")
    write_manifest(tmp_path)
    _write_project(tmp_path, fed_class="indicator")

    with caplog.at_level(logging.WARNING, logger="stel.mcp_server.catalog"):
        catalog = ArtifactCatalog.load(tmp_path)

    institutions = catalog.vocabularies["institutions"]
    assert institutions.labels_in_class("institution") == [
        "Central bank",
        "Federal Reserve",
    ]
    assert institutions.labels_in_class("indicator") == []
    assert institutions.narrower_labels("Central bank") == ["Federal Reserve"]
    assert any(
        "serves the compiled declaration" in record.getMessage()
        for record in caplog.records
    )


def test_a_live_file_that_matches_the_manifest_is_not_warned_about(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _write_project(tmp_path, fed_class="institution")
    write_manifest(tmp_path)

    with caplog.at_level(logging.WARNING, logger="stel.mcp_server.catalog"):
        ArtifactCatalog.load(tmp_path)

    assert not caplog.records


def test_a_manifest_that_cannot_carry_the_declaration_is_refused() -> None:
    """An older artifact is refused, never read as declaring nothing.

    A v2 manifest has no `declarations`; treating that as an empty
    declaration would downgrade a project with a vocabulary to the scanned
    `entity_types` and refuse every `entity_scope` with the wrong reason.
    """
    with pytest.raises(ArtifactCatalogError, match=r"v3 artifact and this one is v2"):
        ArtifactCatalog.from_payloads({"manifest_version": 2, "target": {"name": "dev"}})


def test_a_declaration_that_does_not_validate_is_refused() -> None:
    manifest = {
        "manifest_version": SERVING_MANIFEST_VERSION,
        "declarations": {
            "classes": [],
            # `broader` names a term the vocabulary does not declare -- a
            # payload the compiler would never have written.
            "vocabularies": {"v": {"terms": [{"label": "a", "broader": "b"}]}},
        },
        "target": {"name": "dev"},
        "models": [],
        "dag": {},
    }
    with pytest.raises(ArtifactCatalogError, match=r"vocabulary 'v' does not validate"):
        ArtifactCatalog.from_payloads(manifest)


def test_a_term_class_outside_the_declared_classes_is_refused() -> None:
    """The cross-vocabulary half of the check (#673 review).

    `Vocabulary` cannot see `classes:`, so a payload whose term claims a class
    the artifact does not declare validates vocabulary by vocabulary and would
    be served as a scope nothing else knows about. The catalog runs the same
    `check_declaration` the compiler does.
    """
    manifest = {
        "manifest_version": SERVING_MANIFEST_VERSION,
        "declarations": {
            "classes": [],
            "vocabularies": {
                "v": {"terms": [{"label": "Central bank", "class": "institution"}]}
            },
        },
        "target": {"name": "dev"},
        "models": [],
        "dag": {},
    }
    with pytest.raises(ArtifactCatalogError, match=r"not declared under `classes:`"):
        ArtifactCatalog.from_payloads(manifest)
