"""Graph export as JSON-LD and Turtle (ALE-602, ADR-0063).

The RDF shards are the same graph the JSON Lines export writes, so the test
that matters parses them back with rdflib and compares the result to the JSON
Lines rows. The rest pin the rules around it: the optional extra, a directory
keeping one format, and the identity of a node.
"""
from __future__ import annotations

import importlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

import polars as pl
import pytest
import rdflib
from click.testing import CliRunner

from stel.cli import cli
from stel.config.class_hierarchy import ClassDef
from stel.config.vocabulary import Vocabulary
from stel.graph_export import (
    EDGES_DIRNAME,
    GRAPH_MANIFEST_FILENAME,
    NODES_DIRNAME,
    GraphExportError,
    check_format,
)
from stel.graph_rdf import ENTITY_IRI, RELATION_IRI, TERMS
from tests.support_graph_export import export as _export
from tests.support_graph_export import links_frame as _links
from tests.support_graph_export import relations_frame as _relations
from tests.support_graph_export import snapshot as _snapshot

_RDF_FORMATS = {"jsonld": ("jsonld", "json-ld"), "turtle": ("ttl", "turtle")}
_STEL = rdflib.Namespace(TERMS)


def _vocabularies() -> dict[str, Vocabulary]:
    return {
        "systems": Vocabulary.model_validate(
            {"terms": [{"label": "Café", "class": "cache"}]}
        )
    }


def _rich_export(out: Path, graph_format: str) -> Any:
    """Awkward text everywhere an id or a value can carry it, a class with an
    ancestor, a closed interval, and a document id that is not a file name."""
    document = "docs/a b/é.md"
    links = _links(
        [
            {"document_id": document, "mention_id": "m1", "canonical_id": 'Café "q"\nline'},
            {"document_id": document, "mention_id": "m2", "canonical_id": "B"},
            {"document_id": "d2", "mention_id": "m3", "canonical_id": "Café"},
            {"document_id": "d2", "mention_id": "m4", "canonical_id": "B"},
        ]
    )
    relations = _relations(
        [
            {
                "relation_id": "r/1 ?",
                "document_id": document,
                "subject_mention_id": "m1",
                "object_mention_id": "m2",
                "confidence": 0.123456789,
                "directed": True,
                "relation_type": "depends_on",
            },
            {
                "relation_id": "r2",
                "document_id": "d2",
                "subject_mention_id": "m3",
                "object_mention_id": "m4",
                "sentence_index": 4,
            },
        ]
    ).with_columns(
        pl.Series(
            "valid_from", [datetime(2020, 1, 1, tzinfo=UTC), None], dtype=pl.Datetime("us", "UTC")
        ),
        pl.Series("valid_to", [datetime(2023, 6, 1, 12, 30, 15, 250000, tzinfo=UTC), None],
                  dtype=pl.Datetime("us", "UTC")),
        pl.Series("observed_at", [None, None], dtype=pl.Datetime("us", "UTC")),
        pl.Series("superseded_by", ["r2", None], dtype=pl.String()),
    )
    return _export(
        out,
        links,
        relations,
        graph_format=graph_format,
        vocabularies=_vocabularies(),
        classes=("datastore", ClassDef(name="cache", subclass_of="datastore")),
    )


def _jsonl_rows(root: Path, kind: str) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for shard in sorted((root / kind).rglob("*.jsonl"))
        for line in shard.read_text(encoding="utf-8").splitlines()
    ]


def _parse(root: Path, kind: str, graph_format: str) -> rdflib.Graph:
    extension, parser = _RDF_FORMATS[graph_format]
    graph = rdflib.Graph()
    for shard in sorted((root / kind).rglob(f"*.{extension}")):
        graph.parse(data=shard.read_bytes(), format=parser)
    return graph


def _values(graph: rdflib.Graph, subject: rdflib.URIRef) -> dict[str, Any]:
    """A resource's properties as the JSON Lines column names, nulls omitted."""
    out: dict[str, Any] = {}
    for predicate, obj in graph.predicate_objects(subject):
        if predicate == rdflib.RDF.type:
            continue
        name = str(predicate).removeprefix(TERMS)
        if name == "classAncestor":
            out.setdefault(name, set()).add(str(obj))
        elif not isinstance(obj, rdflib.Literal):
            out[name] = str(obj)
        elif obj.datatype in (
            rdflib.XSD.boolean, rdflib.XSD.double, rdflib.XSD.integer, rdflib.XSD.dateTime
        ):
            out[name] = obj.toPython()
        else:
            out[name] = str(obj)
    return out


def _expected_node(row: dict[str, Any]) -> dict[str, Any]:
    keep = ("entity_namespace", "entity_name", "canonical_id", "class")
    values = {camel(k): row[k] for k in keep if row[k] is not None}
    if row["class_ancestors"]:
        values["classAncestor"] = set(row["class_ancestors"])
    return values


def camel(column: str) -> str:
    head, *rest = column.split("_")
    return head + "".join(word.title() for word in rest)


def _expected_edge(row: dict[str, Any]) -> dict[str, Any]:
    references = {
        "subject_entity_id": ENTITY_IRI,
        "object_entity_id": ENTITY_IRI,
        "superseded_by": RELATION_IRI,
    }
    values: dict[str, Any] = {}
    for key, value in row.items():
        if value is None or key == "relation_id":
            continue
        name = {"subject_entity_id": "subject", "object_entity_id": "object"}.get(
            key, "document" if key == "document_id" else camel(key)
        )
        if key in references:
            values[name] = references[key] + quote(str(value), safe="")
        elif key == "document_id":
            values[name] = "urn:stel:document:" + quote(str(value), safe="")
        elif key in ("valid_from", "valid_to", "observed_at"):
            # A parser reads the instant, so the same moment is what matters.
            values[name] = datetime.fromisoformat(value.replace("Z", "+00:00"))
        else:
            values[name] = value
    return values


@pytest.mark.parametrize("graph_format", ["jsonld", "turtle"])
def test_the_rdf_shards_parse_back_to_the_rows_the_jsonl_export_writes(
    tmp_path: Path, graph_format: str
) -> None:
    """The done-when: both formats are produced from the same graph the JSON
    Lines export writes, and parse back with rdflib. Compared row by row, so a
    dropped column, a mistyped literal or a mangled id fails here."""
    plain = tmp_path / "plain"
    rdf = tmp_path / graph_format
    _rich_export(plain, "jsonl")
    result = _rich_export(rdf, graph_format)
    assert (result.documents, result.node_rows, result.edges) == (2, 4, 2)

    nodes = _parse(rdf, NODES_DIRNAME, graph_format)
    edges = _parse(rdf, EDGES_DIRNAME, graph_format)
    for graph, kind, expected, scheme in (
        (nodes, NODES_DIRNAME, _expected_node, ENTITY_IRI),
        (edges, EDGES_DIRNAME, _expected_edge, RELATION_IRI),
    ):
        id_column = "entity_id" if kind == NODES_DIRNAME else "relation_id"
        rows = {row[id_column]: row for row in _jsonl_rows(plain, kind)}
        subjects = {
            subject for subject in graph.subjects(rdflib.RDF.type, None)
            if isinstance(subject, rdflib.URIRef)
        }
        assert subjects == {rdflib.URIRef(scheme + quote(key, safe="")) for key in rows}
        for key, row in rows.items():
            actual = _values(graph, rdflib.URIRef(scheme + quote(key, safe="")))
            assert actual == expected(row), key


def test_an_entity_is_the_agent_context_id_in_a_urn_not_an_invented_domain(
    tmp_path: Path,
) -> None:
    """What identifies a node is stated: the same `entity_id` the JSON Lines
    export keys nodes by, under a `urn:`. Classes are literals, not IRIs."""
    out = tmp_path / "g"
    _rich_export(out, "turtle")
    plain = tmp_path / "p"
    _rich_export(plain, "jsonl")

    graph = _parse(out, NODES_DIRNAME, "turtle")
    ids = {str(s) for s in graph.subjects(rdflib.RDF.type, _STEL.Entity)}
    assert ids == {ENTITY_IRI + row["entity_id"] for row in _jsonl_rows(plain, NODES_DIRNAME)}
    assert all(i.startswith("urn:stel:entity:") for i in ids)
    classes = set(graph.objects(None, _STEL["class"]))
    assert classes == {rdflib.Literal("cache")}


@pytest.mark.parametrize("graph_format", ["jsonld", "turtle"])
def test_an_unchanged_re_export_writes_nothing(tmp_path: Path, graph_format: str) -> None:
    """The shard fingerprint is over these bytes, so they must not move between
    runs: an unstable serializer would rewrite every shard on every export."""
    out = tmp_path / "g"
    _rich_export(out, graph_format)
    before = _snapshot(out)

    again = _rich_export(out, graph_format)

    assert (again.shards_written, again.shards_unchanged) == (0, 2)
    assert _snapshot(out) == before


@pytest.mark.parametrize("graph_format", ["jsonld", "turtle"])
def test_the_manifest_records_the_format_and_shards_carry_its_extension(
    tmp_path: Path, graph_format: str
) -> None:
    out = tmp_path / "g"
    _rich_export(out, graph_format)

    manifest = json.loads((out / GRAPH_MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert manifest["format"] == graph_format
    extension = _RDF_FORMATS[graph_format][0]
    files = [name for name in _snapshot(out) if name != GRAPH_MANIFEST_FILENAME]
    assert files
    assert all(name.endswith(f".{extension}") for name in files)


def test_a_directory_keeps_one_format_and_is_left_untouched_on_a_switch(tmp_path: Path) -> None:
    out = tmp_path / "g"
    _rich_export(out, "turtle")
    before = _snapshot(out)

    for other in ("jsonl", "jsonld"):
        with pytest.raises(GraphExportError, match="keeps one format"):
            _rich_export(out, other)
        assert _snapshot(out) == before


def test_a_manifest_written_before_format_existed_is_a_jsonl_export(tmp_path: Path) -> None:
    out = tmp_path / "g"
    _rich_export(out, "jsonl")
    manifest_path = out / GRAPH_MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["format"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    again = _rich_export(out, "jsonl")

    assert again.shards_written == 0
    with pytest.raises(GraphExportError, match="keeps one format"):
        _rich_export(out, "turtle")


def test_an_unknown_format_is_refused_by_name(tmp_path: Path) -> None:
    with pytest.raises(GraphExportError, match=r"unknown graph format 'xml'.*jsonl, jsonld"):
        _rich_export(tmp_path / "g", "xml")
    assert not (tmp_path / "g").exists()


@pytest.mark.parametrize("graph_format", ["jsonld", "turtle"])
def test_without_the_rdf_extra_the_export_names_the_install_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, graph_format: str
) -> None:
    """Before anything is written, and before a project is read."""
    monkeypatch.setitem(sys.modules, "rdflib", None)

    with pytest.raises(GraphExportError, match=r"pip install 'stel\[rdf\]'"):
        check_format(graph_format)
    with pytest.raises(GraphExportError, match=r"pip install 'stel\[rdf\]'"):
        _rich_export(tmp_path / "g", graph_format)
    assert not (tmp_path / "g").exists()

    result = CliRunner().invoke(
        cli,
        [
            "--project-dir", str(tmp_path),  # holds no project: it is never read
            "graph-export",
            "--format", graph_format,
            "--linking-model", "l",
            "--relation-model", "r",
            "--entity-name", "system",
            "-o", str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 2
    assert "pip install 'stel[rdf]'" in result.output


def test_the_default_format_never_needs_rdflib(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The core install stays lean: a JSON Lines export runs with rdflib absent."""
    monkeypatch.setitem(sys.modules, "rdflib", None)
    out = tmp_path / "g"

    result = _rich_export(out, "jsonl")

    assert result.edges == 2
    check_format("jsonl")


def test_importing_the_export_does_not_import_rdflib(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Importing the two modules with rdflib unimportable still works: neither
    reaches for it until a shard is serialized."""
    monkeypatch.setitem(sys.modules, "rdflib", None)
    for name in ("stel.graph_export", "stel.graph_rdf"):
        monkeypatch.delitem(sys.modules, name)

    importlib.import_module("stel.graph_export")


def test_a_confidence_keeps_every_digit_in_turtle(tmp_path: Path) -> None:
    """rdflib's Turtle serializer writes a double with six decimals, which would
    turn 0.123456789 into 0.1234568 and let two scores share a fingerprint
    (review of PR #814)."""
    out = tmp_path / "g"
    _rich_export(out, "turtle")

    graph = _parse(out, EDGES_DIRNAME, "turtle")
    confidences = {float(o) for o in graph.objects(None, _STEL.confidence)}
    assert confidences == {0.123456789}
