"""Graph export as Graphiti records, and `stel graph-load` (ALE-603, ADR-0065).

The test that matters loads the export into a real Graphiti store. The file
target is Kuzu, an embedded graph database graphiti-core drives, so the load runs
with no server or credentials; a Neo4j load is credential-gated at the bottom.
The rest pin the rules around it: what fills Graphiti's temporal fields, that a
shard never carries document text, the optional extra, and the loader's
refusals.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import polars as pl
import pytest
from click.testing import CliRunner
from graphiti_core.driver.kuzu_driver import KuzuDriver

from stel.cli import cli
from stel.graph_export import (
    EDGES_DIRNAME,
    GRAPH_MANIFEST_FILENAME,
    NODES_DIRNAME,
    GraphExportError,
    check_format,
    write_graph_export,
)
from stel.graph_graphiti import group_id_for
from stel.graph_load import (
    GraphLoadError,
    GraphLoadResult,
    load_graphiti_export,
    load_into_neo4j,
)
from tests.support_graph_export import context, documents
from tests.support_graph_export import export as _export
from tests.support_graph_export import links_frame as _links
from tests.support_graph_export import relations_frame as _relations
from tests.support_graph_export import snapshot as _snapshot

pytestmark = pytest.mark.filterwarnings(
    "ignore:The Kuzu backend is deprecated:DeprecationWarning"
)

_LOADED_AT = datetime(2030, 1, 1, tzinfo=UTC)
_FACT = (
    "MATCH (a:Entity)-[:RELATES_TO]->(x:RelatesToNode_)-[:RELATES_TO]->(b:Entity) "
    "RETURN a.name AS subject, x.name AS name, x.fact AS fact, x.valid_at AS valid_at, "
    "x.invalid_at AS invalid_at, x.expired_at AS expired_at, x.reference_time AS reference, "
    "b.name AS object ORDER BY x.fact"
)


def _two_documents(out: Path, **kwargs: Any) -> Any:
    """Two documents sharing entity B, one closed fact and one open one."""
    links = _links(
        [
            {"document_id": "d1", "mention_id": "m1", "canonical_id": "A"},
            {"document_id": "d1", "mention_id": "m2", "canonical_id": "B"},
            {"document_id": "d2", "mention_id": "m3", "canonical_id": "B"},
            {"document_id": "d2", "mention_id": "m4", "canonical_id": "C"},
        ]
    )
    relations = _relations(
        [
            {
                "relation_id": "r1", "document_id": "d1", "subject_mention_id": "m1",
                "object_mention_id": "m2", "directed": True, "relation_type": "depends_on",
                "confidence": 0.5,
            },
            {
                "relation_id": "r2", "document_id": "d2", "subject_mention_id": "m3",
                "object_mention_id": "m4", "directed": True, "relation_type": "depends_on",
            },
        ]
    ).with_columns(
        valid_from=pl.Series([datetime(2024, 1, 1), datetime(2025, 1, 1)]),
        valid_to=pl.Series([datetime(2025, 1, 1), None], dtype=pl.Datetime),
        observed_at=pl.Series([datetime(2024, 6, 1), datetime(2025, 6, 1)]),
        superseded_by=pl.Series(["r2", None], dtype=pl.String),
    )
    return _export(out, links, relations, graph_format="graphiti", **kwargs)


def _records(out: Path, kind: str) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for shard in sorted((out / kind).rglob("*.graphiti.jsonl"))
        for line in shard.read_text(encoding="utf-8").splitlines()
    ]


def _load(out: Path, db: Path) -> GraphLoadResult:
    async def run() -> GraphLoadResult:
        driver = KuzuDriver(db=str(db))
        try:
            return await load_graphiti_export(out, driver, loaded_at=_LOADED_AT)
        finally:
            await driver.close()

    return asyncio.run(run())


def _utc(value: datetime | None) -> datetime | None:
    # Kuzu hands a timestamp back without its zone; every value written is UTC.
    return value if value is None or value.tzinfo else value.replace(tzinfo=UTC)


def _query(db: Path, cypher: str) -> list[dict[str, Any]]:
    async def run() -> list[dict[str, Any]]:
        driver = KuzuDriver(db=str(db))
        try:
            rows, _, _ = await driver.execute_query(cypher)
            return [cast(dict[str, Any], row) for row in rows]
        finally:
            await driver.close()

    return asyncio.run(run())


def test_the_export_loads_into_a_graphiti_store_with_no_hand_transformation(
    tmp_path: Path,
) -> None:
    """Done-when of ALE-603: every record the export wrote is saved by
    graphiti-core itself, and the facts come back with their interval."""
    out, db = tmp_path / "g", tmp_path / "store.kuzu"
    _two_documents(out)

    result = _load(out, db)

    # B is in both documents: two shards save it, one node holds it.
    assert (result.documents, result.entities, result.episodes, result.edges) == (2, 4, 2, 2)
    names = _query(db, "MATCH (n:Entity) RETURN n.name AS name ORDER BY n.name")
    assert [row["name"] for row in names] == ["A", "B", "C"]
    episodes = _query(db, "MATCH (e:Episodic) RETURN e.name AS name ORDER BY e.name")
    assert [row["name"] for row in episodes] == ["d1", "d2"]
    mentions = _query(db, "MATCH (:Episodic)-[m:MENTIONS]->(:Entity) RETURN count(m) AS n")
    assert mentions[0]["n"] == 4
    facts = _query(db, _FACT)
    assert [(f["subject"], f["name"], f["object"]) for f in facts] == [
        ("A", "depends_on", "B"),
        ("B", "depends_on", "C"),
    ]
    assert [f["fact"] for f in facts] == ["A depends_on B", "B depends_on C"]


def test_loading_twice_leaves_one_copy(tmp_path: Path) -> None:
    out, db = tmp_path / "g", tmp_path / "store.kuzu"
    _two_documents(out)
    _load(out, db)

    _load(out, db)

    assert len(_query(db, "MATCH (n:Entity) RETURN n.uuid AS u")) == 3
    assert len(_query(db, _FACT)) == 2


def test_graphitis_temporal_fields_are_filled_as_documented(tmp_path: Path) -> None:
    """valid time comes from the file; transaction time is the load's. A closed
    fact is expired when loaded, an open one is not."""
    out, db = tmp_path / "g", tmp_path / "store.kuzu"
    _two_documents(out)
    _load(out, db)

    closed, open_fact = _query(db, _FACT)

    assert _utc(closed["valid_at"]) == datetime(2024, 1, 1, tzinfo=UTC)
    assert _utc(closed["invalid_at"]) == datetime(2025, 1, 1, tzinfo=UTC)
    assert _utc(closed["reference"]) == datetime(2024, 6, 1, tzinfo=UTC)
    assert _utc(closed["expired_at"]) == _LOADED_AT
    assert _utc(open_fact["valid_at"]) == datetime(2025, 1, 1, tzinfo=UTC)
    assert open_fact["invalid_at"] is None
    assert open_fact["expired_at"] is None
    episode = {r["name"]: r for r in _records(out, NODES_DIRNAME) if r["record"] == "episodic_node"}
    assert episode["d1"]["valid_at"] == "2024-06-01T00:00:00Z"


def test_an_episode_with_no_observed_time_is_stamped_by_the_loader(tmp_path: Path) -> None:
    """A relation model with no `observed_at` has no document time to carry: the
    file says null rather than inventing one, and the loader stamps the load."""
    out, db = tmp_path / "g", tmp_path / "store.kuzu"
    links = _links(
        [
            {"mention_id": "m1", "canonical_id": "A"},
            {"mention_id": "m2", "canonical_id": "B"},
        ]
    )
    relations = _relations(
        [{"relation_id": "r1", "subject_mention_id": "m1", "object_mention_id": "m2"}]
    )
    _export(out, links, relations, graph_format="graphiti")

    episode = next(r for r in _records(out, NODES_DIRNAME) if r["record"] == "episodic_node")
    assert episode["valid_at"] is None
    edge = _records(out, EDGES_DIRNAME)[0]
    assert (edge["valid_at"], edge["invalid_at"], edge["reference_time"]) == (None, None, None)
    _load(out, db)
    rows = _query(db, "MATCH (e:Episodic) RETURN e.valid_at AS v")
    assert _utc(rows[0]["v"]) == _LOADED_AT


def test_a_shard_carries_no_document_text_and_no_clock(tmp_path: Path) -> None:
    """The episode's content is the document id and the facts, and the file
    leaves out every field the loader owns, so two exports are byte-identical."""
    out = tmp_path / "g"
    _two_documents(out)
    first = _snapshot(out)

    for record in _records(out, NODES_DIRNAME):
        if record["record"] == "episodic_node":
            assert set(json.loads(record["content"])) == {"document_id", "facts"}
    for record in (*_records(out, NODES_DIRNAME), *_records(out, EDGES_DIRNAME)):
        assert "created_at" not in record and "expired_at" not in record
        assert not any(key.endswith("_embedding") for key in record)

    again = _two_documents(out)
    assert again.shards_written == 0
    assert _snapshot(out) == first


def test_identity_is_a_function_of_the_ids_and_the_group(tmp_path: Path) -> None:
    first, second = tmp_path / "a", tmp_path / "b"
    _two_documents(first)
    _two_documents(second)
    assert _snapshot(first) == _snapshot(second)
    entity_uuids = {
        r["uuid"] for r in _records(first, NODES_DIRNAME) if r["record"] == "entity_node"
    }
    assert len(entity_uuids) == 3

    other = tmp_path / "c"
    links = _links([{"mention_id": "m1", "canonical_id": "A"}])
    write_graph_export(
        documents(links, _relations([])), other, project="other project",
        linking_model="l", relation_model="r", context=context(), graph_format="graphiti",
    )
    (record,) = (r for r in _records(other, NODES_DIRNAME) if r["record"] == "entity_node")
    assert record["group_id"] == "other_project"
    assert record["uuid"] not in entity_uuids


def test_group_ids_use_only_what_graphiti_accepts() -> None:
    assert group_id_for("my project/é") == "my_project__"
    assert group_id_for("") == "stel"


def test_without_the_graphiti_extra_the_export_names_the_install_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Before anything is written, and before a project is read."""
    monkeypatch.setitem(sys.modules, "graphiti_core", None)

    with pytest.raises(GraphExportError, match=r"pip install 'stel\[graphiti\]'"):
        check_format("graphiti")
    with pytest.raises(GraphExportError, match=r"pip install 'stel\[graphiti\]'"):
        _two_documents(tmp_path / "g")
    assert not (tmp_path / "g").exists()

    result = CliRunner().invoke(
        cli,
        [
            "--project-dir", str(tmp_path),  # holds no project: it is never read
            "graph-export", "--format", "graphiti",
            "--linking-model", "l", "--relation-model", "r",
            "--entity-name", "system", "-o", str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 2
    assert "pip install 'stel[graphiti]'" in result.output


def test_without_the_graphiti_extra_the_load_names_the_install_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "g"
    _two_documents(out)
    monkeypatch.setitem(sys.modules, "graphiti_core", None)
    monkeypatch.setenv("NEO4J_USER", "u")
    monkeypatch.setenv("NEO4J_PASSWORD", "p")

    result = CliRunner().invoke(cli, ["graph-load", str(out), "--neo4j-uri", "bolt://x:1"])

    assert result.exit_code == 2
    assert "pip install 'stel[graphiti]'" in result.output


def test_the_other_formats_never_need_graphiti(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "graphiti_core", None)
    check_format("jsonl")
    out = tmp_path / "g"
    _export(
        out,
        _links([{"mention_id": "m1", "canonical_id": "A"}]),
        _relations([]),
    )
    assert (out / GRAPH_MANIFEST_FILENAME).is_file()


def test_the_loader_refuses_an_export_of_another_format(tmp_path: Path) -> None:
    out = tmp_path / "g"
    _export(
        out, _links([{"mention_id": "m1", "canonical_id": "A"}]), _relations([]),
        graph_format="jsonl",
    )
    with pytest.raises(GraphLoadError, match="only `--format graphiti` loads"):
        _load(out, tmp_path / "store.kuzu")


def test_a_shard_changed_since_the_export_fails_before_it_is_loaded(tmp_path: Path) -> None:
    """Loading half an export, or an edited one, is worse than not loading."""
    out, db = tmp_path / "g", tmp_path / "store.kuzu"
    _two_documents(out)
    shard = sorted((out / EDGES_DIRNAME).rglob("*.graphiti.jsonl"))[-1]
    shard.write_bytes(shard.read_bytes().replace(b"depends_on", b"owns"))

    with pytest.raises(GraphLoadError, match="no longer matches the manifest"):
        _load(out, db)


def test_a_shard_the_manifest_names_but_the_directory_lost_fails_the_load(
    tmp_path: Path,
) -> None:
    out = tmp_path / "g"
    _two_documents(out)
    sorted((out / NODES_DIRNAME).rglob("*.graphiti.jsonl"))[0].unlink()

    with pytest.raises(GraphLoadError, match="named by the manifest but missing"):
        _load(out, tmp_path / "store.kuzu")


def test_a_file_the_manifest_does_not_name_is_not_loaded(tmp_path: Path) -> None:
    out, db = tmp_path / "g", tmp_path / "store.kuzu"
    _two_documents(out)
    stray = next((out / NODES_DIRNAME).iterdir()) / ("f" * 32 + ".graphiti.jsonl")
    stray.write_text(
        json.dumps({"record": "entity_node", "uuid": "stray", "name": "Z", "group_id": "p",
                    "labels": ["Entity"], "summary": "", "attributes": {}}) + "\n",
        encoding="utf-8",
    )

    _load(out, db)

    assert "Z" not in [r["name"] for r in _query(db, "MATCH (n:Entity) RETURN n.name AS name")]


def test_the_loader_does_not_follow_a_symlinked_shard_tree(tmp_path: Path) -> None:
    out = tmp_path / "g"
    _two_documents(out)
    elsewhere = tmp_path / "elsewhere"
    (out / EDGES_DIRNAME).rename(elsewhere)
    try:
        (out / EDGES_DIRNAME).symlink_to(elsewhere, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks need privileges on this platform")

    with pytest.raises(GraphLoadError, match="symlink"):
        _load(out, tmp_path / "store.kuzu")


def test_graph_load_needs_neo4j_credentials_in_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "g"
    _two_documents(out)
    monkeypatch.delenv("NEO4J_USER", raising=False)
    monkeypatch.delenv("NEO4J_PASSWORD", raising=False)

    result = CliRunner().invoke(cli, ["graph-load", str(out), "--neo4j-uri", "bolt://x:1"])

    assert result.exit_code == 2
    assert "NEO4J_USER and NEO4J_PASSWORD" in result.output


def test_a_failed_neo4j_connection_does_not_echo_the_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A driver's own error text can carry what it was given."""
    out = tmp_path / "g"
    _two_documents(out)

    class _Refusing:
        def __init__(self, *, uri: str, user: str, password: str, database: str) -> None:
            raise RuntimeError(f"cannot reach {uri} as {user} with {password}")

    monkeypatch.setattr("graphiti_core.driver.neo4j_driver.Neo4jDriver", _Refusing)

    with pytest.raises(GraphLoadError) as raised:
        load_into_neo4j(
            out, uri="bolt://db.internal:7687", user="someone", password="hunter2-secret",
            database=None,
        )

    message = str(raised.value)
    assert "RuntimeError" in message
    assert all(text not in message for text in ("hunter2-secret", "someone", "db.internal"))
    assert raised.value.__cause__ is None and raised.value.__suppress_context__


_NEO4J = {
    name: os.environ.get(f"STEL_NEO4J_TEST_{name.upper()}") for name in ("uri", "user", "password")
}


@pytest.mark.skipif(
    not all(_NEO4J.values()),
    reason="set STEL_NEO4J_TEST_URI, STEL_NEO4J_TEST_USER and STEL_NEO4J_TEST_PASSWORD",
)
def test_live_neo4j_load(tmp_path: Path) -> None:
    """The same export against a live Neo4j-backed Graphiti store (docs/release.md)."""
    out = tmp_path / "g"
    _two_documents(out)

    first = load_into_neo4j(
        out, uri=str(_NEO4J["uri"]), user=str(_NEO4J["user"]),
        password=str(_NEO4J["password"]), database=None,
    )
    second = load_into_neo4j(
        out, uri=str(_NEO4J["uri"]), user=str(_NEO4J["user"]),
        password=str(_NEO4J["password"]), database=None,
    )

    assert first == second
    assert (first.documents, first.episodes, first.edges) == (2, 2, 2)
