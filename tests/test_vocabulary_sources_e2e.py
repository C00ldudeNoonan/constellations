"""External vocabulary changes drive real incremental plans (#718 / ALE-431)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import duckdb
import pytest
import yaml
from click.testing import CliRunner

from stel import runner as runner_module
from stel.adapters import StateScope, create_adapter, parse_warehouse_config
from stel.adapters.duckdb import DuckDBAdapter
from stel.cli import cli
from stel.config.loader import ConfigError
from stel.manifest import build_manifest
from stel.plan import plan_project
from stel.runner import build_project, run_project

pytestmark = pytest.mark.e2e


def _project(tmp_path: Path) -> Path:
    (tmp_path / "stel_project.yml").write_text(
        "name: external_vocab\nprofile: vocab\n"
        "vocabularies:\n  sector:\n    from:\n"
        "      relation: warehouse://main.sectors\n      label_column: id\n",
        encoding="utf-8",
    )
    (tmp_path / "profiles.yml").write_text(
        "vocab:\n  target: dev\n  outputs:\n    dev:\n"
        "      warehouse:\n        type: duckdb\n        path: warehouse.duckdb\n"
        "      llm:\n        provider: deterministic\n",
        encoding="utf-8",
    )
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "m.yml").write_text(
        "version: 2\nmodels:\n  - name: classified\n    source: ref('documents')\n"
        "    materialization: incremental\n"
        "    extraction:\n      backend: llm\n      options:\n"
        "        fields:\n"
        "          - {name: sector, type: enum, values_from: vocab.sector}\n"
        "    fields:\n"
        "      - {name: sector, type: enum, values_from: vocab.sector}\n",
        encoding="utf-8",
    )
    (tmp_path / "sources").mkdir()
    (tmp_path / "sources" / "s.yml").write_text(
        "version: 2\nsources:\n  - name: documents\n    path: data\n",
        encoding="utf-8",
    )
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "one.json").write_text('{"sector":"a"}', encoding="utf-8")
    return tmp_path / "warehouse.duckdb"


@pytest.mark.parametrize("invocation", [run_project, build_project])
def test_relation_label_change_is_planned_and_reprocessed(tmp_path: Path, invocation) -> None:
    warehouse = _project(tmp_path)
    with duckdb.connect(str(warehouse)) as con:
        con.execute("CREATE TABLE sectors AS SELECT 'a' AS id")
    invocation(tmp_path)
    assert plan_project(tmp_path).models[0].status == "unchanged"
    with duckdb.connect(str(warehouse)) as con:
        con.execute("INSERT INTO sectors VALUES ('b')")
    plan = plan_project(tmp_path).models[0]
    assert (plan.status, plan.stale_rows, plan.rows_to_reprocess) == ("changed", 1, 1)
    invocation(tmp_path)
    assert plan_project(tmp_path).models[0].status == "unchanged"
    with duckdb.connect(str(warehouse)) as con:
        con.execute("DELETE FROM sectors WHERE id='b'")
    assert plan_project(tmp_path).models[0].status == "changed"


@pytest.mark.parametrize("invocation", [run_project, build_project, plan_project])
@pytest.mark.parametrize(
    "bad_relation", ["missing", "missing_column", "duplicate", "null", "numeric", "too_many"]
)
def test_bad_relation_fails_before_discovery_or_state_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invocation,
    bad_relation: str,
) -> None:
    warehouse = _project(tmp_path)
    sql = {
        "missing": "SELECT 1",
        "missing_column": "CREATE TABLE sectors AS SELECT 'a' AS wrong",
        "duplicate": (
            "CREATE TABLE sectors AS SELECT 'SENSITIVE' AS id UNION ALL SELECT 'SENSITIVE'"
        ),
        "null": "CREATE TABLE sectors AS SELECT NULL::VARCHAR AS id",
        "numeric": "CREATE TABLE sectors AS SELECT 42 AS id",
        "too_many": "CREATE TABLE sectors AS SELECT 'a' AS id UNION ALL SELECT 'b'",
    }
    if bad_relation == "too_many":
        path = tmp_path / "stel_project.yml"
        path.write_text(path.read_text(encoding="utf-8") + "      max_terms: 1\n", encoding="utf-8")
    with duckdb.connect(str(warehouse)) as con:
        con.execute(sql[bad_relation])

    def forbidden(*args, **kwargs):
        pytest.fail("invalid vocabulary reached document discovery")

    monkeypatch.setattr("stel.runner._discover_sources", forbidden)
    with pytest.raises(ConfigError) as caught:
        invocation(tmp_path)
    assert "SENSITIVE" not in str(caught.value)
    assert caught.value.__cause__ is None
    with duckdb.connect(str(warehouse)) as con:
        assert (
            con.execute(
                "SELECT schema_name FROM information_schema.schemata WHERE schema_name='stel'"
            ).fetchall()
            == []
        )


def test_invalid_model_fails_before_warehouse_connection(tmp_path: Path) -> None:
    warehouse = _project(tmp_path)
    path = tmp_path / "models" / "m.yml"
    path.write_text(
        path.read_text(encoding="utf-8").replace("backend: llm", "backend: missing"),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="unregistered backend"):
        run_project(tmp_path)
    assert not warehouse.exists()


def test_compile_resolves_warehouse_values_without_initializing_state(tmp_path: Path) -> None:
    warehouse = _project(tmp_path)
    with duckdb.connect(str(warehouse)) as con:
        con.execute("CREATE TABLE sectors AS SELECT 'a' AS id")
    result = CliRunner().invoke(cli, ["compile", "--project-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "target" / "manifest.json").exists()
    with duckdb.connect(str(warehouse)) as con:
        assert (
            con.execute(
                "SELECT schema_name FROM information_schema.schemata WHERE schema_name='stel'"
            ).fetchall()
            == []
        )


def test_documented_csv_example_builds_offline_and_invalidates_on_seed_change(
    tmp_path: Path,
) -> None:
    example = Path(__file__).resolve().parents[1] / "examples" / "vocabulary_sources"
    project = tmp_path / "example"
    shutil.copytree(example, project, ignore=shutil.ignore_patterns("target", "__pycache__"))
    result = build_project(project)
    assert not result.skipped
    assert all(not model.errors for model in result.run_results)
    assert plan_project(project).models[0].status == "unchanged"
    seed = project / "seeds" / "competitors.csv"
    with seed.open("a", encoding="utf-8") as handle:
        handle.write("0001652044,Alphabet\n")
    assert plan_project(project).models[0].status == "changed"


def _add_catalog_model(tmp_path: Path) -> None:
    """An unselected context model makes the artifact carry serving declarations."""
    path = tmp_path / "models" / "m.yml"
    model_file = yaml.safe_load(path.read_text(encoding="utf-8"))
    model_file["models"].append(
        {
            "name": "context",
            "depends_on": ["ref('classified')"],
            "transform": {"type": "python", "module": "transforms.context"},
            "agent_context": {"contract": "agent_context/v1", "grain": "document_chunks"},
        }
    )
    path.write_text(yaml.safe_dump(model_file), encoding="utf-8")
    (tmp_path / "transforms").mkdir()
    (tmp_path / "transforms" / "context.py").write_text(
        "def run(inputs):\n    return inputs['classified']\n",
        encoding="utf-8",
    )


@pytest.mark.parametrize("invocation", ["run", "build", "watch"])
@pytest.mark.parametrize("change", ["add_label", "drop_relation"])
def test_cli_manifest_keeps_the_published_vocabulary_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invocation: str,
    change: str,
) -> None:
    """PR #733: post-publication artifacts must not re-read a moving vocabulary."""
    warehouse = _project(tmp_path)
    _add_catalog_model(tmp_path)
    with duckdb.connect(str(warehouse)) as con:
        con.execute("CREATE TABLE sectors AS SELECT 'a' AS id")
    baseline = build_manifest(tmp_path)
    expected_version = baseline["models"][0]["code_version"]
    published = []
    read_limits = []
    original_publish = runner_module._run_extraction_model
    original_read = DuckDBAdapter.read_relation

    def read_relation(self, qualified, *, limit=None):
        read_limits.append(limit)
        return original_read(self, qualified, limit=limit)

    def publish_then_change(**kwargs):
        result = original_publish(**kwargs)
        assert result.rows_written == 1
        assert not (tmp_path / "target" / "manifest.json").exists()
        sql = (
            "INSERT INTO main.sectors VALUES ('b')"
            if change == "add_label"
            else "DROP TABLE main.sectors"
        )
        kwargs["adapter"].execute(sql)
        published.append(result)
        return result

    monkeypatch.setattr(DuckDBAdapter, "read_relation", read_relation)
    monkeypatch.setattr("stel.runner._run_extraction_model", publish_then_change)
    if invocation == "watch":
        monkeypatch.setattr("watchfiles.watch", lambda *args, **kwargs: iter(()))
    command = "run" if invocation == "watch" else invocation
    args = [command, "--project-dir", str(tmp_path), "--select", "classified"]
    if invocation == "watch":
        args.append("--watch")
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 0, result.output
    assert len(published) == 1
    assert read_limits == [10_001]
    manifest_path = tmp_path / "target" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    classified = next(model for model in manifest["models"] if model["name"] == "classified")
    assert classified["fields"][0]["values"] == ["a"]
    assert classified["extraction"]["options"]["fields"][0]["values"] == ["a"]
    assert classified["code_version"] == expected_version
    assert [
        term["label"] for term in manifest["declarations"]["vocabularies"]["sector"]["terms"]
    ] == ["a"]
    config = parse_warehouse_config({"type": "duckdb", "path": str(warehouse)})
    with create_adapter(config, project_dir=tmp_path) as adapter:
        assert adapter.state_code_version_counts(StateScope("classified")) == {expected_version: 1}
    # Standalone compilation still resolves a fresh snapshot, including failures.
    compiled = CliRunner().invoke(cli, ["compile", "--project-dir", str(tmp_path)])
    if change == "add_label":
        assert compiled.exit_code == 0, compiled.output
        refreshed = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert refreshed["models"][0]["fields"][0]["values"] == ["a", "b"]
        assert refreshed["models"][0]["code_version"] != expected_version
    else:
        assert compiled.exit_code == 2, compiled.output
        assert "cannot read source relation" in compiled.output
        assert json.loads(manifest_path.read_text(encoding="utf-8")) == manifest
