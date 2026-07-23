from __future__ import annotations

from pathlib import Path

import pytest

from dbt_ml.adapters.duckdb import DuckDBAdapter, DuckDBWarehouseConfig
from dbt_ml.config.model import TransformConfig
from dbt_ml.sql_models import (
    SqlModelError,
    compile_sql,
    discover_refs,
    read_sql_source,
    validate_single_select,
)

# ── ref discovery ────────────────────────────────────────────────────────────

def test_discover_refs_collects_dedups_and_orders() -> None:
    sql = (
        "select * from {{ ref('a') }} "
        "join {{ ref('b') }} using (id) "
        "join {{ ref('a') }} using (id)"
    )
    assert discover_refs(sql, model_name="m") == ["a", "b"]


def test_discover_refs_rejects_non_literal_ref() -> None:
    with pytest.raises(SqlModelError, match="non-literal ref"):
        discover_refs("select * from {{ ref('a' ~ 'b') }}", model_name="m")


def test_discover_refs_rejects_dynamic_variable_ref() -> None:
    with pytest.raises(SqlModelError):
        discover_refs("select * from {{ ref(model_name) }}", model_name="m")


def test_discover_refs_rejects_unsupported_call() -> None:
    with pytest.raises(SqlModelError, match="unsupported"):
        discover_refs("select {{ source('x', 'y') }}", model_name="m")


# ── single-statement guard ───────────────────────────────────────────────────

@pytest.mark.parametrize(
    "sql",
    [
        "select 1",
        "  SELECT 1  ",
        "with c as (select 1) select * from c",
        "select 1;",
    ],
)
def test_validate_single_select_accepts(sql: str) -> None:
    validate_single_select(sql, model_name="m")


@pytest.mark.parametrize(
    "sql",
    [
        "select 1; drop table t",
        "delete from t",
        "create table t as select 1",
        "   ",
        "-- just a comment\n",
    ],
)
def test_validate_single_select_rejects(sql: str) -> None:
    with pytest.raises(SqlModelError):
        validate_single_select(sql, model_name="m")


def test_validate_single_select_ignores_semicolons_in_comments() -> None:
    validate_single_select("select 1 -- a; b\n", model_name="m")


# ── compilation + sandbox ────────────────────────────────────────────────────

def test_compile_sql_renders_refs_and_target() -> None:
    out = compile_sql(
        "select * from {{ ref('up') }} where t = '{{ target.name }}'",
        model_name="m",
        relations={"up": '"db"."main"."up"'},
        target_name="dev",
        target_type="duckdb",
    )
    assert '"db"."main"."up"' in out
    assert "t = 'dev'" in out


def test_compile_sql_blocks_attribute_escape() -> None:
    # The sandbox must not let a template reach Python internals via target.
    from jinja2.exceptions import SecurityError

    with pytest.raises(SecurityError):
        compile_sql(
            "select {{ target.__class__ }}",
            model_name="m",
            relations={},
            target_name="dev",
            target_type="duckdb",
        )


def test_read_sql_source_rejects_wrong_extension(tmp_path: Path) -> None:
    bad = tmp_path / "q.txt"
    bad.write_text("select 1", encoding="utf-8")
    with pytest.raises(SqlModelError, match="must end in"):
        read_sql_source(bad, model_name="m")


# ── config validator ─────────────────────────────────────────────────────────

def test_transform_config_sql_rejects_module() -> None:
    with pytest.raises(ValueError, match="does not accept a `module`"):
        TransformConfig(type="sql", path="q.sql", module="transforms.x")


def test_transform_config_python_rejects_path() -> None:
    with pytest.raises(ValueError, match="does not accept a `path`"):
        TransformConfig(type="python", module="transforms.x", path="q.sql")


def test_transform_config_sql_rejects_uses_llm() -> None:
    with pytest.raises(ValueError, match="does not accept `uses_llm`"):
        TransformConfig(type="sql", path="q.sql", uses_llm=True)


# ── DuckDB adapter round-trip ────────────────────────────────────────────────

def test_duckdb_materialize_sql_full_and_dry_run(tmp_path: Path) -> None:
    config = DuckDBWarehouseConfig(path=tmp_path / "w.duckdb", schema_name="main")
    with DuckDBAdapter(config) as adapter:
        full = adapter.table_ref("up")
        adapter.connection.execute(
            f"CREATE TABLE {full} AS SELECT * FROM (VALUES (1, 'a'), (2, 'b')) AS t(id, v)"
        )
        select_sql = f"SELECT id, v FROM {full} WHERE id = 1"

        schema = adapter.dry_run_sql(select_sql)
        assert {c.name for c in schema.columns} == {"id", "v"}

        result = adapter.materialize_sql_full("down", select_sql)
        assert result.rows_written == 1
        rows = adapter.connection.execute(
            f"SELECT id, v FROM {adapter.table_ref('down')}"
        ).fetchall()
        assert rows == [(1, "a")]


def test_duckdb_dry_run_rejects_invalid_sql(tmp_path: Path) -> None:
    from dbt_ml.adapters.base import AdapterError

    config = DuckDBWarehouseConfig(path=tmp_path / "w.duckdb", schema_name="main")
    with DuckDBAdapter(config) as adapter:
        with pytest.raises(AdapterError, match="dry-run failed"):
            adapter.dry_run_sql("SELECT * FROM does_not_exist_table")


# ── end-to-end example ───────────────────────────────────────────────────────

def test_governed_chunks_example_builds_end_to_end(tmp_path: Path) -> None:
    import shutil

    from dbt_ml.runner import build_project

    src = Path(__file__).resolve().parents[1] / "examples" / "sql_governed_chunks"
    dst = tmp_path / "proj"
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("target", "__pycache__"))

    result = build_project(dst)
    by_name = {r.model_name: r for r in result.run_results}

    # The SQL model ran in the warehouse (upstream rows never entered Polars).
    governed = by_name["governed_chunks"]
    assert governed.kind == "sql"
    assert governed.rows_written == 3
    assert "governed_chunks" in governed.metrics["relation"]
    assert set(governed.metrics["refs"]) == {"document_chunks", "document_permissions"}
    # Its ref()s ordered it after both upstreams.
    order = [r.model_name for r in result.run_results]
    assert order.index("document_chunks") < order.index("governed_chunks")
    assert order.index("document_permissions") < order.index("governed_chunks")
    # Contracts/tests ran like any warehouse model, nothing skipped.
    assert not result.skipped
    assert result.test_results and all(
        t.status == "pass" for t in result.test_results
    )


def test_load_project_populates_sql_depends_on_for_artifact_dag() -> None:
    # Artifact writers (build_manifest/build_run_results) construct ProjectDAG
    # directly without the compiler; load-time ref population keeps their lineage
    # intact for SQL-only projects.
    from dbt_ml.config import load_project
    from dbt_ml.dag import ProjectDAG, parse_ref

    example = Path(__file__).resolve().parents[1] / "examples" / "sql_governed_chunks"
    _, sources, models = load_project(example)
    governed = next(m for m in models if m.name == "governed_chunks")
    assert {parse_ref(d) for d in (governed.depends_on or [])} == {
        "document_chunks",
        "document_permissions",
    }
    dag = ProjectDAG(sources, models)
    assert dag.predecessors["governed_chunks"] == {
        "document_chunks",
        "document_permissions",
    }


def test_sql_transform_rejects_agent_context(tmp_path: Path) -> None:
    from dbt_ml.agent_context import AgentContextGrain
    from dbt_ml.compiler import _prepare_sql_transform
    from dbt_ml.config.loader import ConfigError
    from dbt_ml.config.model import AgentContextConfig, ModelConfig, TransformConfig

    (tmp_path / "q.sql").write_text("select * from {{ ref('up') }}", encoding="utf-8")
    model = ModelConfig(
        name="gc",
        transform=TransformConfig(type="sql", path="q.sql"),
        agent_context=AgentContextConfig(grain=AgentContextGrain.DOCUMENT_CHUNKS),
    )
    with pytest.raises(ConfigError, match="agent_context"):
        _prepare_sql_transform(model, tmp_path)
