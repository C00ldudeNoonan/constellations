from __future__ import annotations

import json
from pathlib import Path

import pytest

from dbt_ml.adapters.base import AdapterError
from dbt_ml.adapters.duckdb import DuckDBAdapter, DuckDBWarehouseConfig
from dbt_ml.compiler import ConfigError, validate_project_contract
from dbt_ml.config import load_project
from dbt_ml.config.model import ModelConfig, TransformConfig
from dbt_ml.runner import run_project
from dbt_ml.sql_models import compile_sql, discover_refs

# ── template surface: is_incremental() / this ───────────────────────────────

def test_discover_refs_allows_is_incremental_call() -> None:
    sql = "select * from {{ ref('a') }} {% if is_incremental() %} where 1=1 {% endif %}"
    assert discover_refs(sql, model_name="m") == ["a"]


def test_discover_refs_rejects_is_incremental_with_args() -> None:
    from dbt_ml.sql_models import SqlModelError

    with pytest.raises(SqlModelError, match="takes none"):
        discover_refs("select {% if is_incremental(1) %}1{% endif %}", model_name="m")


def test_compile_sql_renders_incremental_branch_true() -> None:
    out = compile_sql(
        "select 1 {% if is_incremental() %} where x > (select max(x) from {{ this }}) {% endif %}",
        model_name="m",
        relations={},
        target_name="dev",
        target_type="duckdb",
        this='"db"."main"."t"',
        is_incremental=True,
    )
    assert '"db"."main"."t"' in out
    assert "where x >" in out


def test_compile_sql_renders_incremental_branch_false() -> None:
    out = compile_sql(
        "select 1 {% if is_incremental() %} where x > 1 {% endif %}",
        model_name="m",
        relations={},
        target_name="dev",
        target_type="duckdb",
        this='"db"."main"."t"',
        is_incremental=False,
    )
    assert "where x > 1" not in out


def test_compile_sql_this_undefined_for_full_model() -> None:
    # A full-type model has no incremental branch; referencing `this` without
    # is_incremental/this being supplied must fail, not silently render blank.
    with pytest.raises(Exception, match="'this' is undefined"):
        compile_sql(
            "select * from {{ this }}",
            model_name="m",
            relations={},
            target_name="dev",
            target_type="duckdb",
        )


# ── config / compiler validation ─────────────────────────────────────────────

def _sql_model(
    name: str,
    tmp_path: Path,
    sql_text: str,
    *,
    materialization: str = "incremental",
    unique_key: str | None = "id",
    on_schema_change: str = "fail",
) -> ModelConfig:
    sql_dir = tmp_path / "sql"
    sql_dir.mkdir(exist_ok=True)
    (sql_dir / f"{name}.sql").write_text(sql_text, encoding="utf-8")
    return ModelConfig(
        name=name,
        transform=TransformConfig(type="sql", path=f"sql/{name}.sql"),
        materialization=materialization,
        unique_key=unique_key,
        on_schema_change=on_schema_change,
    )


_UPSTREAM_SQL = (
    "select id, updated_at from {{ ref('up') }} "
    "{% if is_incremental() %} "
    "where updated_at > (select coalesce(max(updated_at), '1970-01-01') "
    "from {{ this }}) "
    "{% endif %}"
)


def test_incremental_sql_model_requires_unique_key(tmp_path: Path) -> None:
    from dbt_ml.compiler import _prepare_sql_transform

    model = _sql_model("m", tmp_path, _UPSTREAM_SQL, unique_key=None)
    with pytest.raises(ConfigError, match="requires a `unique_key:`"):
        _prepare_sql_transform(model, tmp_path)


def test_unique_key_forbidden_on_full_sql_model(tmp_path: Path) -> None:
    from dbt_ml.compiler import _validate_materialization

    model = _sql_model("m", tmp_path, "select 1 as id", materialization="full")
    with pytest.raises(ConfigError, match="unique_key"):
        _validate_materialization(model)


def test_unique_key_forbidden_on_python_transform() -> None:
    from dbt_ml.compiler import _validate_materialization

    model = ModelConfig(
        name="m",
        transform=TransformConfig(type="python", module="transforms.x"),
        depends_on=["ref('up')"],
        unique_key="id",
    )
    with pytest.raises(ConfigError, match="unique_key"):
        _validate_materialization(model)


def test_python_transform_still_rejects_incremental() -> None:
    from dbt_ml.compiler import _validate_materialization

    model = ModelConfig(
        name="m",
        transform=TransformConfig(type="python", module="transforms.x"),
        depends_on=["ref('up')"],
        materialization="incremental",
    )
    with pytest.raises(ConfigError, match="only supports"):
        _validate_materialization(model)


def test_incremental_sql_transform_passes_materialization_check(tmp_path: Path) -> None:
    from dbt_ml.compiler import _validate_materialization

    model = _sql_model("m", tmp_path, _UPSTREAM_SQL)
    _validate_materialization(model)  # must not raise


# ── DuckDB adapter: incremental merge semantics ──────────────────────────────

@pytest.fixture
def duckdb_adapter(tmp_path: Path):
    config = DuckDBWarehouseConfig(path=tmp_path / "w.duckdb", schema_name="main")
    with DuckDBAdapter(config) as adapter:
        yield adapter


def test_first_incremental_call_creates_and_reports_inserted(duckdb_adapter) -> None:
    duckdb_adapter.connection.execute(
        "CREATE TABLE main.tgt AS SELECT * FROM (VALUES (1, 'a')) AS t(id, v)"
    )
    result = duckdb_adapter.materialize_sql_incremental(
        "tgt",
        "SELECT * FROM (VALUES (2, 'b')) AS t(id, v)",
        unique_key="id",
    )
    assert result.rows_written == 1
    assert result.rows_inserted == 1
    assert result.rows_updated == 0
    rows = duckdb_adapter.connection.execute(
        "SELECT * FROM main.tgt ORDER BY id"
    ).fetchall()
    assert rows == [(1, "a"), (2, "b")]


def test_incremental_upserts_matching_keys(duckdb_adapter) -> None:
    duckdb_adapter.connection.execute(
        "CREATE TABLE main.tgt AS SELECT * FROM (VALUES (1, 'a'), (2, 'b')) AS t(id, v)"
    )
    result = duckdb_adapter.materialize_sql_incremental(
        "tgt",
        "SELECT * FROM (VALUES (2, 'b2'), (3, 'c')) AS t(id, v)",
        unique_key="id",
    )
    assert result.rows_inserted == 1
    assert result.rows_updated == 1
    rows = duckdb_adapter.connection.execute(
        "SELECT * FROM main.tgt ORDER BY id"
    ).fetchall()
    assert rows == [(1, "a"), (2, "b2"), (3, "c")]


def test_incremental_rerun_is_idempotent(duckdb_adapter) -> None:
    duckdb_adapter.connection.execute(
        "CREATE TABLE main.tgt AS SELECT * FROM (VALUES (1, 'a')) AS t(id, v)"
    )
    select_sql = "SELECT * FROM (VALUES (2, 'b')) AS t(id, v)"
    duckdb_adapter.materialize_sql_incremental("tgt", select_sql, unique_key="id")
    before = duckdb_adapter.connection.execute(
        "SELECT * FROM main.tgt ORDER BY id"
    ).fetchall()
    duckdb_adapter.materialize_sql_incremental("tgt", select_sql, unique_key="id")
    after = duckdb_adapter.connection.execute(
        "SELECT * FROM main.tgt ORDER BY id"
    ).fetchall()
    assert before == after == [(1, "a"), (2, "b")]


def test_incremental_empty_batch_is_a_true_noop(duckdb_adapter) -> None:
    duckdb_adapter.connection.execute(
        "CREATE TABLE main.tgt AS SELECT * FROM (VALUES (1, 'a')) AS t(id, v)"
    )
    result = duckdb_adapter.materialize_sql_incremental(
        "tgt",
        "SELECT * FROM (VALUES (1, 'a')) AS t(id, v) WHERE 1=0",
        unique_key="id",
    )
    assert result.rows_written == 0
    rows = duckdb_adapter.connection.execute("SELECT * FROM main.tgt").fetchall()
    assert rows == [(1, "a")]


def test_incremental_rejects_null_key(duckdb_adapter) -> None:
    duckdb_adapter.connection.execute(
        "CREATE TABLE main.tgt AS SELECT * FROM (VALUES (1, 'a')) AS t(id, v)"
    )
    with pytest.raises(AdapterError, match="1 null"):
        duckdb_adapter.materialize_sql_incremental(
            "tgt",
            "SELECT * FROM (VALUES (NULL, 'x')) AS t(id, v)",
            unique_key="id",
        )


def test_incremental_rejects_duplicate_key(duckdb_adapter) -> None:
    duckdb_adapter.connection.execute(
        "CREATE TABLE main.tgt AS SELECT * FROM (VALUES (1, 'a')) AS t(id, v)"
    )
    with pytest.raises(AdapterError, match="1 duplicate"):
        duckdb_adapter.materialize_sql_incremental(
            "tgt",
            "SELECT * FROM (VALUES (2, 'x'), (2, 'y')) AS t(id, v)",
            unique_key="id",
        )
    # Cleanup ran even though materialization failed.
    leftover = duckdb_adapter.connection.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_name LIKE 'dbt_ml_sql_staging%'"
    ).fetchall()
    assert leftover == []


def test_incremental_schema_change_fail_raises(duckdb_adapter) -> None:
    duckdb_adapter.connection.execute(
        "CREATE TABLE main.tgt AS SELECT * FROM (VALUES (1, 'a')) AS t(id, v)"
    )
    with pytest.raises(AdapterError, match="Schema change"):
        duckdb_adapter.materialize_sql_incremental(
            "tgt",
            "SELECT * FROM (VALUES (2, 'b', 'extra')) AS t(id, v, w)",
            unique_key="id",
            on_schema_change="fail",
        )


def test_incremental_schema_change_append_new_columns(duckdb_adapter) -> None:
    duckdb_adapter.connection.execute(
        "CREATE TABLE main.tgt AS SELECT * FROM (VALUES (1, 'a')) AS t(id, v)"
    )
    duckdb_adapter.materialize_sql_incremental(
        "tgt",
        "SELECT * FROM (VALUES (2, 'b', 'extra')) AS t(id, v, w)",
        unique_key="id",
        on_schema_change="append_new_columns",
    )
    rows = duckdb_adapter.connection.execute(
        "SELECT * FROM main.tgt ORDER BY id"
    ).fetchall()
    assert rows == [(1, "a", None), (2, "b", "extra")]


def test_incremental_rejects_key_missing_from_target_even_with_append(
    duckdb_adapter,
) -> None:
    # The target's unique_key changed (or never had this column); appending it
    # as a schema-drift "new column" would leave existing rows with a NULL key,
    # so this must fail under every on_schema_change policy, not only `fail`.
    duckdb_adapter.connection.execute(
        "CREATE TABLE main.tgt AS SELECT * FROM (VALUES ('x', 'a')) AS t(old_id, v)"
    )
    with pytest.raises(AdapterError, match="does not exist in the current target"):
        duckdb_adapter.materialize_sql_incremental(
            "tgt",
            "SELECT * FROM (VALUES (1, 'b')) AS t(id, v)",
            unique_key="id",
            on_schema_change="append_new_columns",
        )
    rows = duckdb_adapter.connection.execute("SELECT * FROM main.tgt").fetchall()
    assert rows == [("x", "a")]  # target untouched


def test_relation_exists(duckdb_adapter) -> None:
    assert duckdb_adapter.relation_exists("nope") is False
    duckdb_adapter.connection.execute("CREATE TABLE main.yep AS SELECT 1 AS x")
    assert duckdb_adapter.relation_exists("yep") is True


# ── end-to-end: build_project with two runs + a mutated source ──────────────

def _write_project(project_dir: Path) -> None:
    (project_dir / "sources").mkdir(parents=True)
    (project_dir / "models").mkdir(parents=True)
    (project_dir / "sql").mkdir(parents=True)
    (project_dir / "data").mkdir(parents=True)

    (project_dir / "dbt_ml_project.yml").write_text(
        "name: incr_demo\n"
        "profile: incr_demo\n"
        "source-paths: [sources]\n"
        "model-paths: [models]\n"
        "target-path: target\n",
        encoding="utf-8",
    )
    (project_dir / "profiles.yml").write_text(
        "incr_demo:\n"
        "  target: dev\n"
        "  outputs:\n"
        "    dev:\n"
        "      warehouse:\n"
        "        type: duckdb\n"
        "        path: ./target/w.duckdb\n"
        "        schema: main\n",
        encoding="utf-8",
    )
    (project_dir / "sources" / "sources.yml").write_text(
        "version: 2\n"
        "sources:\n"
        "  - name: events_raw\n"
        "    path: ./data/\n"
        "    file_pattern: '*.json'\n",
        encoding="utf-8",
    )
    (project_dir / "models" / "events.yml").write_text(
        "version: 2\n"
        "models:\n"
        "  - name: events\n"
        "    source: ref('events_raw')\n"
        "    extraction:\n"
        "      backend: json\n"
        "      options:\n"
        "        fields: [id, name, updated_at]\n"
        "    materialization: full\n"
        "    fields:\n"
        "      - {name: id, data_type: string}\n"
        "      - {name: name, data_type: string}\n"
        "      - {name: updated_at, data_type: string}\n",
        encoding="utf-8",
    )
    (project_dir / "models" / "events_snapshot.yml").write_text(
        "version: 2\n"
        "models:\n"
        "  - name: events_snapshot\n"
        "    transform:\n"
        "      type: sql\n"
        "      path: sql/events_snapshot.sql\n"
        "    materialization: incremental\n"
        "    unique_key: id\n"
        "    fields:\n"
        "      - {name: id, data_type: string}\n"
        "      - {name: name, data_type: string}\n"
        "      - {name: updated_at, data_type: string}\n"
        "    tests:\n"
        "      - unique: id\n"
        "      - not_null: [id, name]\n",
        encoding="utf-8",
    )
    (project_dir / "sql" / "events_snapshot.sql").write_text(
        "select id, name, updated_at\n"
        "from {{ ref('events') }}\n"
        "{% if is_incremental() %}\n"
        "where updated_at > (\n"
        "  select coalesce(max(updated_at), '1970-01-01') from {{ this }}\n"
        ")\n"
        "{% endif %}\n",
        encoding="utf-8",
    )


def _write_event(project_dir: Path, event_id: str, name: str, updated_at: str) -> None:
    (project_dir / "data" / f"{event_id}.json").write_text(
        json.dumps({"id": event_id, "name": name, "updated_at": updated_at}),
        encoding="utf-8",
    )


def _snapshot_rows(project_dir: Path) -> list[tuple]:
    import duckdb

    con = duckdb.connect(str(project_dir / "target" / "w.duckdb"))
    try:
        return con.execute(
            "SELECT id, name, updated_at FROM main.events_snapshot ORDER BY id"
        ).fetchall()
    finally:
        con.close()


def test_incremental_sql_model_end_to_end(tmp_path: Path) -> None:
    project_dir = tmp_path / "proj"
    _write_project(project_dir)
    _write_event(project_dir, "ev1", "a", "2024-01-01")
    _write_event(project_dir, "ev2", "b", "2024-01-01")

    # First run: target doesn't exist yet -> is_incremental() renders False ->
    # a full CTAS over every current row.
    results = run_project(project_dir)
    by_name = {r.model_name: r for r in results}
    snap = by_name["events_snapshot"]
    assert snap.metrics["is_incremental"] is False
    assert snap.rows_written == 2
    assert _snapshot_rows(project_dir) == [
        ("ev1", "a", "2024-01-01"),
        ("ev2", "b", "2024-01-01"),
    ]

    # Second run: update ev1, add ev3, leave ev2 untouched. is_incremental()
    # renders True; the compiled query filters to updated_at > current max, so
    # only ev1 (updated) and ev3 (new) are selected and merged.
    _write_event(project_dir, "ev1", "a-updated", "2024-02-01")
    _write_event(project_dir, "ev3", "c", "2024-02-01")
    results = run_project(project_dir)
    by_name = {r.model_name: r for r in results}
    snap = by_name["events_snapshot"]
    assert snap.metrics["is_incremental"] is True
    assert snap.rows_written == 2  # only the delta, not the full 3-row set
    assert snap.rows_inserted == 1
    assert snap.rows_updated == 1
    assert _snapshot_rows(project_dir) == [
        ("ev1", "a-updated", "2024-02-01"),
        ("ev2", "b", "2024-01-01"),
        ("ev3", "c", "2024-02-01"),
    ]

    # Third run, nothing changed: a true no-op incremental batch.
    results = run_project(project_dir)
    snap = {r.model_name: r for r in results}["events_snapshot"]
    assert snap.metrics["is_incremental"] is True
    assert snap.rows_written == 0
    assert _snapshot_rows(project_dir) == [
        ("ev1", "a-updated", "2024-02-01"),
        ("ev2", "b", "2024-01-01"),
        ("ev3", "c", "2024-02-01"),
    ]

    # --full-refresh forces the full path even though the target exists and
    # the model is configured incremental.
    results = run_project(project_dir, full_refresh=True)
    snap = {r.model_name: r for r in results}["events_snapshot"]
    assert snap.metrics["is_incremental"] is False
    assert snap.rows_written == 3
    assert _snapshot_rows(project_dir) == [
        ("ev1", "a-updated", "2024-02-01"),
        ("ev2", "b", "2024-01-01"),
        ("ev3", "c", "2024-02-01"),
    ]


def test_incremental_sql_model_state_selection_on_unique_key_change(
    tmp_path: Path,
) -> None:
    # unique_key is folded into code_version, so changing it must re-select the
    # model under `state:modified` even though the .sql file is untouched.
    from dbt_ml.versioning import compute_model_code_version

    project_dir = tmp_path / "proj"
    _write_project(project_dir)
    project, _sources, models = load_project(project_dir)
    snap = next(m for m in models if m.name == "events_snapshot")
    v1 = compute_model_code_version(snap, project, project_dir)
    snap.unique_key = "name"
    v2 = compute_model_code_version(snap, project, project_dir)
    assert v1 != v2


def test_incremental_sql_model_compiles_and_validates(tmp_path: Path) -> None:
    project_dir = tmp_path / "proj"
    _write_project(project_dir)
    project, sources, models = load_project(project_dir)
    dag = validate_project_contract(project, sources, models, project_dir)
    assert dag.predecessors["events_snapshot"] == {"events"}
