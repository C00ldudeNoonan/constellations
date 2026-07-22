from __future__ import annotations

import shutil
from pathlib import Path

import polars as pl
import pytest
import yaml

from dbt_ml.dbt_embed import materialize
from dbt_ml.dbt_embed.codegen import generate_dbt_models


@pytest.fixture
def fresh_project(tmp_path: Path, example_project_dir: Path) -> Path:
    dst = tmp_path / "project"
    shutil.copytree(
        example_project_dir,
        dst,
        ignore=shutil.ignore_patterns("data", "target", "__pycache__"),
    )
    return dst


def test_codegen_writes_a_shim_per_embeddable_model(fresh_project: Path, tmp_path: Path) -> None:
    out = tmp_path / "gen"
    written = generate_dbt_models(fresh_project, out)
    names = {p.name for p in written}
    assert {"raw_invoices.py", "invoice_summary.py", "monthly_totals.py", "schema.yml"} <= names
    for py in ("raw_invoices.py", "invoice_summary.py", "monthly_totals.py"):
        assert (out / py).exists()


def test_extraction_shim_has_no_upstreams(fresh_project: Path, tmp_path: Path) -> None:
    out = tmp_path / "gen"
    generate_dbt_models(fresh_project, out)
    shim = (out / "raw_invoices.py").read_text(encoding="utf-8")
    assert "upstreams" not in shim
    assert "materialize(" in shim
    assert "'raw_invoices'," in shim


def test_transform_shim_reads_upstream_from_dbt_ref(fresh_project: Path, tmp_path: Path) -> None:
    out = tmp_path / "gen"
    generate_dbt_models(fresh_project, out)
    shim = (out / "invoice_summary.py").read_text(encoding="utf-8")
    # The transform depends on raw_invoices, so the shim must pull it from
    # dbt.ref(...) and pass it through as an injected upstream frame.
    assert "'raw_invoices': dbt.ref('raw_invoices').pl()," in shim
    assert "upstreams=upstreams" in shim


def test_codegen_schema_carries_fields_and_tests(fresh_project: Path, tmp_path: Path) -> None:
    out = tmp_path / "gen"
    generate_dbt_models(fresh_project, out)
    schema = yaml.safe_load((out / "schema.yml").read_text(encoding="utf-8"))
    assert schema["version"] == 2
    by_name = {m["name"]: m for m in schema["models"]}
    assert {"raw_invoices", "invoice_summary", "monthly_totals"} <= set(by_name)

    raw_cols = {c["name"]: c for c in by_name["raw_invoices"]["columns"]}
    assert "unique" in raw_cols["invoice_id"]["tests"]
    assert "not_null" in raw_cols["vendor"]["tests"]
    assert raw_cols["total"]["data_type"] == "float"


def test_custom_source_name_used_in_schema(fresh_project: Path, tmp_path: Path) -> None:
    out = tmp_path / "gen"
    generate_dbt_models(fresh_project, out, source_name="my_dbt_ml")
    # source_name shapes meta/relation naming, not the model names themselves.
    schema = yaml.safe_load((out / "schema.yml").read_text(encoding="utf-8"))
    assert {m["name"] for m in schema["models"]} == {
        "raw_invoices",
        "invoice_summary",
        "monthly_totals",
    }


def test_materialize_transform_with_injected_upstream(fresh_project: Path) -> None:
    # The whole embedded transform path with no warehouse and no documents:
    # feed the upstream frame the dbt Python model would have read from
    # dbt.ref(...), and assert dbt-ml runs the transform and returns its output.
    upstream = pl.DataFrame(
        {
            "vendor": ["Acme", "Acme", "Globex"],
            "total": [10.0, 5.0, 7.0],
        }
    )
    result = materialize(
        "invoice_summary",
        project_dir=fresh_project,
        upstreams={"raw_invoices": upstream},
    )
    rows = {r["vendor"]: r for r in result.to_dicts()}
    assert rows["Acme"]["invoice_count"] == 2
    assert rows["Acme"]["total_spend"] == pytest.approx(15.0)
    assert rows["Globex"]["invoice_count"] == 1
    assert rows["Globex"]["total_spend"] == pytest.approx(7.0)


def test_materialize_rejects_unknown_model(fresh_project: Path) -> None:
    with pytest.raises(ValueError, match="not defined"):
        materialize("does_not_exist", project_dir=fresh_project)
