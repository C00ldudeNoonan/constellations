from __future__ import annotations

import shutil
from pathlib import Path

import polars as pl
import pytest
import yaml

from dbt_ml.config.model import ModelConfig
from dbt_ml.dbt_embed import materialize
from dbt_ml.dbt_embed.codegen import _embeddable_models, generate_dbt_models


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


def _fake_model(
    name: str,
    *,
    extraction: bool = False,
    transform: bool = False,
    ml: bool = False,
    deps: list[str] | None = None,
) -> ModelConfig:
    # model_construct skips validation so we can exercise the dependency-closure
    # logic without building full extraction/transform/ml block schemas.
    return ModelConfig.model_construct(
        name=name,
        extraction=object() if extraction else None,
        transform=object() if transform else None,
        ml=object() if ml else None,
        depends_on=[f"ref('{d}')" for d in (deps or [])] or None,
    )


def test_embeddable_skips_transforms_that_depend_on_nonemittable_models() -> None:
    models = [
        _fake_model("ex", extraction=True),
        _fake_model("mlmod", ml=True),  # non-emittable kind
        _fake_model("good", transform=True, deps=["ex"]),
        _fake_model("bad", transform=True, deps=["mlmod"]),  # dep not emitted
        _fake_model("downstream", transform=True, deps=["bad"]),  # transitively
    ]
    assert {m.name for m in _embeddable_models(models)} == {"ex", "good"}


# --- dbt_ref source: the reverse dbt->dbt-ml direction (#177) -----------------


@pytest.fixture
def dbt_ref_project(tmp_path: Path) -> Path:
    """A minimal dbt-ml project with one transform whose input is a dbt-built
    table (`source: dbt_ref('vendor_dim')`)."""
    proj = tmp_path / "dbt_ref_proj"
    (proj / "models").mkdir(parents=True)
    (proj / "transforms").mkdir()
    (proj / "dbt_ml_project.yml").write_text(
        "name: refproj\nversion: '0.1.0'\nprofile: refproj\n", encoding="utf-8"
    )
    (proj / "profiles.yml").write_text(
        "refproj:\n"
        "  target: dev\n"
        "  outputs:\n"
        "    dev:\n"
        "      warehouse:\n"
        "        type: duckdb\n"
        "        path: ./target/db.duckdb\n"
        "        schema: main\n",
        encoding="utf-8",
    )
    (proj / "models" / "enriched_vendors.yml").write_text(
        "version: 2\n"
        "models:\n"
        "  - name: enriched_vendors\n"
        "    source: dbt_ref('vendor_dim')\n"
        "    materialization: full\n"
        "    transform:\n"
        "      type: python\n"
        "      module: transforms.enrich\n"
        "    fields:\n"
        "      - {name: vendor, type: string}\n"
        "      - {name: spend_doubled, type: float}\n",
        encoding="utf-8",
    )
    (proj / "transforms" / "enrich.py").write_text(
        "from __future__ import annotations\n\n"
        "import polars as pl\n\n\n"
        "def declared_dependencies(options):\n"
        "    # The dbt_ref target is the transform's declared input; the compiler\n"
        "    # must validate the contract against it even though it forms no\n"
        "    # dbt-ml DAG edge (#177).\n"
        "    return ('vendor_dim',)\n\n\n"
        "def run(deps: dict[str, pl.DataFrame]) -> pl.DataFrame:\n"
        "    frame = deps['vendor_dim']\n"
        "    return frame.with_columns((pl.col('spend') * 2).alias('spend_doubled'))\n",
        encoding="utf-8",
    )
    return proj


def test_dbt_ref_transform_contract_validates_against_the_ref_target(
    dbt_ref_project: Path,
) -> None:
    # Regression: a dbt_ref transform whose module declares its dependency must
    # compile — the contract is validated against the dbt_ref target, not the
    # (empty) depends_on. Compiling the project must not raise.
    from dbt_ml.compiler import validate_project_contract
    from dbt_ml.config import load_project

    project, sources, models = load_project(dbt_ref_project)
    validate_project_contract(project, sources, models, dbt_ref_project)


def test_dbt_ref_shim_reads_the_dbt_built_table(dbt_ref_project: Path, tmp_path: Path) -> None:
    out = tmp_path / "gen"
    generate_dbt_models(dbt_ref_project, out)
    shim = (out / "enriched_vendors.py").read_text(encoding="utf-8")
    # The dbt_ref source is fed in like any upstream: dbt orders vendor_dim first,
    # the shim reads it from dbt.ref(...) and injects it.
    assert "'vendor_dim': dbt.ref('vendor_dim').pl()," in shim
    assert "upstreams=upstreams" in shim
    assert "'enriched_vendors'," in shim


def test_dbt_ref_model_is_embeddable_and_in_schema(dbt_ref_project: Path, tmp_path: Path) -> None:
    out = tmp_path / "gen"
    generate_dbt_models(dbt_ref_project, out)
    schema = yaml.safe_load((out / "schema.yml").read_text(encoding="utf-8"))
    assert "enriched_vendors" in {m["name"] for m in schema["models"]}


def test_materialize_dbt_ref_transform_reads_injected_dbt_table(dbt_ref_project: Path) -> None:
    # The reverse direction end to end: feed the frame the dbt Python model would
    # have read from dbt.ref('vendor_dim') and assert dbt-ml transforms it.
    vendor_dim = pl.DataFrame({"vendor": ["Acme", "Globex"], "spend": [10.0, 7.0]})
    result = materialize(
        "enriched_vendors",
        project_dir=dbt_ref_project,
        upstreams={"vendor_dim": vendor_dim},
    )
    rows = {r["vendor"]: r for r in result.to_dicts()}
    assert rows["Acme"]["spend_doubled"] == pytest.approx(20.0)
    assert rows["Globex"]["spend_doubled"] == pytest.approx(14.0)


def test_standalone_run_rejects_dbt_ref_models(dbt_ref_project: Path) -> None:
    from dbt_ml.execution import RunError
    from dbt_ml.runner import run_project

    with pytest.raises(RunError, match="embedded mode"):
        run_project(dbt_ref_project)
