"""Validity intervals through a real project run (issue #766, ADR-0056).

The unit tests in `test_relation_validity.py` pin the closing rule over frames.
These run the built-in transforms through the runner against DuckDB, for what
frames cannot show: that the timestamps survive the warehouse round trip, that an
incremental extractor keeps the rows (and so the closed intervals) of unchanged
documents, and that building the same corpus in one go or in two steps gives the
same table.
"""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from stel.adapters import create_adapter
from stel.config import load_project
from stel.execution.contracts import RunError
from stel.profile import resolve_profile
from stel.relation_validity import relations_as_of
from stel.runner import run_project

# Runs a whole project, so it belongs to the `e2e` tier (issue #518).
# `test_test_tiers.py` fails if a file that does is missing this.
pytestmark = pytest.mark.e2e

_FILED = {
    "d1": datetime(2020, 1, 1, tzinfo=UTC),
    "d2": datetime(2023, 1, 1, tzinfo=UTC),
    "d3": datetime(2025, 1, 1, tzinfo=UTC),
}
# (document, company, city): the company's headquarters as each document says.
_FACTS = [("d1", "ACME", "PARIS"), ("d2", "ACME", "BERLIN"), ("d3", "ACME", "ROME")]


def _mentions(documents: list[tuple[str, str, str]]) -> pl.DataFrame:
    rows = []
    for doc, company, city in documents:
        rows.append(
            {"entity_id": f"{doc}:co", "document_id": doc, "label": "company",
             "start": 0, "end": 4, "canonical": company}
        )
        rows.append(
            {"entity_id": f"{doc}:city", "document_id": doc, "label": "city",
             "start": 10, "end": 15, "canonical": city}
        )
    return pl.DataFrame(rows).with_columns(
        pl.lit(0, dtype=pl.Int64).alias("sentence_index"),
        pl.col("document_id").replace_strict(_FILED, return_dtype=pl.Datetime("us", "UTC"))
        .alias("filed"),
    )


def _links(mentions: pl.DataFrame) -> pl.DataFrame:
    return mentions.select(
        pl.col("entity_id").alias("mention_id"),
        pl.when(pl.col("label") == "company")
        .then(pl.lit("co"))
        .otherwise(pl.lit("geo"))
        .alias("entity_namespace"),
        pl.col("canonical").alias("canonical_id"),
        pl.lit("matched").alias("status"),
    )


def _write_project(root: Path) -> None:
    for name in ("sources", "models", "data"):
        (root / name).mkdir()
    (root / "data" / "placeholder.json").write_text('{"id": "x"}', encoding="utf-8")
    (root / "stel_project.yml").write_text(
        "name: validity_demo\nversion: '0.1.0'\nprofile: validity_demo\n"
        "source-paths: [sources]\nmodel-paths: [models]\n"
        "classes: [company, city]\n"
        "relations:\n"
        "  - name: hq_in\n    subject_class: company\n    object_class: city\n"
        "    functional: true\n",
        encoding="utf-8",
    )
    (root / "profiles.yml").write_text(
        "validity_demo:\n  target: dev\n  outputs:\n    dev:\n      warehouse:\n"
        "        type: duckdb\n        path: target/demo.duckdb\n        schema: analytics\n",
        encoding="utf-8",
    )
    (root / "sources" / "documents.yml").write_text(
        "version: 2\nsources:\n  - name: documents\n    path: data\n"
        "    file_pattern: '*.json'\n",
        encoding="utf-8",
    )
    _write_models(root, dated=True)


def _write_models(root: Path, *, dated: bool) -> None:
    """`dated` names the mentions column that carries each document's date;
    without it the extractor writes no validity columns (ADR-0056)."""
    date_option = "        effective_date_field: filed\n" if dated else ""
    (root / "models" / "validity.yml").write_text(
        f"""\
version: 2

models:
  - name: mentions
    source: ref('documents')
    extraction:
      backend: json
  - name: links
    source: ref('documents')
    extraction:
      backend: json
  - name: relations
    depends_on: [ref('mentions')]
    transform:
      type: python
      module: stel.text.transforms.extract_relations
      options:
        mentions: mentions
        extractor: rule
        scope: sentence
{date_option}        rules:
          - subject_label: company
            object_label: city
            relation_type: hq_in
    materialization: incremental
  - name: current_relations
    depends_on: [ref('relations'), ref('links')]
    transform:
      type: python
      module: stel.text.transforms.supersede_relations
      options:
        relations: relations
        entity_links: links
    materialization: full
""",
        encoding="utf-8",
    )


def _load_upstream(root: Path, documents: list[tuple[str, str, str]]) -> None:
    mentions = _mentions(documents)
    project, _, _ = load_project(root)
    resolved = resolve_profile(project, root)
    with create_adapter(resolved.warehouse, project_dir=root) as adapter:
        adapter.materialize_full("mentions", mentions)
        adapter.materialize_full("links", _links(mentions))


def _read(root: Path, table: str) -> pl.DataFrame:
    project, _, _ = load_project(root)
    resolved = resolve_profile(project, root)
    with create_adapter(resolved.warehouse, project_dir=root) as adapter:
        return adapter.read_table(table).sort("relation_id")


def _run(root: Path) -> None:
    results = run_project(root, select="relations current_relations")
    assert [r.model_name for r in results if r.errors] == []


def _project(tmp_path: Path, documents: list[tuple[str, str, str]]) -> Path:
    root = tmp_path / "project"
    root.mkdir()
    _write_project(root)
    _load_upstream(root, documents)
    return root


def _city_by_document(frame: pl.DataFrame) -> dict[str, tuple[datetime | None, str | None]]:
    return {
        doc: (closed, closer)
        for doc, closed, closer in zip(
            frame["document_id"], frame["valid_to"], frame["superseded_by"], strict=True
        )
    }


def test_a_changed_headquarters_closes_the_old_one_through_the_warehouse(
    tmp_path: Path,
) -> None:
    root = _project(tmp_path, _FACTS[:2])

    _run(root)

    extracted = _read(root, "relations")
    current = _read(root, "current_relations")
    # The extractor never closes an interval, and its dates survive the warehouse.
    assert extracted["valid_to"].null_count() == 2
    assert dict(zip(extracted["document_id"], extracted["valid_from"], strict=True)) == {
        "d1": _FILED["d1"],
        "d2": _FILED["d2"],
    }
    by_doc = _city_by_document(current)
    d2_row = current.filter(pl.col("document_id") == "d2")["relation_id"][0]
    assert by_doc == {"d1": (_FILED["d2"], d2_row), "d2": (None, None)}
    assert current.height == 2


def test_the_as_of_read_answers_from_the_warehouse_table(tmp_path: Path) -> None:
    root = _project(tmp_path, _FACTS[:2])
    _run(root)
    current = _read(root, "current_relations")

    def docs_at(year: int) -> list[str]:
        at = datetime(year, 6, 1, tzinfo=UTC)
        return sorted(
            relations_as_of(current, at, include_unknown_validity=False)["document_id"]
        )

    assert docs_at(2019) == []
    assert docs_at(2021) == ["d1"]
    assert docs_at(2024) == ["d2"]


def test_an_incremental_run_keeps_closed_rows_and_closes_the_new_latest(
    tmp_path: Path,
) -> None:
    """The issue's "incremental runs keep closed rows": adding a document
    rewrites none of the others' extracted rows, and the supersession table,
    recomputed from all of them, still carries the interval it closed earlier."""
    root = _project(tmp_path, _FACTS[:2])
    _run(root)
    before = _read(root, "relations")

    _load_upstream(root, _FACTS)
    _run(root)

    after = _read(root, "relations")
    assert after.height == 3
    assert after.filter(pl.col("document_id").is_in(["d1", "d2"])).equals(
        before.filter(pl.col("document_id").is_in(["d1", "d2"]))
    )
    by_doc = _city_by_document(_read(root, "current_relations"))
    d2_row = after.filter(pl.col("document_id") == "d2")["relation_id"][0]
    d3_row = after.filter(pl.col("document_id") == "d3")["relation_id"][0]
    d2_closer = by_doc["d1"][1]
    assert by_doc["d1"][0] == _FILED["d2"]
    assert d2_closer == d2_row
    assert by_doc["d2"] == (_FILED["d3"], d3_row)
    assert by_doc["d3"] == (None, None)


def test_building_in_two_steps_equals_building_in_one(tmp_path: Path) -> None:
    """A forced reprocess recomputes intervals deterministically: the table an
    incremental history produced is the table a clean build of the same corpus
    produces."""
    stepwise = tmp_path / "stepwise"
    stepwise.mkdir()
    _write_project(stepwise)
    _load_upstream(stepwise, _FACTS[:2])
    _run(stepwise)
    _load_upstream(stepwise, _FACTS)
    _run(stepwise)

    clean = tmp_path / "clean"
    clean.mkdir()
    _write_project(clean)
    _load_upstream(clean, _FACTS)
    _run(clean)

    assert _read(stepwise, "current_relations").equals(_read(clean, "current_relations"))
    assert _read(stepwise, "relations").equals(_read(clean, "relations"))


def test_a_model_that_never_asked_for_validity_is_untouched_by_a_new_document(
    tmp_path: Path,
) -> None:
    """Upgrading must not break an existing incremental relation model: a column
    added to an incremental table is a schema change the run refuses, so a model
    that names no date keeps exactly the columns it had."""
    root = _project(tmp_path, _FACTS[:2])
    _write_models(root, dated=False)
    results = run_project(root, select="relations")
    assert [r.model_name for r in results if r.errors] == []
    before = _read(root, "relations")

    _load_upstream(root, _FACTS)
    results = run_project(root, select="relations")

    assert [r.model_name for r in results if r.errors] == []
    after = _read(root, "relations")
    assert after.columns == before.columns
    assert not {"valid_from", "valid_to", "observed_at"} & set(after.columns)
    assert after.height == 3


def test_opting_in_is_a_schema_change_that_a_full_refresh_rebuilds(
    tmp_path: Path,
) -> None:
    """The reference docs' upgrade story, end to end: naming a date adds columns
    to an incremental table, which the run refuses rather than guess, and a
    `--full-refresh` rebuilds it with every document stamped."""
    root = _project(tmp_path, _FACTS)
    _write_models(root, dated=False)
    run_project(root, select="relations")

    _write_models(root, dated=True)
    with pytest.raises(RunError, match=r"Schema change.*valid_from.*--full-refresh"):
        run_project(root, select="relations", accept_reprocess=True)

    results = run_project(root, select="relations", full_refresh=True)
    assert [r.model_name for r in results if r.errors] == []
    extracted = _read(root, "relations")
    stamped = dict(zip(extracted["document_id"], extracted["valid_from"], strict=True))
    assert stamped == _FILED
