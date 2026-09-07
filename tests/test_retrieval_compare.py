"""`stel eval --compare` scores variants on one golden set and reports the delta
(issue #532)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb
import pytest
from click.testing import CliRunner

from stel.cli import cli
from stel.compiler import ConfigError, validate_project_contract
from stel.config import load_project
from stel.retrieval_compare import (
    RetrievalComparison,
    Side,
    VariantComparison,
    build_retrieval_compare_artifact,
    compare_results,
    compare_retrieval_variants,
    format_comparison,
)
from stel.retrieval_compare import TestComparison as _TestComparison
from stel.retrieval_eval import RetrievalEvalError, RetrievalTestResult
from stel.retrieval_metrics import QueryDiagnosis, QueryMetrics
from stel.runner import run_project

# Three documents. At a large chunk size the labor report, which carries both
# query terms twice, outranks the release calendar for "payroll unemployment";
# at a small one the report splits into fragments carrying one term each and
# the calendar's single dense sentence wins. That is the chunk-size regression
# a comparison exists to catch. The prices document tops its query either way.
_DOCS = {
    "labor.json": {
        "title": "Employment report",
        "body": (
            "Payroll employment increased and payroll growth was broad across sectors. "
            "Unemployment remained stable and unemployment claims fell again."
        ),
        "category": "labor",
    },
    "calendar.json": {
        "title": "Release calendar",
        "body": (
            "Payroll and unemployment figures, together with the producer price index, "
            "retail sales, housing starts, industrial production, capacity utilization "
            "and the trade balance, are published monthly by the agency according to a "
            "calendar announced at the start of each year."
        ),
        "category": "meta",
    },
    "inflation.json": {
        "title": "Consumer prices",
        "body": "Inflation moderated as consumer price growth slowed.",
        "category": "prices",
    },
}
BIG, SMALL = "1000", "60"

_SEARCH_BODY = """\
    search:
      access: public
      collection: {collection}
      id_field: chunk_id
      document_id_field: {document_id_field}
      chunk_id_field: chunk_id
      text_fields: [text]
      return_text_fields: [text]
      full_text:
        fields: [text]
      query:
        modes: [text]
        consistency: strong
"""

_TEST = """\
    retrieval_tests:
      - name: quality
        golden_set: ref('{golden}')
        mode: text
        at: [1]
        granularity: document
        thresholds:
          recall_at_1: {{min: 1.0, severity: error}}
"""


def _search(
    collection: str,
    *,
    golden: str = "search_golden",
    document_id_field: str = "document_id",
    search_extra: str = "",
) -> str:
    body = _SEARCH_BODY.format(collection=collection, document_id_field=document_id_field)
    return body + search_extra + _TEST.format(golden=golden)


def _write_project(
    tmp_path: Path, models_yaml: str, *, goldens: tuple[str, ...] = ("search_golden",)
) -> Path:
    project = tmp_path / "project"
    project.mkdir()
    (project / "stel_project.yml").write_text(
        "name: compare_demo\nversion: '0.1.0'\nprofile: compare_demo\n", encoding="utf-8"
    )
    (project / "profiles.yml").write_text(
        "compare_demo:\n"
        "  target: dev\n"
        "  outputs:\n"
        "    dev:\n"
        "      warehouse:\n"
        "        type: duckdb\n"
        "        path: target/data.duckdb\n"
        "        schema: analytics\n"
        "      retrieval:\n"
        "        default: local\n"
        "        allow_public_indexes: true\n"
        "        stores:\n"
        "          local:\n"
        "            type: lancedb\n"
        "            path: target/lancedb\n",
        encoding="utf-8",
    )
    (project / "sources").mkdir()
    (project / "sources" / "documents.yml").write_text(
        "version: 2\n"
        "sources:\n"
        "  - name: releases\n"
        "    path: data\n"
        "    file_pattern: '*.json'\n"
        "  - name: golden_queries\n"
        "    path: golden\n"
        "    file_pattern: '*.json'\n",
        encoding="utf-8",
    )
    (project / "models").mkdir()
    (project / "models" / "search.yml").write_text(
        "version: 2\n"
        "models:\n"
        "  - name: release_documents\n"
        "    source: ref('releases')\n"
        "    extraction:\n"
        "      backend: json\n"
        "      options:\n"
        "        fields: [title, body, category]\n"
        "    materialization: incremental\n" + models_yaml,
        encoding="utf-8",
    )
    golden_models = "version: 2\nmodels:\n"
    for name in goldens:
        golden_models += (
            f"  - name: {name}\n"
            "    source: ref('golden_queries')\n"
            "    extraction:\n"
            "      backend: json\n"
            "      options:\n"
            "        fields: [query_id, query_text, relevant_ids]\n"
            "    materialization: full\n"
            "    fields:\n"
            "      - {name: query_id, data_type: string}\n"
            "      - {name: query_text, data_type: string}\n"
            "      - {name: relevant_ids, data_type: json}\n"
        )
    (project / "models" / "golden.yml").write_text(golden_models, encoding="utf-8")
    data = project / "data"
    data.mkdir()
    for name, payload in _DOCS.items():
        (data / name).write_text(json.dumps(payload), encoding="utf-8")
    (project / "golden").mkdir()
    return project


def _chunk_size_variants() -> str:
    return (
        "  - name: release_chunks\n"
        "    for_each:\n"
        f"      chunk_size: [{BIG}, {SMALL}]\n"
        "    depends_on: [ref('release_documents')]\n"
        "    chunk:\n"
        "      text_field: body\n"
        "      chunk_size: ${matrix.chunk_size}\n"
        "      chunk_overlap: 0\n"
        "    materialization: incremental\n"
        "  - name: release_search\n"
        "    for_each:\n"
        f"      chunk_size: [{BIG}, {SMALL}]\n"
        "    depends_on: [\"ref('release_chunks__chunk_size_${matrix.chunk_size}')\"]\n"
        "    materialization: incremental\n" + _search("chunks_${matrix.chunk_size}")
    )


def _document_id(project: Path, stem: str) -> str:
    con = duckdb.connect(str(project / "target" / "data.duckdb"))
    try:
        row = con.execute(
            "select document_id from analytics.release_documents where source_path like ?",
            [f"%{stem}%"],
        ).fetchone()
        assert row is not None
        return str(row[0])
    finally:
        con.close()


def _label(project: Path) -> None:
    """Document-level judgments, so one golden set judges every chunk size."""
    rows = [
        {
            "query_id": "q_labor",
            "query_text": "payroll unemployment",
            "relevant_ids": [_document_id(project, "labor")],
        },
        {
            "query_id": "q_prices",
            "query_text": "consumer prices inflation",
            "relevant_ids": [_document_id(project, "inflation")],
        },
    ]
    for row in rows:
        (project / "golden" / f"{row['query_id']}.json").write_text(
            json.dumps(row), encoding="utf-8"
        )
    run_project(project, select="search_golden")


def _variant(size: str) -> str:
    return f"release_search__chunk_size_{size}"


@pytest.fixture
def chunk_size_project(tmp_path: Path) -> Path:
    project = _write_project(tmp_path, _chunk_size_variants())
    run_project(project)
    _label(project)
    return project


# ─── the comparison ─────────────────────────────────────────────────────────


def test_chunk_size_variants_produce_a_delta_and_the_query_that_moved(
    chunk_size_project: Path,
) -> None:
    comparison = compare_retrieval_variants(
        chunk_size_project,
        compare=f"{_variant(BIG)} {_variant(SMALL)}",
        baseline=None,
        target=None,
        profiles_dir=None,
    )
    assert comparison.baseline == _variant(BIG)
    assert comparison.models == (_variant(BIG), _variant(SMALL))
    [test] = comparison.tests
    assert test.granularity == "document"
    [variant] = test.variants
    # Two queries; the labor one drops from hit to miss at the small size.
    assert variant.deltas["recall"][1] == pytest.approx(-0.5)
    assert variant.deltas["mrr"][1] == pytest.approx(-0.5)
    [moved] = variant.moved
    assert moved.query_id == "q_labor"
    assert moved.deltas["recall"][1] == pytest.approx(-1.0)
    assert moved.baseline_ranked_ids != moved.ranked_ids
    # Document granularity: the ranking is documents, not chunks.
    assert moved.baseline_ranked_ids == (_document_id(chunk_size_project, "labor"),)
    # Each side is tied to the configuration that built it.
    assert test.baseline.code_version != variant.side.code_version
    assert test.baseline.result.status == "pass"
    assert variant.side.result.status == "fail"
    assert comparison.status() == "fail"


def test_cadence_only_variants_produce_an_all_zero_delta(tmp_path: Path) -> None:
    models = (
        "  - name: release_chunks\n"
        "    depends_on: [ref('release_documents')]\n"
        "    chunk:\n"
        "      text_field: body\n"
        "      chunk_size: 1000\n"
        "      chunk_overlap: 0\n"
        "    materialization: incremental\n"
        "  - name: release_search\n"
        "    for_each:\n"
        "      batch_size: [50, 100]\n"
        "    depends_on: [ref('release_chunks')]\n"
        "    materialization: incremental\n"
        + _search(
            "chunks_${matrix.batch_size}",
            search_extra="      batch_size: ${matrix.batch_size}\n",
        )
    )
    project = _write_project(tmp_path, models)
    run_project(project)
    _label(project)

    comparison = compare_retrieval_variants(
        project, compare="tag:release_search", baseline=None, target=None, profiles_dir=None
    )
    # A tag has no order of its own: name order, so 100 sorts before 50.
    assert comparison.models == (
        "release_search__batch_size_100",
        "release_search__batch_size_50",
    )
    [test] = comparison.tests
    [variant] = test.variants
    assert all(
        delta == 0.0 for by_cutoff in variant.deltas.values() for delta in by_cutoff.values()
    )
    assert variant.moved == ()
    assert comparison.status() == "pass"

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["--project-dir", str(project), "eval", "--compare", "tag:release_search",
         "--baseline", "release_search__batch_size_50"],
    )
    assert result.exit_code == 0, result.output
    assert "baseline release_search__batch_size_50" in result.output
    assert "no query moved" in result.output


# ─── refusals, before any query runs ────────────────────────────────────────


def _no_queries(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: Any, **kwargs: Any) -> list[RetrievalTestResult]:
        raise AssertionError("a query ran before comparability was checked")

    monkeypatch.setattr("stel.retrieval_compare.evaluate_search_models", refuse)


def test_different_golden_sets_are_refused_naming_both(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    models = (
        "  - name: release_chunks\n"
        "    depends_on: [ref('release_documents')]\n"
        "    chunk: {text_field: body, chunk_size: 1000, chunk_overlap: 0}\n"
        "    materialization: incremental\n"
        "  - name: search_a\n"
        "    depends_on: [ref('release_chunks')]\n"
        "    materialization: incremental\n" + _search("chunks_a", golden="search_golden")
        + "  - name: search_b\n"
        "    depends_on: [ref('release_chunks')]\n"
        "    materialization: incremental\n" + _search("chunks_b", golden="search_golden_b")
    )
    project = _write_project(tmp_path, models, goldens=("search_golden", "search_golden_b"))
    _no_queries(monkeypatch)
    with pytest.raises(RetrievalEvalError) as excinfo:
        compare_retrieval_variants(
            project, compare="search_a search_b", baseline=None, target=None, profiles_dir=None
        )
    message = str(excinfo.value)
    for name in ("search_a", "search_b", "search_golden", "search_golden_b"):
        assert name in message


def test_different_cutoffs_are_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    models = (
        "  - name: release_chunks\n"
        "    depends_on: [ref('release_documents')]\n"
        "    chunk: {text_field: body, chunk_size: 1000, chunk_overlap: 0}\n"
        "    materialization: incremental\n"
        "  - name: search_a\n"
        "    depends_on: [ref('release_chunks')]\n"
        "    materialization: incremental\n" + _search("chunks_a")
        + "  - name: search_b\n"
        "    depends_on: [ref('release_chunks')]\n"
        "    materialization: incremental\n"
        + _search("chunks_b").replace("at: [1]", "at: [1, 3]")
    )
    project = _write_project(tmp_path, models)
    _no_queries(monkeypatch)
    expected = r"`at: \[1\]` on 'search_a' but `at: \[1, 3\]`"
    with pytest.raises(RetrievalEvalError, match=expected):
        compare_retrieval_variants(
            project, compare="search_a search_b", baseline=None, target=None, profiles_dir=None
        )


def test_compare_needs_two_models_and_a_known_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _write_project(tmp_path, _chunk_size_variants())
    _no_queries(monkeypatch)
    with pytest.raises(RetrievalEvalError, match="at least two"):
        compare_retrieval_variants(
            project, compare=_variant(BIG), baseline=None, target=None, profiles_dir=None
        )
    with pytest.raises(RetrievalEvalError, match="not among the compared models"):
        compare_retrieval_variants(
            project,
            compare="tag:release_search",
            baseline="release_documents",
            target=None,
            profiles_dir=None,
        )
    with pytest.raises(RetrievalEvalError, match="not a search model"):
        compare_retrieval_variants(
            project,
            compare=f"release_documents {_variant(BIG)}",
            baseline=None,
            target=None,
            profiles_dir=None,
        )


def test_document_granularity_requires_a_document_id_field(tmp_path: Path) -> None:
    models = (
        "  - name: release_chunks\n"
        "    depends_on: [ref('release_documents')]\n"
        "    chunk: {text_field: body, chunk_size: 1000, chunk_overlap: 0}\n"
        "    materialization: incremental\n"
        "  - name: search_a\n"
        "    depends_on: [ref('release_chunks')]\n"
        "    materialization: incremental\n" + _search("chunks_a", document_id_field="null")
    )
    project = _write_project(tmp_path, models)
    loaded, sources, model_configs = load_project(project)
    with pytest.raises(ConfigError, match="document_id_field"):
        validate_project_contract(loaded, sources, model_configs, project)


# ─── the CLI and the artifact ───────────────────────────────────────────────


def test_cli_table_and_json_forms(chunk_size_project: Path) -> None:
    runner = CliRunner()
    base = ["--project-dir", str(chunk_size_project), "eval"]
    compare = ["--compare", f"{_variant(BIG)} {_variant(SMALL)}"]

    result = runner.invoke(cli, [*base, *compare])
    # The small-chunk variant fails its threshold, exactly as `stel eval` on
    # it alone would.
    assert result.exit_code == 1, result.output
    assert "recall@1" in result.output
    assert "(-0.500)" in result.output
    assert "q_labor" in result.output
    assert "q_prices" not in result.output.split("moved")[-1]
    assert "1 variant(s) compared against" in result.output

    result = runner.invoke(cli, [*base, *compare, "--json"])
    assert result.exit_code == 1, result.output
    payload = json.loads(result.output)
    assert payload["version"] == 1
    assert payload["models"] == [_variant(BIG), _variant(SMALL)]
    assert (chunk_size_project / "target" / "retrieval_compare.json").exists()
    # The per-model artifact is written too, one entry per side.
    per_model = json.loads(
        (chunk_size_project / "target" / "retrieval_eval.json").read_text(encoding="utf-8")
    )
    assert [entry["model"] for entry in per_model["results"]] == [
        _variant(BIG),
        _variant(SMALL),
    ]


def test_cli_rejects_selectors_with_compare_and_baseline_without_it(tmp_path: Path) -> None:
    project = _write_project(tmp_path, _chunk_size_variants())
    runner = CliRunner()
    base = ["--project-dir", str(project), "eval"]
    result = runner.invoke(cli, [*base, "--compare", "tag:release_search", "--select", "x"])
    assert result.exit_code == 2
    assert "do not apply" in result.output
    result = runner.invoke(cli, [*base, "--baseline", _variant(BIG)])
    assert result.exit_code == 2
    assert "only applies with --compare" in result.output


def test_artifact_carries_identity_and_no_text(chunk_size_project: Path) -> None:
    comparison = compare_retrieval_variants(
        chunk_size_project,
        compare=f"{_variant(BIG)} {_variant(SMALL)}",
        baseline=None,
        target=None,
        profiles_dir=None,
    )
    artifact = build_retrieval_compare_artifact(comparison)
    [test] = artifact["tests"]
    assert test["baseline"]["model"] == _variant(BIG)
    assert test["baseline"]["code_version"]
    [variant] = test["variants"]
    assert variant["code_version"] != test["baseline"]["code_version"]
    assert {q["query_id"] for q in variant["moved_queries"]} == {"q_labor"}
    assert test["golden_set_hash"]
    serialized = json.dumps(artifact)
    for text in ("Payroll employment", "consumer price growth", "data.duckdb", "lancedb"):
        assert text not in serialized


# ─── the pure comparison ────────────────────────────────────────────────────


def _result(
    model: str, values: dict[str, dict[int, float]], ranked: tuple[str, ...]
) -> RetrievalTestResult:
    query = QueryMetrics(
        query_id="q",
        diagnosis=QueryDiagnosis.OK,
        values=values,
        ranked_ids=ranked,
        missing_ids=(),
    )
    return RetrievalTestResult(
        model_name=model,
        test_name="t",
        golden_set="g",
        mode="text",
        per_query=[query],
        aggregate={metric: dict(by_cutoff) for metric, by_cutoff in values.items()},
    )


def test_a_rank_change_that_moves_no_metric_is_not_a_movement() -> None:
    baseline = _result("a", {"recall": {2: 1.0}}, ("x", "y"))
    # Same score, different order among results: noise, not a regression.
    variant = _result("b", {"recall": {2: 1.0 + 1e-12}}, ("y", "x"))
    deltas, moved = compare_results(baseline, variant)
    assert deltas == {"recall": {2: pytest.approx(0.0, abs=1e-9)}}
    assert moved == ()


def test_format_lists_every_model_as_a_column() -> None:
    baseline = _result("a", {"recall": {1: 1.0}}, ("x",))
    variant = _result("b", {"recall": {1: 0.0}}, ("y",))
    deltas, moved = compare_results(baseline, variant)
    comparison = RetrievalComparison(
        project="p",
        baseline="a",
        models=("a", "b"),
        tests=(
            _TestComparison(
                test_name="t",
                golden_set="g",
                golden_set_hash="h",
                granularity="record",
                cutoffs=(1,),
                baseline=Side("a", "v1", baseline),
                variants=(VariantComparison(Side("b", "v2", variant), deltas, moved),),
            ),
        ),
        results=(baseline, variant),
    )
    lines = format_comparison(comparison)
    header = lines[1]
    assert "a" in header and "b" in header
    assert any(line.startswith("recall@1") and "(-1.000)" in line for line in lines)
    assert any("q: recall@1 -1.000" in line for line in lines)
    assert any("ranked [x] -> [y]" in line for line in lines)
