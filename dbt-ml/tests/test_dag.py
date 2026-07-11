from __future__ import annotations

from pathlib import Path

import pytest

from dbt_ml.config import load_project
from dbt_ml.config.model import ModelConfig
from dbt_ml.config.source import SourceConfig
from dbt_ml.dag import DAGError, NodeKind, ProjectDAG, SelectionError, parse_ref


def test_parse_ref_variants() -> None:
    assert parse_ref("ref('foo')") == "foo"
    assert parse_ref('ref("foo")') == "foo"
    assert parse_ref("  ref('foo')  ") == "foo"
    assert parse_ref("foo") == "foo"


def test_example_dag(example_project_dir: Path) -> None:
    _, sources, models = load_project(example_project_dir)
    dag = ProjectDAG(sources, models)

    order = dag.execution_order()
    assert order[0] == "raw_invoices"
    assert set(order) == {"raw_invoices", "invoice_summary", "monthly_totals"}
    assert dag.nodes["vendor_invoices"].kind == NodeKind.SOURCE
    assert dag.nodes["raw_invoices"].kind == NodeKind.MODEL

    mermaid = dag.to_mermaid()
    assert "graph LR" in mermaid
    assert "vendor_invoices --> raw_invoices" in mermaid
    assert "raw_invoices --> invoice_summary" in mermaid
    assert "raw_invoices --> monthly_totals" in mermaid


def test_parallel_batches_groups_independent_siblings(example_project_dir: Path) -> None:
    _, sources, models = load_project(example_project_dir)
    dag = ProjectDAG(sources, models)
    selected = dag.select_models(select="raw_invoices+")

    batches = dag.parallel_batches(selected)
    assert batches[0] == ["raw_invoices"]
    assert set(batches[1]) == {"invoice_summary", "monthly_totals"}
    assert [n for batch in batches for n in batch] == sorted(
        selected, key=dag.execution_order().index
    )


def test_parallel_batches_ignores_unselected_predecessors(
    example_project_dir: Path,
) -> None:
    _, sources, models = load_project(example_project_dir)
    dag = ProjectDAG(sources, models)

    batches = dag.parallel_batches(["invoice_summary", "monthly_totals"])
    assert len(batches) == 1
    assert set(batches[0]) == {"invoice_summary", "monthly_totals"}


def test_required_sources_includes_ancestors_of_downstream_selection(
    example_project_dir: Path,
) -> None:
    _, sources, models = load_project(example_project_dir)
    dag = ProjectDAG(sources, models)

    assert dag.required_sources(["invoice_summary"]) == ["vendor_invoices"]


def test_required_sources_empty_for_empty_selection(
    example_project_dir: Path,
) -> None:
    _, sources, models = load_project(example_project_dir)
    assert ProjectDAG(sources, models).required_sources([]) == []


def test_unknown_ref_raises() -> None:
    models = [
        ModelConfig(name="a", depends_on=["ref('does_not_exist')"]),
    ]
    with pytest.raises(DAGError, match="unknown node 'does_not_exist'"):
        ProjectDAG([], models)


def test_cycle_detection() -> None:
    models = [
        ModelConfig(name="a", depends_on=["ref('b')"]),
        ModelConfig(name="b", depends_on=["ref('a')"]),
    ]
    with pytest.raises(DAGError, match="Cyclic dependency"):
        ProjectDAG([], models)


def test_duplicate_node_name() -> None:
    sources = [SourceConfig(name="x", path="./x/")]
    models = [ModelConfig(name="x")]
    with pytest.raises(DAGError, match="Duplicate node name"):
        ProjectDAG(sources, models)


def test_isolated_source_appears_in_mermaid() -> None:
    sources = [SourceConfig(name="lonely", path="./lonely/")]
    dag = ProjectDAG(sources, [])
    mermaid = dag.to_mermaid()
    assert "lonely" in mermaid


# Selector tests use a fan-out DAG: src -> a -> b, a -> c
@pytest.fixture
def fanout_dag() -> ProjectDAG:
    sources = [SourceConfig(name="src", path="./src/")]
    models = [
        ModelConfig(name="a", source="ref('src')"),
        ModelConfig(name="b", depends_on=["ref('a')"]),
        ModelConfig(name="c", depends_on=["ref('a')"]),
    ]
    return ProjectDAG(sources, models)


def test_select_no_args_returns_all_models(fanout_dag: ProjectDAG) -> None:
    assert fanout_dag.select_models() == ["a", "b", "c"]


def test_select_single_name(fanout_dag: ProjectDAG) -> None:
    assert fanout_dag.select_models(select="a") == ["a"]


def test_select_with_descendants(fanout_dag: ProjectDAG) -> None:
    assert set(fanout_dag.select_models(select="a+")) == {"a", "b", "c"}


def test_select_with_ancestors(fanout_dag: ProjectDAG) -> None:
    # +b includes a (upstream) but not c (sibling)
    assert set(fanout_dag.select_models(select="+b")) == {"a", "b"}


def test_select_both_directions(fanout_dag: ProjectDAG) -> None:
    assert set(fanout_dag.select_models(select="+a+")) == {"a", "b", "c"}


def test_select_multiple_tokens(fanout_dag: ProjectDAG) -> None:
    assert set(fanout_dag.select_models(select="b c")) == {"b", "c"}


def test_exclude_removes_models(fanout_dag: ProjectDAG) -> None:
    assert set(fanout_dag.select_models(exclude="b")) == {"a", "c"}


def test_select_combined_with_exclude(fanout_dag: ProjectDAG) -> None:
    assert set(fanout_dag.select_models(select="a+", exclude="c")) == {"a", "b"}


def test_select_returns_topological_order(fanout_dag: ProjectDAG) -> None:
    out = fanout_dag.select_models(select="a+")
    assert out.index("a") < out.index("b")
    assert out.index("a") < out.index("c")


def test_unknown_selector_raises(fanout_dag: ProjectDAG) -> None:
    with pytest.raises(SelectionError, match="Unknown selector 'nope'"):
        fanout_dag.select_models(select="nope")


def test_source_selector_excludes_source_from_result(fanout_dag: ProjectDAG) -> None:
    # `src+` should pull in everything reachable, but the source itself is filtered out
    out = fanout_dag.select_models(select="src+")
    assert "src" not in out
    assert set(out) == {"a", "b", "c"}


# Tag tests use a DAG where tags overlap meaningfully
@pytest.fixture
def tagged_dag() -> ProjectDAG:
    sources = [SourceConfig(name="src", path="./src/", tags=["external"])]
    models = [
        ModelConfig(name="a", source="ref('src')", tags=["raw", "invoices"]),
        ModelConfig(name="b", depends_on=["ref('a')"], tags=["agg", "invoices"]),
        ModelConfig(name="c", depends_on=["ref('a')"], tags=["agg", "monthly"]),
    ]
    return ProjectDAG(sources, models)


def test_tag_selects_all_matching_models(tagged_dag: ProjectDAG) -> None:
    assert set(tagged_dag.select_models(select="tag:agg")) == {"b", "c"}


def test_tag_selects_single_match(tagged_dag: ProjectDAG) -> None:
    assert tagged_dag.select_models(select="tag:raw") == ["a"]


def test_tag_with_descendants(tagged_dag: ProjectDAG) -> None:
    # tag:raw+ should include a and its descendants (b, c)
    assert set(tagged_dag.select_models(select="tag:raw+")) == {"a", "b", "c"}


def test_tag_with_ancestors(tagged_dag: ProjectDAG) -> None:
    # +tag:agg should include both b and c and their ancestors (a; source filtered out)
    assert set(tagged_dag.select_models(select="+tag:agg")) == {"a", "b", "c"}


def test_unknown_tag_raises(tagged_dag: ProjectDAG) -> None:
    with pytest.raises(SelectionError, match="No nodes tagged 'nonsense'"):
        tagged_dag.select_models(select="tag:nonsense")


def test_empty_tag_raises(tagged_dag: ProjectDAG) -> None:
    with pytest.raises(SelectionError, match="Empty tag"):
        tagged_dag.select_models(select="tag:")


def test_exclude_by_tag(tagged_dag: ProjectDAG) -> None:
    assert set(tagged_dag.select_models(exclude="tag:agg")) == {"a"}


def test_tag_and_name_selectors_compose(tagged_dag: ProjectDAG) -> None:
    # union of "tag:raw" (a) and "c" → {a, c}
    assert set(tagged_dag.select_models(select="tag:raw c")) == {"a", "c"}


def test_source_tag_can_match(tagged_dag: ProjectDAG) -> None:
    # src has tag "external"; selecting tag:external+ pulls in everything downstream
    assert set(tagged_dag.select_models(select="tag:external+")) == {"a", "b", "c"}
