from __future__ import annotations

from pathlib import Path

import pytest

from stel.config import load_project
from stel.config.model import ChunkConfig, EmbedConfig, ModelConfig
from stel.config.source import SourceConfig
from stel.dag import DAGError, NodeKind, ProjectDAG, SelectionError, parse_ref


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


def test_unknown_tag_selects_nothing(tagged_dag: ProjectDAG) -> None:
    assert tagged_dag.select_models(select="tag:nonsense") == []


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


# ─── kind: selection (issue #494) ───────────────────────────────────────────
#
# Selection could name a model, its ancestry, a tag, or its state — never what
# kind of step it is. "Re-run only the search publishes" meant naming each one,
# or having tagged them before anyone knew they would be wanted, which is the
# tag's whole weakness: it has to be applied ahead of the need.


@pytest.fixture
def kinded_dag() -> ProjectDAG:
    """docs -> ex (extraction) -> ch (chunk) -> em, em2 (embed)."""
    embed = EmbedConfig(provider="deterministic", model="contract-v1", dimensions=4)
    sources = [SourceConfig(name="docs", path="./docs/")]
    models = [
        ModelConfig(
            name="ex", source="ref('docs')", extraction={"backend": "heuristic"}
        ),
        ModelConfig(name="ch", depends_on=["ref('ex')"], chunk=ChunkConfig()),
        ModelConfig(name="em", depends_on=["ref('ch')"], embed=embed),
        ModelConfig(name="em2", depends_on=["ref('ch')"], embed=embed),
    ]
    return ProjectDAG(sources, models)


def test_kind_selects_every_model_of_that_kind(kinded_dag: ProjectDAG) -> None:
    """The ask: a class of steps, without naming or pre-tagging each one."""
    assert kinded_dag.select_models(select="kind:embed") == ["em", "em2"]


def test_kind_composes_with_ancestors(kinded_dag: ProjectDAG) -> None:
    """`+kind:embed` has to mean what a reader expects — the embeds and
    everything they need, which is what re-running them actually requires."""
    assert set(kinded_dag.select_models(select="+kind:embed")) == {
        "ex",
        "ch",
        "em",
        "em2",
    }


def test_kind_composes_with_descendants(kinded_dag: ProjectDAG) -> None:
    assert set(kinded_dag.select_models(select="kind:chunk+")) == {"ch", "em", "em2"}


def test_kind_composes_with_other_tokens(kinded_dag: ProjectDAG) -> None:
    assert set(kinded_dag.select_models(select="kind:chunk kind:embed")) == {
        "ch",
        "em",
        "em2",
    }


def test_kind_can_exclude(kinded_dag: ProjectDAG) -> None:
    assert set(kinded_dag.select_models(exclude="kind:embed")) == {"ex", "ch"}


def test_a_kind_with_no_models_selects_nothing(kinded_dag: ProjectDAG) -> None:
    """Valid but unmatched is not an error: the project has no such models.
    Only an unrecognized kind is a mistake, which is the distinction the
    refusal below exists to preserve."""
    assert kinded_dag.select_models(select="kind:llm") == []


def test_an_unknown_kind_names_the_valid_set(kinded_dag: ProjectDAG) -> None:
    """A typo that silently matched nothing would look exactly like a project
    with no models of that kind. `state:` already refuses its unknown methods
    for the same reason."""
    with pytest.raises(SelectionError, match="Unknown kind selector") as error:
        kinded_dag.select_models(select="kind:embeddings")

    # Naming the mistake without naming the alternatives only half helps.
    message = str(error.value)
    assert "embed" in message and "search" in message


def test_an_empty_kind_raises(kinded_dag: ProjectDAG) -> None:
    with pytest.raises(SelectionError, match="Empty kind"):
        kinded_dag.select_models(select="kind:")


def test_a_source_has_no_kind_to_match(kinded_dag: ProjectDAG) -> None:
    """A source runs nothing, so it has no step kind. It can still be reached
    as an ancestor — it is just never selected on its own account."""
    assert kinded_dag.nodes["docs"].model_kind is None
    assert kinded_dag.select_models(select="kind:extraction") == ["ex"]


def test_the_selector_and_the_run_result_label_cannot_disagree(
    kinded_dag: ProjectDAG,
) -> None:
    """Requirement 4 of the issue. Two independent derivations of "what kind is
    this model" are two things that can drift; the run-result label now
    delegates to the same accessor the DAG node was built from."""
    from stel.config.model import ModelKind
    from stel.runner import _model_kind_label

    embed = EmbedConfig(provider="deterministic", model="contract-v1", dimensions=4)
    model = ModelConfig(name="em", depends_on=["ref('ch')"], embed=embed)

    assert _model_kind_label(model) == "embed"
    assert kinded_dag.nodes["em"].model_kind is ModelKind.EMBED
    assert _model_kind_label(model) == kinded_dag.nodes["em"].model_kind


def test_every_model_kind_names_a_real_config_field() -> None:
    """The invariant `declared_kinds` rests on, pinned rather than trusted.

    Kind blocks are located by field name, so a member whose value is not a
    `ModelConfig` field would make that kind permanently unselectable and
    unreportable — a failure with no symptom at any call site.
    """
    from stel.config.model import ModelConfig as MC
    from stel.config.model import ModelKind

    assert {kind.value for kind in ModelKind} <= set(MC.model_fields)


def test_every_kind_the_runner_can_report_is_selectable(
    kinded_dag: ProjectDAG,
) -> None:
    """Every label a run result can carry is a selector that resolves. Adding a
    ninth kind block without extending `ModelKind` would leave the selector
    quietly one short, and this is what says so."""
    from stel.config.model import ModelKind

    for kind in ModelKind:
        # Resolves to a (possibly empty) selection rather than raising.
        assert isinstance(kinded_dag.select_models(select=f"kind:{kind.value}"), list)
