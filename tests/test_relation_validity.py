"""Validity intervals on extracted relations (issue #766, ALE-422, ADR-0056).

Three layers, each tested where it lives: the extractor stamps `valid_from` and
`observed_at` from named mention columns, and writes neither column unless asked;
`supersede_relations` closes the interval of a functional relation a later,
different fact replaced; and `relations_as_of` reads the table at an instant.
"""
from __future__ import annotations

import shutil
from datetime import UTC, date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from pydantic import ValidationError

from stel.adapters import parse_warehouse_config
from stel.compiler import validate_project_contract
from stel.config import load_project
from stel.config.loader import ConfigError
from stel.config.project import ProjectConfig
from stel.config.vocabulary import RelationTypeDef
from stel.relation_validity import (
    SUPERSESSION_VERSION,
    SupersessionResult,
    relations_as_of,
    supersede,
)
from stel.text.transforms import extract_relations, supersede_relations
from stel.transforms import TransformContext

_UTC_US = pl.Datetime("us", "UTC")

# One sentence per document with two mentions, so each document yields exactly
# one co-occurrence relation. `filed` is the document's effective date and
# `seen` the instant it was observed; both repeat on every mention of a document.
_MENTIONS = pl.DataFrame(
    {
        "entity_id": ["m1", "m2", "m3", "m4"],
        "document_id": ["d1", "d1", "d2", "d2"],
        "sentence_index": [0, 0, 0, 0],
        "start": [0, 10, 0, 10],
        "end": [5, 15, 5, 15],
        "label": ["ORG", "GPE", "ORG", "GPE"],
        "entity_text": ["Apple", "France", "Apple", "Spain"],
        "filed": [
            datetime(2024, 1, 1, tzinfo=UTC),
            datetime(2024, 1, 1, tzinfo=UTC),
            datetime(2025, 6, 1, tzinfo=UTC),
            datetime(2025, 6, 1, tzinfo=UTC),
        ],
        "seen": [
            datetime(2024, 1, 2, tzinfo=UTC),
            datetime(2024, 1, 2, tzinfo=UTC),
            datetime(2025, 6, 3, tzinfo=UTC),
            datetime(2025, 6, 3, tzinfo=UTC),
        ],
    }
)


def _ctx(options: dict[str, object] | None = None) -> TransformContext:
    merged: dict[str, object] = {"mentions": "mentions"}
    merged.update(options or {})
    return TransformContext(
        project_dir=Path("."),
        profile_name="test",
        target_name="dev",
        warehouse=parse_warehouse_config(
            {"type": "duckdb", "path": "./test.duckdb", "schema": "main"}
        ),
        llm=None,
        options=merged,
    )


def _run(
    mentions: pl.DataFrame = _MENTIONS, options: dict[str, object] | None = None
) -> pl.DataFrame:
    return extract_relations.run({"mentions": mentions}, _ctx(options))


def _by_document(frame: pl.DataFrame, column: str) -> dict[str, Any]:
    return dict(zip(frame["document_id"], frame[column], strict=True))


# --- the extractor stamps validity from named mention columns -----------------


_VALIDITY_COLUMNS = {"valid_from", "valid_to", "observed_at"}


def test_a_model_that_names_no_date_writes_no_validity_columns() -> None:
    """A column added to an incremental table is a schema change the run refuses
    until the operator accepts it, so a model that did not ask for validity must
    not grow one on upgrade (ADR-0056). The output is exactly what it was."""
    output = _run()

    assert not _VALIDITY_COLUMNS & set(output.columns)
    assert output.height == 2


def test_effective_date_field_stamps_each_document_with_its_own_date() -> None:
    output = _run(options={"effective_date_field": "filed"})

    assert _by_document(output, "valid_from") == {
        "d1": datetime(2024, 1, 1, tzinfo=UTC),
        "d2": datetime(2025, 6, 1, tzinfo=UTC),
    }
    # The extractor never closes an interval: that needs two documents.
    assert output.schema["valid_to"] == _UTC_US
    assert output["valid_to"].null_count() == output.height
    # `observed_at` was not asked for, so it is not written.
    assert "observed_at" not in output.columns


def test_observed_at_field_is_independent_of_the_effective_date() -> None:
    output = _run(options={"observed_at_field": "seen"})

    assert _by_document(output, "observed_at") == {
        "d1": datetime(2024, 1, 2, tzinfo=UTC),
        "d2": datetime(2025, 6, 3, tzinfo=UTC),
    }
    assert not {"valid_from", "valid_to"} & set(output.columns)


def test_a_date_column_reads_as_midnight_utc_and_an_offset_is_normalized() -> None:
    # 12:00 in Berlin on 5 March is 11:00 UTC (CET, UTC+1).
    berlin_noon = (
        pl.Series("seen", [datetime(2024, 3, 5, 12)] * 4)
        .dt.replace_time_zone("Europe/Berlin")
    )
    mentions = _MENTIONS.with_columns(
        pl.Series("filed", [date(2024, 3, 5)] * 2 + [date(2025, 1, 9)] * 2),
        berlin_noon,
    )
    output = _run(mentions, {"effective_date_field": "filed", "observed_at_field": "seen"})

    assert _by_document(output, "valid_from") == {
        "d1": datetime(2024, 3, 5, tzinfo=UTC),
        "d2": datetime(2025, 1, 9, tzinfo=UTC),
    }
    assert set(output["observed_at"]) == {datetime(2024, 3, 5, 11, tzinfo=UTC)}
    assert output.schema["observed_at"] == _UTC_US


def test_a_naive_datetime_column_is_read_as_utc() -> None:
    naive = _MENTIONS.with_columns(pl.col("filed").dt.replace_time_zone(None))

    output = _run(naive, {"effective_date_field": "filed"})

    assert _by_document(output, "valid_from")["d1"] == datetime(2024, 1, 1, tzinfo=UTC)


def test_a_document_with_no_date_on_any_mention_gets_a_null_valid_from() -> None:
    partial = _MENTIONS.with_columns(
        pl.when(pl.col("document_id") == "d2")
        .then(None)
        .otherwise(pl.col("filed"))
        .alias("filed")
    )

    output = _run(partial, {"effective_date_field": "filed"})

    values = _by_document(output, "valid_from")
    assert values["d1"] == datetime(2024, 1, 1, tzinfo=UTC)
    assert values["d2"] is None


def test_a_null_on_one_mention_is_filled_by_its_documents_other_mentions() -> None:
    """A join upstream can leave the date on only some of a document's rows."""
    holey = _MENTIONS.with_columns(
        pl.when(pl.col("entity_id") == "m1")
        .then(None)
        .otherwise(pl.col("filed"))
        .alias("filed")
    )

    output = _run(holey, {"effective_date_field": "filed"})

    assert _by_document(output, "valid_from")["d1"] == datetime(2024, 1, 1, tzinfo=UTC)


def test_two_different_dates_in_one_document_is_an_error() -> None:
    """Which one is the document's effective date is not for stel to pick; the
    message names the column and the document, not a date's content."""
    split = _MENTIONS.with_columns(
        pl.when(pl.col("entity_id") == "m2")
        .then(datetime(2030, 1, 1, tzinfo=UTC))
        .otherwise(pl.col("filed"))
        .alias("filed")
    )

    with pytest.raises(ValueError, match=r"'filed'.*'d1'"):
        _run(split, {"effective_date_field": "filed"})


def test_a_date_column_the_mentions_lack_is_named_in_the_error() -> None:
    with pytest.raises(ValueError, match="nonexistent"):
        _run(options={"effective_date_field": "nonexistent"})


def test_a_text_date_column_is_rejected_rather_than_parsed() -> None:
    """A string could be 03/04 in either order; a wrong guess is a wrong as-of
    answer that looks right."""
    text = _MENTIONS.with_columns(pl.lit("2024-01-01").alias("filed"))

    with pytest.raises(ValueError, match=r"'filed'.*date or datetime"):
        _run(text, {"effective_date_field": "filed"})


def test_naming_a_date_does_not_change_relation_ids() -> None:
    """Identity must not depend on validity, or naming a date would re-key every
    published row (ALE-302: byte-identical rows must republish as such)."""
    plain = _run().sort("relation_id")["relation_id"].to_list()
    dated = (
        _run(options={"effective_date_field": "filed", "observed_at_field": "seen"})
        .sort("relation_id")["relation_id"]
        .to_list()
    )

    assert plain == dated


def test_stamping_is_deterministic_across_runs_and_input_order() -> None:
    options: dict[str, object] = {"effective_date_field": "filed", "observed_at_field": "seen"}

    first = _run(options=options).sort("relation_id")
    again = _run(_MENTIONS.reverse(), options).sort("relation_id")

    assert first.equals(again)


def test_an_empty_mentions_table_keeps_the_columns_the_options_ask_for() -> None:
    both = _run(
        _MENTIONS.clear(), {"effective_date_field": "filed", "observed_at_field": "seen"}
    )
    neither = _run(_MENTIONS.clear())

    assert both.height == 0
    assert {name: both.schema[name] for name in _VALIDITY_COLUMNS} == dict.fromkeys(
        _VALIDITY_COLUMNS, _UTC_US
    )
    assert not _VALIDITY_COLUMNS & set(neither.columns)


@pytest.mark.parametrize("field", ["effective_date_field", "observed_at_field"])
def test_a_blank_date_field_name_is_rejected_at_compile(field: str) -> None:
    with pytest.raises(ValidationError, match="must not be empty"):
        extract_relations.validate_options({"mentions": "m", field: "  "})


# --- `functional` is declared on the relation type ----------------------------


def _declared(*relations: dict[str, Any]) -> ProjectConfig:
    return ProjectConfig.model_validate(
        {"name": "p", "classes": ["org", "gpe"], "relations": list(relations)}
    )


def test_a_relation_is_not_functional_unless_declared() -> None:
    project = _declared({"name": "hq_in", "subject_class": "org", "object_class": "gpe"})

    assert project.relations[0].functional is False


def test_pairings_of_one_relation_must_agree_on_functional() -> None:
    """`functional` is a property of the relation type, not of one pairing, so a
    polymorphic relation cannot be functional for one subject class only."""
    with pytest.raises(ValueError, match=r"'hq_in'.*functional"):
        _declared(
            {"name": "hq_in", "subject_class": "org", "object_class": "gpe", "functional": True},
            {"name": "hq_in", "subject_class": "gpe", "object_class": "gpe"},
        )


def test_agreeing_pairings_may_all_be_functional() -> None:
    project = _declared(
        {"name": "hq_in", "subject_class": "org", "object_class": "gpe", "functional": True},
        {"name": "hq_in", "subject_class": "gpe", "object_class": "gpe", "functional": True},
    )

    assert {relation.functional for relation in project.relations} == {True}


def test_functional_must_be_a_real_boolean() -> None:
    with pytest.raises(ValidationError, match="functional"):
        RelationTypeDef.model_validate(
            {"name": "r", "subject_class": "a", "object_class": "b", "functional": "yes"}
        )


# --- compile-time checks, before any source or warehouse is touched -----------


def _example(tmp_path: Path) -> Path:
    project_dir = tmp_path / "project"
    shutil.copytree(
        Path(__file__).resolve().parents[1] / "examples" / "economic_nlp",
        project_dir,
        ignore=shutil.ignore_patterns("target"),
    )
    return project_dir


def _declare(project_dir: Path, *, functional: dict[str, bool]) -> None:
    """The example's typed rules, plus `co_occurs_with`, declared as a closed set."""
    names = {
        "references_geography": ("ORG", "GPE"),
        "references_amount": ("ORG", "MONEY"),
        "references_topic": ("ORG", "TOPIC"),
        "co_occurs_with": ("ORG", "GPE"),
    }
    lines = ["", "classes: [ORG, GPE, MONEY, TOPIC]", "relations:"]
    for name, (subject, obj) in names.items():
        lines += [f"  - name: {name}", f"    subject_class: {subject}", f"    object_class: {obj}"]
        if functional.get(name):
            lines.append("    functional: true")
    project_file = project_dir / "stel_project.yml"
    project_file.write_text(
        project_file.read_text(encoding="utf-8") + "\n".join(lines) + "\n", encoding="utf-8"
    )


_SUPERSEDE_MODEL = """\
version: 2

models:
  - name: current_relations
    description: "Typed relations with replaced facts closed"
    depends_on: [ref('document_typed_relations'), ref('entity_links')]
    transform:
      type: python
      module: stel.text.transforms.supersede_relations
      options:
        relations: document_typed_relations
        entity_links: entity_links
    materialization: full
"""


def _compile(project_dir: Path) -> None:
    project, sources, models = load_project(project_dir)
    validate_project_contract(project, sources, models, project_dir)


def test_an_undirected_extractor_cannot_assert_a_functional_relation(tmp_path: Path) -> None:
    """Co-occurrence has no subject and object, so "one current object per
    subject" has no meaning for it; refuse at compile, not after a run."""
    project_dir = _example(tmp_path)
    _declare(project_dir, functional={"co_occurs_with": True})

    with pytest.raises(
        ConfigError, match=r"document_relations.*undirected.*co_occurs_with.*functional"
    ):
        _compile(project_dir)


def test_a_supersession_model_needs_a_functional_relation_to_act_on(tmp_path: Path) -> None:
    project_dir = _example(tmp_path)
    (project_dir / "models" / "current_relations.yml").write_text(
        _SUPERSEDE_MODEL, encoding="utf-8"
    )
    _declare(project_dir, functional={})

    with pytest.raises(ConfigError, match=r"current_relations.*functional: true"):
        _compile(project_dir)


def test_a_supersession_model_compiles_beside_a_functional_relation(tmp_path: Path) -> None:
    project_dir = _example(tmp_path)
    (project_dir / "models" / "current_relations.yml").write_text(
        _SUPERSEDE_MODEL, encoding="utf-8"
    )
    _declare(project_dir, functional={"references_geography": True})

    _compile(project_dir)


def test_a_functional_declaration_alone_changes_nothing_for_other_models(
    tmp_path: Path,
) -> None:
    project_dir = _example(tmp_path)
    _declare(project_dir, functional={"references_geography": True})

    _compile(project_dir)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({}, "relations"),
        ({"relations": "r"}, "entity_links"),
        ({"relations": "r", "entity_links": "l", "unexpected": 1}, "Extra inputs"),
        ({"relations": " ", "entity_links": "l"}, "must not be empty"),
        ({"relations": "same", "entity_links": "same"}, "different models"),
    ],
)
def test_supersession_options_are_strict(options: dict[str, object], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        supersede_relations.validate_options(options)


def test_supersession_declares_both_inputs_and_no_incremental_contract() -> None:
    """Closing a row because another document arrived writes outside the changed
    document's parent key, so the step is a full rewrite (ADR-0056)."""
    options = {"relations": "r", "entity_links": "l"}

    assert supersede_relations.declared_dependencies(options) == ("r", "l")
    assert not hasattr(supersede_relations, "declared_incremental_contract")


# --- supersession: a later, different fact closes a functional relation -------

HQ = "hq_in"


def _at(year: int, month: int = 1) -> datetime:
    return datetime(year, month, 1, tzinfo=UTC)


def _relation(
    relation_id: str,
    subject: str,
    obj: str,
    valid_from: datetime | None,
    *,
    relation_type: str = HQ,
    directed: bool = True,
    status: str = "asserted",
    valid_to: datetime | None = None,
    document_id: str | None = None,
) -> dict[str, Any]:
    return {
        "relation_id": relation_id,
        "document_id": document_id or f"doc_{relation_id}",
        "subject_mention_id": subject,
        "object_mention_id": obj,
        "relation_type": relation_type,
        "directed": directed,
        "status": status,
        "valid_from": valid_from,
        "valid_to": valid_to,
        "observed_at": None,
    }


def _relations(*rows: dict[str, Any]) -> pl.DataFrame:
    schema = {
        "relation_id": pl.String,
        "document_id": pl.String,
        "subject_mention_id": pl.String,
        "object_mention_id": pl.String,
        "relation_type": pl.String,
        "directed": pl.Boolean,
        "status": pl.String,
        "valid_from": _UTC_US,
        "valid_to": _UTC_US,
        "observed_at": _UTC_US,
    }
    return pl.DataFrame(list(rows), schema=schema)


def _links(*entries: tuple[str, str, str, str]) -> pl.DataFrame:
    """(mention_id, namespace, canonical_id, status)."""
    return pl.DataFrame(
        list(entries),
        schema={
            "mention_id": pl.String,
            "entity_namespace": pl.String,
            "canonical_id": pl.String,
            "status": pl.String,
        },
        orient="row",
    )


# Mentions `acme_*` all resolve to one company, `paris`/`paris2` to one city,
# `berlin` to another: different surface forms, same entity.
_LINKS = _links(
    ("acme_1", "co", "ACME", "matched"),
    ("acme_2", "co", "ACME", "matched"),
    ("acme_3", "co", "ACME", "matched"),
    ("beta_1", "co", "BETA", "matched"),
    ("paris", "geo", "PARIS", "matched"),
    ("paris2", "geo", "PARIS", "matched"),
    ("berlin", "geo", "BERLIN", "matched"),
    ("rome", "geo", "ROME", "matched"),
)


def _supersede(*rows: dict[str, Any], links: pl.DataFrame = _LINKS) -> SupersessionResult:
    return supersede(_relations(*rows), links, functional=frozenset({HQ}))


def _intervals(result: SupersessionResult) -> dict[str, tuple[Any, Any]]:
    frame = result.frame
    return {
        rid: (closed, closer)
        for rid, closed, closer in zip(
            frame["relation_id"], frame["valid_to"], frame["superseded_by"], strict=True
        )
    }


def test_a_later_conflicting_fact_closes_the_older_one_at_its_valid_from() -> None:
    """The issue's done-when: two dated documents asserting conflicting values
    for a functional relation yield two rows, the older closed at the newer's
    `valid_from`. The two documents name the company through different
    mentions, so identity has to come from the link table."""
    result = _supersede(
        _relation("old", "acme_1", "paris", _at(2020)),
        _relation("new", "acme_2", "berlin", _at(2023)),
    )

    assert result.frame.height == 2
    assert _intervals(result) == {
        "old": (_at(2023), "new"),
        "new": (None, None),
    }
    assert result.closed == 1


def test_restating_the_same_object_closes_nothing() -> None:
    """Two documents agreeing is corroboration, not change. A different mention
    of the same city is still the same object."""
    result = _supersede(
        _relation("a", "acme_1", "paris", _at(2020)),
        _relation("b", "acme_2", "paris2", _at(2022)),
    )

    assert _intervals(result) == {"a": (None, None), "b": (None, None)}
    assert result.closed == 0


def test_a_chain_closes_each_fact_at_the_next_different_one() -> None:
    result = _supersede(
        _relation("a", "acme_1", "paris", _at(2020)),
        _relation("b", "acme_1", "berlin", _at(2021)),
        _relation("c", "acme_1", "paris", _at(2022)),
    )

    assert _intervals(result) == {
        "a": (_at(2021), "b"),
        "b": (_at(2022), "c"),
        "c": (None, None),
    }


def test_every_restatement_before_a_change_is_closed_by_that_change() -> None:
    result = _supersede(
        _relation("a", "acme_1", "paris", _at(2020)),
        _relation("a2", "acme_2", "paris", _at(2021)),
        _relation("b", "acme_3", "berlin", _at(2022)),
    )

    assert _intervals(result) == {
        "a": (_at(2022), "b"),
        "a2": (_at(2022), "b"),
        "b": (None, None),
    }


def test_a_row_is_closed_by_the_earliest_later_different_fact_not_the_latest() -> None:
    result = _supersede(
        _relation("a", "acme_1", "paris", _at(2020)),
        _relation("b", "acme_1", "paris", _at(2021)),
        _relation("c", "acme_1", "berlin", _at(2023)),
        _relation("d", "acme_1", "rome", _at(2025)),
    )

    assert _intervals(result)["a"] == (_at(2023), "c")
    assert _intervals(result)["b"] == (_at(2023), "c")
    assert _intervals(result)["c"] == (_at(2025), "d")


def test_different_facts_with_the_same_valid_from_stay_open_together() -> None:
    """Nothing orders them, so neither is entitled to close the other; the
    count is how an operator learns it happened (a document asserting two
    objects at once is the usual cause)."""
    result = _supersede(
        _relation("a", "acme_1", "paris", _at(2020), document_id="d1"),
        _relation("b", "acme_1", "berlin", _at(2020), document_id="d1"),
    )

    assert _intervals(result) == {"a": (None, None), "b": (None, None)}
    assert result.simultaneous_conflicts == 1


def test_a_simultaneous_conflict_does_not_stop_a_later_fact_closing_both() -> None:
    result = _supersede(
        _relation("a", "acme_1", "paris", _at(2020)),
        _relation("b", "acme_1", "berlin", _at(2020)),
        _relation("c", "acme_1", "rome", _at(2024)),
    )

    assert _intervals(result)["a"] == (_at(2024), "c")
    assert _intervals(result)["b"] == (_at(2024), "c")


def test_the_closer_among_simultaneous_candidates_is_the_lowest_relation_id() -> None:
    """Deterministic whichever order rows arrive in."""
    result = _supersede(
        _relation("z", "acme_1", "berlin", _at(2024)),
        _relation("m", "acme_1", "rome", _at(2024)),
        _relation("old", "acme_1", "paris", _at(2020)),
    )

    assert _intervals(result)["old"] == (_at(2024), "m")


def test_a_fact_is_not_closed_by_a_simultaneous_fact_with_its_own_object() -> None:
    result = _supersede(
        _relation("a", "acme_1", "paris", _at(2020)),
        _relation("b", "acme_1", "paris", _at(2024)),
        _relation("c", "acme_1", "berlin", _at(2024)),
    )

    assert _intervals(result)["a"] == (_at(2024), "c")


def test_a_non_functional_relation_is_untouched_beyond_the_new_column() -> None:
    result = supersede(
        _relations(
            _relation("a", "acme_1", "paris", _at(2020), relation_type="serves"),
            _relation("b", "acme_1", "berlin", _at(2023), relation_type="serves"),
        ),
        _LINKS,
        functional=frozenset({HQ}),
    )

    assert _intervals(result) == {"a": (None, None), "b": (None, None)}
    assert result.frame.columns == [*_relations().columns, "superseded_by"]


def test_subjects_and_relation_types_do_not_interact() -> None:
    result = supersede(
        _relations(
            _relation("a", "acme_1", "paris", _at(2020)),
            _relation("b", "beta_1", "berlin", _at(2023)),
            _relation("c", "acme_1", "berlin", _at(2023), relation_type="serves"),
        ),
        _LINKS,
        functional=frozenset({HQ, "serves"}),
    )

    assert _intervals(result) == {"a": (None, None), "b": (None, None), "c": (None, None)}


def test_the_same_canonical_id_in_two_namespaces_is_two_entities() -> None:
    links = _links(
        ("s1", "ns_a", "X", "matched"),
        ("s2", "ns_b", "X", "matched"),
        ("o1", "geo", "PARIS", "matched"),
        ("o2", "geo", "BERLIN", "matched"),
    )

    result = _supersede(
        _relation("a", "s1", "o1", _at(2020)),
        _relation("b", "s2", "o2", _at(2023)),
        links=links,
    )

    assert _intervals(result) == {"a": (None, None), "b": (None, None)}


def test_rows_that_are_not_asserted_neither_close_nor_are_closed() -> None:
    result = _supersede(
        _relation("a", "acme_1", "paris", _at(2020)),
        _relation("maybe", "acme_1", "berlin", _at(2023), status="ambiguous"),
    )

    assert _intervals(result) == {"a": (None, None), "maybe": (None, None)}
    assert result.skipped == {"not_asserted": 1, "unlinked_mention": 0}


def test_a_row_with_an_unlinked_end_passes_through_and_is_counted() -> None:
    """With no entity at one end there is nothing to compare, so the row cannot
    be a conflict; it is also never silently dropped."""
    result = _supersede(
        _relation("a", "acme_1", "paris", _at(2020)),
        _relation("no_subject", "stranger", "berlin", _at(2023)),
        _relation("no_object", "acme_1", "nowhere", _at(2024)),
    )

    assert _intervals(result) == {
        "a": (None, None),
        "no_subject": (None, None),
        "no_object": (None, None),
    }
    assert result.skipped == {"not_asserted": 0, "unlinked_mention": 2}


def test_an_ambiguous_link_names_no_entity() -> None:
    links = _links(
        ("acme_1", "co", "ACME", "matched"),
        ("paris", "geo", "PARIS", "matched"),
        ("berlin", "geo", "BERLIN", "ambiguous"),
    )

    result = _supersede(
        _relation("a", "acme_1", "paris", _at(2020)),
        _relation("b", "acme_1", "berlin", _at(2023)),
        links=links,
    )

    assert _intervals(result)["a"] == (None, None)
    assert result.skipped["unlinked_mention"] == 1


def test_no_row_is_ever_deleted_or_reordered_by_input_order() -> None:
    rows = [
        _relation("a", "acme_1", "paris", _at(2020)),
        _relation("b", "acme_1", "berlin", _at(2021)),
        _relation("c", "beta_1", "rome", _at(2022)),
    ]

    forward = supersede(_relations(*rows), _LINKS, functional=frozenset({HQ}))
    backward = supersede(_relations(*reversed(rows)), _LINKS, functional=frozenset({HQ}))

    assert sorted(forward.frame["relation_id"]) == ["a", "b", "c"]
    assert forward.frame.equals(backward.frame)
    assert forward.frame["relation_id"].to_list() == ["a", "b", "c"]


def test_a_valid_to_already_on_an_open_row_is_kept() -> None:
    result = _supersede(_relation("a", "acme_1", "paris", _at(2020), valid_to=_at(2030)))

    assert _intervals(result)["a"] == (_at(2030), None)


def test_a_functional_row_with_no_valid_from_is_an_error() -> None:
    """Order is the whole rule; a row that cannot be ordered cannot be left to
    look current. The message names the type and a count, not a value."""
    with pytest.raises(ValueError, match=r"'hq_in'.*1 .*valid_from.*effective_date_field"):
        _supersede(
            _relation("a", "acme_1", "paris", _at(2020)),
            _relation("undated", "acme_1", "berlin", None),
        )


def test_a_row_that_does_not_take_part_may_lack_a_valid_from() -> None:
    result = _supersede(
        _relation("a", "acme_1", "paris", _at(2020)),
        _relation("undated_but_unlinked", "stranger", "berlin", None),
        _relation("undated_but_unasserted", "acme_1", "berlin", None, status="ambiguous"),
    )

    assert result.closed == 0


def test_an_undirected_row_of_a_functional_type_is_an_error() -> None:
    with pytest.raises(ValueError, match=r"'hq_in'.*undirected"):
        _supersede(_relation("a", "acme_1", "paris", _at(2020), directed=False))


def test_a_mention_matched_to_two_entities_is_an_error() -> None:
    links = _links(
        ("acme_1", "co", "ACME", "matched"),
        ("acme_1", "co", "OTHER", "matched"),
        ("paris", "geo", "PARIS", "matched"),
    )

    with pytest.raises(ValueError, match=r"mention 'acme_1'.*two different entities"):
        _supersede(_relation("a", "acme_1", "paris", _at(2020)), links=links)


def test_a_repeated_relation_id_is_an_error() -> None:
    with pytest.raises(ValueError, match=r"'dup'.*unique"):
        _supersede(
            _relation("dup", "acme_1", "paris", _at(2020)),
            _relation("dup", "acme_1", "berlin", _at(2023)),
        )


@pytest.mark.parametrize(
    ("drop_from", "column", "hint"),
    [
        ("relations", "valid_from", "extract_relations"),
        ("relations", "valid_to", "extract_relations"),
        ("relations", "status", "extract_relations"),
        ("links", "canonical_id", "link_entities"),
        ("links", "status", "link_entities"),
    ],
)
def test_a_missing_input_column_names_the_column_and_the_transform(
    drop_from: str, column: str, hint: str
) -> None:
    relations = _relations(_relation("a", "acme_1", "paris", _at(2020)))
    links = _LINKS
    if drop_from == "relations":
        relations = relations.drop(column)
    else:
        links = links.drop(column)

    with pytest.raises(ValueError, match=rf"{column}.*{hint}"):
        supersede(relations, links, functional=frozenset({HQ}))


def test_timestamps_in_other_zones_are_compared_as_instants() -> None:
    """9:00 on 2 Jan in UTC+14 is 19:00 UTC on 1 Jan, so it is *earlier* than a
    fact valid from 23:00 UTC on 1 Jan, whatever the wall clock says."""
    kiritimati = datetime(2024, 1, 2, 9, tzinfo=timezone(timedelta(hours=14)))
    relations = _relations(
        _relation("early", "acme_1", "paris", kiritimati),
        _relation("late", "acme_1", "berlin", datetime(2024, 1, 1, 23, tzinfo=UTC)),
    )
    assert relations.schema["valid_from"] == _UTC_US

    result = supersede(relations, _LINKS, functional=frozenset({HQ}))

    assert _intervals(result)["early"] == (datetime(2024, 1, 1, 23, tzinfo=UTC), "late")
    assert _intervals(result)["late"] == (None, None)


def test_a_naive_valid_from_column_is_read_as_utc() -> None:
    relations = _relations(
        _relation("early", "acme_1", "paris", _at(2020)),
        _relation("late", "acme_1", "berlin", _at(2023)),
    ).with_columns(pl.col("valid_from").dt.replace_time_zone(None))

    result = supersede(relations, _LINKS, functional=frozenset({HQ}))

    assert _intervals(result)["early"] == (_at(2023), "late")
    assert result.frame.schema["valid_from"] == _UTC_US


# --- the as-of read ------------------------------------------------------------


def _timeline() -> pl.DataFrame:
    return supersede(
        _relations(
            _relation("a", "acme_1", "paris", _at(2020)),
            _relation("b", "acme_1", "berlin", _at(2022)),
            _relation("c", "acme_1", "rome", _at(2024)),
        ),
        _LINKS,
        functional=frozenset({HQ}),
    ).frame


def _ids_at(frame: pl.DataFrame, instant: datetime, *, unknown: bool = False) -> list[str]:
    return sorted(
        relations_as_of(frame, instant, include_unknown_validity=unknown)["relation_id"]
    )


def test_as_of_returns_only_the_fact_valid_at_that_instant() -> None:
    frame = _timeline()

    assert _ids_at(frame, _at(2021)) == ["a"]
    assert _ids_at(frame, _at(2023)) == ["b"]
    assert _ids_at(frame, _at(2030)) == ["c"]


def test_as_of_is_half_open_so_only_the_newer_fact_holds_at_the_boundary() -> None:
    frame = _timeline()

    assert _ids_at(frame, _at(2022)) == ["b"]
    assert _ids_at(frame, datetime(2021, 12, 31, 23, 59, 59, tzinfo=UTC)) == ["a"]


def test_as_of_before_a_fact_begins_returns_nothing() -> None:
    assert _ids_at(_timeline(), _at(2019)) == []


def test_as_of_leaves_unknown_validity_out_unless_asked() -> None:
    """A row with no `valid_from` does not claim to hold at any instant."""
    frame = _relations(
        _relation("known", "acme_1", "paris", _at(2020)),
        _relation("unknown", "acme_1", "berlin", None),
    )

    assert _ids_at(frame, _at(2021)) == ["known"]
    assert _ids_at(frame, _at(2021), unknown=True) == ["known", "unknown"]


def test_as_of_does_not_filter_on_status() -> None:
    frame = _relations(_relation("a", "acme_1", "paris", _at(2020), status="ambiguous"))

    assert _ids_at(frame, _at(2021)) == ["a"]


def test_as_of_requires_a_timezone_aware_instant() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        relations_as_of(_timeline(), datetime(2021, 1, 1), include_unknown_validity=False)


def test_as_of_names_the_missing_column() -> None:
    with pytest.raises(ValueError, match="valid_to"):
        relations_as_of(
            _timeline().drop("valid_to"), _at(2021), include_unknown_validity=False
        )


def test_as_of_returns_the_frame_unchanged_in_shape() -> None:
    frame = _timeline()

    selected = relations_as_of(frame, _at(2023), include_unknown_validity=False)

    assert selected.columns == frame.columns
    assert selected.schema == frame.schema


# --- the transform wraps the pure step ----------------------------------------


def _supersede_ctx(
    functional: tuple[str, ...] = (HQ,), options: dict[str, object] | None = None
) -> TransformContext:
    return TransformContext(
        project_dir=Path("."),
        profile_name="test",
        target_name="dev",
        warehouse=parse_warehouse_config(
            {"type": "duckdb", "path": "./test.duckdb", "schema": "main"}
        ),
        llm=None,
        options=options or {"relations": "relations", "entity_links": "links"},
        relations=(
            *(
                RelationTypeDef(
                    name=name, subject_class="company", object_class="city", functional=True
                )
                for name in functional
            ),
            RelationTypeDef(name="serves", subject_class="company", object_class="city"),
        ),
    )


def _transform_deps() -> dict[str, pl.DataFrame]:
    return {
        "relations": _relations(
            _relation("old", "acme_1", "paris", _at(2020)),
            _relation("new", "acme_2", "berlin", _at(2023)),
            _relation("s1", "acme_1", "paris", _at(2020), relation_type="serves"),
            _relation("s2", "acme_1", "berlin", _at(2023), relation_type="serves"),
        ),
        "links": _LINKS,
    }


def test_the_transform_acts_on_the_relations_the_project_declares_functional() -> None:
    output = supersede_relations.run(_transform_deps(), _supersede_ctx())

    closed = dict(zip(output["relation_id"], output["valid_to"], strict=True))
    assert closed["old"] == _at(2023)
    # `serves` is declared without `functional`, so two different objects coexist.
    assert closed["s1"] is None
    assert closed["new"] is None


def test_the_transform_reads_which_types_are_functional_from_the_project_not_the_data() -> None:
    output = supersede_relations.run(_transform_deps(), _supersede_ctx(functional=(HQ, "serves")))

    closed = dict(zip(output["relation_id"], output["valid_to"], strict=True))
    assert closed["s1"] == _at(2023)


def test_the_transform_refuses_to_run_with_nothing_declared_functional() -> None:
    with pytest.raises(ValueError, match=r"no relation to act on.*functional: true"):
        supersede_relations.run(_transform_deps(), _supersede_ctx(functional=()))


def test_the_transform_refuses_dependencies_other_than_the_two_it_names() -> None:
    deps = _transform_deps()
    deps["extra"] = pl.DataFrame()

    with pytest.raises(ValueError, match=r"expects the models.*relations.*entity_links"):
        supersede_relations.run(deps, _supersede_ctx())


def test_the_transform_version_is_part_of_its_code_identity() -> None:
    """A built-in module has no project file to hash, so this is what re-keys a
    model whose closing rule changed (same reason `extract_relations` reports
    its extractor's version)."""
    assert supersede_relations.code_version_identity({"relations": "r", "entity_links": "l"}) == {
        "supersession_version": SUPERSESSION_VERSION
    }


def test_the_run_log_counts_what_the_step_left_out(caplog: pytest.LogCaptureFixture) -> None:
    """The reference docs promise the counts are in the run log, because a short
    count of closures is otherwise indistinguishable from a quiet corpus."""
    deps = {
        "relations": _relations(
            _relation("a", "acme_1", "paris", _at(2020)),
            _relation("b", "acme_1", "berlin", _at(2020)),
            _relation("maybe", "acme_1", "rome", _at(2022), status="ambiguous"),
            _relation("lost", "stranger", "rome", _at(2022)),
        ),
        "links": _LINKS,
    }

    with caplog.at_level("INFO", logger=supersede_relations.log.name):
        supersede_relations.run(deps, _supersede_ctx())

    (message,) = (r.getMessage() for r in caplog.records if r.name == supersede_relations.log.name)
    assert "closed 0 row(s)" in message
    assert "1 not_asserted" in message
    assert "1 unlinked_mention" in message
    assert "1 simultaneous conflict(s)" in message
