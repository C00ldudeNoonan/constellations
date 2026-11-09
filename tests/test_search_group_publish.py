"""A segmented search group publishes one partition per member (ADR-0037, #771).

The group is declared in `stel_project.yml` and each member carries its own
bounds, so these run the whole path a deployment takes: compile, publish through
`run_project`, then read what each member holds, what the manifest records, and
what the serving ledger says was published. The unit tests next door cover every
way a declaration can be wrong.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
import pytest

from stel.cli_services.serving import serving_status
from stel.compiler import validate_project_contract
from stel.config import ConfigError, load_project
from stel.manifest import SERVING_MANIFEST_VERSION, build_manifest
from stel.profile import resolve_profile
from stel.retrieval import StoreRole, create_store
from stel.runner import RunError, run_project
from stel.search_groups import group_availability
from tests.support_retrieval import materialize_upstream, write_project

# Runs a whole project and opens a retrieval store, so it belongs to the `e2e`
# tier (issue #518).
pytestmark = pytest.mark.e2e

_EMBEDDING = [
    "        embedding:",
    "          provider: fixture",
    "          model: deterministic-2d-v1",
    "          provider_contract_version: 2",
    "          provider_implementation: tests:v1",
    "          semantic_config_fingerprint: deterministic-2d-v1",
    "          dimensions: 2",
]


def _member(name: str, partition: str, *, on_index_change: str = "fail") -> list[str]:
    return [
        f"  - name: {name}",
        "    depends_on: [ref('embedding_rows')]",
        "    materialization: incremental",
        "    tags: [filings]",
        "    search:",
        "      access: public",
        "      store: primary",
        f"      collection: {name}",
        "      id_field: chunk_id",
        "      document_id_field: document_id",
        "      chunk_id_field: chunk_id",
        "      text_fields: [text]",
        "      return_text_fields: [text]",
        "      vector:",
        "        field: embedding",
        "        dimensions: 2",
        "        metric: cosine",
        "        search: exact",
        *_EMBEDDING,
        "      full_text:",
        "        fields: [text]",
        "      attributes:",
        "        - name: filing_year",
        "          data_type: integer",
        "          nullable: true",
        "          filter_role: user",
        "          returned: true",
        "      display_fields: [title]",
        "      query:",
        "        modes: [vector, text, filter]",
        "        consistency: strong",
        f"      on_index_change: {on_index_change}",
        "      batch_size: 2",
        f"      partition: {{field: filing_year, type: integer, {partition}}}",
    ]


def _write_group(
    project: Path,
    *,
    mid_upper: int = 2025,
    on_index_change: str = "fail",
) -> None:
    """Three year ranges with open ends and a member for the null year.

    Called again, it rewrites only the members, which is how a test moves a bound.
    """
    if not (project / "models").exists():
        write_project(project)
    members = {
        "filings_old": "upper: 2023",
        "filings_mid": f"lower: 2023, upper: {mid_upper}",
        "filings_new": f"lower: {mid_upper}",
        "filings_unfiled": "nulls: true",
    }
    lines = [
        "version: 2",
        "models:",
        "  - name: embedding_rows",
        "    source: ref('documents')",
        "    extraction:",
        "      backend: json",
    ]
    for name, partition in members.items():
        lines += _member(name, partition, on_index_change=on_index_change)
    (project / "models" / "retrieval.yml").write_text("\n".join(lines), encoding="utf-8")
    project_file = project / "stel_project.yml"
    if "search_groups:" in project_file.read_text(encoding="utf-8"):
        return
    project_file.write_text(
        project_file.read_text(encoding="utf-8")
        + "\n"
        + "\n".join(
            [
                "search_groups:",
                "  - name: filings",
                f"    members: [{', '.join(members)}]",
            ]
        ),
        encoding="utf-8",
    )


def _upstream(years: dict[str, int | None]) -> pl.DataFrame:
    ids = list(years)
    return pl.DataFrame(
        {
            "chunk_id": ids,
            "document_id": [f"d{chunk_id[1:]}" for chunk_id in ids],
            "text": [f"text of {chunk_id}" for chunk_id in ids],
            "embedding": [[1.0, 0.0] for _ in ids],
            "title": [f"title {chunk_id}" for chunk_id in ids],
            "filing_year": pl.Series([years[chunk_id] for chunk_id in ids], dtype=pl.Int64),
        }
    )


_YEARS: dict[str, int | None] = {
    "c1": 2022,
    "c2": 2023,
    "c3": 2024,
    "c4": 2025,
    "c5": 2027,
    "c6": None,
}


def _published_ids(project_dir: Path, collection: str) -> set[str]:
    project, _, _ = load_project(project_dir)
    resolved = resolve_profile(project, project_dir)
    assert resolved.retrieval is not None
    with create_store(
        resolved.retrieval.stores["primary"],
        project_name=project.name,
        target_name=resolved.target_name,
        alias="primary",
        role=StoreRole.PUBLISH,
    ) as store:
        # A rebuild publishes into a new generation and activates it, so what
        # serves is the active collection, not the unsuffixed default.
        entry = serving_status(
            project_dir, profiles_dir=None, target=None, model_name=collection
        ).entry
        name = entry.active_collection or store.physical_collection(collection)
        return {
            record_id
            for page in store.iter_record_ids(name, id_field="chunk_id", page_size=10)
            for record_id in page
        }


def _publish(project: Path) -> None:
    run_project(project, select="tag:filings")


def test_each_member_publishes_only_its_own_partition(tmp_path: Path) -> None:
    """Together the members hold the corpus exactly once: nothing lost to a gap,
    nothing served twice, and the null year is not dropped by a range."""
    _write_group(tmp_path)
    materialize_upstream(tmp_path, _upstream(_YEARS))

    _publish(tmp_path)

    assert _published_ids(tmp_path, "filings_old") == {"c1"}
    assert _published_ids(tmp_path, "filings_mid") == {"c2", "c3"}
    assert _published_ids(tmp_path, "filings_new") == {"c4", "c5"}
    assert _published_ids(tmp_path, "filings_unfiled") == {"c6"}


def test_a_row_re_dated_into_another_member_moves_with_it(tmp_path: Path) -> None:
    """c3 is re-dated from 2024 into 2027: it leaves `filings_mid` for
    `filings_new` and is not left behind in the collection it left (#764)."""
    _write_group(tmp_path)
    materialize_upstream(tmp_path, _upstream(_YEARS))
    _publish(tmp_path)

    materialize_upstream(tmp_path, _upstream({**_YEARS, "c3": 2027}))
    _publish(tmp_path)

    assert _published_ids(tmp_path, "filings_mid") == {"c2"}
    assert _published_ids(tmp_path, "filings_new") == {"c3", "c4", "c5"}


def test_the_manifest_records_the_group_and_each_members_bounds(tmp_path: Path) -> None:
    _write_group(tmp_path)

    manifest = build_manifest(tmp_path)

    assert manifest["manifest_version"] == SERVING_MANIFEST_VERSION
    (group,) = manifest["search_groups"]
    assert group["name"] == "filings"
    assert group["field"] == "filing_year"
    members = {member["model"]: member for member in group["members"]}
    assert members["filings_mid"]["partition"] == {
        "field": "filing_year",
        "type": "integer",
        "lower": 2023,
        "upper": 2025,
        "nulls": False,
    }
    # The member's own serving resource carries the partition too, and nothing
    # else in the manifest does: an ordinary index reads exactly as before.
    resources = {
        node["name"]: node["output"]["serving_resource"]
        for node in manifest["models"]
        if node.get("resource_type") == "search_index"
    }
    assert resources["filings_mid"]["partition"] == members["filings_mid"]["partition"]
    assert resources["filings_mid"]["config_fingerprint"] == (
        members["filings_mid"]["config_fingerprint"]
    )


def test_the_group_serves_once_every_member_has_published_under_the_manifest(
    tmp_path: Path,
) -> None:
    _write_group(tmp_path)
    materialize_upstream(tmp_path, _upstream(_YEARS))
    (group,) = build_manifest(tmp_path)["search_groups"]

    def live() -> dict[str, str | None]:
        return {
            member["model"]: serving_status(
                tmp_path, profiles_dir=None, target=None, model_name=member["model"]
            ).entry.config_fingerprint
            for member in group["members"]
        }

    assert not group_availability(group, live()).available

    run_project(tmp_path, select="filings_old filings_mid filings_new")
    partial = group_availability(group, live())
    assert not partial.available
    assert partial.reasons() == (
        "filings_unfiled: has not published a serving generation",
    )

    run_project(tmp_path, select="filings_unfiled")
    assert group_availability(group, live()).available


def _move_the_boundary(project: Path, *, on_index_change: str) -> None:
    _write_group(project, mid_upper=2026, on_index_change=on_index_change)


def test_moving_a_bound_is_refused_unless_the_member_may_rebuild(tmp_path: Path) -> None:
    """A changed bound is a membership migration, not a silent reuse: the
    collection was cut under the old bounds and `on_index_change: fail` keeps it."""
    _write_group(tmp_path)
    materialize_upstream(tmp_path, _upstream(_YEARS))
    _publish(tmp_path)
    before = build_manifest(tmp_path)["search_groups"][0]

    _move_the_boundary(tmp_path, on_index_change="fail")
    after = build_manifest(tmp_path)["search_groups"][0]
    assert after["version"] != before["version"]

    with pytest.raises(RunError):
        run_project(tmp_path, select="filings_mid")
    # The refused member is still serving what it published under the old bounds.
    assert _published_ids(tmp_path, "filings_mid") == {"c2", "c3"}


def test_a_member_that_may_rebuild_migrates_to_the_new_bounds(tmp_path: Path) -> None:
    _write_group(tmp_path, on_index_change="rebuild")
    materialize_upstream(tmp_path, _upstream(_YEARS))
    _publish(tmp_path)

    _move_the_boundary(tmp_path, on_index_change="rebuild")
    _publish(tmp_path)

    # 2025 moved from `filings_new` into `filings_mid`.
    assert _published_ids(tmp_path, "filings_mid") == {"c2", "c3", "c4"}
    assert _published_ids(tmp_path, "filings_new") == {"c5"}


def test_a_gap_is_refused_before_any_publish(tmp_path: Path) -> None:
    """Compilation names the gap; nothing is read or written first."""
    _write_group(tmp_path)
    path = tmp_path / "models" / "retrieval.yml"
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "lower: 2023, upper: 2025", "lower: 2024, upper: 2025"
        ),
        encoding="utf-8",
    )
    project, sources, models = load_project(tmp_path)

    with pytest.raises(ConfigError, match="with no member"):
        validate_project_contract(project, sources, models, tmp_path)


def _models_by_name(project: Path) -> dict[str, Any]:
    _, _, models = load_project(project)
    return {model.name: model for model in models}


def test_a_for_each_template_can_generate_the_members(tmp_path: Path) -> None:
    """`for_each` may generate members, but only the group declaration makes
    them one: its tag alone groups nothing."""
    _write_group(tmp_path)
    path = tmp_path / "models" / "retrieval.yml"
    text = path.read_text(encoding="utf-8")
    # Turn the open-ended upper member into a template over its bound.
    text = text.replace(
        "  - name: filings_old\n",
        "  - name: filings_old\n    for_each:\n      cut: [2023]\n",
    ).replace("upper: 2023}", "upper: '${matrix.cut}'}")
    path.write_text(text, encoding="utf-8")

    models = _models_by_name(tmp_path)
    expanded = models["filings_old__cut_2023"]

    assert expanded.search is not None
    assert expanded.search.partition is not None
    assert expanded.search.partition.upper == 2023
    assert "filings_old" in expanded.tags
