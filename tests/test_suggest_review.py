"""Regressions from the Codex review of #381.

Each of these is a way a well-meaning suggestion could quietly damage a dbt
project: documenting the wrong object, discarding itself into a duplicate
key, editing outside the project, or emitting malformed YAML.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from stel.suggest import SuggestionError, SuggestionRow, render_suggestion, schema_files


def _row(
    *,
    dbt_model: str = "fct_orders",
    dbt_column: str | None = None,
    suggested_description: str = "One row per order, excluding cancellations",
) -> SuggestionRow:
    return SuggestionRow(
        dbt_model=dbt_model,
        suggested_description=suggested_description,
        evidence_count=4,
        evidence_sessions=("s1", "s2", "s3", "s4"),
        dbt_column=dbt_column,
    )


def _project(tmp_path: Path, schema: str) -> Path:
    project = tmp_path / "dbt"
    (project / "models" / "marts").mkdir(parents=True)
    (project / "models" / "marts" / "schema.yml").write_text(schema, encoding="utf-8")
    return project


# ─── documenting the wrong object ───────────────────────────────────────────


def test_a_source_sharing_the_model_name_is_not_documented_instead() -> None:
    """`sources:` and `models:` live in one file and a source table may share
    a model's name. Scanning the whole file for `- name: fct_orders` finds the
    source first and documents it, leaving the model undocumented while
    reporting success."""
    schema = """version: 2

sources:
  - name: raw
    tables:
      - name: fct_orders

models:
  - name: fct_orders
    columns:
      - name: order_id
"""
    updated, reason = render_suggestion(schema, _row())

    assert updated is not None, reason
    document = yaml.safe_load(updated)
    # The model got the description...
    assert document["models"][0]["description"] == (
        "One row per order, excluding cancellations"
    )
    # ...and the identically named source table did not.
    assert "description" not in document["sources"][0]["tables"][0]


def test_a_column_sharing_the_model_name_is_not_documented_instead() -> None:
    schema = """version: 2

models:
  - name: dim_customers
    columns:
      - name: fct_orders
  - name: fct_orders
    columns:
      - name: order_id
"""
    updated, reason = render_suggestion(schema, _row())

    assert updated is not None, reason
    document = yaml.safe_load(updated)
    assert "description" not in document["models"][0]["columns"][0]
    assert document["models"][1]["description"] == (
        "One row per order, excluding cancellations"
    )


# ─── discarding itself into a duplicate key ─────────────────────────────────


@pytest.mark.parametrize("existing", ['description: ""', "description:"])
def test_an_empty_description_key_is_left_alone(existing: str) -> None:
    """A falsy-but-present description must not get a second `description:`
    inserted beside it: loaders keep the existing empty value, so the
    suggestion would vanish while the command reported it applied."""
    schema = f"""version: 2

models:
  - name: fct_orders
    {existing}
"""
    updated, reason = render_suggestion(schema, _row())

    assert updated is None
    assert "already present but empty" in reason


def test_an_empty_column_description_key_is_left_alone() -> None:
    schema = """version: 2

models:
  - name: fct_orders
    columns:
      - name: order_id
        description: ""
"""
    updated, reason = render_suggestion(schema, _row(dbt_column="order_id"))

    assert updated is None
    assert "already present but empty" in reason


# ─── editing outside the project ────────────────────────────────────────────


def _symlink(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:  # pragma: no cover - platform-dependent privilege
        pytest.skip("creating symlinks requires privileges on this platform")


def test_files_reached_through_a_symlinked_directory_are_not_discovered(
    tmp_path: Path,
) -> None:
    """`rglob` follows symlinked directories, and `is_symlink()` on the file
    it yields sees only the regular destination — so without an ancestry check
    `--write` could edit outside the dbt project."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secrets.yml").write_text("version: 2\n", encoding="utf-8")
    project = _project(tmp_path, "version: 2\n\nmodels:\n  - name: fct_orders\n")
    _symlink(project / "models" / "escape", outside)

    discovered = schema_files(project)

    assert [path.name for path in discovered] == ["schema.yml"]


def test_a_symlinked_models_directory_is_refused(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    (outside / "marts").mkdir(parents=True)
    (outside / "marts" / "schema.yml").write_text("version: 2\n", encoding="utf-8")
    project = tmp_path / "dbt"
    project.mkdir()
    _symlink(project / "models", outside)

    with pytest.raises(SuggestionError, match="symlink"):
        schema_files(project)


# ─── emitting malformed YAML ────────────────────────────────────────────────


def test_an_unterminated_final_line_still_yields_valid_yaml() -> None:
    """`splitlines(keepends=True)` leaves a file with no trailing newline
    unterminated; inserting after that line would splice both onto one."""
    schema = "version: 2\n\nmodels:\n  - name: fct_orders"
    assert not schema.endswith("\n")

    updated, reason = render_suggestion(schema, _row())

    assert updated is not None, reason
    assert "- name: fct_orders\n" in updated
    document = yaml.safe_load(updated)
    assert document["models"][0]["description"] == (
        "One row per order, excluding cancellations"
    )
