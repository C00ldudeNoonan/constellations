"""End-to-end proof for examples/notion_landed_pages (issue #352): landed
Notion block rows become ordered, heading-attributed documents using only
shipped primitives, and the same models run unchanged over `warehouse://`
tables — the position the docs page states."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import duckdb

from stel.checks import run_project_tests
from stel.runner import run_project

_EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "notion_landed_pages"


def _project(tmp_path: Path) -> Path:
    dst = tmp_path / "notion_landed_pages"
    shutil.copytree(_EXAMPLE, dst, ignore=shutil.ignore_patterns("target", "__pycache__"))
    return dst


def _rows(project: Path, sql: str) -> list[tuple[Any, ...]]:
    con = duckdb.connect(str(project / "target" / "stel.duckdb"), read_only=True)
    try:
        return con.execute(sql).fetchall()
    finally:
        con.close()


def _documents(project: Path) -> dict[str, str]:
    return dict(_rows(project, "select page_id, text from notion.page_documents"))


def test_example_runs_and_all_checks_pass(tmp_path: Path) -> None:
    project = _project(tmp_path)
    run_project(project)
    failed = [r for r in run_project_tests(project) if not r.passed]
    assert failed == [], [(r.test_name, r.message) for r in failed]


def test_pages_render_in_order_with_nesting_and_accounting(tmp_path: Path) -> None:
    project = _project(tmp_path)
    run_project(project)

    docs = _documents(project)
    onboarding = docs["pg_3f1c9a"]
    # Blocks were landed out of position order; the document is in reading order.
    assert onboarding.startswith("# Engineering onboarding\n\nWelcome to the team.")
    assert onboarding.index("## Before your first day") < onboarding.index("### Accounts")
    assert onboarding.index("### Accounts") < onboarding.index("## Your first week")
    # Three-deep nesting, with a paragraph under a bullet.
    assert (
        "- Pick a laptop\n  - MacBook Pro 14 (default)\n  - ThinkPad X1 for Linux\n\n"
        "    Linux laptops"
    ) in onboarding
    assert "1. Email and calendar\n2. Slack\n3. GitHub, via the SSO tile" in onboarding
    assert "```bash\ngit clone" in onboarding
    # The orphaned block is at the end, and both imperfections are counted.
    assert onboarding.endswith("Bring a government ID on day one for the badge photo.")
    counts = _rows(
        project,
        "select page_id, block_count, orphan_block_count, unknown_block_count "
        "from notion.page_documents order by page_id",
    )
    assert counts == [("pg_3f1c9a", 24, 1, 1), ("pg_7b2e04", 11, 0, 0), ("pg_c05d71", 10, 0, 0)]


def test_chunks_carry_their_section_and_the_page_attributes(tmp_path: Path) -> None:
    project = _project(tmp_path)
    run_project(project)

    rows = _rows(
        project,
        "select page_id, section, title, parent_page_id, database_id "
        "from notion.page_chunks order by page_id, chunk_index",
    )
    sections = [section for _, section, *_ in rows]
    # Text before the first page heading belongs to the page title.
    assert sections[0] == "Engineering onboarding"
    assert "Before your first day" in sections and "Your first week" in sections
    assert all(section is not None for section in sections)
    laptop = [row for row in rows if row[0] == "pg_7b2e04"]
    assert laptop and all(row[2:] == ("Laptop setup", "pg_3f1c9a", "db_runbooks") for row in laptop)


def test_an_edited_block_re_renders_only_its_page(tmp_path: Path) -> None:
    project = _project(tmp_path)
    run_project(project)
    before = _documents(project)

    block = project / "landed" / "notion_block" / "blk_b2.json"
    row = json.loads(block.read_text(encoding="utf-8"))
    row["text"] = row["text"] + " Bring the box."
    block.write_text(json.dumps(row), encoding="utf-8")
    results = {r.model_name: r for r in run_project(project)}

    assert results["page_documents"].documents_processed == 1
    assert results["page_documents"].documents_skipped == 2
    assert results["page_chunks"].documents_processed == 1
    after = _documents(project)
    assert after["pg_7b2e04"] != before["pg_7b2e04"] and "Bring the box." in after["pg_7b2e04"]
    assert {k: v for k, v in after.items() if k != "pg_7b2e04"} == {
        k: v for k, v in before.items() if k != "pg_7b2e04"
    }


def _land_fixture_as_tables(project: Path) -> None:
    """Load the JSON fixture into DuckDB tables in the project's own warehouse
    and point the sources at them — the production shape."""
    (project / "target").mkdir(exist_ok=True)
    con = duckdb.connect(str(project / "target" / "stel.duckdb"))
    try:
        con.execute("create schema landed")
        for relation, columns in (
            (
                "notion_page",
                "page_id, title, parent_page_id, database_id, properties_json, "
                "last_edited_time",
            ),
            (
                "notion_block",
                "block_id, page_id, parent_block_id, position, type, text, checked, "
                "language",
            ),
        ):
            files = str(project / "landed" / relation / "*.json").replace("\\", "/")
            con.execute(
                f"create table landed.{relation} as "
                f"select {columns} from read_json_auto('{files}', format='auto')"
            )
    finally:
        con.close()
    sources = project / "sources" / "notion.yml"
    sources.write_text(
        "version: 2\nsources:\n"
        "  - name: notion_page_rows\n    path: warehouse://landed.notion_page\n"
        "    key_column: page_id\n"
        "  - name: notion_block_rows\n    path: warehouse://landed.notion_block\n"
        "    key_column: block_id\n",
        encoding="utf-8",
    )


def test_the_same_models_run_unchanged_over_landed_warehouse_tables(tmp_path: Path) -> None:
    local = _project(tmp_path / "local")
    run_project(local)

    landed = _project(tmp_path / "landed")
    _land_fixture_as_tables(landed)
    run_project(landed)
    failed = [r for r in run_project_tests(landed) if not r.passed]
    assert failed == [], [(r.test_name, r.message) for r in failed]

    assert _documents(landed) == _documents(local)
    assert _rows(landed, "select count(*) from notion.page_chunks") == _rows(
        local, "select count(*) from notion.page_chunks"
    )
