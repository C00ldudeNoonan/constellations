"""Shared fixtures for the retrieval and serving tests (issue #518).

These were private helpers in `test_retrieval.py` and `test_online_publication.py`
that four other test modules imported anyway -- two of them from inside test
bodies, which hid the dependency and breaks the module-level import rule in
AGENTS.md. One file's private helper being another file's API is fragile in a
specific way: renaming a leading-underscore function is normally a free local
edit, and here it silently breaks other files.

Not a consolidation of the cluster's *projects*. `test_search.py`,
`test_serving_coordination.py` and `test_retrieval_eval.py` each define their
own `_write_project`, and those are 1-4% similar to this one -- they build
genuinely different projects for different questions, and merging them would
mean a parameterised super-fixture that is harder to read than four honest
ones. Only what was already shared moved here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from stel.adapters import create_adapter
from stel.cli_services.serving import resolve_serving_scope
from stel.config import load_project
from stel.profile import resolve_profile
from stel.runner import run_project


def write_project(tmp_path: Path, *, allow_public: bool = True) -> None:
    (tmp_path / "sources").mkdir()
    (tmp_path / "models").mkdir()
    (tmp_path / "data").mkdir()
    (tmp_path / "stel_project.yml").write_text(
        "\n".join(
            [
                "name: retrieval_demo",
                "version: '0.1.0'",
                "profile: retrieval_demo",
                "source-paths: [sources]",
                "model-paths: [models]",
            ]
        )
    )
    (tmp_path / "profiles.yml").write_text(
        "\n".join(
            [
                "retrieval_demo:",
                "  target: dev",
                "  outputs:",
                "    dev:",
                "      warehouse:",
                "        type: duckdb",
                "        path: target/demo.duckdb",
                "        schema: analytics",
                "      retrieval:",
                "        default: primary",
                f"        allow_public_indexes: {str(allow_public).lower()}",
                "        stores:",
                "          primary:",
                "            type: lancedb",
                "            path: target/lancedb",
                "            collection_template: '{project}__{target}__{collection}'",
            ]
        )
    )
    (tmp_path / "sources" / "documents.yml").write_text(
        "\n".join(
            [
                "version: 2",
                "sources:",
                "  - name: documents",
                "    path: data",
                "    file_pattern: '*.json'",
            ]
        )
    )
    (tmp_path / "models" / "retrieval.yml").write_text(
        "\n".join(
            [
                "version: 2",
                "models:",
                "  - name: embedding_rows",
                "    source: ref('documents')",
                "    extraction:",
                "      backend: json",
                "  - name: context_search",
                "    depends_on: [ref('embedding_rows')]",
                "    materialization: incremental",
                "    tags: [retrieval, economic-data]",
                "    search:",
                "      access: public",
                "      store: primary",
                "      collection: context",
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
                "        embedding:",
                "          provider: fixture",
                "          model: deterministic-2d-v1",
                "          provider_contract_version: 2",
                "          provider_implementation: tests:v1",
                "          semantic_config_fingerprint: deterministic-2d-v1",
                "          dimensions: 2",
                "      full_text:",
                "        fields: [text]",
                "      attributes:",
                "        - name: category",
                "          data_type: string",
                "          filter_role: user",
                "          returned: true",
                "      display_fields: [title]",
                "      query:",
                "        modes: [vector, text, filter]",
                "        consistency: strong",
                "      on_index_change: fail",
                "      batch_size: 2",
            ]
        )
    )


def sample_rows(version: int = 1) -> pl.DataFrame:
    if version == 1:
        return pl.DataFrame(
            {
                "chunk_id": ["c1", "c2"],
                "document_id": ["d1", "d2"],
                "text": ["inflation slowed", "employment increased"],
                "embedding": [[1.0, 0.0], [0.0, 1.0]],
                "category": ["prices", "labor"],
                "title": ["CPI", "Payrolls"],
            }
        )
    return pl.DataFrame(
        {
            "chunk_id": ["c1", "c3"],
            "document_id": ["d1", "d3"],
            "text": ["inflation declined", "output expanded"],
            "embedding": [[0.9, 0.1], [0.5, 0.5]],
            "category": ["prices", "growth"],
            "title": ["CPI revision", "GDP"],
        }
    )


def materialize_upstream(tmp_path: Path, rows: pl.DataFrame) -> None:
    project, _, _ = load_project(tmp_path)
    resolved = resolve_profile(project, tmp_path)
    with create_adapter(resolved.warehouse, project_dir=tmp_path) as adapter:
        adapter.materialize_full("embedding_rows", rows)


def set_index_change_policy(tmp_path: Path, policy: str) -> None:
    path = tmp_path / "models" / "context_search.yml"
    candidates = [path] if path.exists() else list((tmp_path / "models").glob("*.yml"))
    for candidate in candidates:
        text = candidate.read_text()
        if "on_index_change: fail" in text:
            candidate.write_text(
                text.replace("on_index_change: fail", f"on_index_change: {policy}")
            )
            return
    raise AssertionError("fixture no longer declares on_index_change")


def prepare_online_switch(project: Path) -> tuple[Any, Any]:
    write_project(project)
    set_index_change_policy(project, "online")
    materialize_upstream(project, sample_rows())
    run_project(project, select="context_search")
    path = project / "models" / "retrieval.yml"
    path.write_text(
        path.read_text(encoding="utf-8")
        .replace("search: exact", "search: approximate")
        .replace("batch_size: 2", "batch_size: 1"),
        encoding="utf-8",
    )
    return resolve_serving_scope(
        project, profiles_dir=None, target=None, model_name="context_search"
    )
