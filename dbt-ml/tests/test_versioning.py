from __future__ import annotations

from pathlib import Path

from dbt_ml.config.model import ExtractionConfig, TransformConfig
from dbt_ml.versioning import (
    compute_code_version,
    compute_content_hash,
    compute_document_id,
    resolve_module_file,
)


def test_content_hash_changes_with_content(tmp_path: Path) -> None:
    p = tmp_path / "a.json"
    p.write_text("{}")
    h1 = compute_content_hash(p)
    p.write_text('{"x": 1}')
    h2 = compute_content_hash(p)
    assert h1 != h2


def test_document_id_includes_scope() -> None:
    a = compute_document_id("source_a", "invoice.json")
    b = compute_document_id("source_b", "invoice.json")
    assert a != b


def test_code_version_stable(tmp_path: Path) -> None:
    cfg = ExtractionConfig(backend="json", options={"fields": ["a", "b"]})
    v1 = compute_code_version(extraction=cfg, transform=None, project_dir=tmp_path)
    v2 = compute_code_version(extraction=cfg, transform=None, project_dir=tmp_path)
    assert v1 == v2


def test_code_version_changes_with_config(tmp_path: Path) -> None:
    a = compute_code_version(
        extraction=ExtractionConfig(backend="json", options={"fields": ["a"]}),
        transform=None,
        project_dir=tmp_path,
    )
    b = compute_code_version(
        extraction=ExtractionConfig(backend="json", options={"fields": ["b"]}),
        transform=None,
        project_dir=tmp_path,
    )
    assert a != b


def test_code_version_changes_with_module_contents(tmp_path: Path) -> None:
    mod_dir = tmp_path / "transforms"
    mod_dir.mkdir()
    mod = mod_dir / "summary.py"
    mod.write_text("def run(deps): return None\n")
    transform = TransformConfig(type="python", module="transforms.summary")

    v1 = compute_code_version(extraction=None, transform=transform, project_dir=tmp_path)
    mod.write_text("def run(deps): return 42\n")
    v2 = compute_code_version(extraction=None, transform=transform, project_dir=tmp_path)
    assert v1 != v2


def test_resolve_module_file_dotted(tmp_path: Path) -> None:
    assert resolve_module_file("transforms.summary", tmp_path) == (
        tmp_path / "transforms" / "summary.py"
    )


def test_code_version_ignores_flush_every(tmp_path: Path) -> None:
    """flush_every shapes execution, not output — changing it must not
    invalidate incremental state (issue #77)."""
    base = ExtractionConfig(backend="json", options={"fields": ["a"]})
    tuned = ExtractionConfig(
        backend="json", options={"fields": ["a"]}, flush_every=2
    )
    assert compute_code_version(
        extraction=base, transform=None, project_dir=tmp_path
    ) == compute_code_version(extraction=tuned, transform=None, project_dir=tmp_path)
