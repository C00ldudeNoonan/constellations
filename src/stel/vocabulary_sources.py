"""Resolve external vocabulary snapshots before planning or execution (issue #718).

CSV seeds resolve at config load. Warehouse declarations remain explicit until
structural preflight has passed; reads then use the active adapter without
creating stel's schema or state tables. Only validated terms reach consumers.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from .adapters import AdapterError, WarehouseCapability
from .config.loader import (
    ConfigError,
    _reject_symlink_directory_components,
    _resolve_declared_vocabularies,
    _validate_config_file,
)
from .config.source import WAREHOUSE_SOURCE_SCHEME
from .config.vocabulary import Vocabulary, VocabularySource, VocabularyTerm, check_declaration
from .paths import resolve_within_project

if TYPE_CHECKING:
    from .adapters import WarehouseAdapter
    from .config.model import ModelConfig
    from .config.project import ProjectConfig
    from .config.source import SourceConfig


def has_warehouse_vocabularies(project: ProjectConfig) -> bool:
    return any(
        v.source is not None and v.source.relation is not None
        for v in project.vocabularies.values()
    )


def _error(name: str, detail: str) -> ConfigError:
    return ConfigError(f"Vocabulary '{name}': {detail}")


def _check_columns(name: str, source: VocabularySource, columns: Sequence[str]) -> None:
    for column in (source.label_column, source.description_column):
        if column is not None and column not in columns:
            raise _error(name, f"source has no configured column '{column}'")


def _resolve_rows(
    name: str, vocabulary: Vocabulary, rows: Sequence[Mapping[str, Any]]
) -> Vocabulary:
    source = vocabulary.source
    assert source is not None
    if len(rows) + len(vocabulary.extra_terms) > source.max_terms:
        raise _error(
            name,
            f"source exceeds `max_terms: {source.max_terms}`; narrow the source or raise the limit",
        )
    terms: list[VocabularyTerm] = []
    for index, row in enumerate(rows, start=1):
        label = row[source.label_column]
        description = row[source.description_column] if source.description_column else None
        if not isinstance(label, str) or not label.strip():
            raise _error(name, f"source row {index} must have a non-empty string label")
        if description is not None and not isinstance(description, str):
            raise _error(name, f"source row {index} must have a string or null description")
        terms.append(VocabularyTerm(label=label, description=description or None))
    # Relations have no row order. A seed reorder must not reprocess paid work
    # either, so both source kinds use the same canonical label order.
    terms.sort(key=lambda term: term.label)
    terms.extend(vocabulary.extra_terms)
    try:
        return Vocabulary(terms=terms)
    except ValidationError:
        # Duplicate labels and Pydantic input dumps can contain source values.
        raise _error(
            name, "resolved terms must be non-empty, unique, and have valid hierarchy references"
        ) from None


def resolve_csv_vocabularies(project: ProjectConfig, project_dir: Path) -> None:
    for name, vocabulary in project.vocabularies.items():
        source = vocabulary.source
        if source is None or source.path is None:
            continue
        path = resolve_within_project(source.path, project_dir, surface=f"Vocabulary '{name}' seed")
        _reject_symlink_directory_components(
            project_dir / source.path, project_dir, description=f"Vocabulary '{name}' seed"
        )
        _validate_config_file(
            project_dir / source.path,
            allowed_root=project_dir,
            description=f"Vocabulary '{name}' seed",
        )
        project.vocabularies[name] = _resolve_rows(name, vocabulary, _read_csv(name, source, path))
    check_declaration(project.vocabularies, project.classes)


def _read_csv(name: str, source: VocabularySource, path: Path) -> list[Mapping[str, Any]]:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            return _csv_rows(name, source, csv.DictReader(handle, strict=True))
    except (OSError, UnicodeError, csv.Error):
        raise _error(name, "cannot read UTF-8 CSV seed") from None


def _csv_rows(
    name: str, source: VocabularySource, reader: csv.DictReader[str]
) -> list[Mapping[str, Any]]:
    columns = reader.fieldnames or []
    if len(columns) != len(set(columns)):
        raise _error(name, "CSV header contains duplicate columns")
    _check_columns(name, source, columns)
    rows: list[Mapping[str, Any]] = []
    for row in reader:
        if len(rows) >= source.max_terms:
            raise _error(name, f"source exceeds `max_terms: {source.max_terms}`")
        if None in row or any(value is None for value in row.values()):
            raise _error(name, "CSV row does not match its header")
        rows.append(row)
    return rows


def resolve_warehouse_vocabularies(
    project: ProjectConfig,
    sources: list[SourceConfig],
    models: list[ModelConfig],
    project_dir: Path,
    adapter: WarehouseAdapter,
) -> None:
    if not has_warehouse_vocabularies(project):
        return
    # Local imports break compiler -> loader -> vocabulary resolution cycles.
    from .compiler import validate_project_contract, validate_warehouse_capabilities

    validate_project_contract(project, sources, models, project_dir)
    validate_warehouse_capabilities(models, adapter)
    adapter.require_capability(
        WarehouseCapability.TABULAR_READS, operation="reading vocabulary relations"
    )
    resolved: dict[str, Vocabulary] = {}
    with adapter.preflight_reads():
        for name, vocabulary in project.vocabularies.items():
            source = vocabulary.source
            if source is None or source.relation is None:
                continue
            relation = source.relation[len(WAREHOUSE_SOURCE_SCHEME) :]
            try:
                frame = adapter.read_relation(relation, limit=source.max_terms + 1)
            except AdapterError:
                raise _error(
                    name,
                    "cannot read source relation; check that it exists "
                    "and the active target can read it",
                ) from None
            _check_columns(name, source, frame.columns)
            resolved[name] = _resolve_rows(name, vocabulary, frame.to_dicts())
    project.vocabularies.update(resolved)
    _resolve_declared_vocabularies(project, models)
    validate_project_contract(project, sources, models, project_dir)


def require_resolved_vocabularies(project: ProjectConfig) -> None:
    if has_warehouse_vocabularies(project):
        raise ConfigError(
            "Warehouse vocabularies are unresolved. `stel compile`, `plan`, `run`, "
            "and `build` resolve them against the active target. Use a CSV seed "
            "or inline terms for offline exports and embedded execution."
        )
