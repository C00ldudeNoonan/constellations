from __future__ import annotations

import re
from pathlib import PurePosixPath, PureWindowsPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .identifiers import validate_node_name

# A warehouse-table source (issue #322): `warehouse://<relation>` names a
# relation the *active adapter* reads, so the same project can point at
# `economics_raw.reddit_comments_raw` in one target and a dev copy in another
# via per-target `source_paths` overrides, like any other source path.
WAREHOUSE_SOURCE_SCHEME = "warehouse://"

# Relation parts cross from project YAML into SQL, so they are validated here
# and quoted by the adapter. The leading part may be a BigQuery project id or
# an attached DuckDB database, both of which allow dashes; schema/dataset and
# table stay on the strict identifier charset.
_RELATION_PART = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_RELATION_HEAD = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def validate_relation_name(relation: str) -> str:
    parts = relation.split(".")
    if not 1 <= len(parts) <= 3:
        raise ValueError(
            "warehouse source relation must be 'table', 'schema.table', or "
            "'catalog.schema.table'"
        )
    head, tail = parts[:-2], parts[-2:] if len(parts) > 1 else parts
    for part in tail:
        if not _RELATION_PART.match(part):
            raise ValueError(
                f"warehouse source relation part {part!r} is invalid: parts "
                "must start with a letter or underscore and contain only "
                "letters, digits, and underscores"
            )
    for part in head:
        if not _RELATION_HEAD.match(part):
            raise ValueError(
                f"warehouse source catalog/project part {part!r} is invalid"
            )
    return relation


def validate_file_pattern(pattern: str) -> str:
    paths = (PurePosixPath(pattern), PureWindowsPath(pattern))
    if any(path.root or path.drive for path in paths):
        raise ValueError("file_pattern must be a relative path pattern")
    if any(".." in path.parts for path in paths):
        raise ValueError("file_pattern must not contain parent traversal ('..')")
    return pattern


class DurationSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    count: int
    period: Literal["minute", "hour", "day", "week"]

    def to_seconds(self) -> int:
        per = {"minute": 60, "hour": 3600, "day": 86400, "week": 604800}
        return self.count * per[self.period]


class FreshnessConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    warn_after: DurationSpec | None = None
    error_after: DurationSpec | None = None


class SourceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str | None = None
    path: str  # project-relative directory, gs://bucket/prefix, or gdrive://<folderId>
    project: str | None = None
    file_pattern: str = "*.json"
    recursive: bool = True
    # Opt-in for a local path outside the project directory (issue #65).
    # Explicit and per-source so out-of-project reads survive code review.
    external: bool = False
    # Bound on remote listings so a typo'd prefix can't crawl a whole bucket.
    max_objects: int = 5000
    tags: list[str] = Field(default_factory=list)
    freshness: FreshnessConfig | None = None
    # Warehouse-table sources only (issue #322). `key_column` names the column
    # whose value identifies a row across runs — the row-grain analogue of an
    # object path, and the incremental identity. `path_columns` render into the
    # source-relative path ahead of the key, so `--source-filter` globs (and
    # orchestrator partition scoping built on them) address row subsets the
    # same way they address object prefixes: `subreddit/*` works either way.
    key_column: str | None = None
    path_columns: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_warehouse_source(self) -> SourceConfig:
        if self.path.startswith(WAREHOUSE_SOURCE_SCHEME):
            relation = self.path[len(WAREHOUSE_SOURCE_SCHEME) :]
            try:
                validate_relation_name(relation)
            except ValueError as error:
                raise ValueError(f"Source '{self.name}': {error}") from error
            if not self.key_column:
                raise ValueError(
                    f"Source '{self.name}' is a warehouse source and must "
                    "declare `key_column:` — the column that identifies a row "
                    "across runs, like an object path identifies a file"
                )
            # Object-source knobs that are meaningless against a relation are
            # rejected when explicitly set, rather than silently ignored.
            for field in ("file_pattern", "recursive", "external"):
                if field in self.model_fields_set:
                    raise ValueError(
                        f"Source '{self.name}': `{field}:` does not apply to a "
                        "warehouse source"
                    )
        elif self.key_column is not None or self.path_columns:
            raise ValueError(
                f"Source '{self.name}': `key_column:`/`path_columns:` apply "
                "only to warehouse:// sources"
            )
        if self.key_column is not None and self.key_column in self.path_columns:
            raise ValueError(
                f"Source '{self.name}': `path_columns:` must not repeat the "
                "key column; the key is always the path's final segment"
            )
        return self

    @field_validator("name")
    @classmethod
    def _validate_name(cls, v: str) -> str:
        return validate_node_name(v, kind="Source")

    @field_validator("file_pattern")
    @classmethod
    def _validate_file_pattern(cls, v: str) -> str:
        return validate_file_pattern(v)

    @field_validator("project")
    @classmethod
    def _validate_project(cls, v: str | None) -> str | None:
        if v is None:
            return None
        if not v.strip():
            raise ValueError("source project must not be empty")
        return v.strip()


class SourceFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: Literal[2] = 2
    sources: list[SourceConfig]
