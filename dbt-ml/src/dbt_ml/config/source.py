from __future__ import annotations

from pathlib import PurePosixPath, PureWindowsPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .identifiers import validate_node_name


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
    path: str  # project-relative directory, or gs://bucket/prefix
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
