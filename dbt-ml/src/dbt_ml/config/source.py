from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from .identifiers import validate_node_name


class DurationSpec(BaseModel):
    count: int
    period: Literal["minute", "hour", "day", "week"]

    def to_seconds(self) -> int:
        per = {"minute": 60, "hour": 3600, "day": 86400, "week": 604800}
        return self.count * per[self.period]


class FreshnessConfig(BaseModel):
    warn_after: DurationSpec | None = None
    error_after: DurationSpec | None = None


class SourceConfig(BaseModel):
    name: str
    description: str | None = None
    path: str  # project-relative directory, or gs://bucket/prefix
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


class SourceFile(BaseModel):
    version: int = 2
    sources: list[SourceConfig]
