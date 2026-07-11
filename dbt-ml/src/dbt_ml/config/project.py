from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class DuckDBConfig(BaseModel):
    """Deprecated inline warehouse config used only when a project declares no
    `profile:`. Prefer a profiles.yml `warehouse:` block; this is slated for
    removal once the legacy no-profile path goes away."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    path: Path = Path("./target/dbt_ml.duckdb")
    schema_name: str = Field(default="dbt_ml", alias="schema")


class ExtractionDefaults(BaseModel):
    model_config = ConfigDict(extra="forbid")

    default_backend: str = "json"


class ProjectConfig(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True, extra="forbid", protected_namespaces=()
    )

    name: str
    version: str = "0.1.0"
    profile: str | None = None
    duckdb: DuckDBConfig = Field(default_factory=DuckDBConfig)
    extraction: ExtractionDefaults = Field(default_factory=ExtractionDefaults)

    source_paths: list[Path] = Field(
        default_factory=lambda: [Path("sources")], alias="source-paths"
    )
    model_paths: list[Path] = Field(
        default_factory=lambda: [Path("models")], alias="model-paths"
    )
    transform_paths: list[Path] = Field(
        default_factory=lambda: [Path("transforms")], alias="transform-paths"
    )
    target_path: Path = Field(default=Path("target"), alias="target-path")
