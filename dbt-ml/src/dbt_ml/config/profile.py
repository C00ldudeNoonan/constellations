from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class WarehouseConfig(BaseModel):
    """Base for adapter-specific warehouse configs, discriminated by `type:`.

    Each adapter owns a subclass declaring its connection fields (DuckDB: a
    file path; BigQuery: project/dataset; ...) and registers it via
    `WarehouseAdapter.config_model()`. Raw profile blocks are validated
    against the subclass at resolve time, so unknown fields and missing
    credentials fail with the adapter named.
    """

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    type: str = "duckdb"
    schema_name: str = Field(default="dbt_ml", alias="schema")

    def absolutize(self, project_dir: Path) -> WarehouseConfig:
        """Resolve project-relative paths. Default: nothing to resolve."""
        return self

    def storage_location(self) -> str:
        """Human-readable storage location (file path, DSN, dataset, ...)."""
        return ""

    def catalog_name(self) -> str:
        """Catalog/database name for SQL references and emitted dbt sources
        (DuckDB: database file stem; BigQuery: GCP project)."""
        return ""

    def local_path(self) -> Path | None:
        """Filesystem location backing this warehouse, if file-backed
        (DuckDB); None for remote warehouses. Drives `clean`'s --force guard
        for files outside the project directory (issue #65)."""
        return None


class PricingConfig(BaseModel):
    """User-supplied token prices (USD per million tokens) for cost estimates
    in run results. No prices ship with dbt-ml — they drift; you own them."""

    input_usd_per_mtok: float
    output_usd_per_mtok: float
    cache_read_usd_per_mtok: float | None = None
    cache_write_usd_per_mtok: float | None = None


class LLMConfig(BaseModel):
    """Defaults for the LLM extraction backend and LLM-using transforms."""

    provider: Literal["anthropic"] = "anthropic"
    model: str = "claude-haiku-4-5"
    api_key_env: str = "ANTHROPIC_API_KEY"
    cache_path: Path | None = None
    system_prompt: str | None = None
    pricing: PricingConfig | None = None


class TargetConfig(BaseModel):
    # Raw mapping: validated against the adapter's config model at resolve
    # time, once `type:` is known (the adapter registry owns that lookup).
    warehouse: dict[str, Any]
    llm: LLMConfig | None = None


class ProfileConfig(BaseModel):
    """A named profile: one or more targets (e.g. dev/prod), plus default target."""

    target: str = "dev"
    outputs: dict[str, TargetConfig]
