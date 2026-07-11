from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

DEFAULT_LLM_API_KEY_ENV = "ANTHROPIC_API_KEY"
_ENV_NAME_PATTERN = r"^[A-Za-z_][A-Za-z0-9_]*$"
_ENV_NAME_RE = re.compile(_ENV_NAME_PATTERN)


def resolve_llm_credential(options: Mapping[str, Any]) -> tuple[str, str | None]:
    """Return the configured environment-variable name and its current value.

    The secret is deliberately resolved only at the client boundary. Runtime
    options and artifacts carry the variable name, never the credential value.
    """
    api_key_env = options.get("api_key_env", DEFAULT_LLM_API_KEY_ENV)
    if not isinstance(api_key_env, str) or not _ENV_NAME_RE.fullmatch(api_key_env):
        raise ValueError(
            "llm api_key_env must be a valid environment-variable name"
        )
    return api_key_env, os.environ.get(api_key_env)


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
        (DuckDB); None for remote warehouses."""
        return None


class PricingConfig(BaseModel):
    """User-supplied token prices (USD per million tokens) for cost estimates
    in run results. No prices ship with dbt-ml — they drift; you own them."""

    model_config = ConfigDict(extra="forbid")

    input_usd_per_mtok: float
    output_usd_per_mtok: float
    cache_read_usd_per_mtok: float | None = None
    cache_write_usd_per_mtok: float | None = None


class LLMConfig(BaseModel):
    """Defaults for the LLM extraction backend and LLM-using transforms."""

    model_config = ConfigDict(extra="forbid")

    provider: Literal["anthropic"] = "anthropic"
    model: str = "claude-haiku-4-5"
    api_key_env: str = Field(
        default=DEFAULT_LLM_API_KEY_ENV, pattern=_ENV_NAME_PATTERN
    )
    cache_path: Path | None = None
    system_prompt: str | None = None
    pricing: PricingConfig | None = None


class TargetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Raw mapping: validated against the adapter's config model at resolve
    # time, once `type:` is known (the adapter registry owns that lookup).
    warehouse: dict[str, Any]
    llm: LLMConfig | None = None


class ProfileConfig(BaseModel):
    """A named profile: one or more targets (e.g. dev/prod), plus default target."""

    model_config = ConfigDict(extra="forbid")

    target: str = "dev"
    outputs: dict[str, TargetConfig]
