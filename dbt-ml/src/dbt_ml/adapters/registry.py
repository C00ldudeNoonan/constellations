from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ..config.profile import WarehouseConfig
from .base import AdapterError, WarehouseAdapter


class UnknownAdapterError(AdapterError):
    pass


_REGISTRY: dict[str, type[WarehouseAdapter]] = {}


def register(cls: type[WarehouseAdapter]) -> type[WarehouseAdapter]:
    _REGISTRY[cls.adapter_type()] = cls
    return cls


def parse_warehouse_config(raw: dict[str, Any] | WarehouseConfig) -> WarehouseConfig:
    """Validate a raw profiles.yml warehouse block against the config model
    of the adapter named by its `type:` (default duckdb)."""
    if isinstance(raw, WarehouseConfig):
        return raw
    wtype = str(raw.get("type", "duckdb"))
    cls = _REGISTRY.get(wtype)
    if cls is None:
        raise UnknownAdapterError(
            f"No adapter registered for warehouse.type='{wtype}'. "
            f"Known: {sorted(_REGISTRY)}"
        )
    try:
        return cls.config_model().model_validate(raw)
    except ValidationError as e:
        raise AdapterError(
            f"Invalid config for warehouse.type='{wtype}':\n{e}"
        ) from e


def create_adapter(
    config: WarehouseConfig, *, project_dir: Path | None = None
) -> WarehouseAdapter:
    cls = _REGISTRY.get(config.type)
    if cls is None:
        raise UnknownAdapterError(
            f"No adapter registered for warehouse.type='{config.type}'. "
            f"Known: {sorted(_REGISTRY)}"
        )
    return cls(config, project_dir=project_dir)


def list_adapter_types() -> list[str]:
    return sorted(_REGISTRY)
