from __future__ import annotations

from typing import Any

from ..adapters import AdapterError, WarehouseAdapter
from ..config.model import ModelConfig
from .contracts import RunError


def warehouse_options(adapter: WarehouseAdapter, model: ModelConfig) -> Any:
    try:
        return adapter.parse_warehouse_options(
            model.warehouse_options, model_name=model.name
        )
    except AdapterError as error:
        raise RunError(str(error)) from error
