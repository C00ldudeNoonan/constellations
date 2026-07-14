from . import (
    bigquery,  # noqa: F401  # side-effect: registers BigQueryAdapter
    duckdb,  # noqa: F401  # side-effect: registers DuckDBAdapter
)
from .base import (
    AdapterCapabilityError,
    AdapterError,
    WarehouseAdapter,
    WarehouseCapability,
)
from .registry import (
    UnknownAdapterError,
    adapter_capabilities,
    create_adapter,
    list_adapter_types,
    parse_warehouse_config,
    register,
)

__all__ = [
    "AdapterCapabilityError",
    "AdapterError",
    "UnknownAdapterError",
    "WarehouseAdapter",
    "WarehouseCapability",
    "adapter_capabilities",
    "create_adapter",
    "list_adapter_types",
    "parse_warehouse_config",
    "register",
]
