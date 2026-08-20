from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ..config.profile import WarehouseConfig
from ..credentials import CredentialFreeUrl, CredentialReference
from .base import (
    AdapterConfigError,
    AdapterError,
    WarehouseAdapter,
    WarehouseCapability,
)


class UnknownAdapterError(AdapterError):
    pass


_REGISTRY: dict[str, type[WarehouseAdapter]] = {}
_GENERIC_SECRET_REFERENCE_FIELDS = (
    "client_secret",
    "keyfile_json",
    "refresh_token",
    "token",
)


def _protect_unregistered_adapter_input(
    prepared: dict[str, Any],
) -> dict[str, Any]:
    for field_name in _GENERIC_SECRET_REFERENCE_FIELDS:
        value = prepared.get(field_name)
        if value is None or isinstance(value, CredentialReference):
            continue
        try:
            prepared[field_name] = CredentialReference.from_env_var_expression(
                value
            )
        except (TypeError, ValueError):
            raise ValueError(
                f"`{field_name}` must be an exact {{ env_var('NAME') }} "
                "reference with no default"
            ) from None

    for field_name in ("keyfile", "token_uri"):
        value = prepared.get(field_name)
        if value is None or isinstance(value, CredentialReference):
            continue
        if isinstance(value, str) and (
            "{{" in value or "env_var(" in value
        ):
            try:
                prepared[field_name] = (
                    CredentialReference.from_env_var_expression(value)
                )
            except (TypeError, ValueError):
                raise ValueError(
                    f"`{field_name}` environment configuration must be an "
                    "exact {{ env_var('NAME') }} reference with no default"
                ) from None

    token_uri = prepared.get("token_uri")
    if isinstance(token_uri, str):
        prepared["token_uri"] = CredentialFreeUrl(token_uri)
    return prepared


def register[T: WarehouseAdapter](cls: type[T]) -> type[T]:
    _REGISTRY[cls.adapter_type()] = cls
    return cls


def prepare_warehouse_profile_input(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Let an adapter protect credentials before generic interpolation."""
    prepared = dict(raw)
    wtype = str(prepared.get("type", "duckdb"))
    adapter = _REGISTRY.get(wtype)
    result: dict[str, Any] | None = None
    failure: AdapterError | None = None
    try:
        if adapter is None:
            result = _protect_unregistered_adapter_input(prepared)
        else:
            result = adapter.config_model().prepare_profile_input(prepared)
    except ValueError as error:
        failure = AdapterError(
            f"Invalid protected config for warehouse.type='{wtype}': {error}"
        )
    raw = {}
    prepared = {}
    if failure is not None:
        raise failure
    assert result is not None
    return result


def _validation_details(error: ValidationError) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "loc": tuple(item["loc"]),
            "msg": item["msg"],
            "type": item["type"],
        }
        for item in error.errors(
            include_url=False,
            include_context=False,
            include_input=False,
        )
    )


def _format_validation_details(
    validation_details: tuple[dict[str, Any], ...],
) -> str:
    rendered: list[str] = []
    for item in validation_details:
        location = ".".join(str(part) for part in item["loc"])
        prefix = f"{location}: " if location else ""
        rendered.append(f"{prefix}{item['msg']}")
    return "\n".join(rendered)


def _config_error(
    warehouse_type: str,
    error: ValidationError,
) -> AdapterConfigError:
    validation_details = _validation_details(error)
    return AdapterConfigError(
        f"Invalid config for warehouse.type='{warehouse_type}':\n"
        f"{_format_validation_details(validation_details)}",
        validation_details,
    )


def parse_warehouse_config(raw: dict[str, Any] | WarehouseConfig) -> WarehouseConfig:
    """Validate a raw profiles.yml warehouse block against the config model
    of the adapter named by its `type:` (default duckdb)."""
    if isinstance(raw, WarehouseConfig):
        return raw
    wtype = str(raw.get("type", "duckdb"))
    cls = _REGISTRY.get(wtype)
    if cls is None:
        unknown_error = UnknownAdapterError(
            f"No adapter registered for warehouse.type='{wtype}'. "
            f"Known: {sorted(_REGISTRY)}"
        )
        raw = {}
        raise unknown_error
    preparation_error: AdapterError | None = None
    try:
        prepared = prepare_warehouse_profile_input(raw)
    except AdapterError as error:
        preparation_error = error
        prepared = {}
    if preparation_error is not None:
        raw = {}
        raise preparation_error
    try:
        return cls.config_model().model_validate(prepared)
    except ValidationError as error:
        config_error = _config_error(wtype, error)
    raw = {}
    prepared = {}
    raise config_error from None


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


def adapter_capabilities(adapter_type: str) -> frozenset[WarehouseCapability]:
    cls = _REGISTRY.get(adapter_type)
    if cls is None:
        raise UnknownAdapterError(
            f"No adapter registered for warehouse.type='{adapter_type}'. "
            f"Known: {sorted(_REGISTRY)}"
        )
    return cls.capabilities()
