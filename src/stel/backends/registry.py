from __future__ import annotations

from collections.abc import Callable
from typing import overload

from pydantic import BaseModel

from .base import BaseBackend


class BackendNotFoundError(Exception):
    pass


_REGISTRY: dict[str, type[BaseBackend]] = {}


@overload
def register(backend_cls: type[BaseBackend], /) -> type[BaseBackend]: ...


@overload
def register(
    *,
    options_model: type[BaseModel] | None = None,
    native_batch: bool = False,
    requires_credentials: bool = False,
) -> Callable[[type[BaseBackend]], type[BaseBackend]]: ...


def register(
    backend_cls: type[BaseBackend] | None = None,
    /,
    *,
    options_model: type[BaseModel] | None = None,
    native_batch: bool = False,
    requires_credentials: bool = False,
) -> type[BaseBackend] | Callable[[type[BaseBackend]], type[BaseBackend]]:
    """Register a backend and, optionally, its typed option contract.

    Bare ``@register`` remains compatible for third-party backends and installs
    a pass-through option contract. New backends should supply a Pydantic model
    so compile and runtime share the same strict validation boundary.
    """

    def decorator(cls: type[BaseBackend]) -> type[BaseBackend]:
        from .options import (
            BackendOptionsError,
            get_backend_option_contract,
            register_backend_option_contract,
        )

        instance = cls()
        name = instance.name()
        _REGISTRY[name] = cls
        try:
            existing_contract = get_backend_option_contract(name)
        except BackendOptionsError:
            existing_contract = None
        if options_model is not None or existing_contract is None:
            register_backend_option_contract(
                name,
                options_model,
                native_batch=native_batch,
                requires_credentials=requires_credentials,
            )
        elif native_batch or requires_credentials:
            register_backend_option_contract(
                name,
                existing_contract.options_model,
                native_batch=native_batch or existing_contract.native_batch,
                requires_credentials=(
                    requires_credentials or existing_contract.requires_credentials
                ),
            )
        return cls

    return decorator if backend_cls is None else decorator(backend_cls)


def get_backend(name: str) -> BaseBackend:
    cls = _REGISTRY.get(name)
    if cls is None:
        raise BackendNotFoundError(
            f"Backend '{name}' is not registered. Available: {sorted(_REGISTRY)}"
        )
    return cls()


def list_backends() -> list[str]:
    return sorted(_REGISTRY)
