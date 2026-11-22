"""A real `stel mcp serve` process over fixture data, for wire-level tests (ALE-751).

`ContextService.from_project` is the one seam replaced: it would open a warehouse
and load a compiled project, neither of which a protocol test needs. Everything
else is the shipped code -- the Click command, flag validation, resolver and
verifier construction, `create_mcp_server`, and the SDK transport that binds the
socket or reads stdin. So a client talking to this process sees what a client
talking to the real server sees.

Run as `python -m tests.support_mcp_wire <stel args>` from the repository root.

Environment, read once at start:

- `STEL_WIRE_AUTHORIZATION=grants` swaps claim-derived policy for operator-owned
  grants (alice -> tenant `research`, bob -> tenant `restricted`).
- `STEL_WIRE_QUERY_LOG=<path>` appends every query-log row as one JSON line.
- `STEL_WIRE_JWKS=<path>` serves that JWKS file as the issuer's published keys,
  so `--jwt-*` verification runs for real without a network fetch.
"""
from __future__ import annotations

import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from stel.cli import cli
from stel.mcp_server.authorization import (
    ClaimAuthorizationProvider,
    EnvironmentPrincipalResolver,
)
from stel.mcp_server.grants import Grant, GrantAuthorizationProvider, StaticGrantStore
from stel.mcp_server.service import ContextService

# The fixture ids are re-exported for `test_mcp_wire_compat.py`: test modules
# do not import each other (`test_test_tiers.py`), support modules are the seam.
from tests.test_mcp_server import (
    CHUNK_ALLOWED_1,
    CHUNK_HIDDEN,
    CONTEXT_ALLOWED_1,
    DOC_ALLOWED,
    DOC_HIDDEN,
    VERSION_ALLOWED,
    VERSION_HIDDEN,
    FakeRepository,
    FakeSearch,
    _artifact_catalog,
    _fixture_rows,
)

__all__ = [
    "AUTHORIZATION_ENV",
    "CHUNK_ALLOWED_1",
    "CHUNK_HIDDEN",
    "CONTEXT_ALLOWED_1",
    "DOC_ALLOWED",
    "DOC_HIDDEN",
    "JWKS_ENV",
    "QUERY_LOG_ENV",
    "VERSION_ALLOWED",
    "VERSION_HIDDEN",
]

QUERY_LOG_ENV = "STEL_WIRE_QUERY_LOG"
AUTHORIZATION_ENV = "STEL_WIRE_AUTHORIZATION"
JWKS_ENV = "STEL_WIRE_JWKS"


class _LoggingRepository(FakeRepository):
    """Writes each query-log row where the test process can read it."""

    def __init__(self, path: Path | None) -> None:
        super().__init__(_fixture_rows())
        self._path = path

    def log_query(self, row: Mapping[str, Any]) -> None:
        super().log_query(row)
        if self._path is not None:
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(dict(row), default=str) + "\n")


def _authorization() -> Any:
    if os.environ.get(AUTHORIZATION_ENV) != "grants":
        return ClaimAuthorizationProvider()
    grants = [
        Grant(subject, attribute, value)
        for subject, tenant in (("alice", "research"), ("bob", "restricted"))
        for attribute, value in (("tenant_id", tenant), ("classification", "internal"))
    ]
    return GrantAuthorizationProvider(StaticGrantStore(grants))


def _from_project(
    project_dir: Path, *, principal_resolver: Any = None, settings: Any = None, **_: Any
) -> Any:
    log_path = os.environ.get(QUERY_LOG_ENV)
    return ContextService(
        catalog=_artifact_catalog(),
        repository=_LoggingRepository(Path(log_path) if log_path else None),
        context_search=FakeSearch(),
        principal_resolver=principal_resolver or EnvironmentPrincipalResolver(),
        authorization=_authorization(),
        settings=settings,
    )


def _publish_local_jwks(path: Path) -> None:
    """Make the verifier's key fetch return the keys in `path`, never the network."""
    import jwt

    keys = jwt.PyJWKSet.from_dict(json.loads(path.read_text(encoding="utf-8"))).keys
    jwt.PyJWKClient.get_signing_keys = (  # type: ignore[method-assign]
        lambda self, refresh=False: keys
    )


def main() -> None:
    ContextService.from_project = staticmethod(_from_project)  # ty: ignore[invalid-assignment]
    jwks = os.environ.get(JWKS_ENV)
    if jwks:
        _publish_local_jwks(Path(jwks))
    cli.main(args=sys.argv[1:], prog_name="stel")


if __name__ == "__main__":
    main()
