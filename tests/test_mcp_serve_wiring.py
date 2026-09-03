"""Every valid `stel mcp serve` configuration actually reaches a transport.

This file exists because of a bug, and it is the coverage that bug got past.

#488: the MCP SDK refuses a `token_verifier` that arrives without
`AuthSettings`, stel passed one without, and so *every* release carrying token
verification raised `ValueError` at startup the moment `--jwt-*` was used. The
feature had never once worked. What made that survivable for so long was the
shape of the tests around it:

- the verifier classes were tested thoroughly, in isolation;
- the CLI was tested thoroughly, and every one of those tests asserted a
  **refusal** — a bad configuration producing exit code 2;
- the one test that built a server built the unauthenticated one.

Each part was covered. The assembled thing was not, and no test anywhere asked
the only question an operator cares about first: *does it start?*

So these tests are deliberately shallow and deliberately end-to-end. They run
the real CLI, the real flag validation, the real verifier construction, and the
real `create_mcp_server`, and they assert one thing — the transport ran. Only
the two edges are stubbed: the warehouse behind `ContextService.from_project`,
and `FastMCP.run`, which would otherwise block forever serving.

A configuration that an operator can legally type belongs in `CONFIGURATIONS`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

ISSUER = "https://issuer.example"
PUBLIC_URL = "https://stel.example/mcp"
SECRET_ENV = "STEL_TEST_SERVE_WIRING_SECRET"

# Each entry is a configuration an operator can legally type. The parametrize
# ids are what a failure names, so they read as the deployment they describe.
CONFIGURATIONS: dict[str, list[str]] = {
    "stdio": [],
    "http-trusted-proxy": [
        "--transport", "streamable-http",
        "--trust-proxy-principal-headers",
    ],
    "http-jwt": [
        "--transport", "streamable-http",
        "--public-url", PUBLIC_URL,
        "--jwt-issuer", ISSUER,
        "--jwt-audience", PUBLIC_URL,
        "--jwt-jwks-uri", "https://issuer.example/jwks.json",
    ],
    "http-introspection": [
        "--transport", "streamable-http",
        "--public-url", PUBLIC_URL,
        "--introspection-endpoint", "https://issuer.example/introspect",
        "--introspection-issuer", ISSUER,
        "--introspection-audience", PUBLIC_URL,
        "--introspection-client-id", "stel-mcp",
        "--introspection-client-secret-env", SECRET_ENV,
    ],
    "sse-jwt": [
        "--transport", "sse",
        "--public-url", PUBLIC_URL,
        "--jwt-issuer", ISSUER,
        "--jwt-audience", PUBLIC_URL,
        "--jwt-jwks-uri", "https://issuer.example/jwks.json",
    ],
    "sse-trusted-proxy": [
        "--transport", "sse",
        "--trust-proxy-principal-headers",
    ],
}


class _StubService:
    """Stands in for the warehouse-backed service, which is not what is under
    test here — the wiring above it is."""

    def warm_up(self) -> None:
        return None

    def close(self) -> None:
        return None


@pytest.fixture
def started(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[str]:
    """Boot a server without a warehouse or a blocking transport.

    `FastMCP.run` is patched on the class rather than on an instance, so the
    server the CLI builds is the real one: every SDK-side validation that
    rejected stel's own arguments still runs.
    """
    from mcp.server.fastmcp import FastMCP

    from stel.mcp_server import service as service_module

    monkeypatch.setenv(SECRET_ENV, "s3cret")
    monkeypatch.setattr(
        service_module.ContextService,
        "from_project",
        classmethod(lambda cls, *args, **kwargs: _StubService()),
    )
    transports: list[str] = []
    monkeypatch.setattr(
        FastMCP, "run", lambda self, transport: transports.append(transport)
    )
    (tmp_path / "stel_project.yml").write_text(
        "name: p\nversion: '0.1.0'\n", encoding="utf-8"
    )
    return transports


def _serve(args: list[str], tmp_path: Path) -> Any:
    from click.testing import CliRunner

    from stel.cli import cli

    return CliRunner().invoke(
        cli, ["--project-dir", str(tmp_path), "mcp", "serve", *args]
    )


@pytest.mark.parametrize(
    "name", list(CONFIGURATIONS), ids=list(CONFIGURATIONS)
)
def test_a_valid_configuration_reaches_its_transport(
    name: str, started: list[str], tmp_path: Path
) -> None:
    """The question no test was asking: does it start?

    `http-jwt`, `http-introspection` and `sse-jwt` are the three that failed
    before #488 — with an exit code and a `ValueError`, not a subtle wrong
    answer. An operator would have hit it on the first run.
    """
    result = _serve(CONFIGURATIONS[name], tmp_path)

    assert result.exit_code == 0, (
        f"{name} did not start: {result.output.strip()}\n{result.exception!r}"
    )
    assert started, f"{name} exited cleanly without ever running a transport"


def test_every_transport_choice_is_represented() -> None:
    """A transport with no configuration here is a transport nothing boots.

    The gap this file exists for was not a missing assertion, it was a missing
    *case*, so the guard is against the set going stale rather than against any
    one test breaking.
    """
    from stel.mcp_server.server import NETWORK_TRANSPORTS

    covered = {
        args[args.index("--transport") + 1]
        for args in CONFIGURATIONS.values()
        if "--transport" in args
    }

    assert covered == set(NETWORK_TRANSPORTS), (
        "a network transport has no booting configuration under test"
    )


def test_both_verifiers_are_represented() -> None:
    """Same reasoning, for the identity sources. Each of the three ways a
    caller can be identified must have a configuration that boots."""
    flags = {flag for args in CONFIGURATIONS.values() for flag in args}

    assert "--jwt-issuer" in flags
    assert "--introspection-endpoint" in flags
    assert "--trust-proxy-principal-headers" in flags
