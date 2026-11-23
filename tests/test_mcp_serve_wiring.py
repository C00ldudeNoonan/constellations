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
and `MCPServer.run`, which would otherwise block forever serving.

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
    # Warehouse-identity enforcement (#568). A flag the CLI accepts but never
    # boots is the same class of bug this file exists for.
    "stdio-enforced-warehouse-identity": [
        "--grants-relation", "stel_grants",
        "--enforce-warehouse-identity",
    ],
}


# The keywords `MCPServer.run` was last called with, for the tests below.
RUN_ARGUMENTS: list[dict[str, Any]] = []


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

    `MCPServer.run` is patched on the class rather than on an instance, so the
    server the CLI builds is the real one: every SDK-side validation that
    rejected stel's own arguments still runs.
    """
    import inspect

    from mcp.server.mcpserver import MCPServer

    from stel.mcp_server import service as service_module

    monkeypatch.setenv(SECRET_ENV, "s3cret")
    monkeypatch.setattr(
        service_module.ContextService,
        "from_project",
        classmethod(lambda cls, *args, **kwargs: _StubService()),
    )
    transports: list[str] = []
    runners = {
        "stdio": MCPServer.run_stdio_async,
        "sse": MCPServer.run_sse_async,
        "streamable-http": MCPServer.run_streamable_http_async,
    }

    def fake_run(self: Any, transport: str, **kwargs: Any) -> None:
        # SDK v2 forwards `run`'s keywords to the transport and rejects one it
        # does not know only when the transport starts, which this fake would
        # otherwise never reach. Bind them to the real signature so a keyword
        # the SDK would refuse at startup fails here.
        inspect.signature(runners[transport]).bind(self, **kwargs)
        transports.append(transport)
        RUN_ARGUMENTS.append(kwargs)

    RUN_ARGUMENTS.clear()
    monkeypatch.setattr(MCPServer, "run", fake_run)
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


def _without_security(arguments: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in arguments.items() if key != "transport_security"}


def test_the_listen_address_reaches_the_transport_as_run_arguments(
    started: list[str], tmp_path: Path
) -> None:
    """SDK v2 takes `host` and `port` as arguments to `run`; assigning them on
    the server's settings, as v1 allowed, now raises. A server that ignored
    them would bind the SDK default (127.0.0.1:8000) instead of the address
    the operator asked for, with nothing to say so."""
    args = [*CONFIGURATIONS["http-trusted-proxy"], "--host", "0.0.0.0", "--port", "9123"]

    result = _serve(args, tmp_path)

    assert result.exit_code == 0, result.output
    assert len(RUN_ARGUMENTS) == 1
    assert _without_security(RUN_ARGUMENTS[0]) == {
        "host": "0.0.0.0",
        "port": 9123,
        "json_response": True,
    }
    # A wide bind still refuses a foreign Host (review of PR #810).
    assert RUN_ARGUMENTS[0]["transport_security"].enable_dns_rebinding_protection


def test_a_stdio_server_is_given_no_network_arguments(
    started: list[str], tmp_path: Path
) -> None:
    result = _serve(CONFIGURATIONS["stdio"], tmp_path)

    assert result.exit_code == 0, result.output
    assert RUN_ARGUMENTS == [{}]


def test_the_sse_transport_is_not_given_a_json_response_argument(
    started: list[str], tmp_path: Path
) -> None:
    """`json_response` is a streamable-HTTP option; SSE is event-stream by
    definition and the SDK would refuse the keyword."""
    args = [*CONFIGURATIONS["sse-trusted-proxy"], "--host", "0.0.0.0", "--port", "9124"]

    result = _serve(args, tmp_path)

    assert result.exit_code == 0, result.output
    assert len(RUN_ARGUMENTS) == 1
    assert _without_security(RUN_ARGUMENTS[0]) == {"host": "0.0.0.0", "port": 9124}


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
