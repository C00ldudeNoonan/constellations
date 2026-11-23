"""DNS-rebinding protection on every bind (review of PR #810).

SDK 1.x refused a non-loopback `Host` on every bind, because its server was
built with the default host `127.0.0.1`. SDK 2 enables the check only for a
loopback bind, so stel passes `loopback_host_security()` explicitly. These tests
bind `0.0.0.0` in the SDK's own app factory, the case that would otherwise go
unchecked.
"""

import pytest

pytest.importorskip("mcp.server.mcpserver")

from mcp.server.mcpserver import MCPServer
from starlette.testclient import TestClient

from stel.mcp_server.server import loopback_host_security


def _status(host: str, *, secured: bool) -> int:
    app = MCPServer("t").streamable_http_app(
        host="0.0.0.0",
        transport_security=loopback_host_security() if secured else None,
    )
    with TestClient(app, base_url="http://testserver") as client:
        return client.post(
            "/mcp", headers={"Host": host, "Content-Type": "application/json"}, content="{}"
        ).status_code


def test_a_wide_bind_refuses_a_foreign_host() -> None:
    assert _status("stel.example", secured=True) == 421


def test_a_wide_bind_accepts_a_loopback_host() -> None:
    assert _status("localhost:8000", secured=True) != 421


def test_without_the_settings_a_wide_bind_checks_nothing() -> None:
    """The reason the settings are passed: SDK 2's default is no check here."""
    assert _status("stel.example", secured=False) != 421
