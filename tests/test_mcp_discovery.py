"""OAuth discovery metadata for the network transport (issue #464, item 2).

Two things landed together here, because one is the other's mechanism.

**The bug.** The SDK refuses a `token_verifier` that arrives without
`AuthSettings` — `Cannot specify auth_server_provider or token_verifier
without auth settings`. stel passed one without, so *every* release carrying
token verification crashed at startup the moment `--jwt-*` was used, and #487's
introspection flags inherited it. Nothing caught it because the only test that
built a server built the unauthenticated one; the verifier tests all exercised
the verifier classes in isolation, which is exactly the shape of coverage that
proves each part works and never that they fit together.

**The feature.** Those same settings are what a resource server declares, and
the SDK turns `resource_server_url` into the RFC 9728 protected-resource
metadata document and the `resource_metadata` parameter of its
`WWW-Authenticate` challenges. So a spec-compliant client can discover the
authorization server instead of being configured with the issuer out of band.

The first test here is therefore a regression test before it is a feature test.
"""
from __future__ import annotations

from typing import Any

import pytest

from stel.mcp_server.server import _auth_settings, create_mcp_server

ISSUER = "https://issuer.example"
PUBLIC_URL = "https://stel.example/mcp"


class _Verifier:
    """Shape only — nothing here ever verifies a token."""

    async def verify_token(self, token: str) -> Any:
        return None


def _service() -> Any:
    from unittest.mock import MagicMock

    return MagicMock()


def _paths(app: Any) -> list[str]:
    return [getattr(route, "path", "") for route in app.streamable_http_app().routes]


# ─── the regression ────────────────────────────────────────────────────────


def test_a_server_with_a_verifier_starts() -> None:
    """The bug, stated plainly: this raised `ValueError` on every release that
    shipped token verification.

    A verifier is useless if the server carrying it cannot boot, and no test
    built that combination — each half was covered alone.
    """
    app = create_mcp_server(
        _service(),
        token_verifier=_Verifier(),
        issuer_url=ISSUER,
        public_url=PUBLIC_URL,
    )

    assert app is not None
    assert "/mcp" in _paths(app)


def test_a_server_without_a_verifier_still_starts() -> None:
    """The proxy-header and stdio paths must not acquire a requirement they
    have no use for: settings without a verifier make the SDK demand one."""
    app = create_mcp_server(_service())

    assert _paths(app) == ["/mcp"]


# ─── what a client can discover ────────────────────────────────────────────


def test_the_protected_resource_metadata_is_served() -> None:
    """The point of item 2: a client that is handed only this server's URL can
    find out who issues tokens for it."""
    from starlette.testclient import TestClient

    app = create_mcp_server(
        _service(),
        token_verifier=_Verifier(),
        issuer_url=ISSUER,
        public_url=PUBLIC_URL,
    )

    response = TestClient(app.streamable_http_app()).get(
        "/.well-known/oauth-protected-resource/mcp"
    )

    assert response.status_code == 200
    document = response.json()
    assert document["resource"] == PUBLIC_URL
    assert document["authorization_servers"] == [f"{ISSUER}/"]


def test_an_unauthenticated_server_publishes_no_metadata() -> None:
    """Nothing to discover, and saying otherwise would advertise an
    authorization server that does not govern this deployment."""
    app = create_mcp_server(_service())

    assert not [path for path in _paths(app) if "well-known" in path]


# ─── the settings themselves ───────────────────────────────────────────────


def test_a_verifier_without_urls_is_refused() -> None:
    """Caught here rather than as the SDK's own message, which names
    `auth_server_provider` — a parameter stel does not and should not use."""
    with pytest.raises(ValueError, match="public URL"):
        _auth_settings(_Verifier(), issuer_url=ISSUER, public_url=None)

    with pytest.raises(ValueError, match="issuer"):
        _auth_settings(_Verifier(), issuer_url=None, public_url=PUBLIC_URL)


def test_no_verifier_means_no_settings() -> None:
    assert _auth_settings(None, issuer_url=None, public_url=None) is None


def test_urls_are_not_required_without_a_verifier() -> None:
    """A deployment behind an authenticating proxy passes neither, and must
    not be made to."""
    assert _auth_settings(None, issuer_url=ISSUER, public_url=PUBLIC_URL) is None


# ─── the CLI ───────────────────────────────────────────────────────────────


def _serve(args: list[str], tmp_path: Any) -> Any:
    from click.testing import CliRunner

    from stel.cli import cli

    (tmp_path / "stel_project.yml").write_text(
        "name: p\nversion: '0.1.0'\n", encoding="utf-8"
    )
    return CliRunner().invoke(
        cli, ["--project-dir", str(tmp_path), "mcp", "serve", *args]
    )


def test_cli_requires_a_public_url_when_verifying(tmp_path: Any) -> None:
    """It cannot be derived from --host/--port: those are what the process
    binds behind a proxy, not what a caller reaches."""
    result = _serve(
        [
            "--transport", "streamable-http",
            "--jwt-issuer", ISSUER,
            "--jwt-audience", PUBLIC_URL,
            "--jwt-jwks-uri", "https://issuer.example/jwks.json",
        ],
        tmp_path,
    )

    assert result.exit_code != 0
    assert "--public-url" in result.output


def test_cli_refuses_a_public_url_with_nothing_to_verify(tmp_path: Any) -> None:
    """No challenge to issue and no metadata to publish; accepted-and-ignored
    is how a flag comes to look like a feature."""
    result = _serve(
        [
            "--transport", "streamable-http",
            "--trust-proxy-principal-headers",
            "--public-url", PUBLIC_URL,
        ],
        tmp_path,
    )

    assert result.exit_code != 0
    assert "verifies tokens" in result.output
