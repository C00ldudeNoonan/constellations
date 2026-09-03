"""JWT bearer verification for the network transport (issue #392, item 4).

The failure mode this whole issue exists for is silent: a request served under
the wrong identity produces a correct-looking answer out of someone else's
corpus. So these assert refusals, not happy paths — a verifier that accepts
something it should not is indistinguishable from a working one until an
auditor asks.

Tokens are signed with a real RSA key and verified through the real PyJWT
path. A fake verifier would only prove the wiring, and the wiring is not the
part that is dangerous.
"""
from __future__ import annotations

import time
from typing import Any

import pytest

from stel.mcp_server.tokens import (
    ALLOWED_ALGORITHMS,
    JwksTokenVerifier,
    JwtVerifierConfig,
    TokenVerificationError,
)

ISSUER = "https://issuer.example"
AUDIENCE = "https://stel.example/mcp"
JWKS_URI = "https://issuer.example/.well-known/jwks.json"


@pytest.fixture(scope="module")
def rsa_key() -> Any:
    from cryptography.hazmat.primitives.asymmetric import rsa

    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


def _config(**overrides: Any) -> JwtVerifierConfig:
    return JwtVerifierConfig(
        **{"issuer": ISSUER, "audience": AUDIENCE, "jwks_uri": JWKS_URI, **overrides}
    )


def _verifier(rsa_key: Any, monkeypatch: pytest.MonkeyPatch, **overrides: Any) -> Any:
    """A verifier whose key lookup returns `rsa_key`'s public half.

    Only the network fetch is stubbed. Signature, issuer, audience, expiry and
    algorithm checks all run for real.
    """
    verifier = JwksTokenVerifier(_config(**overrides))

    class _Key:
        key = rsa_key.public_key()

    monkeypatch.setattr(
        verifier._keys, "get_signing_key_from_jwt", lambda _token: _Key()
    )
    return verifier


def _token(rsa_key: Any, *, algorithm: str = "RS256", **claims: Any) -> str:
    import jwt

    payload = {
        "iss": ISSUER,
        "aud": AUDIENCE,
        "sub": "svc-analytics",
        "exp": int(time.time()) + 300,
        **claims,
    }
    key: Any = rsa_key if algorithm.startswith(("RS", "ES", "PS")) else "secret"
    return jwt.encode(payload, key, algorithm=algorithm)


async def _verify(verifier: Any, token: str) -> Any:
    return await verifier.verify_token(token)


# ─── configuration is a security boundary, so it has no defaults ────────────


def test_plaintext_jwks_is_refused() -> None:
    """Keys fetched over http can be replaced in transit, which makes every
    other check decorative."""
    with pytest.raises(TokenVerificationError, match="https"):
        _config(jwks_uri="http://issuer.example/jwks.json")


@pytest.mark.parametrize("field", ["issuer", "audience", "jwks_uri"])
def test_every_field_is_required(field: str) -> None:
    with pytest.raises(TokenVerificationError, match=field):
        _config(**{field: "   "})


def test_no_symmetric_algorithm_is_accepted() -> None:
    """The classic JWKS forgery: sign with the published public key as an HMAC
    secret. A verifier that accepts both families takes it."""
    assert not [name for name in ALLOWED_ALGORITHMS if name.startswith("HS")]
    assert "none" not in ALLOWED_ALGORITHMS


# ─── what the verifier refuses ─────────────────────────────────────────────


@pytest.mark.anyio
async def test_a_valid_token_yields_its_subject(
    rsa_key: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    verifier = _verifier(rsa_key, monkeypatch)
    token = await _verify(verifier, _token(rsa_key))
    assert token is not None
    assert token.subject == "svc-analytics"


@pytest.mark.anyio
async def test_a_token_for_another_audience_is_refused(
    rsa_key: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The confused-deputy case the MCP authorization spec calls out: a token
    the caller legitimately holds, for a different service."""
    verifier = _verifier(rsa_key, monkeypatch)
    assert await _verify(verifier, _token(rsa_key, aud="https://other.example")) is None


@pytest.mark.anyio
async def test_a_token_from_another_issuer_is_refused(
    rsa_key: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    verifier = _verifier(rsa_key, monkeypatch)
    assert await _verify(verifier, _token(rsa_key, iss="https://evil.example")) is None


@pytest.mark.anyio
async def test_an_expired_token_is_refused(
    rsa_key: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    verifier = _verifier(rsa_key, monkeypatch)
    stale = _token(rsa_key, exp=int(time.time()) - 3600)
    assert await _verify(verifier, stale) is None


@pytest.mark.anyio
async def test_a_token_signed_by_someone_else_is_refused(
    rsa_key: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cryptography.hazmat.primitives.asymmetric import rsa

    attacker = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    verifier = _verifier(rsa_key, monkeypatch)
    assert await _verify(verifier, _token(attacker)) is None


@pytest.mark.anyio
async def test_a_token_without_a_subject_is_refused(
    rsa_key: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Grants are keyed by subject, so a subjectless token could never be
    authorized — refused here so the reason is stated rather than arriving as
    an anonymous-looking request."""
    verifier = _verifier(rsa_key, monkeypatch)
    assert await _verify(verifier, _token(rsa_key, sub="")) is None


@pytest.mark.anyio
async def test_garbage_is_refused_without_raising(
    rsa_key: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Caller-supplied input must never raise out of the verifier: the message
    would distinguish failure reasons (a probing oracle) and could carry token
    contents into a log."""
    verifier = _verifier(rsa_key, monkeypatch)
    for bad in ("", "not-a-jwt", "a.b.c", "Bearer x"):
        assert await _verify(verifier, bad) is None


# ─── the token carries identity, never authorization ───────────────────────


@pytest.mark.anyio
async def test_claimed_groups_and_tenant_are_not_carried_into_the_principal(
    rsa_key: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#396 made grants the authorization source: groups and tenants are looked
    up by subject, never carried. A token asserting them must gain nothing —
    otherwise a legitimately-issued token becomes a privilege escalation."""
    from stel.mcp_server.authorization import AccessTokenPrincipalResolver

    verifier = _verifier(rsa_key, monkeypatch)
    token = await _verify(
        verifier,
        _token(
            rsa_key,
            tenant_id="someone-elses-tenant",
            access_groups=["admin"],
            groups=["admin"],
        ),
    )
    assert token is not None

    monkeypatch.setattr(
        "stel.mcp_server.authorization._verified_access_token", lambda: token
    )
    principal = AccessTokenPrincipalResolver().resolve()

    assert principal is not None
    assert principal.subject_id == "svc-analytics"
    assert principal.tenant_id is None
    assert principal.access_groups == ()
    assert not principal.policy_claims


def test_no_verified_token_resolves_to_no_principal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Outside a request, or with nothing verified, the service must see an
    unauthenticated call rather than a partially-built principal."""
    from stel.mcp_server.authorization import AccessTokenPrincipalResolver

    monkeypatch.setattr(
        "stel.mcp_server.authorization._verified_access_token", lambda: None
    )
    assert AccessTokenPrincipalResolver().resolve() is None


# ─── the transport refuses a half-configured identity plane ────────────────


def test_a_token_resolver_without_a_verifier_is_refused() -> None:
    """Nothing would verify anything; every call would arrive unauthenticated.
    Noisy rather than dangerous, but it is not the deployment the operator
    configured, so it must not start."""
    from stel.mcp_server.authorization import AccessTokenPrincipalResolver
    from stel.mcp_server.server import _reject_unverified_token_identity

    with pytest.raises(ValueError, match="no token verifier"):
        _reject_unverified_token_identity(AccessTokenPrincipalResolver(), None)


def test_a_verifier_with_a_header_resolver_is_refused() -> None:
    """The dangerous shape: tokens are checked and then ignored, because the
    headers decide. The operator believes tokens gate access; they do not."""
    from stel.mcp_server.authorization import TrustedHeaderPrincipalResolver
    from stel.mcp_server.server import _reject_unverified_token_identity

    with pytest.raises(ValueError, match="checked and then ignored"):
        _reject_unverified_token_identity(TrustedHeaderPrincipalResolver(), object())


def test_a_matched_verifier_and_resolver_pass() -> None:
    from stel.mcp_server.authorization import AccessTokenPrincipalResolver
    from stel.mcp_server.server import _reject_unverified_token_identity

    _reject_unverified_token_identity(AccessTokenPrincipalResolver(), object())


def test_stdio_stays_untouched_by_the_guard() -> None:
    """The guard only ever runs on the network path; the stdio default
    resolves from the environment and has no verifier, which is correct."""
    from stel.mcp_server.authorization import EnvironmentPrincipalResolver
    from stel.mcp_server.server import _reject_unverified_token_identity

    _reject_unverified_token_identity(EnvironmentPrincipalResolver(), None)


# ─── the CLI names every refusal ───────────────────────────────────────────


def _serve(args: list[str], tmp_path: Any) -> Any:
    from click.testing import CliRunner

    from stel.cli import cli

    (tmp_path / "stel_project.yml").write_text(
        "name: p\nversion: '0.1.0'\n", encoding="utf-8"
    )
    return CliRunner().invoke(
        cli, ["--project-dir", str(tmp_path), "mcp", "serve", *args]
    )


def test_cli_refuses_a_partial_jwt_configuration(tmp_path: Any) -> None:
    result = _serve(
        ["--transport", "streamable-http", "--jwt-issuer", ISSUER], tmp_path
    )
    assert result.exit_code != 0
    assert "--jwt-audience" in result.output
    assert "--jwt-jwks-uri" in result.output
    assert "none of them gets a default" in result.output


def test_cli_refuses_both_identity_sources_at_once(tmp_path: Any) -> None:
    """Enabling both means the headers decide and the token checking is
    decoration — the exact shape the server-side guard exists to catch, caught
    one layer earlier with a message that says which one to drop."""
    result = _serve(
        [
            "--transport", "streamable-http",
            "--trust-proxy-principal-headers",
            "--jwt-issuer", ISSUER,
            "--jwt-audience", AUDIENCE,
            "--jwt-jwks-uri", JWKS_URI,
        ],
        tmp_path,
    )
    assert result.exit_code != 0
    assert "Choose one identity source" in result.output


def test_cli_still_refuses_a_network_transport_with_no_identity(tmp_path: Any) -> None:
    """The #394 refusal survives, and now names both ways out."""
    result = _serve(["--transport", "streamable-http"], tmp_path)
    assert result.exit_code != 0
    assert "--jwt-issuer" in result.output
    assert "--trust-proxy-principal-headers" in result.output


def test_cli_refuses_a_plaintext_jwks_uri(tmp_path: Any) -> None:
    result = _serve(
        [
            "--transport", "streamable-http",
            "--jwt-issuer", ISSUER,
            "--jwt-audience", AUDIENCE,
            "--jwt-jwks-uri", "http://issuer.example/jwks.json",
        ],
        tmp_path,
    )
    assert result.exit_code != 0
    assert "https" in result.output
