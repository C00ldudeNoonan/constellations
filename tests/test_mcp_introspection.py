"""OAuth 2.0 token introspection for the network transport (issue #464).

Same failure mode as the JWT verifier's tests, so the same emphasis: a request
served under the wrong identity produces a correct-looking answer out of
someone else's corpus, and a verifier that accepts what it should not is
indistinguishable from a working one until an auditor asks. These assert
refusals first.

The one check with no analogue in the JWT path is the audience. RFC 7662's
`active: true` means the token is live *at that server* — not that it was
minted for this deployment — so the confused-deputy defense is entirely ours
to perform, against a response we did not sign.

Only the HTTP call is faked, at the client. Every check the verifier makes on
the response runs for real.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from stel.mcp_server import introspection as introspection_module
from stel.mcp_server.introspection import (
    IntrospectionTokenVerifier,
    IntrospectionVerifierConfig,
)
from stel.mcp_server.tokens import TokenVerificationError

ISSUER = "https://issuer.example"
AUDIENCE = "https://stel.example/mcp"
ENDPOINT = "https://issuer.example/oauth2/introspect"
CLIENT_ID = "stel-mcp"
SECRET_ENV = "STEL_TEST_INTROSPECTION_SECRET"
CLIENT_SECRET = "s3cret-value-not-in-any-log"


@pytest.fixture(autouse=True)
def _client_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    """The config refuses to build without it, by design."""
    monkeypatch.setenv(SECRET_ENV, CLIENT_SECRET)


def _config(**overrides: Any) -> IntrospectionVerifierConfig:
    return IntrospectionVerifierConfig(
        **{
            "issuer": ISSUER,
            "audience": AUDIENCE,
            "introspection_endpoint": ENDPOINT,
            "client_id": CLIENT_ID,
            "client_secret_env": SECRET_ENV,
            **overrides,
        }
    )


def _body(**overrides: Any) -> dict[str, Any]:
    """What an authorization server returns for a live token minted for us."""
    return {
        "active": True,
        "sub": "svc-analytics",
        "aud": AUDIENCE,
        "iss": ISSUER,
        "exp": int(time.time()) + 300,
        "client_id": "caller-app",
        "scope": "read write",
        **overrides,
    }


class _FakeResponse:
    def __init__(self, status_code: int, body: Any) -> None:
        self.status_code = status_code
        self._body = body

    def json(self) -> Any:
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


class _FakeClient:
    def __init__(self, script: _Script) -> None:
        self._script = script

    async def post(self, url: str, **kwargs: Any) -> _FakeResponse:
        self._script.calls.append({"url": url, **kwargs})
        if self._script.gate is not None:
            await self._script.gate.wait()
        if self._script.error is not None:
            raise self._script.error
        return _FakeResponse(self._script.status_code, self._script.body)


class _Script:
    """The endpoint's scripted behavior, plus what it was asked."""

    def __init__(
        self,
        *,
        body: Any = None,
        status_code: int = 200,
        error: Exception | None = None,
        gate: Any = None,
    ) -> None:
        self.body = body if body is not None else _body()
        self.status_code = status_code
        self.error = error
        # Held open to keep requests in flight, for the concurrency bound.
        self.gate = gate
        self.calls: list[dict[str, Any]] = []
        self.clients_created = 0
        self.limits: dict[str, Any] | None = None

    def Limits(self, **kwargs: Any) -> dict[str, Any]:
        self.limits = kwargs
        return kwargs

    def AsyncClient(self, **kwargs: Any) -> _FakeClient:
        self.clients_created += 1
        return _FakeClient(self)


def _verifier(script: _Script, **overrides: Any) -> IntrospectionTokenVerifier:
    verifier = IntrospectionTokenVerifier(_config(**overrides))
    verifier._httpx = script  # type: ignore[assignment]
    return verifier


# ─── configuration is a security boundary, so it has no defaults ────────────


def test_a_plaintext_endpoint_is_refused() -> None:
    """Stronger than the JWKS case: the caller's token is *sent* to this URL,
    so plaintext hands the credential itself to anyone on the path."""
    with pytest.raises(TokenVerificationError, match="https"):
        _config(introspection_endpoint="http://issuer.example/introspect")


@pytest.mark.parametrize(
    "field",
    ["issuer", "audience", "introspection_endpoint", "client_id", "client_secret_env"],
)
def test_every_field_is_required(field: str) -> None:
    with pytest.raises(TokenVerificationError, match=field):
        _config(**{field: "   "})


def test_an_unset_secret_is_refused_at_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Once, at startup — not one refused caller at a time."""
    monkeypatch.delenv(SECRET_ENV, raising=False)

    with pytest.raises(TokenVerificationError, match="unset or empty"):
        _config()


def test_the_refusal_does_not_name_the_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AGENTS.md keeps credential environment-variable names out of
    diagnostics; the operator just typed this one."""
    monkeypatch.delenv(SECRET_ENV, raising=False)

    with pytest.raises(TokenVerificationError) as excinfo:
        _config()

    assert SECRET_ENV not in str(excinfo.value)


def test_the_config_never_holds_the_resolved_secret() -> None:
    """`repr=False` would hide it from `repr()` and from nothing else.

    A resolved credential in a long-lived object survives `asdict()`,
    `__dict__`, a debugger and any generic dump, so the config carries the
    variable's name and the value is read where the request is built (PR #487
    review).
    """
    import dataclasses

    config = _config()

    assert CLIENT_SECRET not in repr(config)
    assert CLIENT_SECRET not in str(dataclasses.asdict(config))
    assert CLIENT_SECRET not in str(vars(config))
    assert config.client_secret_env == SECRET_ENV


# ─── what the verifier refuses ─────────────────────────────────────────────


@pytest.mark.anyio
async def test_a_live_token_yields_its_subject() -> None:
    script = _Script()

    token = await _verifier(script).verify_token("opaque-token")

    assert token is not None
    assert token.subject == "svc-analytics"
    assert token.client_id == "caller-app"
    assert token.scopes == ["read", "write"]


@pytest.mark.anyio
async def test_an_inactive_token_is_refused() -> None:
    script = _Script(body=_body(active=False))

    assert await _verifier(script).verify_token("opaque-token") is None


@pytest.mark.anyio
async def test_a_string_active_is_not_a_live_session() -> None:
    """RFC 7662 defines a JSON boolean. A truthy string must not read as
    active, which is what a plain `if body.get("active")` would do."""
    script = _Script(body=_body(active="true"))

    assert await _verifier(script).verify_token("opaque-token") is None


@pytest.mark.anyio
async def test_a_token_for_another_service_is_refused() -> None:
    """The check this whole module turns on. The token is genuinely active at
    the issuer — it was simply minted for somebody else."""
    script = _Script(body=_body(aud="https://other.example/api"))

    assert await _verifier(script).verify_token("opaque-token") is None


@pytest.mark.anyio
async def test_a_response_without_an_audience_is_refused() -> None:
    """An authorization server that omits `aud` cannot be used here: there is
    no way left to tell a token minted for this deployment from one the caller
    legitimately holds elsewhere."""
    body = _body()
    del body["aud"]
    script = _Script(body=body)

    assert await _verifier(script).verify_token("opaque-token") is None


@pytest.mark.anyio
async def test_an_audience_list_containing_ours_is_accepted() -> None:
    script = _Script(body=_body(aud=["https://other.example/api", AUDIENCE]))

    assert await _verifier(script).verify_token("opaque-token") is not None


@pytest.mark.anyio
async def test_a_response_from_another_issuer_is_refused() -> None:
    script = _Script(body=_body(iss="https://attacker.example"))

    assert await _verifier(script).verify_token("opaque-token") is None


@pytest.mark.anyio
async def test_a_response_without_an_issuer_is_accepted() -> None:
    """Deliberate, and the one place this is laxer than the JWT path. A JWT
    could have come from anywhere, so its `iss` is load-bearing; this answer
    came over TLS from the one endpoint the operator named, authenticated as a
    client that server registered."""
    body = _body()
    del body["iss"]
    script = _Script(body=body)

    assert await _verifier(script).verify_token("opaque-token") is not None


@pytest.mark.anyio
async def test_a_token_without_a_subject_is_refused() -> None:
    """Verified but unusable: grants are keyed by subject, so this caller
    could never be authorized by any grant."""
    script = _Script(body=_body(sub=""))

    assert await _verifier(script).verify_token("opaque-token") is None


@pytest.mark.anyio
async def test_an_expired_token_is_refused_even_when_called_active() -> None:
    """`active` should have covered it; a server that says both is not one to
    take the generous reading from."""
    script = _Script(body=_body(exp=int(time.time()) - 1))

    assert await _verifier(script).verify_token("opaque-token") is None


@pytest.mark.anyio
@pytest.mark.parametrize("status", [400, 401, 403, 500, 503])
async def test_a_non_200_is_not_an_endorsement(status: int) -> None:
    """Bad client credentials and a down issuer are both 'cannot vouch for
    this caller' — never 'let them in'."""
    script = _Script(status_code=status)

    assert await _verifier(script).verify_token("opaque-token") is None


@pytest.mark.anyio
async def test_a_transport_failure_refuses() -> None:
    script = _Script(error=RuntimeError("connection reset"))

    assert await _verifier(script).verify_token("opaque-token") is None


@pytest.mark.anyio
async def test_a_body_that_is_not_json_refuses() -> None:
    script = _Script(body=ValueError("not json"))

    assert await _verifier(script).verify_token("opaque-token") is None


@pytest.mark.anyio
async def test_a_body_that_is_not_an_object_refuses() -> None:
    script = _Script(body=["active"])

    assert await _verifier(script).verify_token("opaque-token") is None


# ─── the request itself ────────────────────────────────────────────────────


@pytest.mark.anyio
async def test_the_call_authenticates_and_carries_the_token() -> None:
    """An introspection endpoint that accepted unauthenticated calls would let
    anyone test tokens against it, so the client credentials are not optional
    decoration."""
    script = _Script()

    await _verifier(script).verify_token("opaque-token")

    [call] = script.calls
    assert call["url"] == ENDPOINT
    assert call["data"]["token"] == "opaque-token"
    assert call["data"]["token_type_hint"] == "access_token"
    assert call["auth"] == (CLIENT_ID, CLIENT_SECRET)


# ─── the unauthenticated path is bounded ───────────────────────────────────


async def _settle() -> None:
    """Run every ready task to its next real await.

    Turns of the loop rather than a wall-clock sleep, so nothing here depends
    on machine speed: the bound is asserted after the tasks have had every
    chance to exceed it, and extra turns can only make a broken bound show.
    """
    for _ in range(50):
        await asyncio.sleep(0)


@pytest.mark.anyio
async def test_one_pooled_client_serves_every_call() -> None:
    """A client per call meant a TLS handshake per token and no ceiling on the
    sockets an unauthenticated flood could open (PR #487 review)."""
    script = _Script()
    verifier = _verifier(script)

    await verifier.verify_token("token-a")
    await verifier.verify_token("token-b")
    await verifier.verify_token("token-c")

    assert len(script.calls) == 3
    assert script.clients_created == 1
    assert script.limits == {
        "max_connections": introspection_module._MAX_CONNECTIONS,
        "max_keepalive_connections": introspection_module._MAX_CONNECTIONS,
    }


@pytest.mark.anyio
async def test_only_so_many_introspections_run_at_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The SDK verifies before any tool runs, so the service's rate limits are
    downstream of this call. Without a bound of its own, a caller who has
    proven nothing decides how many outbound requests this server makes."""
    monkeypatch.setattr(introspection_module, "_MAX_CONCURRENT_INTROSPECTIONS", 2)
    gate = asyncio.Event()
    script = _Script(gate=gate)
    verifier = _verifier(script)

    tasks = [
        asyncio.create_task(verifier.verify_token(f"token-{n}")) for n in range(6)
    ]
    await _settle()
    in_flight = len(script.calls)
    gate.set()
    await asyncio.gather(*tasks)

    assert in_flight == 2, f"{in_flight} requests in flight past the bound of 2"
    assert len(script.calls) == 6, "every caller was eventually served"


@pytest.mark.anyio
async def test_a_caller_that_cannot_get_a_slot_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Refusing beats queueing without limit: the queue would itself be the
    resource an attacker grows. 'Too busy to ask' is the same answer as every
    other failure here — cannot vouch for this caller."""
    monkeypatch.setattr(introspection_module, "_MAX_CONCURRENT_INTROSPECTIONS", 1)
    monkeypatch.setattr(introspection_module, "_SLOT_WAIT_SECONDS", 0.01)
    gate = asyncio.Event()
    script = _Script(gate=gate)
    verifier = _verifier(script)

    held = asyncio.create_task(verifier.verify_token("token-holding-the-slot"))
    await _settle()
    # Bounded so a verifier that lost its bound fails this test rather than
    # hanging it: without the semaphore this call reaches the gated request
    # and would block until the gate opens, which is after this line.
    refused = await asyncio.wait_for(
        verifier.verify_token("token-arriving-under-load"), timeout=5.0
    )
    gate.set()
    await held

    assert refused is None
    assert len(script.calls) == 1, "the refused caller never reached the issuer"


# ─── the cache, which is also the revocation delay ─────────────────────────


@pytest.mark.anyio
async def test_a_verified_token_is_not_introspected_twice() -> None:
    """The reason the cache exists: otherwise every request costs a round trip
    on a path that already has a timeout budget."""
    script = _Script()
    verifier = _verifier(script)

    await verifier.verify_token("opaque-token")
    second = await verifier.verify_token("opaque-token")

    assert second is not None
    assert len(script.calls) == 1


@pytest.mark.anyio
async def test_a_refusal_is_never_cached() -> None:
    """A rejected token must be asked about again. Caching failures would
    blunt one repeated bad token while doing nothing about the shape that
    matters — distinct invented ones — and hands an unauthenticated caller a
    way to grow memory."""
    script = _Script(body=_body(active=False))
    verifier = _verifier(script)

    await verifier.verify_token("opaque-token")
    await verifier.verify_token("opaque-token")

    assert len(script.calls) == 2


@pytest.mark.anyio
async def test_the_cache_never_outlives_the_token() -> None:
    """A token expiring sooner than the TTL must not be served from the cache
    past its own `exp`."""
    script = _Script(body=_body(exp=int(time.time()) + 1))
    verifier = _verifier(script)
    await verifier.verify_token("opaque-token")

    # Reach past the token's expiry without sleeping through it.
    [entry] = verifier._cache.values()
    assert entry.expires_at <= float(script.body["exp"]), (
        "the entry outlives the token it describes"
    )


@pytest.mark.anyio
async def test_the_cache_is_keyed_by_digest_not_by_the_token() -> None:
    """A cache keyed by the raw token hands a live credential to anything that
    dumps or iterates it."""
    script = _Script()
    verifier = _verifier(script)

    await verifier.verify_token("opaque-token")

    assert "opaque-token" not in verifier._cache
    assert len(verifier._cache) == 1


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


_FULL_INTROSPECTION = [
    "--introspection-endpoint", ENDPOINT,
    "--introspection-issuer", ISSUER,
    "--introspection-audience", AUDIENCE,
    "--introspection-client-id", CLIENT_ID,
    "--introspection-client-secret-env", "STEL_TEST_INTROSPECTION_SECRET",
]


def test_cli_refuses_a_partial_introspection_configuration(tmp_path: Any) -> None:
    result = _serve(
        ["--transport", "streamable-http", "--introspection-endpoint", ENDPOINT],
        tmp_path,
    )

    assert result.exit_code != 0
    assert "--introspection-client-id" in result.output
    assert "none of them gets a default" in result.output


def test_cli_refuses_both_verifiers_at_once(tmp_path: Any) -> None:
    """Which one refused a caller would be ambiguous, and the two answer
    different questions about the same token."""
    result = _serve(
        [
            "--transport", "streamable-http",
            "--jwt-issuer", ISSUER,
            "--jwt-audience", AUDIENCE,
            "--jwt-jwks-uri", "https://issuer.example/jwks.json",
            *_FULL_INTROSPECTION,
        ],
        tmp_path,
    )

    assert result.exit_code != 0
    assert "Choose one token verifier" in result.output


def test_cli_refuses_introspection_flags_on_stdio(tmp_path: Any) -> None:
    """Accepted-and-ignored would look like authentication from the outside."""
    result = _serve(["--introspection-endpoint", ENDPOINT], tmp_path)

    assert result.exit_code != 0
    assert "network transport" in result.output
    assert "verify nothing" in result.output


def test_cli_refuses_introspection_beside_trusted_headers(tmp_path: Any) -> None:
    result = _serve(
        [
            "--transport", "streamable-http",
            "--trust-proxy-principal-headers",
            *_FULL_INTROSPECTION,
        ],
        tmp_path,
    )

    assert result.exit_code != 0
    assert "Choose one identity source" in result.output


def test_cli_refuses_a_plaintext_introspection_endpoint(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The secret is present so the scheme is what fails: the config object is
    # the single place that knows the https rule, and it cannot be built
    # without one.
    monkeypatch.setenv("STEL_TEST_INTROSPECTION_SECRET", CLIENT_SECRET)

    result = _serve(
        [
            "--transport", "streamable-http",
            "--introspection-endpoint", "http://issuer.example/introspect",
            "--introspection-issuer", ISSUER,
            "--introspection-audience", AUDIENCE,
            "--introspection-client-id", CLIENT_ID,
            "--introspection-client-secret-env", "STEL_TEST_INTROSPECTION_SECRET",
        ],
        tmp_path,
    )

    assert result.exit_code != 0
    assert "https" in result.output


def test_cli_refuses_an_unset_secret_without_naming_the_variable(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Actionable without echoing it: AGENTS.md keeps credential
    environment-variable names out of diagnostics, and the operator just typed
    this one."""
    monkeypatch.delenv("STEL_TEST_INTROSPECTION_SECRET", raising=False)

    result = _serve(["--transport", "streamable-http", *_FULL_INTROSPECTION], tmp_path)

    assert result.exit_code != 0
    assert "unset or empty" in result.output
    assert "STEL_TEST_INTROSPECTION_SECRET" not in result.output


def test_cli_still_refuses_a_network_transport_with_no_identity(tmp_path: Any) -> None:
    """The #394 refusal now names all three ways out."""
    result = _serve(["--transport", "streamable-http"], tmp_path)

    assert result.exit_code != 0
    assert "--introspection-" in result.output
    assert "--jwt-issuer" in result.output
    assert "--trust-proxy-principal-headers" in result.output
