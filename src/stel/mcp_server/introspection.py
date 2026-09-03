"""OAuth 2.0 token introspection for the network transport (issue #464).

`JwksTokenVerifier` needs a JWT it can verify locally, which rules out every
authorization server that issues **opaque** tokens. RFC 7662 is the other
option: ask the issuing server about the token instead of reading it. Same
`TokenVerifier` seam, same invariant — a token establishes identity only, and
authorization stays with operator-owned grants (#396).

**What "active" does and does not mean.** RFC 7662's `active: true` says the
token is valid *at that server*, not that it was minted for this deployment.
A caller legitimately holding a token for some other service would introspect
as active, so the audience check the JWT path performs locally has to happen
here too, against the response. It is required for the same reason it is
required there: without it this is a confused deputy with a network hop.

**The response is trusted because of how it was obtained**, not because of
what it contains. It arrives over TLS from an endpoint the operator named,
authenticated as a client that server registered. That is why `iss` is checked
when present but not demanded: unlike a self-contained JWT, which could have
come from anywhere, this answer came from the one server we asked.

**A cache hit is a revocation delay, and that is the whole trade.** Every
request would otherwise cost a network round trip on a path with a timeout
budget. Positive results are cached for at most `_CACHE_TTL_SECONDS`, capped
by the token's own expiry, so a token revoked at the server keeps working here
for up to that long. Shorten it by shortening the constant; there is no way to
have both the round trip and the immediacy.

Failures are not cached. It would blunt a flood of one repeated bad token
while doing nothing about the shape that matters — distinct invented tokens,
which miss every time — and every cache keyed by attacker-supplied input is
memory an unauthenticated caller can grow.

**This whole path is unauthenticated, and it is the one that makes network
calls.** The SDK verifies a bearer token *before any tool runs*, so the
service's own concurrency and rate limits are downstream of everything here: a
caller who has proven nothing decides how many distinct tokens arrive, and each
cache miss is an outbound request to someone else's authorization server.
Unbounded, a small parallel flood becomes unbounded sockets here and a denial
of service there. So the calls share one pooled client with a connection
ceiling, and only `_MAX_CONCURRENT_INTROSPECTIONS` may be in flight; a caller
that cannot get a slot promptly is refused rather than queued indefinitely.
Refusing under load is the correct direction for a verifier — the answer is
"cannot vouch for this caller", which is what every other failure here says
too.

**Nothing here reaches a log.** The token is a credential in a request body,
the response describes a live session, and the client secret authenticates this
server to the issuer. None of the three appear in a message, an exception, or a
cache key: entries are keyed by digest, and the secret is never held at all —
the config carries the *name* of the environment variable, and the value is
read at the moment the request is built. `repr=False` would not have been
enough, since `asdict`, `__dict__`, and a debugger all walk straight past it.
"""
from __future__ import annotations

import asyncio
import hashlib
import os
from dataclasses import dataclass
from threading import Lock
from time import time
from typing import Any
from urllib.parse import urlparse

from ..optional_dependencies import import_optional_dependency
from .tokens import TokenVerificationError

# How long one introspection call may take. The request sits on the path of
# every unverified caller, so a slow issuer must not become this server's
# latency.
_REQUEST_TIMEOUT_SECONDS = 5.0

# In-flight introspections, and the sockets they may hold. Equal on purpose:
# the point of both is that an unauthenticated flood cannot grow either.
_MAX_CONCURRENT_INTROSPECTIONS = 16
_MAX_CONNECTIONS = 16

# How long a caller waits for one of those slots before being refused. Long
# enough to ride out a burst, short enough that the queue itself cannot become
# the resource an attacker grows.
_SLOT_WAIT_SECONDS = 2.0

# Ceiling on how long a verified token is reused without asking again. Also
# the longest a revoked token keeps working; the two are the same number.
_CACHE_TTL_SECONDS = 60.0

# Cap on cached entries. Bounded because an unauthenticated caller decides how
# many distinct tokens arrive, and each verified one would otherwise be a
# permanent allocation.
_CACHE_MAX_ENTRIES = 4096


@dataclass(frozen=True)
class IntrospectionVerifierConfig:
    """Everything the verifier needs, all of it required.

    No defaults, for the reason `JwtVerifierConfig` has none: each field is a
    security boundary, and a default is this module quietly choosing one.
    """

    issuer: str
    audience: str
    introspection_endpoint: str
    client_id: str
    # The variable's *name*, never the secret it holds. A resolved credential
    # living in a long-lived config object survives `asdict()`, `__dict__`, a
    # debugger, and any generic dump; `repr=False` hides it from exactly one of
    # those. AGENTS.md asks for the reference to be preserved and the value
    # revealed only where it is used, which is `_client_auth` below.
    client_secret_env: str

    def __post_init__(self) -> None:
        for name in (
            "issuer",
            "audience",
            "introspection_endpoint",
            "client_id",
            "client_secret_env",
        ):
            if not str(getattr(self, name)).strip():
                raise TokenVerificationError(
                    f"token introspection requires a non-empty {name}"
                )
        if not os.environ.get(self.client_secret_env, "").strip():
            # Checked once, at startup, so a misconfiguration is not discovered
            # one refused caller at a time. The variable's name is not repeated
            # back: AGENTS.md keeps credential environment-variable names out
            # of diagnostics.
            raise TokenVerificationError(
                "the environment variable named by client_secret_env is unset "
                "or empty; it holds the secret this server authenticates to "
                "the introspection endpoint with"
            )
        scheme = urlparse(self.introspection_endpoint).scheme.lower()
        if scheme != "https":
            raise TokenVerificationError(
                f"introspection_endpoint must be https, got "
                f"{scheme or 'no scheme'!r}: the caller's token is sent to this "
                "URL in the request body, so plaintext hands it to anyone on "
                "the path"
            )


@dataclass(frozen=True)
class _CacheEntry:
    claims: dict[str, Any]
    expires_at: float


class IntrospectionTokenVerifier:
    """Verify bearer tokens by asking the issuer about them (RFC 7662).

    Implements the MCP SDK's `TokenVerifier` protocol: returns `None` for any
    token that does not verify, so the SDK answers unauthenticated and the
    service refuses the call. Every rejection is the same `None` — telling an
    "expired" token apart from a "wrong audience" one in the response is a
    probing oracle.
    """

    def __init__(self, config: IntrospectionVerifierConfig) -> None:
        self._config = config
        self._httpx = import_optional_dependency(
            "httpx", extra="mcp", feature="MCP token introspection"
        )
        self._lock = Lock()
        self._cache: dict[str, _CacheEntry] = {}
        self._slots = asyncio.Semaphore(_MAX_CONCURRENT_INTROSPECTIONS)
        self._client: Any = None

    async def verify_token(self, token: str) -> Any:
        access_token = import_optional_dependency(
            "mcp.server.auth.provider",
            extra="mcp",
            feature="MCP token introspection",
        ).AccessToken
        claims = self._cached(token)
        if claims is None:
            response = await self._introspect(token)
            if response is None:
                return None
            claims = self._accepted_claims(response)
            if claims is None:
                return None
            self._remember(token, claims)
        subject = str(claims.get("sub") or "").strip()
        scopes = claims.get("scope")
        return access_token(
            token=token,
            client_id=str(claims.get("client_id") or subject),
            scopes=scopes.split() if isinstance(scopes, str) else [],
            expires_at=claims.get("exp"),
            subject=subject,
            claims=claims,
        )

    def _pooled_client(self) -> Any:
        """One client for the process, with a ceiling on its sockets.

        A client per call meant a TLS handshake per token and no bound at all
        on how many an unauthenticated flood could open (PR #487 review). It is
        created on first use so construction does not require a running loop,
        and never closed: it lives exactly as long as the server does.
        """
        if self._client is None:
            self._client = self._httpx.AsyncClient(
                timeout=_REQUEST_TIMEOUT_SECONDS,
                limits=self._httpx.Limits(
                    max_connections=_MAX_CONNECTIONS,
                    max_keepalive_connections=_MAX_CONNECTIONS,
                ),
            )
        return self._client

    def _client_auth(self) -> tuple[str, str] | None:
        """The client credentials, read at the moment of use.

        Resolved here rather than held in the config, so the plaintext secret
        exists only for the length of one request. Rereading also means a
        rotated secret takes effect without a restart.
        """
        secret = os.environ.get(self._config.client_secret_env, "")
        if not secret:
            return None
        return (self._config.client_id, secret)

    async def _introspect(self, token: str) -> dict[str, Any] | None:
        """The RFC 7662 call, under the concurrency bound. Returns the parsed
        body, or None for anything that is not a clean 200 — a 401 from bad
        client credentials and a 503 from a down issuer are both "cannot vouch
        for this caller", and so is being too busy to ask.
        """
        auth = self._client_auth()
        if auth is None:
            # Verified at startup, so reaching here means the variable was
            # unset under a running server. Unauthenticated introspection would
            # be refused by the issuer anyway; not asking is the same answer
            # without the round trip.
            return None
        try:
            await asyncio.wait_for(self._slots.acquire(), _SLOT_WAIT_SECONDS)
        except TimeoutError:
            return None
        try:
            response = await self._pooled_client().post(
                self._config.introspection_endpoint,
                data={
                    "token": token,
                    "token_type_hint": "access_token",
                },
                # The scheme the RFC names first, and the one every
                # authorization server implements.
                auth=auth,
                headers={"Accept": "application/json"},
            )
            if response.status_code != 200:
                return None
            body = response.json()
        except Exception:
            # Transport failure, a timeout, or a body that is not JSON. The
            # text of any of them can quote the request — which carries the
            # token and the client secret — so none of it is logged or
            # re-raised (AGENTS.md: provider response bodies stay out of logs).
            return None
        finally:
            self._slots.release()
        return body if isinstance(body, dict) else None

    def _accepted_claims(self, body: dict[str, Any]) -> dict[str, Any] | None:
        """The introspection response, checked. None if it fails any check."""
        # `is True`, not truthy: RFC 7662 defines a JSON boolean, and a server
        # answering `"active": "false"` must not read as a live session.
        if body.get("active") is not True:
            return None
        subject = str(body.get("sub") or "").strip()
        if not subject:
            # Verified but unusable: grants are keyed by subject, so this
            # caller could never be authorized by any grant.
            return None
        if not self._audience_matches(body.get("aud")):
            return None
        issuer = body.get("iss")
        if issuer is not None and str(issuer) != self._config.issuer:
            return None
        expires_at = body.get("exp")
        if isinstance(expires_at, int | float) and expires_at <= time():
            # `active` should already have covered this; a server that says
            # both is not one to take the generous reading from.
            return None
        return body

    def _audience_matches(self, audience: Any) -> bool:
        """Required, and matched exactly.

        An authorization server that omits `aud` from its response cannot be
        used here. That is deliberate: without it there is no way to tell a
        token minted for this deployment from one the caller legitimately
        holds for another service, and accepting both is the confused-deputy
        problem the MCP authorization spec calls out.
        """
        wanted = self._config.audience
        if isinstance(audience, str):
            return audience == wanted
        if isinstance(audience, list):
            return any(isinstance(entry, str) and entry == wanted for entry in audience)
        return False

    def _digest(self, token: str) -> str:
        """The cache key. A digest rather than the token itself, so the cache
        cannot hand a live credential to anything that dumps or iterates it."""
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    def _cached(self, token: str) -> dict[str, Any] | None:
        key = self._digest(token)
        now = time()
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            if entry.expires_at <= now:
                del self._cache[key]
                return None
            return entry.claims

    def _remember(self, token: str, claims: dict[str, Any]) -> None:
        """Cache a verified token until the TTL or its own expiry, whichever
        comes first. A token must never outlive its `exp` here."""
        expires_at = time() + _CACHE_TTL_SECONDS
        token_expiry = claims.get("exp")
        if isinstance(token_expiry, int | float):
            expires_at = min(expires_at, float(token_expiry))
        if expires_at <= time():
            return
        with self._lock:
            if len(self._cache) >= _CACHE_MAX_ENTRIES:
                self._evict(time())
            self._cache[self._digest(token)] = _CacheEntry(claims, expires_at)

    def _evict(self, now: float) -> None:
        """Drop what has expired; if that frees nothing, drop the entry closest
        to expiring. Called under the lock."""
        expired = [key for key, entry in self._cache.items() if entry.expires_at <= now]
        for key in expired:
            del self._cache[key]
        if not expired and self._cache:
            soonest = min(self._cache, key=lambda key: self._cache[key].expires_at)
            del self._cache[soonest]
