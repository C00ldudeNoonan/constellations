"""JWT bearer verification for the network transport (issue #392, item 4).

#394 gave the network transport a per-request principal, but only through
headers a proxy in front is trusted to set. That is sound behind an
authenticating proxy and nothing at all without one, so `--transport` has
required naming that trust boundary. This module is the other option: verify
the caller's own bearer token, so a deployment with no proxy in front is not
reduced to auth theater.

**Identity only.** #396 moved authorization to operator-owned grants — groups
and tenants are looked up by subject, never carried — so a verified token has
to establish exactly one thing: who is calling. Nothing here reads a scope, a
group, or a tenant out of a token, and a token that claims them gains nothing
by it. That is what keeps this small: the security question is "is this
subject who it says it is", not "what may it see".

**What is checked, and why each one matters.** All of these are refusals, not
warnings, for the reason the whole issue exists: a mis-scoped answer looks
exactly like a correct one.

- *Signature*, against the issuer's published keys.
- *Asymmetric algorithms only.* Accepting an HMAC algorithm alongside RSA is
  the classic JWKS forgery: an attacker signs a token with the public key as
  the HMAC secret, and a verifier that will take either accepts it. `none` is
  refused for the obvious reason.
- *Issuer*, exactly. A token from another issuer is another system's.
- *Audience*, exactly, and required. This is the confused-deputy defense the
  MCP authorization spec calls out: without it, a token a caller legitimately
  holds for *some other service* is a valid token here.
- *Expiry and not-before*, with a small leeway for clock skew.
- *A subject.* A verified token with no subject cannot be matched to a grant,
  and the caller would arrive as an unauthenticated request rather than a
  wrong one — but refusing here says why.

**HTTPS only for the key source.** Keys fetched over plaintext can be replaced
in transit, which makes every other check above decorative.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from ..optional_dependencies import import_optional_dependency

# RSA, ECDSA and RSA-PSS. Deliberately no HMAC family: a JWKS verifier that
# accepts one can be forged with the published public key as the shared
# secret, and `none` needs no explanation.
ALLOWED_ALGORITHMS = (
    "RS256",
    "RS384",
    "RS512",
    "ES256",
    "ES384",
    "ES512",
    "PS256",
    "PS384",
    "PS512",
)

# Clock skew tolerated on `exp`/`nbf`. Small on purpose: this widens the window
# in which a revoked-by-expiry token still works.
_LEEWAY_SECONDS = 30.0


class TokenVerificationError(Exception):
    """The verifier could not be configured. Not raised per request."""


@dataclass(frozen=True)
class JwtVerifierConfig:
    """Everything the verifier needs, all of it required.

    No defaults anywhere. Each of these is a security boundary, and a default
    would be this module quietly choosing one — the failure mode being a
    server that verifies signatures beautifully against the wrong issuer.
    """

    issuer: str
    audience: str
    jwks_uri: str

    def __post_init__(self) -> None:
        for name in ("issuer", "audience", "jwks_uri"):
            if not str(getattr(self, name)).strip():
                raise TokenVerificationError(
                    f"JWT verification requires a non-empty {name}"
                )
        scheme = urlparse(self.jwks_uri).scheme.lower()
        if scheme != "https":
            raise TokenVerificationError(
                f"jwks_uri must be https, got {scheme or 'no scheme'!r}: keys "
                "fetched in plaintext can be replaced in transit, which makes "
                "verifying against them meaningless"
            )


class JwksTokenVerifier:
    """Verify bearer tokens against an issuer's published JWKS.

    Implements the MCP SDK's `TokenVerifier` protocol. Returns `None` for any
    token that does not verify — the SDK turns that into an unauthenticated
    response, and the service then refuses the call. Returning `None` rather
    than raising is the protocol's contract, and it also means a malformed
    token cannot produce a stack trace carrying its contents.
    """

    def __init__(self, config: JwtVerifierConfig) -> None:
        self._config = config
        jwt = import_optional_dependency(
            "jwt", extra="mcp", feature="MCP JWT token verification"
        )
        self._jwt = jwt
        # Caches keys and refetches on an unknown `kid`, so a routine issuer
        # key rotation does not need a restart.
        self._keys = jwt.PyJWKClient(config.jwks_uri)

    async def verify_token(self, token: str) -> Any:
        access_token = import_optional_dependency(
            "mcp.server.auth.provider",
            extra="mcp",
            feature="MCP JWT token verification",
        ).AccessToken
        claims = self._decode(token)
        if claims is None:
            return None
        subject = str(claims.get("sub") or "").strip()
        if not subject:
            # Verified but unusable: grants are keyed by subject, so this
            # caller could never be authorized. Refused here so the reason is
            # "no subject" rather than an anonymous-looking request.
            return None
        scopes = claims.get("scope")
        return access_token(
            token=token,
            client_id=str(claims.get("client_id") or subject),
            scopes=scopes.split() if isinstance(scopes, str) else [],
            expires_at=claims.get("exp"),
            subject=subject,
            claims=claims,
        )

    def _decode(self, token: str) -> dict[str, Any] | None:
        """Verified claims, or None. Never raises on caller-supplied input."""
        try:
            key = self._keys.get_signing_key_from_jwt(token).key
            return dict(
                self._jwt.decode(
                    token,
                    key,
                    algorithms=list(ALLOWED_ALGORITHMS),
                    issuer=self._config.issuer,
                    audience=self._config.audience,
                    leeway=_LEEWAY_SECONDS,
                    options={
                        "require": ["exp", "iss", "aud", "sub"],
                        "verify_signature": True,
                        "verify_exp": True,
                        "verify_nbf": True,
                        "verify_iss": True,
                        "verify_aud": True,
                    },
                )
            )
        except Exception:
            # Every failure is the same answer to the caller. Distinguishing
            # "bad signature" from "wrong audience" in the response is a
            # probing oracle, and the exception text can carry token contents
            # into a log (AGENTS.md: sensitive exception text stays out of
            # logs and artifacts).
            return None
