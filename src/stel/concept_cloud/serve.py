"""Serve one concept-cloud artifact on loopback, so its cards can reach the
source (issue #678).

The artifact is a static file and the governed MCP is a separate process
speaking JSON-RPC; this module is the join. It serves exactly one HTML file
and exactly one data endpoint from the same origin, which is what lets the
page call a relative path with no token in it and no CORS on the governed
server. The sentence itself comes from `sentences.concept_sentence`, so it is
`ContextService` answering -- the same object `stel mcp serve` wraps.

**The process is the operator.** Identity here is the stdio model: whoever
started the command is the principal, and `ContextService`'s default
`EnvironmentPrincipalResolver` is correct for exactly that reason. It is
therefore a single-operator tool and not a multi-user server, and the
boundary is enforced rather than documented: the bind address is loopback and
cannot be configured, the `Host` header must name loopback, a cross-origin
`Origin` is refused, and no CORS header is ever emitted.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Literal, cast
from urllib.parse import parse_qs, urlparse

from ..mcp_server.catalog import ArtifactCatalog
from ..mcp_server.service import ContextService
from .schema import ConceptCloudExport
from .sentences import (
    Sentence,
    SentenceService,
    SentenceUnavailable,
    concept_sentence,
    declared_labels,
)

log = logging.getLogger(__name__)

# Loopback only, and not a parameter. A `--host` would turn a tool whose
# every caller is the operator into one that answers for an identity it
# cannot establish -- the inversion `serve_network` refuses by making its
# principal resolver required (issue #392). There is no safe value other than
# this one, so there is no option.
_HOST = "127.0.0.1"

# The names the `Host` header may carry, which is the DNS-rebinding defense: a
# page on an attacker's domain that resolves to 127.0.0.1 reaches this socket,
# but its requests carry that domain as `Host`.
_ALLOWED_HOSTNAMES = frozenset({"127.0.0.1", "localhost", "[::1]", "::1"})

_HEALTH_PATH = "/_stel/source/health"
_SOURCE_PATH = "/_stel/source"

# Preference order for the retrieval mode. Resolved once at startup against
# what the model actually declares, rather than per request: a model that
# cannot do `hybrid` would otherwise refuse every click with
# `capability_unavailable` on a request the server could have asked correctly.
_MODE_PREFERENCE: tuple[Literal["hybrid", "vector", "text"], ...] = (
    "hybrid",
    "vector",
    "text",
)


class ConceptCloudServeError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class _Endpoint:
    """Everything a request needs, resolved before the socket opens."""

    service: SentenceService
    model: str
    mode: Literal["vector", "text", "hybrid"]
    declared: frozenset[str]
    # canonical_id -> display, from the bundle being served. The viewer sends
    # an id and nothing else: the search text is this map's value, never a
    # parameter, so the endpoint cannot be driven as a general search proxy by
    # anything else that reaches loopback, and an unknown id is a 404 rather
    # than a query.
    displays: Mapping[str, str]

    def sentence(self, canonical_id: str) -> Sentence | None:
        return concept_sentence(
            self.service,
            model=self.model,
            canonical_id=canonical_id,
            display=self.displays[canonical_id],
            declared=self.declared,
            mode=self.mode,
        )


def _resolve_mode(
    modes: tuple[str, ...], *, model: str
) -> Literal["vector", "text", "hybrid"]:
    for candidate in _MODE_PREFERENCE:
        if candidate in modes:
            return candidate
    raise ConceptCloudServeError(
        f"Context model '{model}' declares no retrieval mode this viewer can "
        f"use (declared: {', '.join(modes) or '(none)'})."
    )


def _prepare(
    export: ConceptCloudExport,
    *,
    project_dir: Path,
    target: str | None,
    profiles_dir: Path | None,
) -> tuple[_Endpoint, ContextService]:
    """Verify the bundle and the project agree, before the socket opens.

    The bundle names a context model but does not promise it exists (see
    `SourceAccess`): the viewer may be served against a different target from
    the one it was exported from. Checking here means a typo or a wrong target
    fails at startup with the available names, not on a reader's first click.
    """
    if export.source_access is None:
        raise ConceptCloudServeError(
            "This bundle declares no context model, so there is no source to "
            "reach. Re-export with `--context-model <model>`."
        )
    model = export.source_access.model
    catalog = ArtifactCatalog.load(project_dir, expected_target=target)
    resource = catalog.get(model)
    if resource is None:
        available = ", ".join(sorted(item.name for item in catalog.all())) or "(none)"
        raise ConceptCloudServeError(
            f"The bundle names context model '{model}', which this project "
            f"does not serve. Available context models: {available}"
        )
    service = ContextService.from_project(
        project_dir, target=target, profiles_dir=profiles_dir
    )
    # Fails loudly at boot on broken credentials rather than as a per-click
    # error, for the reason `serve_stdio` does it (issue #365).
    service.warm_up()
    endpoint = _Endpoint(
        service=service,
        model=model,
        mode=_resolve_mode(resource.modes, model=model),
        declared=declared_labels(catalog.vocabularies),
        displays={c.canonical_id: c.display for c in export.concepts},
    )
    return endpoint, service


def _handler_class(endpoint: _Endpoint, html: bytes) -> type[BaseHTTPRequestHandler]:
    class _Handler(BaseHTTPRequestHandler):
        # No `directory` and no file lookup anywhere below: the one artifact
        # is held in memory and every other path is a 404, so there is no
        # path to traverse.
        server_version = "stel-concept-cloud"
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:
            # The default writes every request to stderr, which would bury
            # the one line the operator needs. Kept at debug so `-vv` can
            # still see the traffic.
            log.debug("concept-cloud serve: " + format, *args)

        def _own_names(self) -> tuple[frozenset[str], frozenset[str]]:
            """The `Host` values and origins that name this server.

            Read from the bound socket rather than from a port passed in, so
            there is no second copy of the port to disagree with the one the
            server actually listens on.
            """
            # `BaseRequestHandler.server` is annotated as the base class,
            # which carries no bound port; every concrete server here is an
            # `HTTPServer`, which does.
            port = cast(HTTPServer, self.server).server_port
            hosts = {f"{name}:{port}" for name in _ALLOWED_HOSTNAMES}
            return (
                frozenset(hosts | _ALLOWED_HOSTNAMES),
                frozenset(f"http://{name}:{port}" for name in _ALLOWED_HOSTNAMES),
            )

        def _reject_foreign_caller(self) -> bool:
            """True when the request was refused.

            Two checks, for two different attacks. `Host` catches DNS
            rebinding, where a hostname an attacker controls resolves here and
            the browser treats their page as same-origin. `Origin` catches the
            plain cross-origin GET: CORS stops the attacker *reading* the
            reply, but the request would still run, and running it means a
            warehouse read and a query-log row attributed to the operator.
            """
            expected_hosts, expected_origins = self._own_names()
            if self.headers.get("Host", "") not in expected_hosts:
                self._refuse(HTTPStatus.FORBIDDEN, "unexpected Host header")
                return True
            origin = self.headers.get("Origin")
            if origin is not None and origin not in expected_origins:
                self._refuse(HTTPStatus.FORBIDDEN, "cross-origin request")
                return True
            return False

        # Named for http.server's contract, not this project's.
        def do_GET(self) -> None:
            if self._reject_foreign_caller():
                return
            parsed = urlparse(self.path)
            if parsed.path in ("/", "/index.html"):
                self._send(HTTPStatus.OK, "text/html; charset=utf-8", html)
                return
            if parsed.path == _HEALTH_PATH:
                self._send_json(
                    HTTPStatus.OK, {"ok": True, "model": endpoint.model}
                )
                return
            if parsed.path == _SOURCE_PATH:
                self._serve_sentence(parse_qs(parsed.query))
                return
            self._refuse(HTTPStatus.NOT_FOUND, "no such path")

        def _serve_sentence(self, query: Mapping[str, list[str]]) -> None:
            concepts = query.get("concept", ())
            if len(concepts) != 1:
                self._refuse(
                    HTTPStatus.BAD_REQUEST, "pass exactly one `concept` id"
                )
                return
            canonical_id = concepts[0]
            if canonical_id not in endpoint.displays:
                self._refuse(
                    HTTPStatus.NOT_FOUND, "no such concept in this bundle"
                )
                return
            # The error boundary for a click: `ContextService` already turns a
            # refusal into a sanitized `ToolError`, and `SentenceUnavailable`
            # carries that pair through. Anything else is a defect and is
            # reported as one without its text, which may name a relation or
            # a credential.
            try:
                sentence = endpoint.sentence(canonical_id)
            except SentenceUnavailable as refusal:
                self._send_json(
                    HTTPStatus.OK,
                    {"error": {"code": refusal.code, "message": refusal.message}},
                )
                return
            except Exception:
                log.exception("concept-cloud sentence lookup failed")
                self._send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {
                        "error": {
                            "code": "internal",
                            "message": (
                                "The sentence lookup failed; see the server log."
                            ),
                        }
                    },
                )
                return
            self._send_json(
                HTTPStatus.OK,
                {"sentence": sentence.as_payload()} if sentence else {"sentence": None},
            )

        def _send_json(self, status: HTTPStatus, payload: Mapping[str, Any]) -> None:
            self._send(
                status,
                "application/json; charset=utf-8",
                json.dumps(payload).encode("utf-8"),
            )

        def _refuse(self, status: HTTPStatus, message: str) -> None:
            self._send_json(status, {"error": {"code": status.phrase, "message": message}})

        def _send(self, status: HTTPStatus, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            # Deliberately no `Access-Control-Allow-Origin`: the page is
            # same-origin, and the absence is what stops any other page in the
            # browser from reading this reply.
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

    return _Handler


def serve_concept_cloud(
    export: ConceptCloudExport,
    artifact: Path,
    *,
    project_dir: Path,
    target: str | None = None,
    profiles_dir: Path | None = None,
    port: int = 8081,
) -> None:
    """Serve `artifact` and its source endpoint on loopback. Ctrl-C to stop.

    `export` is the bundle that produced the file: the server reads concept
    ids and display names from it, and the file from disk, so what a reader
    sees is what was written rather than a second rendering of it.
    """
    html = artifact.read_bytes()
    endpoint, service = _prepare(
        export, project_dir=project_dir, target=target, profiles_dir=profiles_dir
    )
    handler_cls = _handler_class(endpoint, html)
    # Threading so one slow retrieval cannot stall the page itself.
    # `ContextService` is built for concurrent callers -- it is what the MCP
    # transports serve through -- and bounds them in its own limiter.
    httpd = ThreadingHTTPServer((_HOST, port), handler_cls)
    try:
        print(
            f"Serving {artifact.name} at http://{_HOST}:{port}/  "
            f"(sentences from '{endpoint.model}' via {endpoint.mode}; Ctrl-C to stop)"
        )
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        httpd.server_close()
        # Flushes the query log the clicks wrote (issue #528).
        service.close()
