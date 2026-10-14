"""The loopback viewer server, and the boundary it enforces (issue #678).

`stel concept-cloud --serve` exists so a static artifact can reach a governed
service from the same origin, with no token in the page and no CORS on the
governed server. That makes the server itself the security boundary, and most
of what is pinned here is that boundary rather than the feature: the endpoint
answers only for concepts the bundle contains, only to requests that name
this server, and never with a header that would let another page read the
reply.

Driven over a real socket with `http.client` rather than against the handler
class, because `Host` and `Origin` are the subject of three of these tests
and a direct call would not carry them.
"""

from __future__ import annotations

import inspect
import json
import threading
from collections.abc import Iterator
from http import HTTPStatus
from http.client import HTTPConnection, HTTPResponse
from http.server import ThreadingHTTPServer
from typing import Any

import pytest

from stel.concept_cloud.schema import SourceAccess, placeholder_export
from stel.concept_cloud.serve import (
    ConceptCloudServeError,
    _Endpoint,
    _handler_class,
    _prepare,
    _resolve_mode,
    serve_concept_cloud,
)
from stel.manifest import SERVING_MANIFEST_VERSION
from stel.mcp_server.catalog import ArtifactCatalog
from stel.mcp_server.contracts import (
    MCPErrorCode,
    SearchContextResponse,
    ToolError,
)
from tests.support_concept_cloud import FakeContextService, search_result

_HTML = b"<!doctype html><title>star map</title><body>map</body>"
_MODEL = "sec_context_chunks"


def _endpoint(response: SearchContextResponse) -> _Endpoint:
    return _Endpoint(
        service=FakeContextService(response),
        model=_MODEL,
        mode="hybrid",
        declared=frozenset({"Federal Reserve"}),
        displays={"Federal Reserve": "Federal Reserve", "Acme": "Acme Corp"},
    )


class _Client:
    """One HTTP connection, with `Host` and `Origin` under the test's control."""

    def __init__(self, port: int) -> None:
        self.port = port

    def get(
        self, path: str, *, host: str | None = None, origin: str | None = None
    ) -> tuple[HTTPResponse, bytes]:
        conn = HTTPConnection("127.0.0.1", self.port, timeout=10)
        try:
            conn.putrequest("GET", path, skip_host=True, skip_accept_encoding=True)
            conn.putheader("Host", host or f"127.0.0.1:{self.port}")
            if origin is not None:
                conn.putheader("Origin", origin)
            conn.putheader("Connection", "close")
            conn.endheaders()
            response = conn.getresponse()
            return response, response.read()
        finally:
            conn.close()

    def json(self, path: str, **kwargs: str) -> tuple[int, dict[str, Any]]:
        response, body = self.get(path, **kwargs)
        return response.status, json.loads(body)


@pytest.fixture
def served(request: pytest.FixtureRequest) -> Iterator[_Client]:
    """A running server for the response the test asks for via `indirect`."""
    response = getattr(request, "param", None) or SearchContextResponse(
        results=(search_result(),)
    )
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _handler_class(_endpoint(response), _HTML))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield _Client(httpd.server_port)
    finally:
        httpd.shutdown()
        thread.join(timeout=10)
        httpd.server_close()


def test_the_artifact_is_served_at_the_root(served: _Client) -> None:
    for path in ("/", "/index.html"):
        response, body = served.get(path)
        assert response.status == HTTPStatus.OK
        assert body == _HTML
        assert response.headers["Content-Type"] == "text/html; charset=utf-8"


def test_the_health_probe_names_the_model_the_viewer_will_search(
    served: _Client,
) -> None:
    """The viewer gates its button on this, so it has to answer before any
    concept is asked for -- and name the model, so an operator can see which
    one a click will reach."""
    status, body = served.json("/_stel/source/health")

    assert status == HTTPStatus.OK
    assert body == {"ok": True, "model": _MODEL}


def test_a_concept_in_the_bundle_yields_its_sentence(served: _Client) -> None:
    status, body = served.json("/_stel/source?concept=Federal%20Reserve")

    assert status == HTTPStatus.OK
    assert body["sentence"]["snippet"].startswith("The Federal Reserve raised")
    assert body["sentence"]["scope"] == "term"


def test_a_concept_outside_the_bundle_is_refused_rather_than_searched(
    served: _Client,
) -> None:
    """The closed set is the point. The endpoint takes an id and looks the
    search text up in the bundle, so nothing else that reaches loopback can
    drive it as a general search proxy over the corpus."""
    status, body = served.json("/_stel/source?concept=anything%20at%20all")

    assert status == HTTPStatus.NOT_FOUND
    assert "no such concept" in body["error"]["message"]


def test_a_missing_or_repeated_concept_parameter_is_a_bad_request(
    served: _Client,
) -> None:
    for query in ("", "?concept=", "?concept=Acme&concept=Federal%20Reserve"):
        status, body = served.json(f"/_stel/source{query}")
        # An empty value is a well-formed parameter naming no concept, so it
        # is a 404 on the lookup rather than a malformed request.
        assert status in (HTTPStatus.BAD_REQUEST, HTTPStatus.NOT_FOUND), query
        assert body["error"]


def test_a_host_header_this_server_does_not_answer_to_is_refused(
    served: _Client,
) -> None:
    """DNS rebinding: a name an attacker controls can resolve to 127.0.0.1,
    and their page is then same-origin as far as the browser is concerned.
    What it cannot do is send a `Host` that names loopback."""
    status, body = served.json(
        "/_stel/source?concept=Federal%20Reserve", host="evil.example.com"
    )

    assert status == HTTPStatus.FORBIDDEN
    assert "Host" in body["error"]["message"]


def test_a_cross_origin_request_is_refused_before_it_reads_anything(
    served: _Client,
) -> None:
    """CORS stops another page *reading* the reply, but the request would
    still have run -- and running it means a warehouse read and a query-log
    row attributed to the operator."""
    status, _ = served.json(
        "/_stel/source?concept=Federal%20Reserve", origin="https://evil.example.com"
    )

    assert status == HTTPStatus.FORBIDDEN


def test_the_page_origin_is_accepted(served: _Client) -> None:
    """The counterpart to the refusal above: the viewer's own fetch carries
    an `Origin`, so refusing every one would refuse the only caller."""
    status, body = served.json(
        "/_stel/source?concept=Federal%20Reserve",
        origin=f"http://127.0.0.1:{served.port}",
    )

    assert status == HTTPStatus.OK
    assert body["sentence"]


def test_no_response_ever_carries_a_cors_header(served: _Client) -> None:
    """The absence is the defense, so it is pinned on every kind of response
    rather than on the one a test happened to ask for."""
    for path in (
        "/",
        "/_stel/source/health",
        "/_stel/source?concept=Federal%20Reserve",
        "/_stel/source?concept=nope",
        "/nothing-here",
    ):
        response, _ = served.get(path)
        assert "Access-Control-Allow-Origin" not in response.headers, path
        assert "Access-Control-Allow-Credentials" not in response.headers, path


def test_an_unknown_path_is_not_found(served: _Client) -> None:
    """There is no file lookup behind any path: the one artifact is held in
    memory, so there is no directory to traverse out of."""
    for path in ("/nothing-here", "/../../etc/passwd", "/_stel/", "/_stel/source/x"):
        response, _ = served.get(path)
        assert response.status == HTTPStatus.NOT_FOUND, path


@pytest.mark.parametrize(
    "served",
    [
        SearchContextResponse(
            error=ToolError(
                code=MCPErrorCode.NOT_FOUND_OR_DENIED,
                message="The context model is not available to this caller",
            )
        )
    ],
    indirect=True,
)
def test_a_refusal_reaches_the_card_as_an_error_not_a_crash(
    served: _Client,
) -> None:
    """A refusal is an answer the card can render -- "source unavailable" plus
    the service's own sanitized message -- so it is a 200 carrying an error,
    not a 500 the viewer would report as the server being down."""
    status, body = served.json("/_stel/source?concept=Federal%20Reserve")

    assert status == HTTPStatus.OK
    assert body["error"]["code"] == "not_found_or_denied"
    assert body["error"]["message"] == (
        "The context model is not available to this caller"
    )


@pytest.mark.parametrize(
    "served", [SearchContextResponse(results=())], indirect=True
)
def test_nothing_matching_is_a_null_sentence(served: _Client) -> None:
    status, body = served.json("/_stel/source?concept=Federal%20Reserve")

    assert status == HTTPStatus.OK
    assert body == {"sentence": None}


# ── preflight ───────────────────────────────────────────────────────────────


def test_a_bundle_with_no_context_model_cannot_be_served(tmp_path) -> None:
    """Checked before the project is read, so the message is about the bundle
    rather than whatever the project turns out to be missing."""
    with pytest.raises(ConceptCloudServeError, match=r"--context-model"):
        _prepare(
            placeholder_export(),
            project_dir=tmp_path,
            target=None,
            profiles_dir=None,
        )


def test_a_model_the_project_does_not_serve_is_refused_at_startup(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bundle names a model but does not promise it exists: the viewer may
    be served against a different target from the export's. The refusal names
    what is available, because a typo and a wrong target look identical
    otherwise."""
    catalog = ArtifactCatalog.from_payloads(
        {
            "manifest_version": SERVING_MANIFEST_VERSION,
            "declarations": {"vocabularies": {}, "classes": []},
            "target": {"name": "dev"},
            "models": [],
            "dag": {},
        }
    )
    monkeypatch.setattr(
        "stel.concept_cloud.serve.ArtifactCatalog.load",
        classmethod(lambda cls, project_dir, *, expected_target=None: catalog),
    )
    export = placeholder_export().model_copy(
        update={"source_access": SourceAccess(model="missing_model")}
    )

    with pytest.raises(ConceptCloudServeError) as caught:
        _prepare(export, project_dir=tmp_path, target=None, profiles_dir=None)

    assert "missing_model" in str(caught.value)
    assert "Available context models: (none)" in str(caught.value)


def test_the_retrieval_mode_is_resolved_against_what_the_model_declares() -> None:
    """Resolved once at startup rather than per click: a model that cannot do
    `hybrid` would otherwise refuse every click with `capability_unavailable`
    on a request the server could have asked correctly."""
    assert _resolve_mode(("text", "vector", "hybrid"), model="m") == "hybrid"
    assert _resolve_mode(("text", "vector"), model="m") == "vector"
    assert _resolve_mode(("text",), model="m") == "text"

    with pytest.raises(ConceptCloudServeError, match=r"no retrieval mode"):
        _resolve_mode((), model="m")


def test_the_bind_address_is_not_configurable() -> None:
    """A `--host` would turn a tool whose every caller is the operator into
    one answering for an identity it cannot establish -- the inversion
    `serve_network` refuses by making its principal resolver required (#392).
    There is no safe value other than loopback, so there is no parameter: this
    fails if one is ever added without reading that argument."""
    assert "host" not in inspect.signature(serve_concept_cloud).parameters
