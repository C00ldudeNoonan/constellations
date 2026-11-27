"""The wire contract stel's MCP server presents to clients (ALE-751, issue #805).

stel builds its server on the MCP Python SDK v1 and means to move to v2
(ALE-752). The tests elsewhere call the service, or the SDK's in-memory
session, and so say nothing about what a *client* sees: the bytes on stdin and
stdout, or on a socket. This file is that baseline. It runs a real
`stel mcp serve` process (see `tests/support_mcp_wire.py`) and speaks to it as
a client does, so the upgrade can be checked against behavior clients already
depend on rather than against whatever the SDK's internals happen to be.

**The clients here are raw JSON-RPC.** Newline-delimited JSON on stdio, plain
HTTP POSTs for streamable-http, an event stream plus POSTs for SSE. None of
them imports `mcp`: a test written against the SDK's client would move with
the SDK and could not tell the upgrade it was guarding had changed anything.

What is pinned is what an operator documents (docs/mcp.md) or a client reads:
the five tool names, each tool's argument and result contract, the shape of the
`mcp_context/v2` responses, the refusals at the HTTP boundary, the OAuth
discovery document, and the client identity a query-log row records. What is
not pinned is the SDK's own wording: `serverInfo.version` is the SDK's version
and a validation message is pydantic's, so neither is asserted.

No live credentials are involved. Tokens are signed with a throwaway RSA key
whose public half is the only thing the verifier is told about.
"""
from __future__ import annotations

import json
import os
import queue
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Protocol

import httpx
import pytest

from stel.mcp_server.contracts import MCP_CONTEXT_SCHEMA_VERSION
from tests.support_mcp_wire import (
    AUTHORIZATION_ENV,
    CHUNK_ALLOWED_1,
    CHUNK_HIDDEN,
    CONTEXT_ALLOWED_1,
    DOC_ALLOWED,
    DOC_HIDDEN,
    JWKS_ENV,
    QUERY_LOG_ENV,
    VERSION_ALLOWED,
    VERSION_HIDDEN,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

PROTOCOL_VERSION = "2025-06-18"
CLIENT_NAME = "wire-client"
CLIENT_VERSION = "9.9.9"

ISSUER = "https://issuer.example"
PUBLIC_URL = "https://stel.example/mcp"
JWKS_URI = "https://issuer.example/.well-known/jwks.json"
METADATA_URL = "https://stel.example/.well-known/oauth-protected-resource/mcp"

TOOL_NAMES = {
    "list_context_models",
    "search_context",
    "get_document",
    "get_context_lineage",
    "get_entity_neighbors",
}

# Each tool as a client reads it from `tools/list`: argument name -> the
# default it advertises (`REQUIRED` when it has none), which arguments must be
# given, and the top-level fields of its structured result. Written out by
# hand, not derived from the server, because a table derived from the thing it
# checks would pass whatever the thing became.
REQUIRED = "<required>"
TOOL_CONTRACT: dict[str, dict[str, Any]] = {
    "list_context_models": {
        "arguments": {"limit": 20, "cursor": None, "schema_version": "mcp_context/v2"},
        "required": [],
        "result": ["schema_version", "models", "next_cursor", "error"],
    },
    "search_context": {
        "arguments": {
            "model": REQUIRED,
            "query": REQUIRED,
            "mode": "hybrid",
            "limit": 10,
            "candidate_limit": None,
            "filters": None,
            "entity_scope": None,
            "detail": "lean",
            "schema_version": "mcp_context/v2",
        },
        "required": ["model", "query"],
        "result": [
            "schema_version",
            "results",
            "detail",
            "degraded",
            "entity_scope_applied",
            "safe_error_code",
            "error",
        ],
    },
    "get_document": {
        "arguments": {
            "model": REQUIRED,
            "document_id": REQUIRED,
            "document_version_id": REQUIRED,
            "limit": 20,
            "cursor": None,
            "schema_version": "mcp_context/v2",
        },
        "required": ["model", "document_id", "document_version_id"],
        "result": [
            "schema_version",
            "document_id",
            "document_version_id",
            "source",
            "interval",
            "freshness",
            "lineage",
            "chunks",
            "next_cursor",
            "error",
        ],
    },
    "get_context_lineage": {
        "arguments": {
            "model": REQUIRED,
            "reference_type": REQUIRED,
            "reference_id": REQUIRED,
            "schema_version": "mcp_context/v2",
        },
        "required": ["model", "reference_type", "reference_id"],
        "result": ["schema_version", "record", "error"],
    },
    "get_entity_neighbors": {
        "arguments": {
            "model": REQUIRED,
            "entity_id": None,
            "term": None,
            "hops": 1,
            "relation_types": None,
            "as_of": None,
            "limit": 25,
            "schema_version": "mcp_context/v2",
        },
        "required": ["model"],
        "result": [
            "schema_version",
            "start",
            "edges",
            "hops",
            "as_of",
            "truncated",
            "error",
        ],
    },
}

RESEARCH = {
    "x-stel-principal-id": "local-user",
    "x-stel-tenant-id": "research",
    "x-stel-policy-claims": json.dumps({"classification": "internal"}),
}
RESTRICTED = {
    "x-stel-principal-id": "other-user",
    "x-stel-tenant-id": "restricted",
    "x-stel-policy-claims": json.dumps({"classification": "internal"}),
}

STARTUP_SECONDS = 90
REPLY_SECONDS = 20


class WireError(Exception):
    """A JSON-RPC error object, as the server sent it."""

    def __init__(self, error: Mapping[str, Any]) -> None:
        super().__init__(error.get("message"))
        self.code = error.get("code")


# ─── clients ───────────────────────────────────────────────────────────────


class WireClient(Protocol):
    headers: dict[str, str]
    last_request_id: int
    initialized: Mapping[str, Any]

    def rpc(self, method: str, params: Mapping[str, Any] | None = None) -> Any: ...

    def close(self) -> None: ...


def _initialize_params() -> dict[str, Any]:
    return {
        "protocolVersion": PROTOCOL_VERSION,
        "capabilities": {},
        "clientInfo": {"name": CLIENT_NAME, "version": CLIENT_VERSION},
    }


def _unwrap(message: Mapping[str, Any]) -> Any:
    if "error" in message:
        raise WireError(message["error"])
    return message["result"]


class _StdioClient:
    """Newline-delimited JSON-RPC to a child's stdin and stdout."""

    def __init__(self, process: subprocess.Popen[str]) -> None:
        self._process = process
        self.headers: dict[str, str] = {}
        self.last_request_id = 0
        self._replies: queue.Queue[dict[str, Any]] = queue.Queue()
        threading.Thread(target=self._read, daemon=True).start()
        self.initialized = self.rpc("initialize", _initialize_params())
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def _read(self) -> None:
        assert self._process.stdout is not None
        for line in self._process.stdout:
            if line.strip():
                self._replies.put(json.loads(line))

    def _send(self, message: Mapping[str, Any]) -> None:
        assert self._process.stdin is not None
        self._process.stdin.write(json.dumps(message) + "\n")
        self._process.stdin.flush()

    def rpc(self, method: str, params: Mapping[str, Any] | None = None) -> Any:
        self.last_request_id += 1
        self._send(
            {
                "jsonrpc": "2.0",
                "id": self.last_request_id,
                "method": method,
                **({"params": params} if params is not None else {}),
            }
        )
        reply = self._replies.get(timeout=REPLY_SECONDS)
        assert reply["id"] == self.last_request_id
        return _unwrap(reply)

    def close(self) -> None:
        assert self._process.stdin is not None
        self._process.stdin.close()
        self._process.wait(timeout=30)


class _StreamableHttpClient:
    """One session of plain HTTP POSTs to `/mcp`.

    The server runs with `json_response=True`, so each reply is a JSON body
    and no event-stream parsing is needed to read it.
    """

    def __init__(self, base_url: str, headers: Mapping[str, str]) -> None:
        self._http = httpx.Client(base_url=base_url, timeout=REPLY_SECONDS)
        self.headers = dict(headers)
        self.last_request_id = 0
        self.session_id: str | None = None
        self.initialized = self.rpc("initialize", _initialize_params())
        self._post({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def _post(self, message: Mapping[str, Any]) -> httpx.Response:
        headers = {
            "accept": "application/json, text/event-stream",
            "content-type": "application/json",
            **self.headers,
        }
        if self.session_id is not None:
            headers["mcp-session-id"] = self.session_id
            headers["mcp-protocol-version"] = PROTOCOL_VERSION
        return self._http.post("/mcp", json=message, headers=headers)

    def rpc(self, method: str, params: Mapping[str, Any] | None = None) -> Any:
        self.last_request_id += 1
        response = self._post(
            {
                "jsonrpc": "2.0",
                "id": self.last_request_id,
                "method": method,
                **({"params": params} if params is not None else {}),
            }
        )
        assert response.status_code == 200, response.text
        self.session_id = response.headers.get("mcp-session-id", self.session_id)
        return _unwrap(response.json())

    def close(self) -> None:
        self._http.close()


class _SseClient:
    """The legacy SSE transport: replies arrive on a stream, requests are POSTed."""

    def __init__(self, base_url: str, headers: Mapping[str, str]) -> None:
        self._http = httpx.Client(base_url=base_url, timeout=None)
        self.headers = dict(headers)
        self.last_request_id = 0
        self._endpoint: str | None = None
        self._ready = threading.Event()
        self._messages: queue.Queue[dict[str, Any]] = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()
        assert self._ready.wait(REPLY_SECONDS), "the SSE stream never named an endpoint"
        self.initialized = self.rpc("initialize", _initialize_params())
        self._post({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def _pump(self) -> None:
        event = ""
        try:
            with self._http.stream("GET", "/sse", headers=self.headers) as response:
                for line in response.iter_lines():
                    if line.startswith("event:"):
                        event = line.removeprefix("event:").strip()
                    elif line.startswith("data:"):
                        data = line.removeprefix("data:").strip()
                        if event == "endpoint":
                            self._endpoint = data
                            self._ready.set()
                        elif event == "message":
                            self._messages.put(json.loads(data))
        except httpx.HTTPError:  # the client closed the stream
            return

    def _post(self, message: Mapping[str, Any]) -> None:
        assert self._endpoint is not None
        response = self._http.post(
            self._endpoint,
            json=message,
            headers={"content-type": "application/json", **self.headers},
            timeout=REPLY_SECONDS,
        )
        assert response.status_code == 202, response.text

    def rpc(self, method: str, params: Mapping[str, Any] | None = None) -> Any:
        self.last_request_id += 1
        self._post(
            {
                "jsonrpc": "2.0",
                "id": self.last_request_id,
                "method": method,
                **({"params": params} if params is not None else {}),
            }
        )
        reply = self._messages.get(timeout=REPLY_SECONDS)
        assert reply["id"] == self.last_request_id
        return _unwrap(reply)

    def close(self) -> None:
        try:
            self._http.close()
        except RuntimeError:  # the pump thread still holds the stream
            return


# ─── processes ─────────────────────────────────────────────────────────────


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _wait_for_port(port: int, process: subprocess.Popen[Any], log: Path) -> None:
    deadline = time.monotonic() + STARTUP_SECONDS
    while time.monotonic() < deadline:
        if process.poll() is not None:
            pytest.fail(f"stel mcp serve exited early:\n{log.read_text(encoding='utf-8')}")
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.2)
    pytest.fail(f"stel mcp serve never listened:\n{log.read_text(encoding='utf-8')}")


def _spawn(
    project_dir: Path,
    serve_args: list[str],
    env: Mapping[str, str],
    *,
    pipes: bool,
    stderr: Any,
) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "tests.support_mcp_wire",
            "--project-dir",
            str(project_dir),
            "mcp",
            "serve",
            *serve_args,
        ],
        cwd=REPO_ROOT,
        env={**os.environ, **env},
        stdin=subprocess.PIPE if pipes else subprocess.DEVNULL,
        stdout=subprocess.PIPE if pipes else subprocess.DEVNULL,
        stderr=stderr,
        text=True,
        encoding="utf-8",
    )


class Served:
    """A running server, how to reach it, and where its query log lands."""

    def __init__(
        self,
        transport: str,
        port: int | None,
        query_log: Path,
        connect: Callable[[Mapping[str, str]], WireClient],
    ) -> None:
        self.transport = transport
        self.port = port
        self.query_log = query_log
        self.connect = connect

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def mark(self) -> int:
        """How many log rows exist now, so a later `logged` reads only newer ones."""
        if not self.query_log.exists():
            return 0
        return len(self.query_log.read_text(encoding="utf-8").splitlines())

    def logged(self, request_id: int, *, tool: str, since: int) -> dict[str, Any]:
        """The row a call wrote after `since`, waiting for the writer's batch to land.

        Request ids restart at each client, and the server keeps every earlier
        session's rows, so matching on id alone can return a previous test's row.
        """
        deadline = time.monotonic() + REPLY_SECONDS
        while time.monotonic() < deadline:
            if self.query_log.exists():
                lines = self.query_log.read_text(encoding="utf-8").splitlines()
                for line in lines[since:]:
                    row = json.loads(line)
                    if row.get("request_id") == str(request_id) and row.get("tool") == tool:
                        return dict(row)
            time.sleep(0.2)
        pytest.fail(f"no query-log row for request {request_id} ({tool})")


@contextmanager
def _network_server(
    tmp_path: Path,
    transport: str,
    extra_args: list[str],
    extra_env: Mapping[str, str],
) -> Iterator[Served]:
    project = tmp_path / "project"
    project.mkdir()
    (project / "stel_project.yml").write_text("name: p\nversion: '0.1.0'\n", encoding="utf-8")
    query_log = tmp_path / "query_log.jsonl"
    port = _free_port()
    stderr_path = tmp_path / "stderr.txt"
    with stderr_path.open("w", encoding="utf-8") as stderr:
        process = _spawn(
            project,
            ["--transport", transport, "--host", "127.0.0.1", "--port", str(port), *extra_args],
            {QUERY_LOG_ENV: str(query_log), **extra_env},
            pipes=False,
            stderr=stderr,
        )
        clients: list[WireClient] = []
        try:
            _wait_for_port(port, process, stderr_path)
            factory = _StreamableHttpClient if transport == "streamable-http" else _SseClient

            def connect(headers: Mapping[str, str]) -> WireClient:
                client = factory(f"http://127.0.0.1:{port}", headers)
                clients.append(client)
                return client

            yield Served(transport, port, query_log, connect)
        finally:
            for client in clients:
                client.close()
            process.terminate()
            process.wait(timeout=30)


@contextmanager
def _stdio_server(tmp_path: Path) -> Iterator[Served]:
    project = tmp_path / "project"
    project.mkdir()
    (project / "stel_project.yml").write_text("name: p\nversion: '0.1.0'\n", encoding="utf-8")
    query_log = tmp_path / "query_log.jsonl"
    clients: list[_StdioClient] = []

    def connect(headers: Mapping[str, str]) -> WireClient:
        # The operator running the process is the principal; there are no
        # request headers on stdio, so `headers` is deliberately unused. One
        # process serves the whole module: starting stel costs seconds, and a
        # session's request ids only ever increase.
        if clients:
            return clients[0]
        process = _spawn(
            project,
            [],
            {
                QUERY_LOG_ENV: str(query_log),
                "STEL_MCP_PRINCIPAL_ID": "local-user",
                "STEL_MCP_TENANT_ID": "research",
                "STEL_MCP_POLICY_CLAIMS": json.dumps({"classification": "internal"}),
            },
            pipes=True,
            stderr=subprocess.DEVNULL,
        )
        client = _StdioClient(process)
        clients.append(client)
        return client

    try:
        yield Served("stdio", None, query_log, connect)
    finally:
        for client in clients:
            client.close()


# ─── fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def stdio_server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Served]:
    with _stdio_server(tmp_path_factory.mktemp("stdio")) as served:
        yield served


@pytest.fixture(scope="module")
def http_server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Served]:
    with _network_server(
        tmp_path_factory.mktemp("http"),
        "streamable-http",
        ["--trust-proxy-principal-headers"],
        {},
    ) as served:
        yield served


@pytest.fixture(scope="module")
def sse_server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Served]:
    with _network_server(
        tmp_path_factory.mktemp("sse"), "sse", ["--trust-proxy-principal-headers"], {}
    ) as served:
        yield served


@pytest.fixture(scope="module")
def rsa_key() -> Any:
    from cryptography.hazmat.primitives.asymmetric import rsa

    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


def _mint(key: Any, **overrides: Any) -> str:
    import jwt

    claims = {
        "iss": ISSUER,
        "aud": PUBLIC_URL,
        "sub": "alice",
        "exp": int(time.time()) + 300,
        **overrides,
    }
    return jwt.encode(claims, key, algorithm="RS256", headers={"kid": "wire-key"})


@pytest.fixture(scope="module")
def jwt_server(tmp_path_factory: pytest.TempPathFactory, rsa_key: Any) -> Iterator[Served]:
    """Token verification and operator-owned grants, the documented hosted setup."""
    from jwt.algorithms import RSAAlgorithm

    tmp_path = tmp_path_factory.mktemp("jwt")
    jwk = json.loads(RSAAlgorithm.to_jwk(rsa_key.public_key()))
    jwk.update({"kid": "wire-key", "alg": "RS256", "use": "sig"})
    jwks = tmp_path / "jwks.json"
    jwks.write_text(json.dumps({"keys": [jwk]}), encoding="utf-8")
    with _network_server(
        tmp_path,
        "streamable-http",
        [
            "--public-url", PUBLIC_URL,
            "--jwt-issuer", ISSUER,
            "--jwt-audience", PUBLIC_URL,
            "--jwt-jwks-uri", JWKS_URI,
        ],
        {JWKS_ENV: str(jwks), AUTHORIZATION_ENV: "grants"},
    ) as served:
        yield served


@pytest.fixture(params=["stdio", "streamable-http", "sse"])
def server(request: pytest.FixtureRequest) -> Served:
    """The same assertions, run against every transport `stel mcp serve` offers."""
    served: Served = request.getfixturevalue(
        {"stdio": "stdio_server", "streamable-http": "http_server", "sse": "sse_server"}[
            request.param
        ]
    )
    return served


@pytest.fixture
def client(server: Served) -> Iterator[WireClient]:
    session = server.connect(RESEARCH)
    try:
        yield session
    finally:
        if server.transport != "stdio":
            session.close()


def _call(session: WireClient, name: str, arguments: Mapping[str, Any]) -> Any:
    return session.rpc("tools/call", {"name": name, "arguments": dict(arguments)})


# ─── the handshake ─────────────────────────────────────────────────────────


def test_the_handshake_negotiates_the_protocol_and_names_the_server(
    client: WireClient,
) -> None:
    """What a client reads before it sends a tool call."""
    handshake = client.initialized

    assert handshake["protocolVersion"] == PROTOCOL_VERSION
    assert handshake["serverInfo"]["name"] == "stel"
    assert handshake["instructions"] == (
        "Read-only governed document context. Use dbt MCP, not this server, "
        "for semantic-layer metrics."
    )
    # Tools only: the server advertises nothing a client could be waiting on.
    assert "tools" in handshake["capabilities"]
    assert handshake["capabilities"]["tools"].get("listChanged") is False


def test_an_unknown_method_is_a_protocol_error_not_a_tool_result(
    client: WireClient,
) -> None:
    with pytest.raises(WireError) as raised:
        client.rpc("stel/not-a-method")

    # v1 answers an unrecognised method with "invalid params" rather than
    # "method not found"; either is the protocol-level refusal clients handle.
    assert raised.value.code in (-32601, -32602)


# ─── the five tools ────────────────────────────────────────────────────────


def test_exactly_five_tools_are_advertised(client: WireClient) -> None:
    tools = client.rpc("tools/list")["tools"]

    assert {tool["name"] for tool in tools} == TOOL_NAMES
    assert len(tools) == 5
    # Every tool says what it is for: a client shows this to its model.
    assert all(tool["description"].strip() for tool in tools)


def test_each_tool_advertises_its_argument_and_result_contract(
    client: WireClient,
) -> None:
    tools = {tool["name"]: tool for tool in client.rpc("tools/list")["tools"]}

    for name, expected in TOOL_CONTRACT.items():
        schema = tools[name]["inputSchema"]
        arguments = {
            argument: spec.get("default", REQUIRED)
            for argument, spec in schema["properties"].items()
        }
        assert arguments == expected["arguments"], name
        assert schema.get("required", []) == expected["required"], name
        assert list(tools[name]["outputSchema"]["properties"]) == expected["result"], name


def test_the_policy_scope_is_not_an_argument_a_caller_can_supply(
    client: WireClient,
) -> None:
    """Tenant and groups come from the principal. A tool that offered them as
    arguments would let a caller pick their own tenant."""
    tools = client.rpc("tools/list")["tools"]

    for tool in tools:
        names = set(tool["inputSchema"]["properties"])
        assert not names & {"tenant_id", "access_groups", "principal", "subject_id"}, tool["name"]


def test_search_filters_arrive_as_a_closed_operator_set(client: WireClient) -> None:
    """The operators `list_context_models` advertises are an enum a client can
    offer to a model, so a new spelling is a contract change."""
    search = next(
        tool for tool in client.rpc("tools/list")["tools"] if tool["name"] == "search_context"
    )

    operators = search["inputSchema"]["$defs"]["FilterOperator"]["enum"]

    assert operators == ["eq", "ne", "lt", "le", "gt", "ge", "in", "array_contains_any"]
    assert search["inputSchema"]["properties"]["mode"]["enum"] == ["vector", "text", "hybrid"]


# ─── mcp_context/v2 responses ──────────────────────────────────────────────


def _structured(result: Mapping[str, Any]) -> dict[str, Any]:
    """A tool result as an agent receives it, after checking both renderings.

    A client that reads `content` and one that reads `structuredContent` must
    see the same answer: older clients only have the first.
    """
    assert result["isError"] is False if "isError" in result else True
    structured = result["structuredContent"]
    assert [item["type"] for item in result["content"]] == ["text"]
    assert json.loads(result["content"][0]["text"]) == structured
    assert structured["schema_version"] == MCP_CONTEXT_SCHEMA_VERSION
    return dict(structured)


def test_list_context_models_response(client: WireClient) -> None:
    body = _structured(_call(client, "list_context_models", {}))

    assert body["error"] is None
    assert body["next_cursor"] is None
    (model,) = body["models"]
    assert model["name"] == "context_search"
    assert model["contract"] == "agent_context/v1"
    assert model["grain"] == "document_chunks"
    assert model["access"] == "governed"
    assert model["retrieval"]["modes"] == ["hybrid", "text", "vector"]
    assert [field["field"] for field in model["retrieval"]["filter_fields"]] == ["category"]
    assert model["entity_types"] == ["series"]
    assert model["last_successful_materialization"] == "2026-07-20T12:00:00Z"
    assert {"chunk_id", "document_id", "text"} <= {f["name"] for f in model["schema_fields"]}


def test_search_context_response(client: WireClient) -> None:
    body = _structured(
        _call(client, "search_context", {"model": "context_search", "query": "inflation"})
    )

    assert body["error"] is None
    assert body["degraded"] is False
    assert body["entity_scope_applied"] is None
    # Two hits come back from the index; the caller may read one of them.
    (hit,) = body["results"]
    assert hit["rank"] == 1
    assert hit["document_id"] == DOC_ALLOWED
    assert hit["document_version_id"] == VERSION_ALLOWED
    assert hit["context_id"] == CONTEXT_ALLOWED_1
    assert hit["chunk_id"] == CHUNK_ALLOWED_1
    assert hit["snippet"].startswith("Consumer prices slowed")
    assert hit["citation"]["chunk_index"] == 0
    assert hit["entities"][0]["name"] == "series"
    assert hit["served_model"] == "context_search"
    # Lean is the default (issue #812): the provenance blocks are null, and
    # the response says why.
    assert body["detail"] == "lean"
    assert hit["interval"] is None
    assert hit["freshness"] is None
    assert hit["lineage"] is None
    assert set(hit) == {
        "rank", "score", "document_id", "document_version_id", "context_id", "chunk_id",
        "snippet", "snippet_truncated", "source_version", "entities", "interval",
        "freshness", "citation", "lineage", "attributes", "matched_terms",
        "served_model", "served_generation",
    }  # fmt: skip
    assert CHUNK_HIDDEN not in json.dumps(body)


def test_search_context_full_detail_inlines_provenance(client: WireClient) -> None:
    """`detail: full` restores the blocks lean leaves null (issue #812)."""
    body = _structured(
        _call(
            client,
            "search_context",
            {"model": "context_search", "query": "inflation", "detail": "full"},
        )
    )

    assert body["detail"] == "full"
    (hit,) = body["results"]
    assert hit["freshness"]["status"] == "fresh"
    assert hit["interval"]["recorded_from"]
    assert hit["lineage"]["search_index_unique_id"] == "search_index.context_demo.context_search"


def test_get_document_response(client: WireClient) -> None:
    body = _structured(
        _call(
            client,
            "get_document",
            {
                "model": "context_search",
                "document_id": DOC_ALLOWED,
                "document_version_id": VERSION_ALLOWED,
                "limit": 1,
            },
        )
    )

    assert body["error"] is None
    assert body["document_id"] == DOC_ALLOWED
    assert body["source"]["source_system"] == "fred"
    assert [chunk["chunk_id"] for chunk in body["chunks"]] == [CHUNK_ALLOWED_1]
    # A page of one out of two chunks: the cursor is what a client pages with.
    assert isinstance(body["next_cursor"], str)


def test_get_context_lineage_response(client: WireClient) -> None:
    body = _structured(
        _call(
            client,
            "get_context_lineage",
            {"model": "context_search", "reference_type": "chunk", "reference_id": CHUNK_ALLOWED_1},
        )
    )

    assert body["error"] is None
    assert body["record"]["chunk_id"] == CHUNK_ALLOWED_1
    assert body["record"]["document_id"] == DOC_ALLOWED


# ─── how refusals reach the client ─────────────────────────────────────────


def test_a_service_refusal_is_a_structured_error_not_a_failed_call(
    client: WireClient,
) -> None:
    """The distinction agents act on: `isError` means the call was malformed,
    while `error.code` means it was understood and answered with a refusal."""
    result = _call(
        client,
        "get_document",
        {
            "model": "context_search",
            "document_id": DOC_HIDDEN,
            "document_version_id": VERSION_HIDDEN,
        },
    )

    assert result.get("isError", False) is False
    body = result["structuredContent"]
    assert body["error"]["code"] == "not_found_or_denied"
    assert body["error"]["retryable"] is False
    assert body["chunks"] == []


def test_a_malformed_call_is_a_tool_error(client: WireClient) -> None:
    """Wording is the SDK's and is not pinned; the flag is what clients read."""
    missing = _call(client, "search_context", {"model": "context_search"})
    wrong_type = _call(
        client, "search_context", {"model": "context_search", "query": "x", "limit": "many"}
    )
    other_version = _call(
        client,
        "search_context",
        {"model": "context_search", "query": "x", "schema_version": "mcp_context/v1"},
    )
    unknown_tool = _call(client, "delete_everything", {})

    for result in (missing, wrong_type, other_version, unknown_tool):
        assert result["isError"] is True
        assert result["content"][0]["type"] == "text"
    assert "query" in missing["content"][0]["text"]


# ─── identity and the query log ────────────────────────────────────────────


def test_a_served_call_logs_the_request_and_the_client_that_made_it(
    server: Served, client: WireClient
) -> None:
    """Issue #528: the handshake's `clientInfo` and the JSON-RPC id are what the
    protocol supplies, and they are what the query log records."""
    since = server.mark()
    _call(client, "search_context", {"model": "context_search", "query": "inflation"})
    request_id = client.last_request_id

    row = server.logged(request_id, tool="search_context", since=since)

    assert row["request_id"] == str(request_id)
    assert row["client_name"] == CLIENT_NAME
    assert row["client_version"] == CLIENT_VERSION
    assert row["transport"] == server.transport
    assert row["principal_id"] == "local-user"


def test_every_call_in_a_session_logs_its_own_request_id(
    server: Served, client: WireClient
) -> None:
    since = server.mark()
    _call(client, "list_context_models", {})
    first = client.last_request_id
    _call(client, "list_context_models", {})
    second = client.last_request_id

    assert first != second
    logged_first = server.logged(first, tool="list_context_models", since=since)
    logged_second = server.logged(second, tool="list_context_models", since=since)
    assert logged_first["request_id"] == str(first)
    assert logged_second["request_id"] == str(second)


# ─── the HTTP boundary: proxy-supplied identity ────────────────────────────


@pytest.mark.parametrize("transport_fixture", ["http_server", "sse_server"])
def test_a_request_with_no_identity_is_refused_as_missing_principal(
    request: pytest.FixtureRequest, transport_fixture: str
) -> None:
    served: Served = request.getfixturevalue(transport_fixture)
    anonymous = served.connect({})

    body = _call(anonymous, "list_context_models", {})["structuredContent"]

    assert body["error"]["code"] == "missing_principal"
    assert body["models"] == []


@pytest.mark.parametrize("transport_fixture", ["http_server", "sse_server"])
def test_identity_is_read_per_request_not_per_session(
    request: pytest.FixtureRequest, transport_fixture: str
) -> None:
    """One session, two callers' headers in turn. Whatever the first request
    established must not carry to the second: the failure is a correct-looking
    answer out of the wrong tenant."""
    served: Served = request.getfixturevalue(transport_fixture)
    session = served.connect(RESEARCH)
    arguments = {
        "model": "context_search",
        "document_id": DOC_HIDDEN,
        "document_version_id": VERSION_HIDDEN,
    }

    as_research = _call(session, "get_document", arguments)["structuredContent"]
    session.headers = dict(RESTRICTED)
    as_restricted = _call(session, "get_document", arguments)["structuredContent"]
    session.headers = dict(RESEARCH)
    back_as_research = _call(session, "get_document", arguments)["structuredContent"]

    assert as_research["error"]["code"] == "not_found_or_denied"
    assert as_restricted["error"] is None
    assert [chunk["chunk_id"] for chunk in as_restricted["chunks"]] == [CHUNK_HIDDEN]
    assert back_as_research["error"]["code"] == "not_found_or_denied"


def test_a_proxy_server_publishes_no_oauth_metadata(http_server: Served) -> None:
    """Nothing to discover: advertising an authorization server that does not
    govern this deployment would send clients to the wrong place."""
    response = httpx.get(f"{http_server.base_url}/.well-known/oauth-protected-resource/mcp")

    assert response.status_code == 404


# ─── the HTTP boundary: bearer tokens, discovery and grants ────────────────


def _bearer(token: str) -> dict[str, str]:
    return {"authorization": f"Bearer {token}"}


def _initialize_request(headers: Mapping[str, str]) -> httpx.Response:
    return httpx.post(
        f"{JWT_BASE}/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": _initialize_params(),
        },
        headers={
            "accept": "application/json, text/event-stream",
            "content-type": "application/json",
            **headers,
        },
        timeout=REPLY_SECONDS,
    )


JWT_BASE = ""  # set per test from the running server; see `_at`


@pytest.fixture
def at_jwt(jwt_server: Served) -> Served:
    global JWT_BASE
    JWT_BASE = jwt_server.base_url
    return jwt_server


def test_the_protected_resource_metadata_is_served_without_a_token(at_jwt: Served) -> None:
    """RFC 9728: a client handed only this server's URL finds who issues tokens.

    One deliberate difference from SDK v1 (ALE-752): v1 normalized a path-less
    issuer to `https://issuer.example/`; v2 keeps the configured string, so the
    advertised authorization server is exactly what the operator passed to
    `--jwt-issuer`. RFC 8414 compares issuers as exact strings, which makes the
    v2 form the one that matches a token's `iss`.
    """
    response = httpx.get(f"{at_jwt.base_url}/.well-known/oauth-protected-resource/mcp")

    assert response.status_code == 200
    assert response.json() == {
        "resource": PUBLIC_URL,
        "authorization_servers": [ISSUER],
        "bearer_methods_supported": ["header"],
    }


def test_a_call_without_a_token_is_challenged_and_names_the_metadata(
    at_jwt: Served,
) -> None:
    response = _initialize_request({})

    assert response.status_code == 401
    challenge = response.headers["www-authenticate"]
    assert challenge.startswith("Bearer ")
    assert f'resource_metadata="{METADATA_URL}"' in challenge


def test_every_kind_of_bad_token_gets_the_same_refusal(at_jwt: Served, rsa_key: Any) -> None:
    """The documented rule: bad signature, wrong audience, wrong issuer and an
    expired token are indistinguishable, because telling them apart is a
    probing oracle."""
    from cryptography.hazmat.primitives.asymmetric import rsa

    attacker = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    bad_tokens = {
        "garbage": "not-a-jwt",
        "signed by someone else": _mint(attacker),
        "another audience": _mint(rsa_key, aud="https://other.example"),
        "another issuer": _mint(rsa_key, iss="https://evil.example"),
        "expired": _mint(rsa_key, exp=int(time.time()) - 3600),
        "no subject": _mint(rsa_key, sub=""),
    }

    refusals = {label: _initialize_request(_bearer(token)) for label, token in bad_tokens.items()}

    assert {label: r.status_code for label, r in refusals.items()} == dict.fromkeys(bad_tokens, 401)
    assert len({r.headers["www-authenticate"] for r in refusals.values()}) == 1
    assert len({r.text for r in refusals.values()}) == 1


def test_a_valid_token_is_served_as_its_subject(at_jwt: Served, rsa_key: Any) -> None:
    session = at_jwt.connect(_bearer(_mint(rsa_key, sub="alice")))

    body = _call(
        session, "search_context", {"model": "context_search", "query": "inflation"}
    )["structuredContent"]

    assert body["error"] is None
    assert [hit["chunk_id"] for hit in body["results"]] == [CHUNK_ALLOWED_1]


def test_grants_isolate_callers_request_by_request(at_jwt: Served, rsa_key: Any) -> None:
    """Alice is granted `research`, bob `restricted`. Their calls interleave on
    one server: what one caller's request established must not reach the next."""
    alice = at_jwt.connect(_bearer(_mint(rsa_key, sub="alice")))
    bob = at_jwt.connect(_bearer(_mint(rsa_key, sub="bob")))
    arguments = {
        "model": "context_search",
        "document_id": DOC_HIDDEN,
        "document_version_id": VERSION_HIDDEN,
    }

    as_alice = _call(alice, "get_document", arguments)["structuredContent"]
    as_bob = _call(bob, "get_document", arguments)["structuredContent"]
    alice_again = _call(alice, "get_document", arguments)["structuredContent"]

    assert as_alice["error"]["code"] == "not_found_or_denied"
    assert as_bob["error"] is None
    assert [chunk["chunk_id"] for chunk in as_bob["chunks"]] == [CHUNK_HIDDEN]
    assert alice_again["error"]["code"] == "not_found_or_denied"


def test_a_session_cannot_be_carried_over_to_another_callers_token(
    at_jwt: Served, rsa_key: Any
) -> None:
    """Possessing alice's session id does not make bob's token alice's session
    or the other way round: the server answers as if that session did not exist,
    and serves nothing."""
    alice = at_jwt.connect(_bearer(_mint(rsa_key, sub="alice")))
    assert isinstance(alice, _StreamableHttpClient)

    response = httpx.post(
        f"{at_jwt.base_url}/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 99,
            "method": "tools/call",
            "params": {"name": "list_context_models", "arguments": {}},
        },
        headers={
            "accept": "application/json, text/event-stream",
            "content-type": "application/json",
            "mcp-session-id": str(alice.session_id),
            "mcp-protocol-version": PROTOCOL_VERSION,
            **_bearer(_mint(rsa_key, sub="bob")),
        },
        timeout=REPLY_SECONDS,
    )

    assert response.status_code == 404
    assert "result" not in response.json()


def test_a_token_that_claims_a_tenant_gains_nothing_by_claiming_it(
    at_jwt: Served, rsa_key: Any
) -> None:
    """A token answers who; the grant store answers what. Alice's token asking
    for bob's tenant is still alice."""
    session = at_jwt.connect(
        _bearer(_mint(rsa_key, sub="alice", tenant_id="restricted", groups=["admins"]))
    )

    body = _call(
        session,
        "get_document",
        {
            "model": "context_search",
            "document_id": DOC_HIDDEN,
            "document_version_id": VERSION_HIDDEN,
        },
    )["structuredContent"]

    assert body["error"]["code"] == "not_found_or_denied"


def test_a_verified_subject_with_no_grants_is_refused_not_defaulted(
    at_jwt: Served, rsa_key: Any
) -> None:
    """Valid token, unknown subject: authenticated and unauthorized. Failing
    open here would make an unknown caller the most privileged one."""
    session = at_jwt.connect(_bearer(_mint(rsa_key, sub="mallory")))

    body = _call(
        session, "search_context", {"model": "context_search", "query": "inflation"}
    )["structuredContent"]

    assert body["results"] == []
    assert body["error"] is not None
    assert body["error"]["code"] != "internal"


def test_the_log_row_for_a_token_call_names_the_verified_subject(
    at_jwt: Served, rsa_key: Any
) -> None:
    since = at_jwt.mark()
    session = at_jwt.connect(_bearer(_mint(rsa_key, sub="bob")))
    _call(session, "list_context_models", {})

    row = at_jwt.logged(session.last_request_id, tool="list_context_models", since=since)

    assert row["principal_id"] == "bob"
    assert row["client_name"] == CLIENT_NAME
    assert row["transport"] == "streamable-http"


# ─── the newer protocol (ALE-752) ───────────────────────────────────────────
#
# SDK v2 serves the 2026-07-28 protocol beside the handshake versions the rest
# of this file pins. That version has no `initialize`: the client sends its
# identity and capabilities in each request's envelope, so the per-request
# context the caller-identity and principal code read is built a different way.
# The client below is the SDK's own, pinned to the newest protocol, talking to
# the same spawned `stel mcp serve` as every other test here. What must not
# change with the protocol is the tool's `mcp_context/v2` answer, the identity
# the transport supplies, and the client the query log names.


def _call_on_the_newest_protocol(
    served: Served, headers: Mapping[str, str], arguments: Mapping[str, Any]
) -> tuple[str, str, Any]:
    """(negotiated version, newest version, structured result) of one search."""
    import anyio
    import httpx2
    from mcp import Client
    from mcp.client.streamable_http import streamable_http_client
    from mcp.types import Implementation
    from mcp.types.version import MODERN_PROTOCOL_VERSIONS

    newest = MODERN_PROTOCOL_VERSIONS[-1]

    async def call() -> tuple[str, str, Any]:
        async with httpx2.AsyncClient(headers=dict(headers), timeout=REPLY_SECONDS) as http:
            transport = streamable_http_client(f"{served.base_url}/mcp", http_client=http)
            async with Client(
                transport,
                mode=newest,
                client_info=Implementation(name=CLIENT_NAME, version=CLIENT_VERSION),
            ) as sdk_client:
                result = await sdk_client.call_tool("search_context", dict(arguments))
                assert result.is_error is False
                return sdk_client.protocol_version, newest, result.structured_content

    return anyio.run(call)


def _first_logged_row(served: Served, *, since: int) -> dict[str, Any]:
    """The first query-log row written after `since`, waiting for the batch."""
    deadline = time.monotonic() + REPLY_SECONDS
    while time.monotonic() < deadline:
        if served.query_log.exists():
            lines = served.query_log.read_text(encoding="utf-8").splitlines()
            if len(lines) > since:
                return dict(json.loads(lines[since]))
        time.sleep(0.2)
    pytest.fail("no query-log row was written")


def test_a_call_negotiated_on_the_newer_protocol_returns_the_same_contract(
    http_server: Served,
) -> None:
    since = http_server.mark()

    negotiated, newest, structured = _call_on_the_newest_protocol(
        http_server, RESEARCH, {"model": "context_search", "query": "inflation"}
    )

    assert negotiated == newest
    assert structured["schema_version"] == MCP_CONTEXT_SCHEMA_VERSION
    assert structured["error"] is None
    assert structured["results"]
    row = _first_logged_row(http_server, since=since)
    assert row["client_name"] == CLIENT_NAME
    assert row["client_version"] == CLIENT_VERSION
    assert row["request_id"] is not None
    assert row["principal_id"] == "local-user"


def test_a_verified_token_is_the_identity_on_the_newer_protocol(
    at_jwt: Served, rsa_key: Any
) -> None:
    """Token verification runs in the HTTP layer and hands the subject to the
    tool through a contextvar; the newer protocol must not lose it on the way."""
    since = at_jwt.mark()

    _, _, structured = _call_on_the_newest_protocol(
        at_jwt,
        _bearer(_mint(rsa_key, sub="alice")),
        {"model": "context_search", "query": "inflation"},
    )

    assert structured["error"] is None
    assert [hit["chunk_id"] for hit in structured["results"]] == [CHUNK_ALLOWED_1]
    assert _first_logged_row(at_jwt, since=since)["principal_id"] == "alice"
