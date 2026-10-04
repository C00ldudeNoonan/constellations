#!/bin/bash
# SessionStart hook for Claude Code on the web: make the full test suite runnable
# in a cloud sandbox whose network policy blocks three download hosts the suite
# otherwise reaches at test time. Without this, 65 tests fail or error on every
# cloud session for reasons that have nothing to do with the code, and an agent
# learns to read "some tests fail" as normal.
#
#   extensions.duckdb.org          -> DuckDB `vss` and `fts` (the DuckDB search store)
#   openaipublic.blob.core.windows.net -> tiktoken's cl100k_base encoding
#
# Both fall back to PyPI, which the sandbox allows, and both fallbacks are
# verified by the consuming library's own integrity check, never by trusting
# the package: DuckDB refuses an extension binary that is not DuckDB-signed
# (signature checking is on by default and this script does not turn it off),
# and tiktoken refuses a cached encoding whose SHA-256 differs from the hash it
# pins. Only data and signed binaries are taken from those wheels; no package is
# installed and no code from them runs.
#
# Local sessions are untouched: this exits immediately unless remote.
set -euo pipefail

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "${CLAUDE_PROJECT_DIR:-$(git rev-parse --show-toplevel)}"

# The project's own documented setup (AGENTS.md, CONTRIBUTING.md).
uv sync --all-extras --dev --locked

PY=.venv/bin/python
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# Persisted for every shell the session opens, so tiktoken finds the cache
# regardless of TMPDIR.
TIKTOKEN_CACHE_DIR="${HOME}/.cache/tiktoken"
mkdir -p "$TIKTOKEN_CACHE_DIR"
if [ -n "${CLAUDE_ENV_FILE:-}" ] && ! grep -qs "TIKTOKEN_CACHE_DIR=" "$CLAUDE_ENV_FILE"; then
  echo "export TIKTOKEN_CACHE_DIR=\"$TIKTOKEN_CACHE_DIR\"" >> "$CLAUDE_ENV_FILE"
fi
export TIKTOKEN_CACHE_DIR

# --- DuckDB extensions --------------------------------------------------------
# Pinned to the locked duckdb version: an extension built for another version
# will not load. Each step is best-effort; a failure here costs those tests,
# not the session.
"$PY" - "$WORK" <<'PYEOF' || echo "session-start: DuckDB extensions not provisioned; DuckDB-store tests will fail" >&2
import glob, os, subprocess, sys, zipfile
import duckdb

work = sys.argv[1]
version = duckdb.__version__
needed = ("vss", "fts")

def installed() -> set[str]:
    rows = duckdb.connect().execute(
        "SELECT extension_name FROM duckdb_extensions() WHERE installed"
    ).fetchall()
    return {name for (name,) in rows}

missing = [e for e in needed if e not in installed()]
for ext in list(missing):
    try:  # the official repository, if the network policy allows it
        duckdb.connect().execute(f"INSTALL {ext}")
        missing.remove(ext)
    except duckdb.Error:
        pass

if missing:
    subprocess.run(
        [sys.executable, "-m", "pip", "download", "--quiet", "--no-deps",
         "--only-binary=:all:", "--dest", work,
         *[f"duckdb-extension-{e}=={version}" for e in missing]],
        check=True,
    )
    for ext in missing:
        (wheel,) = glob.glob(os.path.join(work, f"duckdb_extension_{ext}-{version}-*.whl"))
        member = f"duckdb_extension_{ext}/extensions/v{version}/{ext}.duckdb_extension"
        with zipfile.ZipFile(wheel) as zf:
            path = zf.extract(member, work)
        # INSTALL from a file still verifies DuckDB's signature on LOAD; an
        # unsigned or tampered binary raises here and is not left installed.
        con = duckdb.connect()
        con.execute(f"INSTALL '{path}'")
        con.execute(f"LOAD {ext}")

still = [e for e in needed if e not in installed()]
if still:
    raise SystemExit(f"not installed: {still}")
print(f"session-start: DuckDB {version} extensions ready: {', '.join(needed)}")
PYEOF

# --- tiktoken encodings -------------------------------------------------------
"$PY" - "$WORK" <<'PYEOF' || echo "session-start: tiktoken cl100k_base not provisioned; token-counting tests will fail" >&2
import glob, hashlib, os, subprocess, sys, zipfile
import tiktoken

work = sys.argv[1]
url = "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
cache = os.path.join(os.environ["TIKTOKEN_CACHE_DIR"], hashlib.sha1(url.encode()).hexdigest())

def usable() -> bool:
    try:
        # Verifies the cached bytes against tiktoken's pinned SHA-256.
        tiktoken.get_encoding("cl100k_base").encode("ok")
        return True
    except Exception:
        return False

if not usable():
    if os.path.exists(cache):
        os.remove(cache)  # a bad or partial file would keep failing the hash
    subprocess.run(
        [sys.executable, "-m", "pip", "download", "--quiet", "--no-deps",
         "--only-binary=:all:", "--dest", work, "tiktoken-offline==0.1.1"],
        check=True,
    )
    (wheel,) = glob.glob(os.path.join(work, "tiktoken_offline-*.whl"))
    with zipfile.ZipFile(wheel) as zf, open(cache, "wb") as out:
        out.write(zf.read("tiktoken_ext/data/cl100k_base.tiktoken"))
    tiktoken.registry.ENCODINGS.pop("cl100k_base", None)
    if not usable():
        os.remove(cache)
        raise SystemExit("cl100k_base failed tiktoken's hash check; removed")
print("session-start: tiktoken cl100k_base ready")
PYEOF
