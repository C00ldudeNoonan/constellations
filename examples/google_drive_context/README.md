# Google Drive, from sign-in to an MCP answer

A shared Drive folder of Docs and Slides becomes governed, section-attributed
context in DuckDB, then answers questions through the stel MCP server. Three
steps: sign in once with `gcloud`, run the project, serve it. No connector,
no service account for a personal folder, and the default test suite never
touches the network.

```
gdrive://<folderId>     Docs export as markdown; Slides render one heading per slide
      │
      ▼
drive_documents         markdown extraction: body, word_count, Drive lineage
      ▼
drive_chunks            chunk: with headings: — every chunk knows its section or slide
      ▼
document_registry ──► document_chunks      the two agent_context/v1 wrappers
                            ▼
                      chunk_embeddings     embed: deterministic (offline)
                            ▼
                      context_search       search: access governed, store local
                            ▼
                      stel mcp serve       search_context / get_document / lineage
```

## 1. Sign in

Enable the **Google Drive API** and the **Google Slides API** in a Google
Cloud project you can use for quota (the API Library in the Cloud console),
then sign in with Application Default Credentials carrying the Drive read
scope. The scope must be on the login; a plain `gcloud auth
application-default login` does not grant Drive access.

```bash
gcloud auth application-default login \
  --scopes=https://www.googleapis.com/auth/drive.readonly,https://www.googleapis.com/auth/cloud-platform
gcloud auth application-default set-quota-project <your-gcp-project>
```

This is the same posture the `gs://` source uses. In CI, point
`GOOGLE_APPLICATION_CREDENTIALS` at a service-account key instead and share
the folder with that account's email. stel never sees a token: `google-auth`
holds the credential and stel makes four read-only REST calls with it.

## 2. Run

The folder id is the last path segment of the folder's URL in Drive. It
reaches the project through the profile's `source_paths` override, so the
checked-in source file never carries anyone's id:

```bash
export STEL_DRIVE_FOLDER=gdrive://1AbCdEfGhIjKlMnOpQrStUvWxYz
uv sync --extra gdrive --extra lancedb --extra mcp
uv run stel --project-dir examples/google_drive_context run
uv run stel --project-dir examples/google_drive_context test
```

```
model               kind        mater.        processed   skipped  deleted   rows
drive_documents     extraction  incremental          14         0        0     14
drive_chunks        chunk       incremental          14         0        0     97
document_registry   transform   full                 14         0        0     14
document_chunks     transform   incremental          14         0        0     97
chunk_embeddings    embed       incremental          97         0        0     97
context_search      search      incremental          97         0        0     97
```

Run it again and every model skips: native files carry a change token
(their modified time), uploaded files an md5, so an untouched folder costs a
listing and nothing else. Edit one Doc and exactly one document re-extracts,
re-chunks, and re-indexes.

Check what landed before serving it:

```bash
uv run stel --project-dir examples/google_drive_context search \
  --model context_search --query "disk encryption" --mode text
```

## 3. Serve and ask

The stdio server takes its principal from the environment: the operator
running the process is the caller.

```bash
STEL_MCP_PRINCIPAL_ID=$USER STEL_MCP_TENANT_ID=drive \
  uv run stel --project-dir examples/google_drive_context mcp serve
```

Register it in Claude Desktop, Claude Code, or any MCP client as a stdio
server with the same command and environment, then ask a question that lives
in the folder. `search_context` returns snippets with a citation naming the
document, the section (or the slide), and the `gdrive://<fileId>#v<version>`
the text came from; `get_document` pages through one document; and
`get_context_lineage` traces a hit back through every model to the source.

```json
{
  "mcpServers": {
    "drive-context": {
      "command": "uv",
      "args": ["run", "stel", "--project-dir", "examples/google_drive_context", "mcp", "serve"],
      "env": {"STEL_MCP_PRINCIPAL_ID": "me", "STEL_MCP_TENANT_ID": "drive",
              "STEL_DRIVE_FOLDER": "gdrive://1AbCdEfGhIjKlMnOpQrStUvWxYz"}
    }
  }
}
```

## What the test proves without a Google account

`tests/test_google_drive_context_example.py` runs this exact project against
an in-memory Drive holding two Docs and a deck, then builds the MCP service
from the project and asks it about disk encryption and API quota. The top
hits cite `("Laptop setup", "Encrypt the disk")` and `("Q3 kickoff", "2.
Risks")` with their `gdrive://` URIs. The live path is covered separately by
a credential-gated test in `tests/test_gdrive_source.py`.

## Choices worth knowing

- **Chunks have no overlap.** Heading attribution names the section where a
  chunk *starts* (#343). With overlap, every chunk would start inside the
  previous section's tail and headings would never name the chunk that holds
  their text. Structured documents want boundaries, not overlap.
- **Slides are headings.** A deck renders as `# Deck title` and `## 3. Slide
  title` per slide, with body bullets, tables, and speaker notes, so a
  citation can say which slide.
- **Sheets are skipped.** Docs and Slides export to text; a spreadsheet is a
  table and belongs in the warehouse, not in a document pipeline. Skipped
  native types are counted in the run log.
- **Uploaded PDFs are a second source.** Add a source with `file_pattern:
  "*.pdf"` over the same folder and an extraction model with `backend: pdf`;
  their identity is the md5 Drive lists, verified on download.
- **The embed provider is offline.** Swap `provider: deterministic` for
  `vertex` and add an `embedding:` block to profiles.yml for real semantic
  search; text mode works either way.
