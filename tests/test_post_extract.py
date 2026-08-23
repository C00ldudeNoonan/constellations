from __future__ import annotations

import json
import shutil
import traceback
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import duckdb
import pytest

from stel.backends import BaseBackend, ExtractionResult
from stel.backends.base import BatchExtractionOutput
from stel.budget import BudgetGuard
from stel.config.loader import ConfigError
from stel.config.model import ExtractionConfig, ModelConfig
from stel.config.project import ProjectConfig
from stel.config.source import SourceConfig
from stel.execution.extraction import _extract_batched
from stel.manifest import write_manifest
from stel.post_extract import PostExtractError, load_post_extract
from stel.runner import run_project
from stel.sources import DocumentRef, DocumentSource, SourceScan
from stel.versioning import compute_model_code_version


def _write_hook(project_dir: Path, source: str) -> Path:
    hook_dir = project_dir / "post_extract"
    hook_dir.mkdir(exist_ok=True)
    hook = hook_dir / "derive.py"
    hook.write_text(source)
    return hook


def test_post_extract_config_accepts_shorthand_and_options() -> None:
    shorthand = ExtractionConfig.model_validate(
        {"backend": "json", "post_extract": "post_extract.derive"}
    )
    configured = ExtractionConfig.model_validate(
        {
            "backend": "json",
            "post_extract": {
                "module": "post_extract.derive",
                "options": {"output_field": "text"},
            },
        }
    )

    assert shorthand.post_extract is not None
    assert shorthand.post_extract.module == "post_extract.derive"
    assert shorthand.post_extract.options == {}
    assert configured.post_extract is not None
    assert configured.post_extract.options == {"output_field": "text"}


def test_hook_derives_fields_with_verified_context_and_preserves_accounting(
    tmp_path: Path,
) -> None:
    local_path = tmp_path / "verified.json"
    local_path.write_text('{"content": "raw"}')
    _write_hook(
        tmp_path,
        """
def validate_options(options):
    if options.get("output_field") != "text":
        raise ValueError("output_field must be text")

def run(fields, ctx):
    assert ctx.local_path.read_text() == '{"content": "raw"}'
    return {
        ctx.options["output_field"]: fields["content"].upper(),
        "source": ctx.source_path,
    }
""".lstrip(),
    )
    loaded = load_post_extract(
        "post_extract.derive", tmp_path, {"output_field": "text"}
    )
    document = DocumentRef(
        source_name="docs",
        relative_path="filing.json",
        document_id="doc-1",
        content_hash="hash-1",
        source_uri="gs://bucket/filing.json#1",
        source_metadata={"generation": "1"},
    )

    result = loaded.apply(
        ExtractionResult(
            fields={"content": "raw"},
            warnings=["backend warning"],
            metrics={"input_tokens": 3},
        ),
        document=document,
        local_path=local_path,
    )

    assert result.fields == {"text": "RAW", "source": "filing.json"}
    assert result.warnings == ["backend warning"]
    assert result.metrics == {"input_tokens": 3}


def test_hook_failure_does_not_retain_raw_payload_in_diagnostics(tmp_path: Path) -> None:
    secret = "raw-document-secret-that-must-not-survive"
    local_path = tmp_path / "verified.json"
    local_path.write_text("{}")
    _write_hook(
        tmp_path,
        "def run(fields):\n    raise ValueError(fields['content'])\n",
    )
    loaded = load_post_extract("post_extract.derive", tmp_path, {})
    document = DocumentRef("docs", "filing.json", "doc-1", "hash-1")

    with pytest.raises(PostExtractError) as exc_info:
        loaded.apply(
            ExtractionResult(fields={"content": secret}),
            document=document,
            local_path=local_path,
        )

    rendered = "".join(
        traceback.format_exception(
            exc_info.type, exc_info.value, exc_info.value.__traceback__
        )
    )
    assert secret not in rendered
    assert str(exc_info.value) == "Post-extract hook 'post_extract.derive' failed"


def test_invalid_hook_fails_compilation_before_source_discovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "sources").mkdir()
    (tmp_path / "models").mkdir()
    (tmp_path / "stel_project.yml").write_text(
        "name: p\nsource-paths: [sources]\nmodel-paths: [models]\n"
    )
    (tmp_path / "sources" / "docs.yml").write_text(
        "version: 2\nsources:\n  - name: docs\n    path: gs://bucket/docs\n"
    )
    (tmp_path / "models" / "raw.yml").write_text(
        """version: 2
models:
  - name: raw_docs
    source: ref('docs')
    extraction:
      backend: json
      post_extract: post_extract.missing
"""
    )
    discovered = False

    def _unexpected_discovery(*args: object, **kwargs: object) -> None:
        nonlocal discovered
        discovered = True
        raise AssertionError("source discovery must not run")

    monkeypatch.setattr(
        "stel.sources.gcs.GCSDocumentSource.discover", _unexpected_discovery
    )

    with pytest.raises(ConfigError, match=r"post_extract\.missing"):
        run_project(tmp_path)

    assert not discovered


def test_hook_validation_failure_severs_sensitive_exception_chain(
    tmp_path: Path,
) -> None:
    sensitive_prompt = "private-hook-prompt-that-must-not-survive"
    (tmp_path / "sources").mkdir()
    (tmp_path / "models").mkdir()
    (tmp_path / "stel_project.yml").write_text(
        "name: p\nsource-paths: [sources]\nmodel-paths: [models]\n"
    )
    (tmp_path / "sources" / "docs.yml").write_text(
        "version: 2\nsources:\n  - name: docs\n    path: data/docs\n"
    )
    (tmp_path / "models" / "raw.yml").write_text(
        f"""version: 2
models:
  - name: raw_docs
    source: ref('docs')
    extraction:
      backend: json
      post_extract:
        module: post_extract.derive
        options:
          prompt: {sensitive_prompt}
"""
    )
    _write_hook(
        tmp_path,
        """
def validate_options(options):
    raise ValueError(options["prompt"])

def run(fields):
    return fields
""".lstrip(),
    )

    with pytest.raises(ConfigError) as exc_info:
        run_project(tmp_path)

    error = exc_info.value
    rendered = "".join(
        traceback.format_exception(type(error), error, error.__traceback__)
    )
    assert sensitive_prompt not in str(error)
    assert sensitive_prompt not in rendered
    assert error.__cause__ is None
    assert error.__context__ is None


def test_manifest_omits_sensitive_hook_options(tmp_path: Path) -> None:
    sensitive_prompt = "private-manifest-hook-prompt"
    credential_reference = "PRIVATE_HOOK_CREDENTIAL_ENV"
    (tmp_path / "sources").mkdir()
    (tmp_path / "models").mkdir()
    (tmp_path / "stel_project.yml").write_text(
        "name: p\nsource-paths: [sources]\nmodel-paths: [models]\n"
    )
    (tmp_path / "sources" / "docs.yml").write_text(
        "version: 2\nsources:\n  - name: docs\n    path: data/docs\n"
    )
    (tmp_path / "models" / "raw.yml").write_text(
        f"""version: 2
models:
  - name: raw_docs
    source: ref('docs')
    extraction:
      backend: json
      post_extract:
        module: post_extract.derive
        options:
          prompt: {sensitive_prompt}
          api_key_env: {credential_reference}
"""
    )
    _write_hook(tmp_path, "def run(fields):\n    return fields\n")

    manifest_path = write_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    model = next(item for item in manifest["models"] if item["name"] == "raw_docs")
    serialized = manifest_path.read_text()

    assert model["extraction"]["post_extract"] == {
        "module": "post_extract.derive"
    }
    assert sensitive_prompt not in serialized
    assert credential_reference not in serialized


class _BatchSource(DocumentSource):
    def __init__(self, source: Path) -> None:
        self.source = source
        self.fetched: Path | None = None

    def discover(
        self,
        source: SourceConfig,
        project_dir: Path,
        *,
        source_filter: Sequence[str] = (),
    ) -> list[DocumentRef]:
        del source, project_dir, source_filter
        return []

    def fetch(self, ref: DocumentRef, work_dir: Path) -> Path:
        del ref
        destination = work_dir / "fetched.json"
        shutil.copy2(self.source, destination)
        self.fetched = destination
        return destination

    def scan(self, source: SourceConfig, project_dir: Path) -> SourceScan:
        del source, project_dir
        return SourceScan(True, 1, None, None)


class _BatchBackend(BaseBackend):
    def __init__(self) -> None:
        self.output: BatchExtractionOutput | None = None

    def name(self) -> str:
        return "batch-test"

    def supported_formats(self) -> list[str]:
        return [".json"]

    def extract(self, path: Path, options: dict[str, Any]) -> ExtractionResult:
        del path, options
        raise AssertionError("batch path expected")

    def extract_batch_with_metrics(
        self,
        paths: list[Path],
        options: dict[str, Any],
        *,
        budget: BudgetGuard | None = None,
    ) -> BatchExtractionOutput:
        del options, budget
        self.output = BatchExtractionOutput(
            [ExtractionResult({"content": path.read_text()}) for path in paths],
            {"batch_submissions": 1},
        )
        return self.output


def test_batch_hook_runs_before_verified_snapshot_cleanup(tmp_path: Path) -> None:
    source_file = tmp_path / "source.json"
    source_file.write_text("raw batch payload")
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    _write_hook(
        tmp_path,
        """
def run(fields, ctx):
    assert ctx.local_path.exists()
    return {"text": ctx.local_path.read_text().upper()}
""".lstrip(),
    )
    hook = load_post_extract("post_extract.derive", tmp_path, {})
    source = _BatchSource(source_file)
    backend = _BatchBackend()
    document = DocumentRef("docs", "source.json", "doc-1", "hash-1")

    extracted, metrics = _extract_batched(
        [document],
        source,
        backend,
        {},
        work_dir,
        "raw_docs",
        post_extract=hook,
    )

    assert extracted[0][1] is not None
    assert extracted[0][1].fields == {"text": "RAW BATCH PAYLOAD"}
    assert metrics == {"batch_submissions": 1}
    assert backend.output is not None
    retained = backend.output.items[0]
    assert isinstance(retained, ExtractionResult)
    assert retained.fields == {"text": "RAW BATCH PAYLOAD"}
    assert source.fetched is not None
    assert not source.fetched.exists()


def test_hook_source_and_options_change_incremental_code_version(tmp_path: Path) -> None:
    hook = _write_hook(tmp_path, "def run(fields):\n    return fields\n")
    project = ProjectConfig(name="p")

    def version(options: dict[str, Any]) -> str:
        return compute_model_code_version(
            ModelConfig(
                name="raw_docs",
                source="ref('docs')",
                extraction=ExtractionConfig.model_validate(
                    {
                        "backend": "json",
                        "post_extract": {
                            "module": "post_extract.derive",
                            "options": options,
                        },
                    }
                ),
            ),
            project,
            tmp_path,
        )

    baseline = version({"output_field": "text"})
    assert baseline != version({"output_field": "body"})
    hook.write_text("def run(fields):\n    return {'text': fields.get('content')}\n")
    assert baseline != version({"output_field": "text"})


def test_end_to_end_hook_keeps_raw_envelope_out_of_warehouse(tmp_path: Path) -> None:
    (tmp_path / "sources").mkdir()
    (tmp_path / "models").mkdir()
    (tmp_path / "data").mkdir()
    (tmp_path / "stel_project.yml").write_text(
        """name: envelope_project
profile: envelope_project
source-paths: [sources]
model-paths: [models]
target-path: target
"""
    )
    (tmp_path / "profiles.yml").write_text(
        """envelope_project:
  target: dev
  outputs:
    dev:
      warehouse:
        type: duckdb
        path: ./target/stel.duckdb
        schema: stel
"""
    )
    (tmp_path / "sources" / "filings.yml").write_text(
        """version: 2
sources:
  - name: filings
    path: data
    file_pattern: "*.json"
"""
    )
    (tmp_path / "models" / "filing_text.yml").write_text(
        """version: 2
models:
  - name: filing_text
    source: ref('filings')
    extraction:
      backend: json
      options:
        fields: [company, content]
      post_extract:
        module: post_extract.sec_text
        options:
          output_field: text
    materialization: full
"""
    )
    raw_html = "<html><body>PRIVATE RAW HTML <b>Useful filing text</b></body></html>"
    (tmp_path / "data" / "filing.json").write_text(
        json.dumps({"company": "ACME", "content": raw_html})
    )
    hook_dir = tmp_path / "post_extract"
    hook_dir.mkdir()
    (hook_dir / "sec_text.py").write_text(
        """import re

def validate_options(options):
    if set(options) != {"output_field"}:
        raise ValueError("expected output_field")

def run(fields, ctx):
    text = re.sub(r"<[^>]+>", " ", fields["content"])
    return {
        "company": fields["company"],
        ctx.options["output_field"]: " ".join(text.split()),
    }
"""
    )

    result = run_project(tmp_path, select="filing_text")

    assert result[0].errors == []
    database = tmp_path / "target" / "stel.duckdb"
    with duckdb.connect(str(database), read_only=True) as connection:
        cursor = connection.execute('SELECT * FROM "stel".stel.filing_text')
        columns = [description[0] for description in cursor.description]
        row = cursor.fetchone()
    assert "content" not in columns
    assert "text" in columns
    assert row is not None
    assert raw_html not in repr(row)
    assert row[columns.index("text")] == "PRIVATE RAW HTML Useful filing text"
