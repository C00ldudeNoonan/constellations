from __future__ import annotations

import importlib
import subprocess
import sys
import tomllib
from collections.abc import Callable
from pathlib import Path

import pytest
from click.testing import CliRunner

from stel import chunking, optional_dependencies
from stel.backends import html_backend, options, pdf_backend
from stel.cli import cli
from stel.optional_dependencies import OptionalDependencyError
from stel.synth import invoice_pdfs
from stel.text import dedup, encoding, language, nlp, pii, tokens


def _reset_optional_caches() -> None:
    html_backend._bs4.cache_clear()
    tokens._get_encoding.cache_clear()
    pii._get_analyzer.cache_clear()
    pii._get_anonymizer.cache_clear()
    nlp._spacy_provider.cache_clear()


@pytest.mark.parametrize(
    ("blocked", "extra", "use_feature"),
    [
        ("pypdf", "pdf", pdf_backend._pypdf),
        ("fpdf", "pdf", invoice_pdfs._fpdf),
        ("bs4", "html", html_backend._bs4),
        ("soupsieve", "html", lambda: options._validate_css_selector("p.title")),
        ("tiktoken", "text", lambda: tokens.count_tokens("hello")),
        ("ftfy", "text", lambda: encoding.clean_encoding("hello")),
        (
            "langdetect",
            "text",
            lambda: language.detect_language("This input is long enough to detect."),
        ),
        ("datasketch", "text", lambda: dedup.minhash_signature("hello world")),
        (
            "presidio_analyzer",
            "pii",
            lambda: pii.detect_pii("Contact alex@example.com for details."),
        ),
        (
            "spacy",
            "nlp",
            lambda: nlp.get_nlp_provider(nlp.NLPTokenOptions()),
        ),
        (
            "tiktoken",
            "text",
            lambda: chunking._split_tokens("hello world", 10, 0, "cl100k_base"),
        ),
    ],
)
def test_optional_feature_errors_name_install_extra(
    blocked: str,
    extra: str,
    use_feature: Callable[[], object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = importlib.import_module

    def import_without_optional(name: str, package: str | None = None) -> object:
        if name == blocked or name.startswith(f"{blocked}."):
            raise ModuleNotFoundError(name=name)
        return real_import(name, package)

    _reset_optional_caches()
    monkeypatch.setattr(importlib, "import_module", import_without_optional)

    with pytest.raises(OptionalDependencyError, match=rf"stel\[{extra}\]"):
        use_feature()


def test_core_cli_imports_without_optional_packages() -> None:
    script = """
import builtins

blocked = {
    'bs4', 'datasketch', 'fpdf', 'ftfy', 'langdetect', 'presidio_analyzer',
    'presidio_anonymizer', 'pypdf', 'spacy', 'tiktoken', 'mcp',
}
real_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name.split('.', 1)[0] in blocked:
        raise ModuleNotFoundError(name=name)
    return real_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
import stel.cli
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_optional_dependency_version_does_not_import_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_distribution(_distribution: str) -> str:
        raise optional_dependencies.PackageNotFoundError

    monkeypatch.setattr(optional_dependencies, "version", missing_distribution)

    assert optional_dependencies.optional_dependency_version("pypdf") == "not-installed"


def test_compile_reports_missing_optional_dependency_without_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "stel_project.yml").write_text("name: optional_html\n")
    (tmp_path / "sources").mkdir()
    (tmp_path / "sources" / "docs.yml").write_text(
        "version: 2\nsources:\n  - name: docs\n    path: data\n"
    )
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "raw.yml").write_text(
        "version: 2\nmodels:\n  - name: raw_docs\n"
        "    source: ref('docs')\n    extraction:\n      backend: html\n"
        "      options:\n        selectors:\n          title: h1\n"
    )
    real_import = importlib.import_module

    def import_without_soupsieve(name: str, package: str | None = None) -> object:
        if name == "soupsieve":
            raise ModuleNotFoundError(name=name)
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", import_without_soupsieve)

    result = CliRunner().invoke(
        cli, ["--project-dir", str(tmp_path), "compile"]
    )

    assert result.exit_code == 2, result.output
    assert "pip install 'stel[html]'" in result.output
    assert "Traceback" not in result.output


def test_run_reports_missing_optional_dependency_without_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_run(*args: object, **kwargs: object) -> None:
        raise OptionalDependencyError(
            "PDF extraction requires an optional dependency. "
            "Install it with: pip install 'stel[pdf]'"
        )

    monkeypatch.setattr("stel.cli.run_project", fail_run)

    result = CliRunner().invoke(cli, ["--project-dir", str(tmp_path), "run"])

    assert result.exit_code == 2, result.output
    assert "pip install 'stel[pdf]'" in result.output
    assert "Traceback" not in result.output


def test_seed_reports_missing_pdf_extra_without_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "stel_project.yml").write_text(
        "name: optional_pdf\nprofile: optional_pdf\n"
    )
    (tmp_path / "profiles.yml").write_text(
        "optional_pdf:\n  target: dev\n  outputs:\n    dev:\n"
        "      warehouse:\n        type: duckdb\n"
        "        path: target/optional.duckdb\n        schema: optional_pdf\n"
    )
    (tmp_path / "sources").mkdir()
    (tmp_path / "sources" / "docs.yml").write_text(
        "version: 2\nsources:\n  - name: docs\n    path: data/pdfs\n"
    )
    real_import = importlib.import_module

    def import_without_fpdf(name: str, package: str | None = None) -> object:
        if name == "fpdf":
            raise ModuleNotFoundError(name=name)
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", import_without_fpdf)

    result = CliRunner().invoke(
        cli,
        [
            "--project-dir",
            str(tmp_path),
            "seed",
            "--type",
            "invoice_pdfs",
            "--count",
            "1",
        ],
    )

    assert result.exit_code == 2, result.output
    assert "pip install 'stel[pdf]'" in result.output
    assert "Traceback" not in result.output


def test_unselected_pdf_model_does_not_import_pdf_extra_during_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "stel_project.yml").write_text(
        "name: mixed_parsers\nprofile: mixed_parsers\n"
    )
    (tmp_path / "profiles.yml").write_text(
        "mixed_parsers:\n  target: dev\n  outputs:\n    dev:\n"
        "      warehouse:\n        type: duckdb\n"
        "        path: target/mixed.duckdb\n        schema: mixed_parsers\n"
    )
    (tmp_path / "sources").mkdir()
    (tmp_path / "sources" / "docs.yml").write_text(
        "version: 2\nsources:\n"
        "  - name: json_docs\n    path: data/json\n"
        "    file_pattern: '*.json'\n"
        "  - name: pdf_docs\n    path: data/pdf\n"
        "    file_pattern: '*.pdf'\n"
    )
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "models.yml").write_text(
        "version: 2\nmodels:\n"
        "  - name: raw_json\n    source: ref('json_docs')\n"
        "    extraction:\n      backend: json\n"
        "      options:\n        fields: [title]\n"
        "  - name: raw_pdf\n    source: ref('pdf_docs')\n"
        "    extraction:\n      backend: pdf\n"
    )
    json_dir = tmp_path / "data" / "json"
    json_dir.mkdir(parents=True)
    (json_dir / "one.json").write_text('{"title": "Selected"}')

    def fail_if_imported() -> object:
        raise AssertionError("unselected PDF backend must not import pypdf")

    monkeypatch.setattr(pdf_backend, "_pypdf", fail_if_imported)

    result = CliRunner().invoke(
        cli,
        ["--project-dir", str(tmp_path), "run", "--select", "raw_json"],
    )

    assert result.exit_code == 0, result.output
    manifest = (tmp_path / "target" / "manifest.json").read_text()
    assert '"raw_pdf"' in manifest
    assert "Traceback" not in result.output


def test_unselected_html_model_does_not_import_html_extra_during_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "stel_project.yml").write_text(
        "name: mixed_parsers\nprofile: mixed_parsers\n"
    )
    (tmp_path / "profiles.yml").write_text(
        "mixed_parsers:\n  target: dev\n  outputs:\n    dev:\n"
        "      warehouse:\n        type: duckdb\n"
        "        path: target/mixed.duckdb\n        schema: mixed_parsers\n"
    )
    (tmp_path / "sources").mkdir()
    (tmp_path / "sources" / "docs.yml").write_text(
        "version: 2\nsources:\n"
        "  - name: json_docs\n    path: data/json\n"
        "    file_pattern: '*.json'\n"
        "  - name: html_docs\n    path: data/html\n"
        "    file_pattern: '*.html'\n"
    )
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "models.yml").write_text(
        "version: 2\nmodels:\n"
        "  - name: raw_json\n    source: ref('json_docs')\n"
        "    extraction:\n      backend: json\n"
        "      options:\n        fields: [title]\n"
        "  - name: raw_html\n    source: ref('html_docs')\n"
        "    extraction:\n      backend: html\n"
    )
    json_dir = tmp_path / "data" / "json"
    json_dir.mkdir(parents=True)
    (json_dir / "one.json").write_text('{"title": "Selected"}')

    def fail_if_parser_validation_runs() -> tuple[str, ...]:
        raise AssertionError("unselected HTML backend must not inspect bs4 builders")

    def fail_if_imported() -> object:
        raise AssertionError("unselected HTML backend must not import BeautifulSoup")

    monkeypatch.setattr(options, "_available_html_parsers", fail_if_parser_validation_runs)
    monkeypatch.setattr(html_backend, "_bs4", fail_if_imported)

    result = CliRunner().invoke(
        cli,
        ["--project-dir", str(tmp_path), "run", "--select", "raw_json"],
    )

    assert result.exit_code == 0, result.output
    manifest = (tmp_path / "target" / "manifest.json").read_text()
    assert '"raw_html"' in manifest
    assert "Traceback" not in result.output


def test_heavy_dependencies_live_only_in_extras() -> None:
    pyproject_path = Path(__file__).parents[1] / "pyproject.toml"
    project = tomllib.loads(pyproject_path.read_text())["project"]
    core = " ".join(project["dependencies"])
    extras = project["optional-dependencies"]

    for package in (
        "beautifulsoup4",
        "datasketch",
        "fpdf2",
        "ftfy",
        "langdetect",
        "mcp",
        "presidio-analyzer",
        "presidio-anonymizer",
        "pypdf",
        "spacy",
        "tiktoken",
    ):
        assert package not in core

    assert set(extras) >= {
        "all",
        "bigquery",
        "gcs",
        "html",
        "mcp",
        "nlp",
        "pdf",
        "pii",
        "text",
    }
