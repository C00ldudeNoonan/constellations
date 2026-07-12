from __future__ import annotations

import importlib
import subprocess
import sys
import tomllib
from collections.abc import Callable
from pathlib import Path

import pytest

from dbt_ml import chunking
from dbt_ml.backends import html_backend, options, pdf_backend
from dbt_ml.optional_dependencies import OptionalDependencyError
from dbt_ml.synth import invoice_pdfs
from dbt_ml.text import dedup, encoding, language, pii, tokens


def _reset_optional_caches() -> None:
    html_backend._bs4.cache_clear()
    tokens._get_encoding.cache_clear()
    pii._get_analyzer.cache_clear()
    pii._get_anonymizer.cache_clear()


@pytest.mark.parametrize(
    ("blocked", "extra", "use_feature"),
    [
        ("pypdf", "pdf", lambda: pdf_backend.PdfBackend().version()),
        ("fpdf", "pdf", invoice_pdfs._fpdf),
        ("bs4", "html", lambda: html_backend.HtmlBackend().version()),
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

    with pytest.raises(OptionalDependencyError, match=rf"dbt-ml\[{extra}\]"):
        use_feature()


def test_core_cli_imports_without_optional_packages() -> None:
    script = """
import builtins

blocked = {
    'bs4', 'datasketch', 'fpdf', 'ftfy', 'langdetect', 'presidio_analyzer',
    'presidio_anonymizer', 'pypdf', 'tiktoken',
}
real_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name.split('.', 1)[0] in blocked:
        raise ModuleNotFoundError(name=name)
    return real_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
import dbt_ml.cli
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


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
        "presidio-analyzer",
        "presidio-anonymizer",
        "pypdf",
        "tiktoken",
    ):
        assert package not in core

    assert set(extras) >= {"all", "bigquery", "gcs", "html", "pdf", "pii", "text"}
