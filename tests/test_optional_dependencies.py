from __future__ import annotations

import importlib
import json
import subprocess
import sys
import tomllib
from collections.abc import Callable
from pathlib import Path

import duckdb
import pytest
from click.testing import CliRunner
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

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
    project = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))["project"]
    core = " ".join(project["dependencies"])
    extras = project["optional-dependencies"]

    for package in (
        "anthropic",
        "beautifulsoup4",
        "cryptography",
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


def test_cryptography_security_floor_is_declared_on_affected_extras() -> None:
    """Keep published extras past CVE-2026-69247 without burdening base installs (ALE-626)."""
    pyproject_path = Path(__file__).parents[1] / "pyproject.toml"
    extras = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))["project"][
        "optional-dependencies"
    ]
    for extra in ("bigquery", "gcs", "gdrive", "vertex", "mcp", "pii", "all"):
        assert "cryptography>=50.0.0" in extras[extra]


def _requirement_key(requirement: str) -> tuple[str, frozenset[str], str]:
    """A requirement as the three things that have to match: which package,
    with which extras, at which versions."""
    parsed = Requirement(requirement)
    return (canonicalize_name(parsed.name), frozenset(parsed.extras), str(parsed.specifier))


def test_the_all_extra_is_a_superset_of_every_other_extra() -> None:
    """`all` is hand-maintained, so it drifts silently (issue #597).

    It omitted the `mcp` extra's `pyjwt[crypto]` and `httpx` floors -- the
    network transport's two token verifiers -- which made the everything-install
    the one install whose authentication depended on `mcp`'s dependency graph
    rather than on stel's own. Latent rather than live, and invisible: a user
    reads `all` as a superset because that is what the name says.
    """
    pyproject_path = Path(__file__).parents[1] / "pyproject.toml"
    extras = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))["project"][
        "optional-dependencies"
    ]
    everything = {_requirement_key(r) for r in extras["all"]}

    missing = {
        name: sorted(r for r in requirements if _requirement_key(r) not in everything)
        for name, requirements in extras.items()
        if name != "all"
    }
    missing = {name: absent for name, absent in missing.items() if absent}

    assert not missing, (
        f"the `all` extra is missing requirements declared by other extras: {missing}. "
        "`stel[all]` is the everything-install, so anything an extra declares it "
        "needs, `all` needs too -- add the line rather than relying on it arriving "
        "transitively (issue #597)."
    )


def test_anthropic_extra_and_http_security_floors() -> None:
    """Published HTTP extras retain the CA bundle floor (ALE-631)."""
    project = tomllib.loads(
        (Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]
    extras = project["optional-dependencies"]
    assert "anthropic>=0.103.1" in extras["anthropic"]
    for extra in ("anthropic", "bigquery", "gcs", "gdrive", "vertex", "mcp",
                  "pii", "text", "nlp", "all"):
        assert "certifi>=2026.10.19" in extras[extra]


def test_compile_missing_anthropic_fails_before_source_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An LLM project reports the extra before touching sources (ALE-631)."""
    (tmp_path / "stel_project.yml").write_text("name: optional_llm\n", encoding="utf-8")
    (tmp_path / "sources").mkdir()
    (tmp_path / "sources" / "docs.yml").write_text(
        "version: 2\nsources:\n  - name: docs\n    path: data\n", encoding="utf-8"
    )
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "raw.yml").write_text(
        "version: 2\nmodels:\n  - name: raw_docs\n"
        "    source: ref('docs')\n    extraction:\n      backend: llm\n"
        "      options:\n        fields:\n          - {name: title, type: string}\n",
        encoding="utf-8",
    )
    real_import = importlib.import_module

    def without_anthropic(name: str, package: str | None = None) -> object:
        if name == "anthropic":
            raise ModuleNotFoundError(name=name)
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", without_anthropic)
    for command in ("compile", "run"):
        result = CliRunner().invoke(cli, ["--project-dir", str(tmp_path), command])
        assert result.exit_code == 2, result.output
        assert "pip install 'stel[anthropic]'" in result.output
        assert not (tmp_path / "target").exists()


def test_pdf_only_profile_does_not_require_anthropic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A configured but unused provider does not block a PDF-only project (ALE-631)."""
    (tmp_path / "stel_project.yml").write_text("name: p\nprofile: p\n", encoding="utf-8")
    (tmp_path / "profiles.yml").write_text(
        "p:\n  target: dev\n  outputs:\n    dev:\n"
        "      warehouse: {type: duckdb, path: target/p.duckdb, schema: p}\n"
        "      llm: {provider: anthropic}\n", encoding="utf-8"
    )
    real_import = importlib.import_module

    def without_anthropic(name: str, package: str | None = None) -> object:
        if name == "anthropic":
            raise ModuleNotFoundError(name=name)
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", without_anthropic)
    result = CliRunner().invoke(cli, ["--project-dir", str(tmp_path), "compile"])
    assert result.exit_code == 0, result.output


def test_unselected_anthropic_model_does_not_require_sdk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Selecting an offline model does not require an unused SDK (ALE-631)."""
    (tmp_path / "stel_project.yml").write_text("name: p\n", encoding="utf-8")
    (tmp_path / "sources").mkdir()
    (tmp_path / "sources" / "docs.yml").write_text(
        "version: 2\nsources:\n  - name: docs\n    path: data\n", encoding="utf-8"
    )
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "raw.yml").write_text(
        "version: 2\nmodels:\n  - name: raw_json\n"
        "    source: ref('docs')\n    extraction:\n      backend: json\n"
        "      options: {fields: [title]}\n"
        "  - name: raw_llm\n    source: ref('docs')\n    extraction:\n      backend: llm\n"
        "      options:\n        fields:\n          - {name: title, type: string}\n",
        encoding="utf-8",
    )
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "one.json").write_text('{"title": "hello"}', encoding="utf-8")
    real_import = importlib.import_module

    def without_anthropic(name: str, package: str | None = None) -> object:
        if name == "anthropic":
            raise ModuleNotFoundError(name=name)
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", without_anthropic)
    result = CliRunner().invoke(
        cli, ["--project-dir", str(tmp_path), "run", "--select", "raw_json"]
    )
    assert result.exit_code == 0, result.output


@pytest.mark.parametrize("command", ["run", "build"])
@pytest.mark.parametrize("warehouse_vocabulary", [False, True])
@pytest.mark.parametrize("modified_model", ["raw_json", "raw_llm"])
def test_state_selection_checks_only_selected_inference_dependencies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, command: str,
    warehouse_vocabulary: bool, modified_model: str,
) -> None:
    """State selection checks the selected provider before execution (ALE-631)."""
    project = "name: p\nduckdb: {path: warehouse.duckdb}\n"
    if warehouse_vocabulary:
        project += (
            "vocabularies:\n  sector:\n    from:\n"
            "      relation: warehouse://main.sectors\n      label_column: id\n"
        )
        with duckdb.connect(str(tmp_path / "warehouse.duckdb")) as con:
            con.execute("CREATE TABLE sectors AS SELECT 'a' AS id")
    (tmp_path / "stel_project.yml").write_text(project, encoding="utf-8")
    (tmp_path / "sources").mkdir()
    (tmp_path / "sources" / "docs.yml").write_text(
        "version: 2\nsources:\n  - name: docs\n    path: data\n", encoding="utf-8"
    )
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "raw.yml").write_text(
        "version: 2\nmodels:\n  - name: raw_json\n"
        "    source: ref('docs')\n    extraction:\n      backend: json\n"
        "      options: {fields: [title]}\n"
        "  - name: raw_llm\n    source: ref('docs')\n    extraction:\n      backend: llm\n"
        "      options:\n        fields:\n          - {name: title, type: string}\n",
        encoding="utf-8",
    )
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "one.json").write_text('{"title": "hello"}', encoding="utf-8")
    runner = CliRunner()
    compiled = runner.invoke(cli, ["--project-dir", str(tmp_path), "compile"])
    assert compiled.exit_code == 0, compiled.output
    manifest = json.loads((tmp_path / "target" / "manifest.json").read_text(encoding="utf-8"))
    for model in manifest["models"]:
        if model["name"] == modified_model:
            model["code_version"] = "different"
    state = tmp_path / "state.json"
    state.write_text(json.dumps(manifest), encoding="utf-8")
    real_import = importlib.import_module

    def without_anthropic(name: str, package: str | None = None) -> object:
        if name == "anthropic":
            raise ModuleNotFoundError(name=name)
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", without_anthropic)
    result = runner.invoke(
        cli,
        [
            "--project-dir", str(tmp_path), command, "--select", "state:modified",
            "--state", str(state),
        ],
    )
    if modified_model == "raw_llm":
        assert result.exit_code == 2, result.output
        assert "pip install 'stel[anthropic]'" in result.output
        warehouse = tmp_path / "warehouse.duckdb"
        if warehouse.exists():
            with duckdb.connect(str(warehouse), read_only=True) as con:
                rows = con.execute(
                    "SELECT count(*) FROM information_schema.tables WHERE table_name = 'raw_llm'"
                ).fetchone()
                assert rows is not None and rows[0] == 0
    else:
        assert result.exit_code == 0, result.output
