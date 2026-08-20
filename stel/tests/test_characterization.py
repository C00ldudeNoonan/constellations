"""Characterization tests for public contracts that precede the #190 runner
executor extractions.

These pin behavior that the rest of the suite only checks piecemeal, so a
later refactor that silently changes the run-results schema or the
error-sanitization boundary fails loudly. A deliberate contract change must
update the expected sets here (and version the schema); that conscious update
is the point of a characterization test.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import fields
from pathlib import Path

import pytest

from stel.execution import ModelRunResult
from stel.manifest import build_run_results, write_run_results
from stel.providers import (
    ProviderBatchError,
    ProviderConfigurationError,
    ProviderRequestError,
    ProviderResponseError,
)
from stel.runner import _artifact_error_text, run_project
from stel.synth import generate_invoices


@pytest.fixture
def fresh_project(tmp_path: Path, example_project_dir: Path) -> Path:
    dst = tmp_path / "project"
    shutil.copytree(
        example_project_dir,
        dst,
        ignore=shutil.ignore_patterns("data", "target", "__pycache__"),
    )
    return dst


# --- run-results schema shape (issue #195, gap 1) --------------------------

# The serialized run_results payload is the contract Dagster reads (#87). Pin
# the exact key sets so a ModelRunResult field or an injected key cannot be
# dropped, renamed, or added without this test being updated on purpose.
_EXPECTED_RESULT_ROW_KEYS = {f.name for f in fields(ModelRunResult)} | {
    "test_failures",
    "relation",
}
_EXPECTED_METADATA_KEYS = {
    "dbt_ml_version",
    "generated_at",
    "invocation",
    "status",
    "elapsed_seconds",
    "target",
    "sources_considered",
    "counts",
}
_EXPECTED_COUNTS_KEYS = {"total", "success", "error", "skipped", "warnings"}
_EXPECTED_TARGET_KEYS = {
    "profile",
    "name",
    "adapter_type",
    "schema",
    "catalog",
    "location",
}
_EXPECTED_RELATION_KEYS = {"catalog", "schema", "name", "fully_qualified"}


def test_run_results_result_row_shape_is_pinned(fresh_project: Path) -> None:
    generate_invoices(3, fresh_project / "data" / "invoices", seed=1)
    results = run_project(fresh_project)
    payload = json.loads(
        write_run_results(fresh_project, results, elapsed_seconds=1.0).read_text()
    )

    assert set(payload) == {"metadata", "results"}
    assert payload["results"], "invoice_pipeline should materialize models"
    # The example exercises both the extraction and transform executors; the
    # serialized row shape must be identical across kinds.
    assert {row["kind"] for row in payload["results"]} == {"extraction", "transform"}
    for row in payload["results"]:
        assert set(row) == _EXPECTED_RESULT_ROW_KEYS
        assert set(row["relation"]) == _EXPECTED_RELATION_KEYS


def test_run_results_metadata_shape_is_pinned(fresh_project: Path) -> None:
    generate_invoices(3, fresh_project / "data" / "invoices", seed=1)
    results = run_project(fresh_project)
    payload = json.loads(
        write_run_results(fresh_project, results, elapsed_seconds=1.0).read_text()
    )

    meta = payload["metadata"]
    assert set(meta) == _EXPECTED_METADATA_KEYS
    assert set(meta["counts"]) == _EXPECTED_COUNTS_KEYS
    assert set(meta["target"]) == _EXPECTED_TARGET_KEYS


def test_run_results_empty_run_keeps_metadata_shape(fresh_project: Path) -> None:
    # An empty run still emits the full metadata envelope; a skipped model
    # still serializes the complete row shape.
    payload = build_run_results(fresh_project, [], skipped=["invoice_summary"])

    assert set(payload) == {"metadata", "results"}
    assert set(payload["metadata"]) == _EXPECTED_METADATA_KEYS
    assert set(payload["metadata"]["counts"]) == _EXPECTED_COUNTS_KEYS
    (skipped_row,) = payload["results"]
    assert skipped_row["status"] == "skipped"
    assert set(skipped_row) == _EXPECTED_RESULT_ROW_KEYS


# --- error sanitization boundary (issue #195, gap 2) -----------------------

# _artifact_error_text converts any executor exception into a display string.
# Provider-authored detail must never survive; each provider-error family maps
# to a fixed safe prefix. Pinning the exact strings here protects the helper
# the extraction/embed/llm executors all depend on.


def test_artifact_error_text_sanitizes_configuration_error() -> None:
    error = ProviderConfigurationError("SECRET_ENV is missing from the shell")
    text = _artifact_error_text(error)
    assert text == "ProviderConfigurationError: provider configuration is invalid"
    assert "SECRET_ENV" not in text


def test_artifact_error_text_sanitizes_request_error() -> None:
    # The request branch delegates to sanitized_provider_error and is the real
    # leak risk: only a safe provider label, operation, and code may appear.
    error = ProviderRequestError(
        "anthropic", "messages.create", code="rate_limited", retryable=True
    )
    text = _artifact_error_text(error)
    assert text.startswith("ProviderRequestError: ")
    assert "rate_limited" in text
    assert "anthropic" in text


def test_artifact_error_text_request_error_drops_unsafe_labels() -> None:
    # Unsafe provider/operation labels are normalized away by the boundary; the
    # raw text must not survive into the display string.
    error = ProviderRequestError(
        "prod-key=sk-abc123", "POST https://api/secret", code="boom"
    )
    text = _artifact_error_text(error)
    assert "sk-abc123" not in text
    assert "api/secret" not in text
    assert text.startswith("ProviderRequestError: ")


def test_artifact_error_text_sanitizes_response_error() -> None:
    error = ProviderResponseError("raw body: {\"account\": \"acct_123\"}")
    text = _artifact_error_text(error)
    assert text == "ProviderResponseError: provider response is invalid"
    assert "acct_123" not in text


def test_artifact_error_text_sanitizes_batch_error() -> None:
    error = ProviderBatchError("batch job/xyz internal endpoint failed")
    text = _artifact_error_text(error)
    assert text == "ProviderBatchError: provider batch operation failed"
    assert "job/xyz" not in text


def test_artifact_error_text_sanitizes_chained_provider_error() -> None:
    # A provider error reached through __cause__ is still recognized and its
    # underlying SDK text is not surfaced.
    try:
        try:
            raise ProviderConfigurationError("PRIVATE_TOKEN not set")
        except ProviderConfigurationError as cause:
            raise RuntimeError("transform failed while calling model") from cause
    except RuntimeError as chained:
        text = _artifact_error_text(chained)
    assert text == "ProviderConfigurationError: provider configuration is invalid"
    assert "PRIVATE_TOKEN" not in text


def test_artifact_error_text_passes_through_non_provider_error() -> None:
    text = _artifact_error_text(ValueError("bad column count"))
    assert text == "ValueError: bad column count"
