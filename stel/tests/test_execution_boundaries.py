from __future__ import annotations

import subprocess
import sys

from stel.execution import ModelRunResult as ExecutionModelRunResult
from stel.execution import RunError as ExecutionRunError
from stel.execution import artifact_error_text
from stel.execution.chunk import chunk_document_ids, chunk_input_hash, run_chunk_model
from stel.execution.cost import budget_cost_estimator, estimate_cost
from stel.execution.embed import run_embed_model
from stel.execution.extraction import (
    EXTRACTION_FIELD_DTYPES,
    DiscoveredSource,
    run_extraction_model,
)
from stel.execution.llm import run_llm_model
from stel.execution.ml import run_ml_model
from stel.execution.search import run_search_model
from stel.execution.transform import run_sql_model, run_transform_model
from stel.execution.usage import add_provider_usage
from stel.runner import (
    _EXTRACTION_FIELD_DTYPES,
    ModelRunResult,
    RunError,
    _add_provider_usage,
    _artifact_error_text,
    _budget_cost_estimator,
    _chunk_document_ids,
    _chunk_input_hash,
    _estimate_cost,
    _run_chunk_model,
    _run_embed_model,
    _run_extraction_model,
    _run_llm_model,
    _run_ml_model,
    _run_search_model,
    _run_sql_model,
    _run_transform_model,
)
from stel.runner import DiscoveredSource as RunnerDiscoveredSource


def test_runner_preserves_execution_contract_and_chunk_helper_imports() -> None:
    assert ModelRunResult is ExecutionModelRunResult
    assert RunError is ExecutionRunError
    assert _run_chunk_model is run_chunk_model
    assert _chunk_document_ids is chunk_document_ids
    assert _chunk_input_hash is chunk_input_hash


def test_runner_preserves_transform_executor_and_error_helper_imports() -> None:
    assert _run_sql_model is run_sql_model
    assert _run_transform_model is run_transform_model
    assert _artifact_error_text is artifact_error_text


def test_runner_preserves_extraction_executor_and_cost_imports() -> None:
    assert _run_extraction_model is run_extraction_model
    assert RunnerDiscoveredSource is DiscoveredSource
    assert _EXTRACTION_FIELD_DTYPES is EXTRACTION_FIELD_DTYPES
    assert _estimate_cost is estimate_cost
    assert _budget_cost_estimator is budget_cost_estimator


def test_runner_preserves_embed_executor_and_usage_imports() -> None:
    assert _run_embed_model is run_embed_model
    assert _add_provider_usage is add_provider_usage


def test_runner_preserves_llm_executor_import() -> None:
    assert _run_llm_model is run_llm_model


def test_runner_preserves_search_and_ml_executor_imports() -> None:
    assert _run_search_model is run_search_model
    assert _run_ml_model is run_ml_model


def test_manifest_import_does_not_load_runner() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import stel.manifest; "
            "assert 'stel.runner' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
