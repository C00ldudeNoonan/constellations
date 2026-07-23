from __future__ import annotations

import subprocess
import sys

from dbt_ml.execution import ModelRunResult as ExecutionModelRunResult
from dbt_ml.execution import RunError as ExecutionRunError
from dbt_ml.execution.chunk import chunk_document_ids, chunk_input_hash, run_chunk_model
from dbt_ml.runner import (
    ModelRunResult,
    RunError,
    _chunk_document_ids,
    _chunk_input_hash,
    _run_chunk_model,
)


def test_runner_preserves_execution_contract_and_chunk_helper_imports() -> None:
    assert ModelRunResult is ExecutionModelRunResult
    assert RunError is ExecutionRunError
    assert _run_chunk_model is run_chunk_model
    assert _chunk_document_ids is chunk_document_ids
    assert _chunk_input_hash is chunk_input_hash


def test_manifest_import_does_not_load_runner() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import dbt_ml.manifest; "
            "assert 'dbt_ml.runner' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
