from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def example_project_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "examples" / "invoice_pipeline"
