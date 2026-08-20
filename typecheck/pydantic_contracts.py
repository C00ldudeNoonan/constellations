"""Static compatibility checks for Pydantic behavior used by stel."""

from typing import assert_type

from stel.adapters.bigquery import BigQueryWarehouseConfig
from stel.config.model import ModelConfig
from stel.config.project import ProjectConfig
from stel.endpoints import OpenAICompatibleBaseUrl


def check_model_validation(payload: object) -> None:
    assert_type(ProjectConfig.model_validate(payload), ProjectConfig)
    assert_type(ModelConfig.model_validate(payload), ModelConfig)
    assert_type(
        BigQueryWarehouseConfig.model_validate(payload),
        BigQueryWarehouseConfig,
    )


def check_model_construction() -> None:
    assert_type(ProjectConfig(name="example"), ProjectConfig)
    assert_type(
        BigQueryWarehouseConfig(project="example-project"),
        BigQueryWarehouseConfig,
    )
    assert_type(
        OpenAICompatibleBaseUrl("https://example.invalid/v1"),
        OpenAICompatibleBaseUrl,
    )
