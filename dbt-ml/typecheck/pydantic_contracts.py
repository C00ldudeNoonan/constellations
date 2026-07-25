"""Static compatibility checks for Pydantic behavior used by dbt-ml."""

from typing import assert_type

from dbt_ml.adapters.bigquery import BigQueryWarehouseConfig
from dbt_ml.config.model import ModelConfig
from dbt_ml.config.project import ProjectConfig
from dbt_ml.endpoints import OpenAICompatibleBaseUrl


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
