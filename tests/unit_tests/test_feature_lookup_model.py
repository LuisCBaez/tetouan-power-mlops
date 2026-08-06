"""Unit tests for FeatureLookUpModel (Databricks-independent surface only).

The Feature Engineering client only runs on Databricks, so these tests exercise
name construction and the SQL emitted by create_feature_table/define_feature_function
using a mocked SparkSession. The fe-client methods are validated on Serverless.
"""

from unittest.mock import MagicMock

from tetouan_power.config import ProjectConfig, Tags
from tetouan_power.models.feature_lookup_model import FeatureLookUpModel


def _model(config: ProjectConfig, tags: Tags) -> FeatureLookUpModel:
    """Build a model with a mocked Spark session (no Databricks required)."""
    return FeatureLookUpModel(config=config, tags=tags, spark=MagicMock())


def test_init_builds_names(config: ProjectConfig, tags: Tags) -> None:
    """Verify derived UC names and feature lists."""
    model = _model(config, tags)
    assert model.feature_table_name == f"{config.catalog_name}.{config.schema_name}.power_features"
    assert model.function_name == f"{config.catalog_name}.{config.schema_name}.calculate_is_weekend"
    assert model.lookup_features == [
        "temperature",
        "humidity",
        "wind_speed",
        "general_diffuse_flows",
        "diffuse_flows",
    ]
    assert isinstance(model.tags, dict)


def test_fe_client_is_lazy(config: ProjectConfig, tags: Tags) -> None:
    """Verify the Databricks-only FE client is NOT created on init."""
    model = _model(config, tags)
    assert model._fe is None


def test_create_feature_table_emits_sql(config: ProjectConfig, tags: Tags) -> None:
    """Verify create_feature_table emits CREATE/PK/CDF + one INSERT per split."""
    model = _model(config, tags)
    model.create_feature_table()

    sql_calls = [call.args[0] for call in model.spark.sql.call_args_list]
    joined = "\n".join(sql_calls)

    assert any("CREATE OR REPLACE TABLE" in s for s in sql_calls)
    assert any("PRIMARY KEY(id)" in s for s in sql_calls)
    assert any("enableChangeDataFeed = true" in s for s in sql_calls)
    assert joined.count("INSERT INTO") == 3  # train_set, val_set, test_set
    assert "temperature, humidity, wind_speed, general_diffuse_flows, diffuse_flows" in joined


def test_define_feature_function_emits_sql(config: ProjectConfig, tags: Tags) -> None:
    """Verify define_feature_function registers calculate_is_weekend."""
    model = _model(config, tags)
    model.define_feature_function()

    sql = model.spark.sql.call_args.args[0]
    assert "CREATE OR REPLACE FUNCTION" in sql
    assert "calculate_is_weekend" in sql
    assert "day_of_week >= 5" in sql
