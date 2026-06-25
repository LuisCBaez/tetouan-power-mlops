"""Unit tests for CustomModel."""

import mlflow
import pandas as pd
from loguru import logger
from mlflow.tracking import MlflowClient
from pyspark.sql import SparkSession
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from tests.conftest import CATALOG_DIR, TRACKING_URI
from tetouan_power.config import ProjectConfig, Tags
from tetouan_power.models.custom_model import CustomModel

mlflow.set_tracking_uri(TRACKING_URI)


def test_custom_model_init(config: ProjectConfig, tags: Tags, spark_session: SparkSession) -> None:
    """Verify constructor stores config, tags, spark, and code_paths."""
    model = CustomModel(config=config, tags=tags, spark=spark_session, code_paths=[])
    assert isinstance(model, CustomModel)
    assert isinstance(model.config, ProjectConfig)
    assert isinstance(model.tags, dict)
    assert isinstance(model.spark, SparkSession)
    assert isinstance(model.code_paths, list)
    assert not model.code_paths


def test_load_data_validate_df_assignment(mock_custom_model: CustomModel) -> None:
    """Verify train and test DataFrames are loaded from mocked Spark."""
    train_data = pd.read_csv((CATALOG_DIR / "train_set.csv").as_posix())
    test_data = pd.read_csv((CATALOG_DIR / "test_set.csv").as_posix())

    mock_custom_model.load_data()

    pd.testing.assert_frame_equal(mock_custom_model.train_set, train_data)
    pd.testing.assert_frame_equal(mock_custom_model.test_set, test_data)


def test_load_data_validate_splits(mock_custom_model: CustomModel) -> None:
    """Verify feature/target splits match config."""
    train_data = pd.read_csv((CATALOG_DIR / "train_set.csv").as_posix())
    test_data = pd.read_csv((CATALOG_DIR / "test_set.csv").as_posix())

    mock_custom_model.load_data()

    expected_features = mock_custom_model.num_features
    pd.testing.assert_frame_equal(mock_custom_model.X_train, train_data[expected_features])
    pd.testing.assert_series_equal(mock_custom_model.y_train, train_data[mock_custom_model.target])
    pd.testing.assert_frame_equal(mock_custom_model.X_test, test_data[expected_features])
    pd.testing.assert_series_equal(mock_custom_model.y_test, test_data[mock_custom_model.target])


def test_prepare_features(mock_custom_model: CustomModel) -> None:
    """Verify pipeline has StandardScaler + LGBMRegressor steps."""
    mock_custom_model.prepare_features()

    assert isinstance(mock_custom_model.pipeline, Pipeline)
    assert isinstance(mock_custom_model.pipeline.steps[0][1], StandardScaler)


def test_train(mock_custom_model: CustomModel) -> None:
    """Verify pipeline trains and learns the correct number of features."""
    mock_custom_model.load_data()
    mock_custom_model.prepare_features()
    mock_custom_model.train()

    expected_feature_count = len(mock_custom_model.config.num_features)
    assert mock_custom_model.pipeline.n_features_in_ == expected_feature_count


def test_log_model_with_pandas_dataset(mock_custom_model: CustomModel) -> None:
    """Verify MLflow experiment and run are created with metrics."""
    mock_custom_model.load_data()
    mock_custom_model.prepare_features()
    mock_custom_model.train()
    mock_custom_model.log_model(dataset_type="PandasDataset")

    client = MlflowClient()
    experiment = mlflow.get_experiment_by_name(mock_custom_model.experiment_name)
    assert experiment is not None
    assert experiment.name == mock_custom_model.experiment_name

    runs = client.search_runs(experiment.experiment_id, order_by=["start_time desc"], max_results=1)
    assert len(runs) == 1


def test_register_model(mock_custom_model: CustomModel) -> None:
    """Verify model is registered locally with 'latest-model' alias."""
    mock_custom_model.load_data()
    mock_custom_model.prepare_features()
    mock_custom_model.train()
    mock_custom_model.log_model(dataset_type="PandasDataset")
    mock_custom_model.register_model()

    client = MlflowClient()
    model_name = f"{mock_custom_model.catalog_name}.{mock_custom_model.schema_name}.tetouan_power_model_custom"

    model = client.get_registered_model(model_name)
    logger.info(f"Registered model: {model.name}, aliases: {model.aliases}")
    alias, _version = model.aliases.popitem()
    assert alias == "latest-model"


def test_retrieve_current_run_metadata(mock_custom_model: CustomModel) -> None:
    """Verify metrics and params are retrievable from the logged run."""
    mock_custom_model.load_data()
    mock_custom_model.prepare_features()
    mock_custom_model.train()
    mock_custom_model.log_model(dataset_type="PandasDataset")

    metrics, params = mock_custom_model.retrieve_current_run_metadata()
    assert isinstance(metrics, dict)
    assert "mae" in metrics
    assert "rmse" in metrics
    assert isinstance(params, dict)
    assert "learning_rate" in params


def test_load_latest_model_and_predict(mock_custom_model: CustomModel) -> None:
    """Verify model loads by alias and produces predictions."""
    mock_custom_model.load_data()
    mock_custom_model.prepare_features()
    mock_custom_model.train()
    mock_custom_model.log_model(dataset_type="PandasDataset")
    mock_custom_model.register_model()

    input_data = mock_custom_model.X_test.iloc[0:1]
    predictions = mock_custom_model.load_latest_model_and_predict(input_data=input_data)
    assert len(predictions) == 1
    assert all(p >= 0 for p in predictions)
