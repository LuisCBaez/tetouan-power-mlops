"""CustomModel: pyfunc wrapper with post-processing and code_paths packaging."""

from datetime import UTC, datetime
from typing import Literal

import mlflow
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from loguru import logger
from mlflow.data.dataset_source import DatasetSource
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from mlflow.utils.environment import _mlflow_conda_env
from pyspark.sql import SparkSession
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from tetouan_power.config import ProjectConfig, Tags
from tetouan_power.mlflow_pip_deps import pyspark_pip_requirement
from tetouan_power.utils import adjust_predictions


class PowerConsumptionModelWrapper(mlflow.pyfunc.PythonModel):
    """Pyfunc wrapper that applies post-processing to predictions.

    This class wraps a trained sklearn pipeline and applies adjust_predictions()
    (clip negatives to zero) before returning results. It is what gets serialized
    and served by MLflow serving endpoints.
    """

    def __init__(self, model: object) -> None:
        """Store the trained pipeline."""
        self.model = model

    def predict(self, context: mlflow.pyfunc.PythonModelContext, model_input) -> np.ndarray:
        """Run prediction with post-processing.

        Args:
            context: MLflow context (unused, required by interface).
            model_input: Input features as DataFrame or array.

        Returns:
            Predictions with negatives clipped to zero.
        """
        predictions = self.model.predict(model_input)
        return adjust_predictions(predictions)


class CustomModel:
    """Train, log, and register a model using pyfunc wrapper with code_paths.

    Unlike BasicModel, this class:
    - Wraps the pipeline in PowerConsumptionModelWrapper for custom predict()
    - Bundles the tetouan_power .whl via code_paths so the model is self-contained
    - Uses mlflow.pyfunc.log_model() instead of mlflow.sklearn.log_model()
    """

    def __init__(self, config: ProjectConfig, tags: Tags, spark: SparkSession, code_paths: list[str]) -> None:
        """Initialize with config, tags, SparkSession, and code_paths to the built .whl."""
        self.config = config
        self.spark = spark
        self.num_features = config.num_features
        self.target = config.target
        self.parameters = config.parameters
        self.catalog_name = config.catalog_name
        self.schema_name = config.schema_name
        self.experiment_name = config.experiment_name_custom
        self.tags = tags.model_dump()
        self.code_paths = code_paths

    def load_data(self) -> None:
        """Load train and test sets from Unity Catalog Delta tables."""
        logger.info("Loading data from catalog tables...")
        self.train_set_spark = self.spark.table(f"{self.catalog_name}.{self.schema_name}.train_set")
        self.train_set = self.train_set_spark.toPandas()
        self.test_set = self.spark.table(f"{self.catalog_name}.{self.schema_name}.test_set").toPandas()
        self.data_version = "0"

        self.X_train = self.train_set[self.num_features]
        self.y_train = self.train_set[self.target]
        self.X_test = self.test_set[self.num_features]
        self.y_test = self.test_set[self.target]
        logger.info("Data loaded successfully.")

    def prepare_features(self) -> None:
        """Build the sklearn pipeline: StandardScaler + LGBMRegressor."""
        logger.info("Defining preprocessing pipeline...")
        self.pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("regressor", LGBMRegressor(**self.parameters)),
            ]
        )
        logger.info("Pipeline defined.")

    def train(self) -> None:
        """Fit the pipeline on training data."""
        logger.info("Starting training...")
        self.pipeline.fit(self.X_train, self.y_train)
        logger.info("Training complete.")

    def log_model(self, dataset_type: Literal["PandasDataset", "SparkDataset"] = "SparkDataset") -> None:
        """Log the trained model, metrics, and dataset lineage to MLflow.

        Args:
            dataset_type: Use "PandasDataset" for local tests (no Spark catalog),
                "SparkDataset" for Databricks (links to Delta table version).
        """
        mlflow.set_experiment(self.experiment_name)

        run_name = f"Custom-model-{datetime.now(UTC):%Y%m%d-%H%M%S}"

        additional_pip_deps = [pyspark_pip_requirement()]
        for package in self.code_paths:
            whl_name = package.split("/")[-1]
            additional_pip_deps.append(f"./code/{whl_name}")

        with mlflow.start_run(run_name=run_name, tags=self.tags) as run:
            self.run_id = run.info.run_id

            y_pred = self.pipeline.predict(self.X_test)

            mse = mean_squared_error(self.y_test, y_pred)
            mae = mean_absolute_error(self.y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(self.y_test, y_pred)

            logger.info(f"MAE: {mae:.2f} | RMSE: {rmse:.2f} | R2: {r2:.4f}")

            mlflow.log_param("model_type", "LightGBM with StandardScaler (pyfunc)")
            mlflow.log_params(self.parameters)
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2_score", r2)

            signature = infer_signature(
                model_input=self.X_train,
                model_output=self.pipeline.predict(self.X_train),
            )

            if dataset_type == "PandasDataset":
                dataset = mlflow.data.from_pandas(self.train_set, name="train_set")
            elif dataset_type == "SparkDataset":
                dataset = mlflow.data.from_spark(
                    self.train_set_spark,
                    table_name=f"{self.catalog_name}.{self.schema_name}.train_set",
                    version=self.data_version,
                )
            else:
                raise ValueError(f"Unsupported dataset_type: {dataset_type}")

            mlflow.log_input(dataset, context="training")

            conda_env = _mlflow_conda_env(additional_pip_deps=additional_pip_deps)

            mlflow.pyfunc.log_model(
                python_model=PowerConsumptionModelWrapper(self.pipeline),
                artifact_path="pyfunc-power-consumption-model",
                code_paths=self.code_paths,
                conda_env=conda_env,
                signature=signature,
                input_example=self.X_train.iloc[0:1],
            )

    def register_model(self) -> None:
        """Register the model in Unity Catalog and set the 'latest-model' alias."""
        logger.info("Registering model in Unity Catalog...")
        model_name = f"{self.catalog_name}.{self.schema_name}.tetouan_power_model_custom"

        registered_model = mlflow.register_model(
            model_uri=f"runs:/{self.run_id}/pyfunc-power-consumption-model",
            name=model_name,
            tags=self.tags,
        )

        client = MlflowClient()
        client.set_registered_model_alias(
            name=model_name,
            alias="latest-model",
            version=registered_model.version,
        )
        logger.info(f"Model registered as version {registered_model.version} with alias 'latest-model'.")

    def retrieve_current_run_dataset(self) -> DatasetSource:
        """Retrieve the dataset source linked to the current MLflow run."""
        run = mlflow.get_run(self.run_id)
        dataset_info = run.inputs.dataset_inputs[0].dataset
        dataset_source = mlflow.data.get_source(dataset_info)
        return dataset_source.load()

    def retrieve_current_run_metadata(self) -> tuple[dict, dict]:
        """Retrieve metrics and params from the current MLflow run."""
        run = mlflow.get_run(self.run_id)
        metrics = run.data.to_dictionary()["metrics"]
        params = run.data.to_dictionary()["params"]
        return metrics, params

    def load_latest_model_and_predict(self, input_data: pd.DataFrame) -> np.ndarray:
        """Load the model by alias and make predictions.

        Args:
            input_data: DataFrame with the same columns as training features.

        Returns:
            Predictions (with post-processing applied by the wrapper).
        """
        model_uri = f"models:/{self.catalog_name}.{self.schema_name}.tetouan_power_model_custom@latest-model"
        model = mlflow.pyfunc.load_model(model_uri)
        return model.predict(input_data)
