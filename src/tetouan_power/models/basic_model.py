"""BasicModel: sklearn pipeline with native MLflow logging."""

from datetime import UTC, datetime

import mlflow
import numpy as np
from lightgbm import LGBMRegressor
from loguru import logger
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from pyspark.sql import SparkSession
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from tetouan_power.config import ProjectConfig, Tags


class BasicModel:
    """Train, log, and register a model using native sklearn MLflow logging."""

    def __init__(self, config: ProjectConfig, tags: Tags, spark: SparkSession) -> None:
        """Initialize with config, tags, and a SparkSession."""
        self.config = config
        self.spark = spark
        self.num_features = config.num_features
        self.target = config.target
        self.parameters = config.parameters
        self.catalog_name = config.catalog_name
        self.schema_name = config.schema_name
        self.experiment_name = config.experiment_name_basic
        self.tags = tags.model_dump()

    def load_data(self) -> None:
        """Load train and test sets from Unity Catalog Delta tables."""
        logger.info("Loading data from catalog tables...")
        self.train_set_spark = self.spark.table(f"{self.catalog_name}.{self.schema_name}.train_set")
        self.train_set = self.train_set_spark.toPandas()
        self.test_set = self.spark.table(f"{self.catalog_name}.{self.schema_name}.test_set").toPandas()

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

    def log_model(self) -> None:
        """Log the trained model, metrics, and dataset lineage to MLflow."""
        mlflow.set_experiment(self.experiment_name)
        run_name = f"Basic-model-{datetime.now(UTC):%Y%m%d-%H%M%S}"
        with mlflow.start_run(run_name=run_name, tags=self.tags) as run:
            self.run_id = run.info.run_id

            y_pred = self.pipeline.predict(self.X_test)

            mse = mean_squared_error(self.y_test, y_pred)
            mae = mean_absolute_error(self.y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(self.y_test, y_pred)

            logger.info(f"MAE: {mae:.2f} | RMSE: {rmse:.2f} | R2: {r2:.4f}")

            mlflow.log_param("model_type", "LightGBM with StandardScaler")
            mlflow.log_params(self.parameters)
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2_score", r2)

            signature = infer_signature(model_input=self.X_train, model_output=y_pred)

            dataset = mlflow.data.from_spark(
                self.train_set_spark,
                table_name=f"{self.catalog_name}.{self.schema_name}.train_set",
                version="0",
            )

            mlflow.log_input(dataset, context="training")

            mlflow.sklearn.log_model(
                sk_model=self.pipeline,
                artifact_path="lightgbm-pipeline-model",
                signature=signature,
                input_example=self.X_train.iloc[0:1],
            )
            logger.info(f"Model logged to MLflow: runs:/{self.run_id}/lightgbm-pipeline-model")

    def register_model(self) -> None:
        """Register the model in Unity Catalog and set the 'latest-model' alias."""
        logger.info("Registering model in Unity Catalog...")
        model_name = f"{self.catalog_name}.{self.schema_name}.tetouan_power_model_basic"

        registered_model = mlflow.register_model(
            model_uri=f"runs:/{self.run_id}/lightgbm-pipeline-model",
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
