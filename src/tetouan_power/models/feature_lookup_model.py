"""FeatureLookUpModel: Databricks Feature Engineering for Tetouan power forecasting.

Builds a training set from a Unity Catalog feature table (weather, looked up by
timestamp) and an on-demand feature function (is_weekend). The Feature Engineering
client only runs inside Databricks; this module imports it lazily so it can be
imported and partially unit-tested locally with mocks.
"""

from typing import TYPE_CHECKING

import mlflow
import numpy as np
from lightgbm import LGBMRegressor
from loguru import logger
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from pyspark.sql import DataFrame, SparkSession
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from tetouan_power.config import ProjectConfig, Tags
from tetouan_power.utils import adjust_predictions

if TYPE_CHECKING:  # import only for type hints; not at runtime (Databricks-only package)
    from databricks.feature_engineering import FeatureEngineeringClient


class FeatureLookUpModel:
    """Train, log, and register a feature-engineered model on Databricks.

    Uses a Unity Catalog feature table (weather, keyed by the `id` timestamp) via
    FeatureLookup, and a UC feature function (is_weekend, computed on demand) via
    FeatureFunction. fe.log_model packages the feature graph into the model so that
    fe.score_batch can later score rows that contain only keys + non-stored columns.
    """

    def __init__(self, config: ProjectConfig, tags: Tags, spark: SparkSession) -> None:
        """Initialize with config, tags, and a SparkSession (names derived here)."""
        self.config = config
        self.spark = spark
        self.num_features = config.num_features
        self.cat_features = config.cat_features
        self.target = config.target
        self.parameters = config.parameters
        self.catalog_name = config.catalog_name
        self.schema_name = config.schema_name

        # Feature store object names (derived from catalog/schema).
        self.feature_table_name = f"{self.catalog_name}.{self.schema_name}.power_features"
        self.function_name = f"{self.catalog_name}.{self.schema_name}.calculate_is_weekend"

        # The 5 weather features live in the feature table; is_weekend comes from the UDF.
        self.lookup_features = [
            "temperature",
            "humidity",
            "wind_speed",
            "general_diffuse_flows",
            "diffuse_flows",
        ]

        self.experiment_name = config.experiment_name_fe
        self.tags = tags.model_dump()

        # Lazy: the FE client only works on Databricks. Created on first use.
        self._fe: FeatureEngineeringClient | None = None

    @property
    def fe(self) -> "FeatureEngineeringClient":
        """Lazily create the Feature Engineering client (Databricks-only)."""
        if self._fe is None:
            from databricks.feature_engineering import FeatureEngineeringClient

            self._fe = FeatureEngineeringClient()
        return self._fe

    def create_feature_table(self) -> None:
        """Create (or replace) the power_features weather table and populate it.

        Keyed by `id` (the timestamp string). Change Data Feed is enabled so Phase 4
        can sync this table to an online store for real-time serving.
        """
        self.spark.sql(f"""
        CREATE OR REPLACE TABLE {self.feature_table_name}
        (id STRING NOT NULL,
         temperature DOUBLE,
         humidity DOUBLE,
         wind_speed DOUBLE,
         general_diffuse_flows DOUBLE,
         diffuse_flows DOUBLE);
        """)
        self.spark.sql(f"ALTER TABLE {self.feature_table_name} ADD CONSTRAINT power_pk PRIMARY KEY(id);")
        self.spark.sql(f"ALTER TABLE {self.feature_table_name} SET TBLPROPERTIES (delta.enableChangeDataFeed = true);")

        cols = "id, " + ", ".join(self.lookup_features)
        for split in ("train_set", "val_set", "test_set"):
            self.spark.sql(
                f"INSERT INTO {self.feature_table_name} "
                f"SELECT {cols} FROM {self.catalog_name}.{self.schema_name}.{split}"
            )
        logger.info(f"Feature table {self.feature_table_name} created and populated.")

    def define_feature_function(self) -> None:
        """Register the on-demand UDF that derives is_weekend from day_of_week.

        Time features are pure functions of the timestamp, so they need not be stored.
        Computing is_weekend on demand guarantees no training/serving skew.
        """
        self.spark.sql(f"""
        CREATE OR REPLACE FUNCTION {self.function_name}(day_of_week INT)
        RETURNS INT
        LANGUAGE PYTHON AS
        $$
        return 1 if day_of_week >= 5 else 0
        $$
        """)
        logger.info(f"Feature function {self.function_name} defined.")

    def load_data(self) -> None:
        """Load train/test from Delta, dropping columns sourced from the feature store.

        Weather (looked up) and is_weekend (computed by the UDF) are dropped from the
        base training frame so they come from the feature table / function instead.
        """
        drop_cols = self.lookup_features + ["is_weekend"]
        self.train_set = self.spark.table(f"{self.catalog_name}.{self.schema_name}.train_set").drop(*drop_cols)

        # Cast lookup key + UDF input to the types the feature store expects.
        self.train_set = self.train_set.withColumn("id", self.train_set["id"].cast("string"))
        self.train_set = self.train_set.withColumn("day_of_week", self.train_set["day_of_week"].cast("int"))

        # Test set keeps all columns: used for an offline metric (already has weather + is_weekend).
        self.test_set = self.spark.table(f"{self.catalog_name}.{self.schema_name}.test_set").toPandas()
        self.data_version = "0"
        logger.info("Data loaded.")

    def feature_engineering(self) -> None:
        """Build the training set: weather via FeatureLookup, is_weekend via FeatureFunction."""
        from databricks.feature_engineering import FeatureFunction, FeatureLookup

        self.training_set = self.fe.create_training_set(
            df=self.train_set,
            label=self.target,
            feature_lookups=[
                FeatureLookup(
                    table_name=self.feature_table_name,
                    feature_names=self.lookup_features,
                    lookup_key="id",
                ),
                FeatureFunction(
                    udf_name=self.function_name,
                    output_name="is_weekend",
                    input_bindings={"day_of_week": "day_of_week"},
                ),
            ],
            exclude_columns=["update_timestamp_utc", "datetime"],
        )

        self.training_df = self.training_set.load_df().toPandas()

        self.X_train = self.training_df[self.num_features]
        self.y_train = self.training_df[self.target]
        self.X_test = self.test_set[self.num_features]
        self.y_test = self.test_set[self.target]
        logger.info("Feature engineering completed.")

    def train(self) -> None:
        """Train StandardScaler + LGBM, log metrics, and log the model with feature metadata."""
        logger.info("Starting training...")
        pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("regressor", LGBMRegressor(**self.parameters)),
            ]
        )

        mlflow.set_experiment(self.experiment_name)

        with mlflow.start_run(tags=self.tags) as run:
            self.run_id = run.info.run_id

            pipeline.fit(self.X_train, self.y_train)
            # Clip-to-zero for the offline metric; see the note on serve-time post-processing.
            y_pred = adjust_predictions(pipeline.predict(self.X_test))

            mse = mean_squared_error(self.y_test, y_pred)
            mae = mean_absolute_error(self.y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(self.y_test, y_pred)

            logger.info(f"MAE: {mae:.2f} | RMSE: {rmse:.2f} | R2: {r2:.4f}")

            mlflow.log_param("model_type", "LightGBM with StandardScaler (feature lookup)")
            mlflow.log_params(self.parameters)
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2_score", r2)

            signature = infer_signature(model_input=self.X_train, model_output=y_pred)

            self.fe.log_model(
                model=pipeline,
                flavor=mlflow.sklearn,
                artifact_path="lightgbm-pipeline-model-fe",
                training_set=self.training_set,
                signature=signature,
            )
        logger.info("Model logged with feature metadata.")

    def register_model(self) -> str:
        """Register the FE model in Unity Catalog and set the 'latest-model' alias."""
        logger.info("Registering feature-engineered model...")
        model_name = f"{self.catalog_name}.{self.schema_name}.tetouan_power_model_fe"

        registered_model = mlflow.register_model(
            model_uri=f"runs:/{self.run_id}/lightgbm-pipeline-model-fe",
            name=model_name,
            tags=self.tags,
        )

        client = MlflowClient()
        client.set_registered_model_alias(
            name=model_name,
            alias="latest-model",
            version=registered_model.version,
        )
        logger.info(f"Registered version {registered_model.version} with alias 'latest-model'.")
        return registered_model.version

    def load_latest_model_and_predict(self, X: DataFrame) -> DataFrame:
        """Batch-score with the FE client. X needs only keys + non-stored columns.

        fe.score_batch performs the lookups and runs the UDF automatically, so X must
        contain the lookup key (`id`), the UDF input (`day_of_week`), and the base
        features not stored in the feature table (`hour`, `month`). Weather and
        is_weekend are filled in by the feature graph.

        Args:
            X: Spark DataFrame of rows to score (keys + non-stored columns).

        Returns:
            Spark DataFrame with a `prediction` column.
        """
        model_uri = f"models:/{self.catalog_name}.{self.schema_name}.tetouan_power_model_fe@latest-model"
        return self.fe.score_batch(model_uri=model_uri, df=X)
