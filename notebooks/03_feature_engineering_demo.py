# Databricks notebook source
# Run on Serverless. The Feature Engineering client is preinstalled on Databricks ML runtimes.

# COMMAND ----------

# If the package is not on the cluster, install the built wheel (synced via the bundle in Phase 4):
# %pip install tetouan_power_mlops-0.1.0-py3-none-any.whl
# %restart_python

# COMMAND ----------

import sys
from pathlib import Path

# Make the src/ package importable when running the notebook from the repo (src layout).
sys.path.append(str(Path.cwd().parent / "src"))

import mlflow
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

from tetouan_power.config import ProjectConfig, Tags
from tetouan_power.models.feature_lookup_model import FeatureLookUpModel

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

config = ProjectConfig.from_yaml(config_path="../project_config.yaml", env="dev")
tags = Tags(git_sha="demo", branch="feature/phase3", job_run_id="0")
spark = SparkSession.builder.getOrCreate()

fe_model = FeatureLookUpModel(config=config, tags=tags, spark=spark)

# COMMAND ----------

# 1) Create the weather feature table (PK = id, CDF enabled) and populate it.
fe_model.create_feature_table()

# COMMAND ----------

# 2) Define the on-demand UDF that derives is_weekend from day_of_week.
fe_model.define_feature_function()
spark.sql(f"SELECT {fe_model.function_name}(6) AS is_weekend").show()  # Sunday -> 1

# COMMAND ----------

# 3) Build the training set (weather via lookup, is_weekend via function).
fe_model.load_data()
fe_model.feature_engineering()
fe_model.X_train.head()

# COMMAND ----------

# 4) Train + log the model WITH its feature metadata, then register it.
fe_model.train()
fe_model.register_model()

# COMMAND ----------

# 5) Score a batch passing ONLY keys + non-stored columns (no weather, no is_weekend).
score_cols = ["id", "hour", "day_of_week", "month"]
X_score = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set").select(*score_cols).limit(10)
predictions = fe_model.load_latest_model_and_predict(X_score)
predictions.select("id", "prediction").show()

# COMMAND ----------

# 6) Gotcha: score a row whose key does NOT exist in the feature table.
# The lookup finds no weather -> nulls -> the model errors or returns null.
# This is why Phase 4 adds an online store + default handling.
X_missing = X_score.withColumn("id", col("id").cast("string")).withColumn("id", col("id") + "_missing")
fe_model.load_latest_model_and_predict(X_missing).select("id", "prediction").show()
