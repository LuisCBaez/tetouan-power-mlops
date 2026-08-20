# Databricks notebook source
# MAGIC %md
# MAGIC # Phase 2: Model experimentation
# MAGIC
# MAGIC Run `01_databricks_validation_demo.py` first. This notebook trains both Phase 2 model variants, logs
# MAGIC their runs to MLflow, registers them in Unity Catalog, verifies the `latest-model` aliases, and scores a
# MAGIC small batch with the custom pyfunc model.

# COMMAND ----------

# MAGIC %pip install mlflow==3.10.1 lightgbm==4.6.0 loguru==0.7.3 scikit-learn==1.8.0
# MAGIC %restart_python

# COMMAND ----------

import subprocess
import sys
import tempfile
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from pyspark.sql import SparkSession


def find_repo_root(start: Path) -> Path:
    """Find the project root from a notebook running inside the repository."""
    for candidate in (start, *start.parents):
        if (candidate / "project_config.yaml").is_file() and (candidate / "src").is_dir():
            return candidate
    raise FileNotFoundError("Could not find project_config.yaml and src/. Run this notebook from the cloned repo.")


repo_root = find_repo_root(Path.cwd())
sys.path.insert(0, str(repo_root / "src"))

from tetouan_power.config import ProjectConfig, Tags  # noqa: E402
from tetouan_power.models.basic_model import BasicModel  # noqa: E402
from tetouan_power.models.custom_model import CustomModel  # noqa: E402

# COMMAND ----------

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

config = ProjectConfig.from_yaml(config_path=str(repo_root / "project_config.yaml"), env="dev")
tags = Tags(git_sha="demo", branch="feature/databricks-demo-notebooks", job_run_id="interactive")
spark = SparkSession.builder.getOrCreate()
client = MlflowClient()

base_table_name = f"{config.catalog_name}.{config.schema_name}"
for split_name in ("train_set", "val_set", "test_set"):
    assert spark.catalog.tableExists(f"{base_table_name}.{split_name}"), (
        f"Missing {base_table_name}.{split_name}. Run 01_databricks_validation_demo.py first."
    )

print(f"Using Phase 1 tables from {base_table_name}.")

# COMMAND ----------

# BasicModel logs the native sklearn pipeline.
basic_model = BasicModel(config=config, tags=tags, spark=spark)
basic_model.load_data()
basic_model.prepare_features()
basic_model.train()
basic_model.log_model()
basic_model.register_model()

basic_model_name = f"{base_table_name}.tetouan_power_model_basic"
basic_run = mlflow.get_run(basic_model.run_id)
basic_alias = client.get_model_version_by_alias(basic_model_name, "latest-model")

assert basic_run.info.run_name.startswith("Basic-model-")
assert {"mae", "mse", "rmse", "r2_score"}.issubset(basic_run.data.metrics)
assert basic_alias.run_id == basic_model.run_id

loaded_basic_model = mlflow.sklearn.load_model(f"models:/{basic_model_name}@latest-model")
basic_predictions = loaded_basic_model.predict(basic_model.X_test.head(10))
assert len(basic_predictions) == 10
assert np.isfinite(basic_predictions).all()

print(f"BasicModel run: {basic_model.run_id}")
print(f"BasicModel registry version: {basic_alias.version}")

# COMMAND ----------

# CustomModel packages project code with the model. Build a fresh wheel outside the repo to avoid stale artifacts.
wheel_dir = Path(tempfile.mkdtemp(prefix="tetouan-power-demo-wheel-"))
subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "wheel",
        "--no-deps",
        "--wheel-dir",
        str(wheel_dir),
        str(repo_root),
    ],
    check=True,
)

wheel_paths = sorted(wheel_dir.glob("tetouan_power_mlops-*.whl"))
assert len(wheel_paths) == 1, f"Expected one project wheel in {wheel_dir}, found {wheel_paths}"
wheel_path = wheel_paths[0]
print(f"Built wheel: {wheel_path}")

# COMMAND ----------

custom_model = CustomModel(
    config=config,
    tags=tags,
    spark=spark,
    code_paths=[wheel_path.as_posix()],
)
custom_model.load_data()
custom_model.prepare_features()
custom_model.train()
custom_model.log_model(dataset_type="SparkDataset")
custom_model.register_model()

custom_model_name = f"{base_table_name}.tetouan_power_model_custom"
custom_run = mlflow.get_run(custom_model.run_id)
custom_alias = client.get_model_version_by_alias(custom_model_name, "latest-model")
custom_metrics, custom_params = custom_model.retrieve_current_run_metadata()

assert custom_run.info.run_name.startswith("Custom-model-")
assert custom_run.inputs.dataset_inputs
assert {"mae", "mse", "rmse", "r2_score"}.issubset(custom_metrics)
assert custom_params["model_type"] == "LightGBM with StandardScaler (pyfunc)"
assert custom_alias.run_id == custom_model.run_id

print(f"CustomModel run: {custom_model.run_id}")
print(f"CustomModel registry version: {custom_alias.version}")

# COMMAND ----------

score_input = custom_model.X_test.head(10)
custom_predictions = custom_model.load_latest_model_and_predict(score_input)

assert len(custom_predictions) == len(score_input)
assert np.isfinite(custom_predictions).all()
assert (custom_predictions >= 0).all(), "Power predictions must be non-negative"

prediction_preview = pd.DataFrame(
    {
        "id": custom_model.test_set.loc[score_input.index, "id"].to_numpy(),
        "prediction": custom_predictions,
    }
)
display(prediction_preview)  # noqa: F821 - Databricks injects display into the notebook runtime.

# COMMAND ----------

run_summary = pd.DataFrame(
    [
        {
            "model": "basic",
            "run_id": basic_model.run_id,
            "registered_version": basic_alias.version,
            **basic_run.data.metrics,
        },
        {
            "model": "custom",
            "run_id": custom_model.run_id,
            "registered_version": custom_alias.version,
            **custom_metrics,
        },
    ]
)
display(run_summary)  # noqa: F821
print("Phase 2 model experimentation demo succeeded. Continue with 03_feature_engineering_demo.py.")
