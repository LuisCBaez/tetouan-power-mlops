# Databricks notebook source
# MAGIC %md
# MAGIC # Phase 1d: Databricks validation
# MAGIC
# MAGIC Run this notebook on Serverless before the Phase 2 and Phase 3 demos. It reads the raw CSV from the
# MAGIC Unity Catalog Volume, applies the production `DataProcessor`, overwrites the train/validation/test Delta
# MAGIC tables, enables Change Data Feed, and verifies the resulting data contract.

# COMMAND ----------

import sys
from pathlib import Path

from pyspark.sql import SparkSession


def find_repo_root(start: Path) -> Path:
    """Find the project root from a notebook running inside the repository."""
    for candidate in (start, *start.parents):
        if (candidate / "project_config.yaml").is_file() and (candidate / "src").is_dir():
            return candidate
    raise FileNotFoundError("Could not find project_config.yaml and src/. Run this notebook from the cloned repo.")


repo_root = find_repo_root(Path.cwd())
sys.path.insert(0, str(repo_root / "src"))

from tetouan_power.config import ProjectConfig  # noqa: E402
from tetouan_power.data_processor import DataProcessor  # noqa: E402

# COMMAND ----------

config = ProjectConfig.from_yaml(config_path=str(repo_root / "project_config.yaml"), env="dev")
spark = SparkSession.builder.getOrCreate()

base_table_name = f"{config.catalog_name}.{config.schema_name}"
raw_path = f"/Volumes/{config.catalog_name}/{config.schema_name}/data/raw/tetouan-power-consumption.csv"

print(f"Environment: {base_table_name}")
print(f"Raw input: {raw_path}")

# COMMAND ----------

# Read through Spark so Unity Catalog Volume permissions are enforced, then use the pandas-based processor.
raw_df = spark.read.option("header", True).option("inferSchema", True).csv(raw_path).toPandas()
assert not raw_df.empty, f"No rows were read from {raw_path}"

processor = DataProcessor(pandas_df=raw_df, config=config, spark=spark)
processor.preprocess()

expected_columns = {
    "datetime",
    *config.num_features,
    *config.cat_features,
    config.target,
    "id",
}
assert set(processor.df.columns) == expected_columns
assert processor.df["id"].is_unique
assert not processor.df.isnull().any().any()

print(f"Validated {len(processor.df):,} processed rows and {len(processor.df.columns)} columns.")

# COMMAND ----------

train_set, val_set, test_set = processor.split_data()
split_frames = {
    "train_set": train_set,
    "val_set": val_set,
    "test_set": test_set,
}

assert all(not frame.empty for frame in split_frames.values())
assert sum(len(frame) for frame in split_frames.values()) == len(processor.df)
assert train_set["datetime"].max() < val_set["datetime"].min()
assert val_set["datetime"].max() < test_set["datetime"].min()

for split_name, frame in split_frames.items():
    print(f"{split_name}: {len(frame):,} rows")

# COMMAND ----------

# This is intentionally destructive for the dev tables: every demo run replaces all three splits.
processor.save_to_catalog(train_set, val_set, test_set)
processor.enable_change_data_feed()

# COMMAND ----------

validation_rows = []

for split_name, expected_frame in split_frames.items():
    table_name = f"{base_table_name}.{split_name}"
    table_df = spark.table(table_name)
    actual_count = table_df.count()
    cdf_property = spark.sql(f"SHOW TBLPROPERTIES {table_name} ('delta.enableChangeDataFeed')").first()
    cdf_enabled = cdf_property["value"].lower() == "true"

    assert actual_count == len(expected_frame), f"Unexpected row count for {table_name}"
    assert "update_timestamp_utc" in table_df.columns
    assert cdf_enabled, f"Change Data Feed is not enabled for {table_name}"

    validation_rows.append(
        {
            "table": table_name,
            "rows": actual_count,
            "cdf_enabled": cdf_enabled,
        }
    )

validation_df = spark.createDataFrame(validation_rows).select("table", "rows", "cdf_enabled")
display(validation_df)  # noqa: F821 - Databricks injects display into the notebook runtime.

# COMMAND ----------

display(spark.table(f"{base_table_name}.train_set").limit(10))  # noqa: F821
print("Phase 1d Databricks validation succeeded. Continue with 02_model_experimentation_demo.py.")
