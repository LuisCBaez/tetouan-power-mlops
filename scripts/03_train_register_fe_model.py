"""Train and register the feature-engineered model on Databricks."""

import argparse

import mlflow
from loguru import logger
from pyspark.sql import SparkSession

from tetouan_power.config import ProjectConfig, Tags
from tetouan_power.models.feature_lookup_model import FeatureLookUpModel


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments passed by the Databricks job."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, required=True)
    parser.add_argument("--env", type=str, default="dev")
    parser.add_argument("--git_sha", type=str, default="local")
    parser.add_argument("--branch", type=str, default="local")
    parser.add_argument("--job_run_id", type=str, default="")
    return parser.parse_args()


def main() -> None:
    """Run the full feature-engineering training pipeline."""
    args = parse_args()

    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")

    config_path = f"{args.root_path}/files/project_config.yaml"
    config = ProjectConfig.from_yaml(config_path=config_path, env=args.env)

    spark = SparkSession.builder.getOrCreate()
    tags = Tags(git_sha=args.git_sha, branch=args.branch, job_run_id=args.job_run_id)

    model = FeatureLookUpModel(config=config, tags=tags, spark=spark)
    logger.info("FeatureLookUpModel initialized.")

    model.create_feature_table()
    logger.info("Feature table ready.")

    model.define_feature_function()
    logger.info("Feature function ready.")

    model.load_data()
    model.feature_engineering()
    logger.info("Training set built from feature store.")

    model.train()
    logger.info("Model trained and logged with feature metadata.")

    model.register_model()
    logger.info("Feature-engineered model registered in Unity Catalog.")


if __name__ == "__main__":
    main()
