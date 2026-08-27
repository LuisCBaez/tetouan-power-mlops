"""Train and register the CustomModel on Databricks."""

import argparse

import mlflow
from loguru import logger
from pyspark.sql import SparkSession

from tetouan_power.config import ProjectConfig, Tags
from tetouan_power.models.custom_model import CustomModel


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments passed by Databricks job."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, required=True)
    parser.add_argument("--env", type=str, default="dev")
    parser.add_argument("--git_sha", type=str, default="local")
    parser.add_argument("--branch", type=str, default="local")
    parser.add_argument("--job_run_id", type=str, default="")
    return parser.parse_args()


def main() -> None:
    """Run the full training pipeline."""
    args = parse_args()

    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")

    config_path = f"{args.root_path}/files/project_config.yaml"
    config = ProjectConfig.from_yaml(config_path=config_path, env=args.env)

    spark = SparkSession.builder.getOrCreate()
    tags = Tags(git_sha=args.git_sha, branch=args.branch, job_run_id=args.job_run_id)

    whl_path = f"{args.root_path}/dist/tetouan_power_mlops-0.1.0-py3-none-any.whl"

    model = CustomModel(config=config, tags=tags, spark=spark, code_paths=[whl_path])
    logger.info("Model initialized.")

    model.load_data()
    logger.info("Data loaded.")

    model.prepare_features()
    model.train()
    logger.info("Training complete.")

    model.log_model(dataset_type="SparkDataset")
    logger.info("Model logged to MLflow.")

    model.register_model()
    logger.info("Model registered in Unity Catalog.")


if __name__ == "__main__":
    main()
