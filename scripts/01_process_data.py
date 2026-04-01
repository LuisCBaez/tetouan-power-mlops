"""Preprocess raw data and save to Unity Catalog Delta tables."""

import argparse

from loguru import logger
from pyspark.sql import SparkSession

from tetouan_power.config import ProjectConfig
from tetouan_power.data_processor import DataProcessor


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments passed by Databricks job."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, required=True)
    parser.add_argument("--env", type=str, default="dev")
    return parser.parse_args()


def main() -> None:
    """Run the full preprocessing pipeline."""
    args = parse_args()

    config_path = f"{args.root_path}/files/project_config.yaml"
    config = ProjectConfig.from_yaml(config_path=config_path, env=args.env)
    logger.info(f"Config loaded for env={args.env}: {config.catalog_name}.{config.schema_name}")

    spark = SparkSession.builder.getOrCreate()

    raw_path = f"/Volumes/{config.catalog_name}/{config.schema_name}/data/raw/tetouan-power-consumption.csv"
    df = spark.read.csv(raw_path, header=True, inferSchema=True).toPandas()
    logger.info(f"Loaded {len(df)} rows from {raw_path}")

    processor = DataProcessor(pandas_df=df, config=config, spark=spark)
    processor.preprocess()
    logger.info(f"Preprocessed: {len(processor.df)} rows, {list(processor.df.columns)}")

    train, val, test = processor.split_data()
    logger.info(f"Split sizes: train={len(train)}, val={len(val)}, test={len(test)}")

    processor.save_to_catalog(train, val, test)
    logger.info("Data saved to catalog.")

    processor.enable_change_data_feed()
    logger.info("Change Data Feed enabled on all tables.")


if __name__ == "__main__":
    main()