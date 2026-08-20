"""Configuration for the Tetouan power pipeline."""

from typing import Any

import yaml
from pydantic import BaseModel


class SplitConfig(BaseModel):
    """Time-based split dates for train/val/test (Tetouan time series)."""

    train_end: str  # e.g. "2017-09-30" — train = data before this
    val_end: str  # e.g. "2017-11-30" — val = train_end to val_end, test = val_end to end


class ProjectConfig(BaseModel):
    """Project configuration loaded from YAML with environment-specific overrides."""

    num_features: list[str]
    cat_features: list[str]
    target: str
    catalog_name: str
    schema_name: str
    parameters: dict[str, Any]
    split: SplitConfig | None = None
    experiment_name_basic: str | None = None
    experiment_name_custom: str | None = None
    experiment_name_fe: str | None = None

    @classmethod
    def from_yaml(cls, config_path: str, env: str = "dev") -> "ProjectConfig":
        """Load YAML and merge catalog/schema from the chosen environment."""
        if env not in ("prd", "acc", "dev"):
            raise ValueError(f"Invalid env: {env}. Expected 'prd', 'acc', or 'dev'.")

        with open(config_path, encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)

        config_dict["catalog_name"] = config_dict[env]["catalog_name"]
        config_dict["schema_name"] = config_dict[env]["schema_name"]

        return cls(**config_dict)


class Tags(BaseModel):
    """Tags for MLflow runs: git_sha, branch, job_run_id."""

    git_sha: str
    branch: str
    job_run_id: str = ""  # optional for local runs
