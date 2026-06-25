"""Post-processing utilities for power consumption predictions."""

import numpy as np


def adjust_predictions(predictions: np.ndarray) -> np.ndarray:
    """Clip negative predictions to zero.

    Power consumption cannot be negative. Raw model predictions may
    occasionally go below zero (especially for low-consumption periods
    near zero). This function enforces the physical constraint.

    Args:
        predictions: Raw model predictions (may contain negatives).

    Returns:
        Predictions with all negative values replaced by zero.
    """
    return np.clip(predictions, a_min=0, a_max=None)
