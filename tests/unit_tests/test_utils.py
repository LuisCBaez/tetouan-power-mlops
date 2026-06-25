"""Unit tests for utils.py."""

import numpy as np

from tetouan_power.utils import adjust_predictions


def test_adjust_predictions_clips_negatives() -> None:
    """Verify negative values are clipped to zero."""
    predictions = np.array([100.0, -5.0, 200.0, -0.1, 0.0])
    result = adjust_predictions(predictions)
    expected = np.array([100.0, 0.0, 200.0, 0.0, 0.0])
    np.testing.assert_array_equal(result, expected)


def test_adjust_predictions_preserves_positives() -> None:
    """Verify positive values are unchanged."""
    predictions = np.array([100.0, 200.0, 300.0])
    result = adjust_predictions(predictions)
    np.testing.assert_array_equal(result, predictions)


def test_adjust_predictions_returns_ndarray() -> None:
    """Verify return type is always np.ndarray."""
    predictions = np.array([1.0, 2.0, 3.0])
    result = adjust_predictions(predictions)
    assert isinstance(result, np.ndarray)
