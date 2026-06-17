"""
Evaluation metrics for forecasting models

Functions for computing MAE, RMSE, MAPE, and other forecast accuracy metrics.
"""

import numpy as np
import pandas as pd
from typing import Union
import logging

from pneumonia.utils import setup_logger

logger = setup_logger(__name__)


def mean_absolute_error(
    actual: Union[np.ndarray, pd.Series],
    predicted: Union[np.ndarray, pd.Series]
) -> float:
    """
    Calculate Mean Absolute Error (MAE).
    
    MAE = (1/n) * Σ|actual - predicted|
    
    Args:
        actual: Actual values
        predicted: Predicted values
        
    Returns:
        MAE value
    """
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    
    if actual.shape != predicted.shape:
        raise ValueError(f"Shape mismatch: {actual.shape} vs {predicted.shape}")
    
    mae = np.mean(np.abs(actual - predicted))
    return float(mae)


def root_mean_squared_error(
    actual: Union[np.ndarray, pd.Series],
    predicted: Union[np.ndarray, pd.Series]
) -> float:
    """
    Calculate Root Mean Squared Error (RMSE).
    
    RMSE = √((1/n) * Σ(actual - predicted)²)
    
    Args:
        actual: Actual values
        predicted: Predicted values
        
    Returns:
        RMSE value
    """
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    
    if actual.shape != predicted.shape:
        raise ValueError(f"Shape mismatch: {actual.shape} vs {predicted.shape}")
    
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    return float(rmse)


def mean_absolute_percentage_error(
    actual: Union[np.ndarray, pd.Series],
    predicted: Union[np.ndarray, pd.Series],
    epsilon: float = 1e-10
) -> float:
    """
    Calculate Mean Absolute Percentage Error (MAPE).
    
    MAPE = (1/n) * Σ|actual - predicted| / |actual|
    
    WARNING: This metric is problematic when actual values are close to zero.
    MAPE will exclude observations where |actual| <= epsilon to avoid division by zero.
    If all values are excluded, returns NaN with warning.
    
    Consider using SMAPE (Symmetric MAPE) instead for more robust results.
    
    Args:
        actual: Actual values
        predicted: Predicted values
        epsilon: Threshold for excluding near-zero values (default 1e-10)
        
    Returns:
        MAPE value (as percentage), or NaN if all values are excluded
        
    Warning:
        If some observations are excluded due to near-zero actuals,
        the returned MAPE is biased and may not be comparable across datasets.
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    
    if actual.shape != predicted.shape:
        raise ValueError(f"Shape mismatch: {actual.shape} vs {predicted.shape}")
    
    # Identify near-zero values
    nonzero_mask = np.abs(actual) > epsilon
    n_excluded = (~nonzero_mask).sum()
    
    if n_excluded > 0:
        exclusion_rate = n_excluded / len(actual) * 100
        logger.warning(
            f"MAPE: Excluding {n_excluded}/{len(actual)} observations ({exclusion_rate:.1f}%) "
            f"with |actual| <= {epsilon} to avoid division by zero. "
            f"Consider using SMAPE instead."
        )
    
    if nonzero_mask.sum() == 0:
        logger.error(
            f"MAPE: All actual values are zero or near-zero (|actual| <= {epsilon}). "
            "MAPE is undefined. Returning NaN. "
            "This typically indicates: (1) very low disease counts, or (2) epsilon too large. "
            "Consider using SMAPE instead."
        )
        return np.nan
    
    actual_nz = actual[nonzero_mask]
    predicted_nz = predicted[nonzero_mask]
    
    mape = np.mean(np.abs((actual_nz - predicted_nz) / np.abs(actual_nz))) * 100
    return float(mape)


def symmetric_mean_absolute_percentage_error(
    actual: Union[np.ndarray, pd.Series],
    predicted: Union[np.ndarray, pd.Series]
) -> float:
    """
    Calculate Symmetric Mean Absolute Percentage Error (SMAPE).
    
    SMAPE = (1/n) * Σ|actual - predicted| / ((|actual| + |predicted|) / 2)
    
    More robust than MAPE for asymmetric data.
    
    Args:
        actual: Actual values
        predicted: Predicted values
        
    Returns:
        SMAPE value (as percentage)
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    
    if actual.shape != predicted.shape:
        raise ValueError(f"Shape mismatch: {actual.shape} vs {predicted.shape}")
    
    denominator = (np.abs(actual) + np.abs(predicted)) / 2.0
    
    # Handle zero denominators
    nonzero_mask = denominator > 1e-10
    
    if nonzero_mask.sum() == 0:
        logger.warning("All denominators are zero; SMAPE undefined")
        return np.nan
    
    smape = np.mean(
        np.abs(actual[nonzero_mask] - predicted[nonzero_mask]) / denominator[nonzero_mask]
    ) * 100
    
    return float(smape)


def mean_directional_accuracy(
    actual: Union[np.ndarray, pd.Series],
    predicted: Union[np.ndarray, pd.Series]
) -> float:
    """
    Calculate Mean Directional Accuracy (MDA).
    
    Percentage of time series points where direction of change matches.
    
    Args:
        actual: Actual values
        predicted: Predicted values
        
    Returns:
        MDA value (as percentage)
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    
    if actual.shape != predicted.shape:
        raise ValueError(f"Shape mismatch: {actual.shape} vs {predicted.shape}")
    
    if len(actual) < 2:
        logger.warning("Need at least 2 observations for MDA")
        return np.nan
    
    # Direction of actual change: did actual go up from t to t+1?
    actual_changes = np.diff(actual)
    # Direction of predicted change: did the model forecast an increase over the last known actual?
    predicted_changes = predicted[1:] - actual[:-1]

    mda = np.mean(np.sign(actual_changes) == np.sign(predicted_changes)) * 100
    
    return float(mda)


def mean_error(
    actual: Union[np.ndarray, pd.Series],
    predicted: Union[np.ndarray, pd.Series]
) -> float:
    """
    Calculate Mean Error (ME) or Bias.
    
    ME = (1/n) * Σ(actual - predicted)
    
    Args:
        actual: Actual values
        predicted: Predicted values
        
    Returns:
        ME value
    """
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    
    if actual.shape != predicted.shape:
        raise ValueError(f"Shape mismatch: {actual.shape} vs {predicted.shape}")
    
    me = np.mean(actual - predicted)
    return float(me)


def r2_score(
    actual: Union[np.ndarray, pd.Series],
    predicted: Union[np.ndarray, pd.Series]
) -> float:
    """
    Calculate Coefficient of Determination (R2).
    
    Args:
        actual: Actual values
        predicted: Predicted values
        
    Returns:
        R2 value
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    
    if actual.shape != predicted.shape:
        raise ValueError(f"Shape mismatch: {actual.shape} vs {predicted.shape}")
    
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    if ss_tot < 1e-10:
        return 0.0
    return float(1.0 - (ss_res / ss_tot))


def mean_absolute_scaled_error(
    actual: Union[np.ndarray, pd.Series],
    predicted: Union[np.ndarray, pd.Series],
    training_actual: Union[np.ndarray, pd.Series] = None,
    seasonality: int = 1
) -> float:
    """
    Calculate Mean Absolute Scaled Error (MASE).
    
    Args:
        actual: Actual values
        predicted: Predicted values
        training_actual: Optional training values to estimate naive baseline error
        seasonality: Seasonal period (default 1)
        
    Returns:
        MASE value
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    
    if actual.shape != predicted.shape:
        raise ValueError(f"Shape mismatch: {actual.shape} vs {predicted.shape}")
    
    denom_data = np.asarray(training_actual, dtype=float) if training_actual is not None else actual
    if len(denom_data) <= seasonality:
        return np.nan
    
    mae_naive = np.mean(np.abs(denom_data[seasonality:] - denom_data[:-seasonality]))
    if mae_naive < 1e-10:
        return np.nan
    
    mae_model = np.mean(np.abs(actual - predicted))
    return float(mae_model / mae_naive)


def compute_all_metrics(
    actual: Union[np.ndarray, pd.Series],
    predicted: Union[np.ndarray, pd.Series],
    warn_on_nan: bool = True,
    training_actual: Union[np.ndarray, pd.Series] = None,
) -> dict:
    """
    Compute all evaluation metrics.
    
    Args:
        actual: Actual values
        predicted: Predicted values
        warn_on_nan: If True, log warning for any NaN metric values
        training_actual: Optional training values for MASE computation
        
    Returns:
        Dictionary with metric names and values (may contain NaN)
        
    Raises:
        ValueError: If shapes don't match
    """
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    
    if actual.shape != predicted.shape:
        raise ValueError(f"Shape mismatch: {actual.shape} vs {predicted.shape}")
    
    metrics = {
        "mae": mean_absolute_error(actual, predicted),
        "rmse": root_mean_squared_error(actual, predicted),
        "mape": mean_absolute_percentage_error(actual, predicted),
        "smape": symmetric_mean_absolute_percentage_error(actual, predicted),
        "mda": mean_directional_accuracy(actual, predicted),
        "me": mean_error(actual, predicted),
        "r2": r2_score(actual, predicted),
        "mase": mean_absolute_scaled_error(actual, predicted, training_actual=training_actual),
    }
    
    # Check for NaN values
    nan_metrics = [k for k, v in metrics.items() if np.isnan(v)]
    if nan_metrics and warn_on_nan:
        logger.warning(f"Metrics with NaN values: {nan_metrics}. Check data quality.")
    
    return metrics


def baseline_metrics(
    actual: Union[np.ndarray, pd.Series],
    baseline_value: float = None
) -> dict:
    """
    Compute metrics using a baseline (e.g., historical mean).
    
    Useful for comparison with actual forecasts.
    
    Args:
        actual: Actual values
        baseline_value: Baseline prediction. Uses historical mean if None.
        
    Returns:
        Dictionary with baseline metrics
    """
    actual = np.asarray(actual)
    
    if baseline_value is None:
        baseline_value = np.mean(actual)
    
    predictions = np.full_like(actual, baseline_value, dtype=float)
    
    baseline = {
        "method": "baseline",
        "baseline_value": float(baseline_value),
    }
    
    baseline.update(compute_all_metrics(actual, predictions))
    
    return baseline
