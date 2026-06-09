"""
Feature engineering for weekly pneumonia time series.

Builds a supervised learning dataset suitable for tree-based models
(RandomForest, XGBoost) from a univariate weekly time series.

Convention (no look-ahead bias):
  Row t predicts series[t] from values strictly before t.
    lag_k[t]           = series[t-k]               (series.shift(k))
    rolling_mean_w[t]  = mean(series[t-w : t])      (series.shift(1).rolling(w).mean())
    rolling_std_w[t]   = std(series[t-w : t])

  build_step_features() mirrors this exactly for recursive prediction so
  the feature representation is identical between training and inference.
"""

from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from pneumonia.utils import setup_logger

logger = setup_logger(__name__)

_DEFAULT_LAGS    = [1, 2, 4, 8, 13, 26, 52]
_DEFAULT_WINDOWS = [4, 13, 26]


def build_features(
    series: pd.Series,
    lags: Optional[List[int]] = None,
    windows: Optional[List[int]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Convert a weekly time series into a supervised learning dataset.

    Args:
        series: Weekly pneumonia cases with DatetimeIndex.
        lags: Lag periods (default: [1,2,4,8,13,26,52]).
        windows: Rolling window sizes for mean/std (default: [4,13,26]).

    Returns:
        X: Feature DataFrame (NaN rows already dropped).
        y: Target Series aligned to X.
    """
    if lags is None:
        lags = _DEFAULT_LAGS
    if windows is None:
        windows = _DEFAULT_WINDOWS

    df = pd.DataFrame(index=series.index)

    for k in lags:
        df[f'lag_{k}'] = series.shift(k)

    # shift(1) so rolling stats use only values before the target row
    shifted = series.shift(1)
    for w in windows:
        df[f'rolling_mean_{w}'] = shifted.rolling(w).mean()
        df[f'rolling_std_{w}']  = shifted.rolling(w).std()   # ddof=1

    woy = series.index.isocalendar().week.astype(int)
    df['week_of_year'] = woy
    df['sin_week']     = np.sin(2 * np.pi * woy / 52.1775)
    df['cos_week']     = np.cos(2 * np.pi * woy / 52.1775)
    df['month']        = series.index.month
    df['quarter']      = series.index.quarter
    df['trend']        = np.arange(len(series))

    valid = df.notna().all(axis=1)
    X = df[valid].copy()
    y = series[valid].copy()

    logger.info(
        f"build_features: {len(X)} usable rows from {len(series)} "
        f"({(~valid).sum()} NaN rows dropped). "
        f"Columns: {list(X.columns)}"
    )
    return X, y


def build_step_features(
    history: np.ndarray,
    target_date: pd.Timestamp,
    trend_idx: int,
    feature_names: List[str],
    lags: List[int],
    windows: List[int],
) -> np.ndarray:
    """
    Build one feature vector for a single recursive prediction step.

    Mirrors build_features() exactly so training and prediction produce
    identical feature representations.

    Args:
        history: 1-D array of known values (real + previously predicted).
                 Must have at least max(max_lag, max_window) elements.
        target_date: Timestamp of the value being predicted.
        trend_idx: Linear trend index for this step.
        feature_names: Ordered feature column names from training.
        lags: Same lag periods used during training.
        windows: Same rolling windows used during training.

    Returns:
        1-D float array of length len(feature_names).
    """
    values: dict = {}

    for k in lags:
        values[f'lag_{k}'] = float(history[-k]) if k <= len(history) else np.nan

    for w in windows:
        chunk = history[-w:] if w <= len(history) else history
        values[f'rolling_mean_{w}'] = float(np.mean(chunk))
        values[f'rolling_std_{w}']  = float(np.std(chunk, ddof=1)) if len(chunk) > 1 else 0.0

    iso = target_date.isocalendar()
    woy = int(iso[1])
    values['week_of_year'] = woy
    values['sin_week']     = float(np.sin(2 * np.pi * woy / 52.1775))
    values['cos_week']     = float(np.cos(2 * np.pi * woy / 52.1775))
    values['month']        = int(target_date.month)
    values['quarter']      = int(target_date.quarter)
    values['trend']        = int(trend_idx)

    return np.array([values[name] for name in feature_names], dtype=float)


def prepare_features_for_model(
    train_X: pd.DataFrame,
    test_X: pd.DataFrame,
    scaler=None,
) -> Tuple[np.ndarray, np.ndarray, Any]:
    """
    Scale train and test features consistently.

    Tree-based models are scale-invariant, but scaling is useful when
    features feed into ensemble or linear meta-learners.

    Args:
        train_X: Training feature DataFrame.
        test_X:  Test feature DataFrame (same columns as train_X).
        scaler:  Sklearn-compatible scaler. Defaults to StandardScaler.

    Returns:
        (scaled_train, scaled_test, fitted_scaler)
    """
    if set(train_X.columns) != set(test_X.columns):
        raise ValueError(
            f"Column mismatch — train: {list(train_X.columns)}, "
            f"test: {list(test_X.columns)}"
        )

    test_X = test_X[train_X.columns]  # align column order

    if scaler is None:
        scaler = StandardScaler()

    scaled_train = scaler.fit_transform(train_X.values)
    scaled_test  = scaler.transform(test_X.values)

    logger.info(
        f"prepare_features_for_model: "
        f"train {scaled_train.shape}, test {scaled_test.shape}"
    )
    return scaled_train, scaled_test, scaler
