"""
Walk-Forward Validation module for time series forecasting models.

Provides utilities to run expanding and sliding window backtesting (walk-forward validation)
across multiple forecast horizons, with support for univariate (e.g., SARIMA) and
multivariate/machine learning models.
"""

import inspect
import logging
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd

from pneumonia.evaluation.metrics import compute_all_metrics
from pneumonia.models.base import BaseForecaster
from pneumonia.utils import setup_logger

logger = setup_logger(__name__)


class WalkForwardValidator:
    """
    Class to execute walk-forward validation (rolling-origin backtesting) on forecasting models.

    Supports expanding/sliding windows, configurable refitting frequencies, multiple forecast
    horizons, and automatic feature alignment for machine learning models.
    """

    def __init__(
        self,
        model_class: Type[BaseForecaster],
        model_params: Dict[str, Any],
        initial_train_size: Union[int, str, pd.Timestamp],
        horizon: int = 1,
        step_size: int = 1,
        window_type: Literal["expanding", "sliding"] = "expanding",
        refit_frequency: int = 1,
        min_train_size: int = 104,
    ):
        """
        Initialize the walk-forward validator.

        Args:
            model_class: Class of the model to evaluate (subclass of BaseForecaster).
            model_params: Dictionary of parameters to initialize the model.
            initial_train_size: Number of steps (int) or end date (str/Timestamp) of the first training set.
            horizon: Forecast horizon (number of steps ahead) at each origin.
            step_size: Number of steps to advance the origin at each iteration.
            window_type: 'expanding' (use all history) or 'sliding' (use a fixed-size window of history).
            refit_frequency: Frequency of model retraining. If 1, refits at every step.
                            If N, refits every N steps. If 0 or None, fits only once at the beginning.
            min_train_size: Minimum required size for training set (default: 104 weeks / 2 years).
        """
        self.model_class = model_class
        self.model_params = model_params
        self.initial_train_size_input = initial_train_size
        self.horizon = horizon
        self.step_size = step_size
        self.window_type = window_type
        self.refit_frequency = refit_frequency or 0
        self.min_train_size = min_train_size

        # Validation of basic parameters
        if self.horizon <= 0:
            raise ValueError(f"horizon must be positive, got {self.horizon}")
        if self.step_size <= 0:
            raise ValueError(f"step_size must be positive, got {self.step_size}")
        if self.window_type not in ["expanding", "sliding"]:
            raise ValueError(f"window_type must be 'expanding' or 'sliding', got {self.window_type}")

    def run(
        self,
        y: pd.Series,
        X: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        fit_kwargs: Optional[Dict[str, Any]] = None,
        predict_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute the walk-forward validation loop.

        Args:
            y: Target time series (pandas Series with DatetimeIndex).
            X: Feature matrix of the same length as y (optional, for ML models).
            fit_kwargs: Keyword arguments passed to the model's fit method.
            predict_kwargs: Keyword arguments passed to the model's predict method.

        Returns:
            Dictionary containing:
                - 'metrics_by_horizon': Dictionary mapping horizon (1..H) to overall metrics.
                - 'step_results': List of details for each evaluation step.
                - 'predictions': DataFrame with actuals and predictions per horizon.
        """
        if not isinstance(y, pd.Series):
            raise TypeError(f"y must be a pandas Series, got {type(y)}")
        if not isinstance(y.index, pd.DatetimeIndex):
            raise TypeError(f"y index must be a DatetimeIndex, got {type(y.index)}")

        fit_kwargs = fit_kwargs or {}
        predict_kwargs = predict_kwargs or {}

        # Resolve initial_train_size to integer index
        if isinstance(self.initial_train_size_input, (str, pd.Timestamp)):
            initial_train_dt = pd.to_datetime(self.initial_train_size_input)
            initial_train_size = int((y.index <= initial_train_dt).sum())
        else:
            initial_train_size = int(self.initial_train_size_input)

        if initial_train_size < self.min_train_size:
            logger.warning(
                f"initial_train_size ({initial_train_size}) is less than min_train_size ({self.min_train_size})"
            )

        n_samples = len(y)
        if initial_train_size >= n_samples:
            raise ValueError(
                f"initial_train_size ({initial_train_size}) must be less than number of samples ({n_samples})"
            )

        # Realign features if provided
        features = None
        if X is not None:
            if len(X) != n_samples:
                raise ValueError(f"X length ({len(X)}) must match y length ({n_samples})")
            if isinstance(X, pd.DataFrame):
                if not X.index.equals(y.index):
                    logger.warning("Feature matrix X index does not match target y index. Realigning...")
                    features = X.reindex(y.index)
                else:
                    features = X
            else:
                # Convert numpy array to DataFrame to preserve index alignment
                features = pd.DataFrame(X, index=y.index)

        # Initialize the predictions DataFrame
        # Evaluated indices are all steps from initial_train_size to the end of target series y
        eval_indices = y.index[initial_train_size:]
        predictions_df = pd.DataFrame(index=eval_indices)
        predictions_df["actual"] = y.iloc[initial_train_size:]

        # Add columns for each horizon
        for h in range(1, self.horizon + 1):
            predictions_df[f"pred_h{h}"] = np.nan

        step_results = []
        model = None
        step_idx = 0

        # Walk-forward loop
        while True:
            train_end = initial_train_size + step_idx * self.step_size
            if train_end >= n_samples:
                break

            train_start = 0 if self.window_type == "expanding" else (train_end - initial_train_size)
            forecast_start = train_end
            forecast_end = min(train_end + self.horizon, n_samples)
            current_horizon = forecast_end - forecast_start

            if current_horizon <= 0:
                break

            # Slices
            y_train = y.iloc[train_start:train_end]
            y_val = y.iloc[forecast_start:forecast_end]

            train_dates = (y_train.index.min().strftime('%Y-%m-%d'), y_train.index.max().strftime('%Y-%m-%d'))
            val_dates = (y_val.index.min().strftime('%Y-%m-%d'), y_val.index.max().strftime('%Y-%m-%d'))

            logger.info(
                f"Step {step_idx}: Train [{train_dates[0]} to {train_dates[1]}] (size {len(y_train)}), "
                f"Forecast [{val_dates[0]} to {val_dates[1]}]"
            )

            # Determine if model retraining is needed
            should_refit = (
                model is None or
                (self.refit_frequency > 0 and step_idx % self.refit_frequency == 0)
            )

            if should_refit:
                logger.info(f"Refitting model at step {step_idx}")
                model = self.model_class(**self.model_params)

                # Dynamically determine fit signature to support ML/univariate model types
                current_fit_kwargs = fit_kwargs.copy()
                if features is not None:
                    X_train = features.iloc[train_start:train_end]
                    sig = inspect.signature(model.fit)
                    if "train_features" in sig.parameters:
                        current_fit_kwargs["train_features"] = X_train.values
                    elif "features" in sig.parameters:
                        current_fit_kwargs["features"] = X_train.values

                model.fit(y_train, **current_fit_kwargs)

            # Predict
            current_predict_kwargs = predict_kwargs.copy()
            if features is not None:
                X_val = features.iloc[forecast_start:forecast_end]
                sig_pred = inspect.signature(model.predict)
                if "test_features" in sig_pred.parameters:
                    current_predict_kwargs["test_features"] = X_val.values
                elif "features" in sig_pred.parameters:
                    current_predict_kwargs["features"] = X_val.values

            forecast = model.predict(steps=current_horizon, **current_predict_kwargs)
            forecast = np.asarray(forecast).flatten()

            # Align predictions into predictions_df
            for j in range(len(forecast)):
                h = j + 1
                target_dt = y.index[train_end + j]
                predictions_df.loc[target_dt, f"pred_h{h}"] = forecast[j]

            # Compute step metrics (if we have matching forecast values)
            step_metrics = {}
            if len(forecast) > 0:
                try:
                    step_metrics = compute_all_metrics(y_val.values, forecast, warn_on_nan=False)
                except Exception as e:
                    logger.warning(f"Failed to compute metrics at step {step_idx}: {e}")

            step_results.append({
                "step": step_idx,
                "train_start_date": train_dates[0],
                "train_end_date": train_dates[1],
                "forecast_start_date": val_dates[0],
                "forecast_end_date": val_dates[1],
                "actuals": y_val.values.tolist(),
                "predictions": forecast.tolist(),
                "metrics": step_metrics,
            })

            step_idx += 1

        # Compute overall metrics by horizon
        metrics_by_horizon = {}
        for h in range(1, self.horizon + 1):
            h_col = f"pred_h{h}"
            valid_df = predictions_df[["actual", h_col]].dropna()
            if len(valid_df) > 0:
                try:
                    metrics = compute_all_metrics(
                        valid_df["actual"].values,
                        valid_df[h_col].values,
                        warn_on_nan=False
                    )
                    metrics_by_horizon[h] = metrics
                    logger.info(f"Overall Metrics for Horizon {h} (N={len(valid_df)}): MAE={metrics['mae']:.2f}, RMSE={metrics['rmse']:.2f}")
                except Exception as e:
                    logger.error(f"Error computing overall metrics for horizon {h}: {e}")
                    metrics_by_horizon[h] = {}
            else:
                logger.warning(f"No valid predictions found for horizon {h}")
                metrics_by_horizon[h] = {}

        return {
            "metrics_by_horizon": metrics_by_horizon,
            "step_results": step_results,
            "predictions": predictions_df,
        }


def walkforward_validation(
    data: pd.Series,
    model_class: Type[BaseForecaster],
    model_params: Dict[str, Any],
    initial_train_size: Union[int, str, pd.Timestamp],
    X: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    horizon: int = 1,
    step_size: int = 1,
    window_type: Literal["expanding", "sliding"] = "expanding",
    refit_frequency: int = 1,
    min_train_size: int = 104,
    fit_kwargs: Optional[Dict[str, Any]] = None,
    predict_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Helper function to run walk-forward validation on a time series model.

    Args:
        data: Target time series (pandas Series with DatetimeIndex).
        model_class: Class of the model to evaluate (subclass of BaseForecaster).
        model_params: Dictionary of parameters to initialize the model.
        initial_train_size: Number of steps (int) or end date (str/Timestamp) of first training set.
        X: Feature matrix of the same length as data (optional, for ML models).
        horizon: Forecast horizon (number of steps ahead) at each origin.
        step_size: Number of steps to advance the origin at each iteration.
        window_type: 'expanding' (use all history) or 'sliding' (use a fixed-size window of history).
        refit_frequency: Frequency of model retraining. If 1, refits at every step.
        min_train_size: Minimum required size for training set (default: 104 weeks / 2 years).
        fit_kwargs: Keyword arguments passed to the model's fit method.
        predict_kwargs: Keyword arguments passed to the model's predict method.

    Returns:
        Dictionary with overall metrics, step-by-step details, and predictions dataframe.
    """
    validator = WalkForwardValidator(
        model_class=model_class,
        model_params=model_params,
        initial_train_size=initial_train_size,
        horizon=horizon,
        step_size=step_size,
        window_type=window_type,
        refit_frequency=refit_frequency,
        min_train_size=min_train_size,
    )
    return validator.run(
        y=data,
        X=X,
        fit_kwargs=fit_kwargs,
        predict_kwargs=predict_kwargs,
    )
