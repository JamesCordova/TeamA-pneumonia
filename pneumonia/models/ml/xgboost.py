"""
XGBoost forecasting model for pneumonia case prediction.

Follows the same recursive multi-step approach as RandomForestModel:
  fit()     — builds supervised (X, y) from training series, trains XGBRegressor.
  predict() — seeds a history buffer from real observations, then iterates one
              step at a time, feeding each prediction back as a lag for the next.

Feature set (from pneumonia.features.build):
  lags, rolling mean/std, week_of_year, sin_week, cos_week, month, quarter, trend.
  sin/cos week encode the circular seasonal pattern continuously — this prevents
  the model from treating week 52 and week 1 as distant when they are adjacent.
"""

from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from pneumonia.features.build import build_features, build_step_features
from pneumonia.models.base import BaseForecaster
from pneumonia.models.ml.config import (
    DEPARTMENTAL_CONFIGS,
    FEATURE_ENGINEERING_CONFIG,
    XGBOOST_DEFAULT_PARAMS,
)
from pneumonia.utils import setup_logger

logger = setup_logger(__name__)


class XGBoostModel(BaseForecaster):
    """
    XGBoost gradient boosting forecasting model.

    Parameters
    ----------
    department   : str
    age_group    : str  ('under5' or '60plus')
    xgb_params   : dict, optional
        Hyperparameter overrides applied on top of XGBOOST_DEFAULT_PARAMS
        and any department-specific config.
    lags         : list of int, optional
    windows      : list of int, optional
    """

    def __init__(
        self,
        department: str,
        age_group: str,
        xgb_params: Optional[Dict] = None,
        lags: Optional[List[int]] = None,
        windows: Optional[List[int]] = None,
    ):
        super().__init__(name="XGBoost", department=department, age_group=age_group)

        self.lags    = lags    or FEATURE_ENGINEERING_CONFIG["lag_periods"]
        self.windows = windows or FEATURE_ENGINEERING_CONFIG["rolling_windows"]

        # Merge: defaults → department config → caller overrides
        params = {**XGBOOST_DEFAULT_PARAMS}
        params.update(
            DEPARTMENTAL_CONFIGS.get(self.department, {}).get("xgboost_params", {})
        )
        if xgb_params:
            params.update(xgb_params)
        self._params = params

        self._xgb: Optional[XGBRegressor] = None
        self.feature_names_: Optional[List[str]] = None
        self._min_history: int = 0
        self._fit_size: int = 0

    def get_params(self) -> dict:
        return {
            "xgb_params": dict(self._params),
            "lags":       self.lags,
            "windows":    self.windows,
        }

    # ------------------------------------------------------------------
    # BaseForecaster interface
    # ------------------------------------------------------------------

    def fit(self, train_data: pd.Series, **kwargs) -> None:
        """
        Build supervised features and train XGBRegressor.

        Args:
            train_data: Weekly time series with DatetimeIndex.
            **kwargs:   Passed to XGBRegressor.fit() (e.g. eval_set,
                        early_stopping_rounds, verbose).
        """
        if not isinstance(train_data, pd.Series):
            raise TypeError(f"Expected pd.Series, got {type(train_data)}")

        max_lag = max(self.lags)
        if len(train_data) <= max_lag:
            raise ValueError(
                f"Training data ({len(train_data)} obs) must be longer "
                f"than max lag ({max_lag})"
            )

        logger.info(
            f"XGBoost fit — {self.department}/{self.age_group}, "
            f"{len(train_data)} observations"
        )

        X, y = build_features(train_data, lags=self.lags, windows=self.windows)

        self._xgb = XGBRegressor(**self._params)
        self._xgb.fit(X.values, y.values, **kwargs)

        self.feature_names_ = list(X.columns)
        self._min_history   = max(max_lag, max(self.windows))
        self._fit_size      = len(train_data)
        self.is_fitted      = True
        self.fitted_date    = datetime.now().isoformat()

        importances = dict(zip(self.feature_names_, self._xgb.feature_importances_))
        top3 = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)[:3]

        self.metadata.update({
            "n_estimators":  self._params["n_estimators"],
            "max_depth":     self._params["max_depth"],
            "learning_rate": self._params["learning_rate"],
            "train_size":    len(train_data),
            "n_features":    len(self.feature_names_),
            "fit_method":    "recursive_supervised",
            "lags":          self.lags,
            "windows":       self.windows,
            "top3_features": top3,
        })

        logger.info(
            f"XGBoost fitted — {self._params['n_estimators']} trees, "
            f"{len(self.feature_names_)} features. "
            f"Top feature: {top3[0][0]} ({top3[0][1]:.3f})"
        )

    def predict(self, data: pd.Series, steps: int = 52) -> np.ndarray:
        """
        Recursive multi-step forecast anchored to the end of data.

        Each step predicts the next value from the current history buffer,
        appends it, and repeats — identical strategy to RandomForestModel.

        Args:
            data:  Real observations available at prediction time.
            steps: Weeks to forecast ahead (default: 52).

        Returns:
            Array of shape (steps,) with forecast values (clipped to >= 0).
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predicting")

        if len(data) < self._min_history:
            logger.warning(
                f"data has {len(data)} observations; need {self._min_history} "
                f"for full feature coverage. Edge features may be approximate."
            )

        logger.info(
            f"XGBoost recursive predict — {steps} steps from {len(data)} observations"
        )

        buffer     = list(data.values[-self._min_history:].astype(float))
        last_date  = data.index[-1]
        step_delta = (
            data.index[-1] - data.index[-2] if len(data) > 1
            else pd.Timedelta(weeks=1)
        )
        trend_base = len(data)

        predictions: List[float] = []

        for step in range(steps):
            target_date = last_date + step_delta * (step + 1)
            trend_idx   = trend_base + step

            x = build_step_features(
                history=np.array(buffer),
                target_date=target_date,
                trend_idx=trend_idx,
                feature_names=self.feature_names_,
                lags=self.lags,
                windows=self.windows,
            )

            y_hat = max(0.0, float(self._xgb.predict(x.reshape(1, -1))[0]))
            predictions.append(y_hat)
            buffer.append(y_hat)

        return np.array(predictions)

    # ------------------------------------------------------------------
    # Extra methods
    # ------------------------------------------------------------------

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Normalized feature importance scores (gain-based), sorted descending.

        Returns:
            Dict mapping feature_name -> importance (sums to 1.0).
        """
        if not self.is_fitted or self._xgb is None:
            raise ValueError("Model must be fitted before getting importances")

        raw   = self._xgb.feature_importances_
        total = raw.sum()
        normalized = (raw / total).tolist() if total > 0 else raw.tolist()
        result = dict(zip(self.feature_names_, normalized))
        return dict(sorted(result.items(), key=lambda kv: kv[1], reverse=True))
