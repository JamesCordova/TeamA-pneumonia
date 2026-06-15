"""
RandomForest forecasting model for pneumonia case prediction.

Approach — recursive multi-step supervised learning:
  1. fit():    converts the training series into (X, y) pairs using lag /
               rolling / calendar features, then trains a RandomForestRegressor
               to predict y[t] from X[t].
  2. predict(): anchors to the last real observations in `data`, then iterates
               one step at a time — each predicted value is appended to the
               history buffer and used as a lag for subsequent steps.
"""

from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from pneumonia.features.build import build_features, build_step_features
from pneumonia.models.base import BaseForecaster
from pneumonia.models.ml.config import (
    DEPARTMENTAL_CONFIGS,
    FEATURE_ENGINEERING_CONFIG,
    RANDOM_FOREST_DEFAULT_PARAMS,
)
from pneumonia.utils import setup_logger

logger = setup_logger(__name__)


class RandomForestModel(BaseForecaster):
    """
    RandomForest forecasting model.

    Parameters
    ----------
    department : str
    age_group  : str  ('under5' or '60plus')
    rf_params  : dict, optional
        Hyperparameter overrides applied on top of RANDOM_FOREST_DEFAULT_PARAMS
        and any department-specific config.
    lags : list of int, optional
        Lag periods to use as features (default from FEATURE_ENGINEERING_CONFIG).
    windows : list of int, optional
        Rolling window sizes for mean/std features (default from config).
    """

    def __init__(
        self,
        department: str,
        age_group: str,
        rf_params: Optional[Dict] = None,
        lags: Optional[List[int]] = None,
        windows: Optional[List[int]] = None,
    ):
        super().__init__(name="RandomForest", department=department, age_group=age_group)

        self.lags    = lags    or FEATURE_ENGINEERING_CONFIG['lag_periods']
        self.windows = windows or FEATURE_ENGINEERING_CONFIG['rolling_windows']

        # Merge: defaults → department config → caller overrides
        params = {**RANDOM_FOREST_DEFAULT_PARAMS}
        params.update(DEPARTMENTAL_CONFIGS.get(self.department, {}).get('random_forest_params', {}))
        if rf_params:
            params.update(rf_params)
        self._params = params

        self._rf: Optional[RandomForestRegressor] = None
        self.feature_names_: Optional[List[str]] = None
        self._min_history: int = 0
        self._fit_size: int    = 0

    # ------------------------------------------------------------------
    # BaseForecaster interface
    # ------------------------------------------------------------------

    def fit(self, train_data: pd.Series, **kwargs) -> None:
        """
        Build supervised features from train_data and train RandomForest.

        Args:
            train_data: Weekly time series with DatetimeIndex.
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
            f"RandomForest fit — {self.department}/{self.age_group}, "
            f"{len(train_data)} observations"
        )

        X, y = build_features(train_data, lags=self.lags, windows=self.windows)

        self._rf = RandomForestRegressor(**self._params)
        self._rf.fit(X.values, y.values)

        self.feature_names_ = list(X.columns)
        self._min_history   = max(max_lag, max(self.windows))
        self._fit_size      = len(train_data)
        self.is_fitted      = True
        self.fitted_date    = datetime.now().isoformat()

        importances = dict(zip(self.feature_names_, self._rf.feature_importances_))
        top3 = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)[:3]

        self.metadata.update({
            'n_estimators':  self._params['n_estimators'],
            'max_depth':     self._params['max_depth'],
            'train_size':    len(train_data),
            'n_features':    len(self.feature_names_),
            'fit_method':    'recursive_supervised',
            'lags':          self.lags,
            'windows':       self.windows,
            'top3_features': top3,
        })

        logger.info(
            f"RandomForest fitted — {self._params['n_estimators']} trees, "
            f"{len(self.feature_names_)} features. "
            f"Top feature: {top3[0][0]} ({top3[0][1]:.3f})"
        )

    def predict(self, data: pd.Series, steps: int = 52) -> np.ndarray:
        """
        Recursive multi-step forecast anchored to the end of data.

        Each step predicts the next value from the current history buffer,
        appends it, and repeats.

        Args:
            data:  Real observations available at prediction time.
            steps: Number of weeks to forecast ahead (default: 52).

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
            f"RandomForest recursive predict — {steps} steps "
            f"from {len(data)} observations"
        )

        # Seed buffer with the minimum history required for features
        buffer = list(data.values[-self._min_history:].astype(float))
        last_date = data.index[-1]

        # Infer step delta from the last two index entries
        step_delta = (data.index[-1] - data.index[-2]) if len(data) > 1 else pd.Timedelta(weeks=1)
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

            y_hat = max(0.0, float(self._rf.predict(x.reshape(1, -1))[0]))
            predictions.append(y_hat)
            buffer.append(y_hat)

        return np.array(predictions)

    # ------------------------------------------------------------------
    # Extra methods
    # ------------------------------------------------------------------

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Normalized feature importance scores, sorted descending.

        Returns:
            Dict mapping feature_name -> importance (sums to 1.0).
        """
        if not self.is_fitted or self._rf is None:
            raise ValueError("Model must be fitted before getting importances")

        raw   = self._rf.feature_importances_
        total = raw.sum()
        normalized = (raw / total).tolist() if total > 0 else raw.tolist()
        result = dict(zip(self.feature_names_, normalized))
        return dict(sorted(result.items(), key=lambda kv: kv[1], reverse=True))
