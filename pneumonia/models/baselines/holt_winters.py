"""
Holt-Winters (Triple Exponential Smoothing) forecaster.

Uses additive trend with damping + additive seasonality, which is robust for
weekly epidemiological data with a strong annual cycle.

predict() re-fits the smoothing parameters on the current data window so the
seasonal state is always up to date — this is correct for walk-forward use.
"""

from datetime import datetime

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from pneumonia.models.base import BaseForecaster
from pneumonia.utils import setup_logger

logger = setup_logger(__name__)


class HoltWintersForecaster(BaseForecaster):
    """
    Triple Exponential Smoothing (Holt-Winters) forecaster.

    Parameters
    ----------
    department       : str
    age_group        : str  ('under5' or '60plus')
    seasonal_periods : int  (default 52 — weekly annual cycle)
    trend            : str  ('add' or 'mul', default 'add')
    damped_trend     : bool (default True — prevents drift on long horizons)
    seasonal         : str  ('add' or 'mul', default 'add')
    """

    def __init__(
        self,
        department: str,
        age_group: str,
        seasonal_periods: int = 52,
        trend: str = "add",
        damped_trend: bool = True,
        seasonal: str = "add",
    ):
        super().__init__("HoltWinters", department, age_group)
        self.seasonal_periods = seasonal_periods
        self.trend = trend
        self.damped_trend = damped_trend
        self.seasonal = seasonal
        self._fitted = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(self, train_data: pd.Series, **kwargs) -> None:
        if len(train_data) < self.seasonal_periods * 2:
            raise ValueError(
                f"HoltWinters needs at least {self.seasonal_periods * 2} observations "
                f"(2 full seasons), got {len(train_data)}."
            )
        try:
            self._fitted = self._build_model(train_data.values.astype(float))
            self.is_fitted = True
            self.fitted_date = datetime.now().isoformat()
            self.metadata.update({
                "seasonal_periods": self.seasonal_periods,
                "trend":            self.trend,
                "damped_trend":     self.damped_trend,
                "seasonal":         self.seasonal,
                "n_obs":            len(train_data),
                "aic":              float(self._fitted.aic),
                "fit_method":       "optimized",
            })
            logger.info(
                f"HoltWinters fitted — {len(train_data)} obs, AIC={self._fitted.aic:.2f}"
            )
        except Exception as exc:
            logger.error(f"HoltWinters fit failed: {exc}")
            raise

    def predict(self, data: pd.Series, steps: int = 52) -> np.ndarray:
        """
        Re-fit on `data` and forecast `steps` ahead.

        Re-fitting is necessary because HoltWinters has no apply() mechanism
        equivalent to SARIMAX; the seasonal state must be re-estimated from
        the current window.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predicting")

        try:
            fitted = self._build_model(data.values.astype(float))
            preds = np.array(fitted.forecast(steps))
            return np.maximum(0.0, preds)
        except Exception as exc:
            logger.warning(f"HoltWinters predict failed ({exc}) — falling back to SeasonalNaive")
            return self._seasonal_naive_fallback(data.values, steps)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_model(self, values: np.ndarray):
        return ExponentialSmoothing(
            values,
            trend=self.trend,
            damped_trend=self.damped_trend,
            seasonal=self.seasonal,
            seasonal_periods=self.seasonal_periods,
            initialization_method="estimated",
        ).fit(optimized=True)

    def _seasonal_naive_fallback(self, values: np.ndarray, steps: int) -> np.ndarray:
        season = values[-self.seasonal_periods:].astype(float)
        n_full = (steps + len(season) - 1) // len(season)
        return np.maximum(0.0, np.tile(season, n_full)[:steps])
