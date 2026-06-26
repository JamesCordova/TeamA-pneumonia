"""
Prophet Forecasting Model for pneumonia case prediction.

Uses the prophet package from Meta (Facebook). Prophet models the series as a
combination of trend, yearly/weekly seasonality, and holidays.
In this case, since we work with weekly data:
- Yearly seasonality is enabled (critical for annual patterns).
- Weekly and daily seasonalities are disabled.
- Peru's official country holidays ('PE') are added automatically to capture
  effects of holiday weeks.
"""

from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from pneumonia.models.base import BaseForecaster
from pneumonia.models.prophet.config import (
    DEPARTMENTAL_CONFIGS,
    PROPHET_DEFAULT_PARAMS,
)
from pneumonia.utils import setup_logger

logger = setup_logger(__name__)


class ProphetModel(BaseForecaster):
    """
    Prophet forecasting model.

    Parameters
    ----------
    department               : str
    age_group                : str  ('under5' or '60plus')
    growth                   : str, optional
    yearly_seasonality       : bool or str, optional
    weekly_seasonality       : bool or str, optional
    daily_seasonality        : bool or str, optional
    seasonality_mode         : str, optional
    changepoint_prior_scale  : float, optional
    seasonality_prior_scale  : float, optional
    """

    def __init__(
        self,
        department: str,
        age_group: str,
        growth: Optional[str] = None,
        yearly_seasonality: Optional[Any] = None,
        weekly_seasonality: Optional[Any] = None,
        daily_seasonality: Optional[Any] = None,
        seasonality_mode: Optional[str] = None,
        changepoint_prior_scale: Optional[float] = None,
        seasonality_prior_scale: Optional[float] = None,
        interval_width: float = 0.95,
        **kwargs,
    ):
        super().__init__(name="Prophet", department=department, age_group=age_group)

        # Merge config parameters: defaults -> department config -> caller overrides
        params = {**PROPHET_DEFAULT_PARAMS}
        dept_cfg = DEPARTMENTAL_CONFIGS.get(self.department, {})
        params.update(dept_cfg)

        if growth is not None:
            params["growth"] = growth
        if yearly_seasonality is not None:
            params["yearly_seasonality"] = yearly_seasonality
        if weekly_seasonality is not None:
            params["weekly_seasonality"] = weekly_seasonality
        if daily_seasonality is not None:
            params["daily_seasonality"] = daily_seasonality
        if seasonality_mode is not None:
            params["seasonality_mode"] = seasonality_mode
        if changepoint_prior_scale is not None:
            params["changepoint_prior_scale"] = changepoint_prior_scale
        if seasonality_prior_scale is not None:
            params["seasonality_prior_scale"] = seasonality_prior_scale
        params["interval_width"] = interval_width

        params.update(kwargs)
        self._params = params
        self._model = None

        logger.info(
            f"Initialized Prophet Model for {self.department} ({self.age_group}) — "
            f"growth={params['growth']}, seasonality={params['seasonality_mode']}"
        )

    # ------------------------------------------------------------------
    # BaseForecaster interface
    # ------------------------------------------------------------------

    def fit(self, train_data: pd.Series, **kwargs) -> None:
        """
        Fit Prophet model on train_data.

        Args:
            train_data: Weekly time series with DatetimeIndex.
            **kwargs:   Additional parameters passed to Prophet.fit().
        """
        if not isinstance(train_data, pd.Series):
            raise TypeError(f"Expected pd.Series, got {type(train_data)}")

        try:
            from prophet import Prophet
        except ImportError:
            raise ImportError(
                "The 'prophet' package is not installed. "
                "Please run `pip install prophet` or check SETUP_GUIDE.md."
            )

        logger.info(
            f"Prophet fit — {self.department}/{self.age_group}, "
            f"{len(train_data)} observations"
        )

        # Format DataFrame as required by Prophet (columns 'ds' and 'y')
        df = pd.DataFrame({
            "ds": train_data.index,
            "y": train_data.values,
        })

        # Initialize Prophet model
        self._model = Prophet(
            growth=self._params["growth"],
            yearly_seasonality=self._params["yearly_seasonality"],
            weekly_seasonality=self._params["weekly_seasonality"],
            daily_seasonality=self._params["daily_seasonality"],
            seasonality_mode=self._params["seasonality_mode"],
            changepoint_prior_scale=self._params["changepoint_prior_scale"],
            seasonality_prior_scale=self._params["seasonality_prior_scale"],
            interval_width=self._params["interval_width"],
        )

        # Add official Peru holidays
        self._model.add_country_holidays(country_name="PE")

        # Fit model
        self._model.fit(df, **kwargs)

        self.is_fitted = True
        self.fitted_date = datetime.now().isoformat()

        # Update metadata
        self.metadata.update({
            "growth": self._params["growth"],
            "yearly_seasonality": self._params["yearly_seasonality"],
            "weekly_seasonality": self._params["weekly_seasonality"],
            "daily_seasonality": self._params["daily_seasonality"],
            "seasonality_mode": self._params["seasonality_mode"],
            "changepoint_prior_scale": self._params["changepoint_prior_scale"],
            "seasonality_prior_scale": self._params["seasonality_prior_scale"],
            "n_obs": len(train_data),
            "fit_method": "prophet_fit",
        })

        logger.info(f"Prophet fit complete for {self.department}/{self.age_group}")

    def predict(self, data: pd.Series, steps: int = 52) -> np.ndarray:
        """
        Predict steps ahead starting after the end of data.

        Args:
            data:  Real observations available at prediction time.
            steps: Weeks to forecast ahead.

        Returns:
            Array of shape (steps,) with point forecasts (clipped to >= 0).
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predicting")

        logger.info(
            f"Prophet predict — {steps} steps from {len(data)} observations"
        )

        try:
            last_date = data.index[-1]
            future_dates = pd.date_range(
                start=last_date,
                periods=steps + 1,
                freq=pd.infer_freq(data.index) or "7D",
            )[1:]

            future_df = pd.DataFrame({"ds": future_dates})
            forecast = self._model.predict(future_df)
            return np.maximum(0.0, forecast["yhat"].values)

        except Exception as e:
            logger.error(f"Prophet prediction failed: {e}")
            raise

    def get_forecast_interval(
        self,
        data: pd.Series,
        steps: int = 52,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Point forecasts plus confidence interval anchored to data.
        The interval width is determined by the interval_width parameter set at construction.

        Args:
            data:  Real observations available at prediction time.
            steps: Weeks to forecast ahead.

        Returns:
            (predictions, lower_bound, upper_bound)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predicting")

        logger.info(
            f"Prophet CI forecast — {steps} steps, "
            f"{self._params['interval_width']*100:.0f}% CI"
        )

        try:
            last_date = data.index[-1]
            future_dates = pd.date_range(
                start=last_date,
                periods=steps + 1,
                freq=pd.infer_freq(data.index) or "7D",
            )[1:]

            future_df = pd.DataFrame({"ds": future_dates})
            forecast = self._model.predict(future_df)

            preds = np.maximum(0.0, forecast["yhat"].values)
            lower = np.maximum(0.0, forecast["yhat_lower"].values)
            upper = np.maximum(0.0, forecast["yhat_upper"].values)

            return preds, lower, upper

        except Exception as e:
            logger.error(f"Prophet forecast interval failed: {e}")
            raise
