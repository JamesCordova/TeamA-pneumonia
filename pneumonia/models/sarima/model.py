"""
SARIMA / Fourier-SARIMAX Model for forecasting pneumonia cases.

Two seasonality modes (controlled by use_fourier):

  use_fourier=True  (default, recommended for s=52):
    SARIMAX(p,d,q)x(0,0,0,0) with K Fourier pairs as exogenous regressors.
    sin(2πkt/52) and cos(2πkt/52) for k=1..K.
    Avoids the 52-lag SAR/SMA estimation problem and the D=1 collapse toward
    Seasonal Naive.

  use_fourier=False:
    Classic SARIMA(p,d,q)x(P,D,Q,52). Useful as a comparison baseline but
    tends to underperform on weekly data with s=52.
"""

from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from pneumonia.models.base import BaseForecaster
from pneumonia.models.sarima.config import (
    DEFAULT_SARIMA_ORDER,
    DEFAULT_SARIMA_SEASONAL_ORDER,
    DEPARTMENTAL_CONFIGS,
    N_FOURIER_TERMS,
    USE_FOURIER_SEASONALITY,
)
from pneumonia.utils import setup_logger

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Fourier term utilities
# ---------------------------------------------------------------------------

def _fourier_terms(index: pd.DatetimeIndex, n_terms: int, period: float = 52.1775) -> np.ndarray:
    """
    Build a (len(index), 2*n_terms) matrix of sin/cos Fourier regressors.

    Each column k gives sin(2πkt/period) or cos(2πkt/period) where t is the
    fractional week number within the year inferred from the ISO week number.

    Args:
        index:   DatetimeIndex of the time series.
        n_terms: Number of sin/cos pairs (K).
        period:  Seasonal period in weeks (52.1775 = average weeks per year).

    Returns:
        NumPy array of shape (len(index), 2*n_terms).
    """
    # Use cumulative week count so terms are continuous across year boundaries
    t = np.arange(len(index), dtype=float)
    cols = []
    for k in range(1, n_terms + 1):
        cols.append(np.sin(2 * np.pi * k * t / period))
        cols.append(np.cos(2 * np.pi * k * t / period))
    return np.column_stack(cols)


def _fourier_df(index: pd.DatetimeIndex, n_terms: int, period: float = 52.1775) -> pd.DataFrame:
    """Return Fourier terms as a DataFrame (column names: sin_k, cos_k)."""
    arr = _fourier_terms(index, n_terms, period)
    cols = [f"{fn}_{k}" for k in range(1, n_terms + 1) for fn in ("sin", "cos")]
    return pd.DataFrame(arr, index=index, columns=cols)


def _future_fourier(last_index: pd.DatetimeIndex, steps: int, n_terms: int,
                    period: float = 52.1775) -> np.ndarray:
    """
    Build Fourier terms for `steps` future periods following last_index.

    The cumulative t counter continues from the last training index so
    sin/cos phases are aligned with the fitted exogenous regressors.
    """
    n_train = len(last_index)
    t = np.arange(n_train, n_train + steps, dtype=float)
    cols = []
    for k in range(1, n_terms + 1):
        cols.append(np.sin(2 * np.pi * k * t / period))
        cols.append(np.cos(2 * np.pi * k * t / period))
    return np.column_stack(cols)


# ---------------------------------------------------------------------------
# Model class
# ---------------------------------------------------------------------------

class SARIMAModel(BaseForecaster):
    """
    Seasonal ARIMA forecasting model.

    Parameters
    ----------
    department : str
    age_group  : str  ('under5' or '60plus')
    order      : tuple (p, d, q)  — uses config default if None
    seasonal_order : tuple (P, D, Q, s) — ignored when use_fourier=True
    use_fourier : bool — use Fourier exog instead of SAR/SMA (default True)
    n_fourier_terms : int — number of sin/cos pairs (default from config)
    """

    def __init__(
        self,
        department: str,
        age_group: str,
        order: Optional[Tuple[int, int, int]] = None,
        seasonal_order: Optional[Tuple[int, int, int, int]] = None,
        use_fourier: Optional[bool] = None,
        n_fourier_terms: Optional[int] = None,
    ):
        super().__init__(name="SARIMA", department=department, age_group=age_group)

        dept_cfg = DEPARTMENTAL_CONFIGS.get(self.department, {})

        self.order = order or dept_cfg.get("order", DEFAULT_SARIMA_ORDER)
        self.seasonal_order = seasonal_order or dept_cfg.get(
            "seasonal_order", DEFAULT_SARIMA_SEASONAL_ORDER
        )
        self.use_fourier = (
            use_fourier if use_fourier is not None
            else dept_cfg.get("use_fourier", USE_FOURIER_SEASONALITY)
        )
        self.n_fourier_terms = (
            n_fourier_terms if n_fourier_terms is not None
            else dept_cfg.get("n_fourier_terms", N_FOURIER_TERMS)
        )

        self._sm_results = None   # always a raw SARIMAXResults (statsmodels)
        self._train_len: int = 0  # length of data the model was fitted on

        # results kept for diagnostics (may be pmdarima or statsmodels)
        self.results = None

        logger.info(
            f"Initialized SARIMA for {self.department} ({self.age_group}) — "
            f"{'Fourier(K=%d)' % self.n_fourier_terms if self.use_fourier else 'Seasonal ARIMA'}"
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(
        self,
        train_data: pd.Series,
        enforce_stationarity: bool = False,
        enforce_invertibility: bool = False,
        use_auto_arima: bool = False,
        **kwargs,
    ) -> None:
        """
        Fit the SARIMA/Fourier-SARIMAX model.

        Args:
            train_data:            Weekly Series with DatetimeIndex.
            enforce_stationarity:  Passed to SARIMAX.
            enforce_invertibility: Passed to SARIMAX.
            use_auto_arima:        If True, search for the best (p,d,q) with
                                   pmdarima.auto_arima.
        """
        if not isinstance(train_data, pd.Series):
            raise TypeError(f"Expected pd.Series, got {type(train_data)}")
        if len(train_data) < 52:
            logger.warning(f"Short training set ({len(train_data)} < 52 weeks).")

        if use_auto_arima:
            try:
                self._fit_auto_arima(train_data, enforce_stationarity, enforce_invertibility)
            except ImportError:
                logger.warning("pmdarima not installed — falling back to manual order.")
                self._fit_manual(train_data, enforce_stationarity, enforce_invertibility)
        else:
            self._fit_manual(train_data, enforce_stationarity, enforce_invertibility)

    def predict(self, data: pd.Series, steps: int = 52) -> np.ndarray:
        """
        Generate forecasts anchored to the last observations in data.

        Applies the fitted parameters (no refit) to `data` so the Kalman
        filter state is updated to the end of `data` before forecasting.

        Args:
            data:  Real observations available at prediction time.
            steps: Weeks to forecast ahead.

        Returns:
            Array of shape (steps,) with point forecasts (clipped to >= 0).
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        logger.info(f"SARIMA predict — {steps} steps from {len(data)} observations")

        try:
            if self.use_fourier:
                exog_hist = _fourier_terms(data.index, self.n_fourier_terms)
                exog_fut  = _future_fourier(data.index, steps, self.n_fourier_terms)

                updated = self._sm_results.apply(data, exog=exog_hist, refit=False)
                preds   = updated.get_forecast(steps=steps, exog=exog_fut).predicted_mean.values
            else:
                updated = self._sm_results.apply(data, refit=False)
                preds   = updated.get_forecast(steps=steps).predicted_mean.values

            return np.maximum(0.0, np.array(preds))

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise

    def get_forecast_interval(
        self,
        data: pd.Series,
        steps: int = 52,
        alpha: float = 0.05,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Point forecasts plus confidence interval anchored to data.

        Returns:
            (predictions, lower_bound, upper_bound)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        logger.info(f"SARIMA CI forecast — {steps} steps, {(1-alpha)*100:.0f}% CI")

        try:
            if self.use_fourier:
                exog_hist = _fourier_terms(data.index, self.n_fourier_terms)
                exog_fut  = _future_fourier(data.index, steps, self.n_fourier_terms)
                updated   = self._sm_results.apply(data, exog=exog_hist, refit=False)
                forecast  = updated.get_forecast(steps=steps, exog=exog_fut)
            else:
                updated  = self._sm_results.apply(data, refit=False)
                forecast = updated.get_forecast(steps=steps)

            preds = np.maximum(0.0, forecast.predicted_mean.values)
            ci    = forecast.conf_int(alpha=alpha)
            lower = np.maximum(0.0, ci.iloc[:, 0].values)
            upper = np.maximum(0.0, ci.iloc[:, 1].values)

            return preds, lower, upper

        except Exception as e:
            logger.error(f"Forecast interval failed: {e}")
            raise

    def get_residuals(self) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted")
        resid = self._sm_results.resid
        return np.array(resid() if callable(resid) else resid)

    def diagnostics(self) -> Dict[str, Any]:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before diagnostics")

        try:
            from statsmodels.tsa.stattools import acf, adfuller
        except ImportError:
            return {}

        residuals = self.get_residuals()

        try:
            aic_val = float(self._sm_results.aic)
            bic_val = float(self._sm_results.bic)
        except Exception:
            aic_val = bic_val = np.nan

        diag: Dict[str, Any] = {
            "aic": aic_val,
            "bic": bic_val,
            "rmse": float(np.sqrt(np.mean(residuals ** 2))),
            "residuals": {
                "mean": float(residuals.mean()),
                "std": float(residuals.std()),
                "min": float(residuals.min()),
                "max": float(residuals.max()),
            },
        }

        try:
            adf = adfuller(residuals, autolag="AIC")
            diag["adf_test"] = {
                "statistic": float(adf[0]),
                "p_value": float(adf[1]),
                "stationary": adf[1] < 0.05,
            }
        except Exception as exc:
            logger.warning(f"ADF test failed: {exc}")

        try:
            acf_vals = acf(residuals, nlags=min(40, len(residuals) // 2))
            diag["acf"] = {
                "lag_1": float(acf_vals[1]),
                "significant_lags": int(
                    (np.abs(acf_vals) > 1.96 / np.sqrt(len(residuals))).sum()
                ),
            }
        except Exception as exc:
            logger.warning(f"ACF calculation failed: {exc}")

        return diag

    # ------------------------------------------------------------------
    # Private fitting helpers
    # ------------------------------------------------------------------

    def _fit_manual(
        self,
        train_data: pd.Series,
        enforce_stationarity: bool,
        enforce_invertibility: bool,
    ) -> None:
        seasonal_order = (0, 0, 0, 0) if self.use_fourier else self.seasonal_order
        exog = _fourier_terms(train_data.index, self.n_fourier_terms) if self.use_fourier else None

        logger.info(
            f"Fitting SARIMA{self.order}×{seasonal_order} "
            f"{'+ Fourier(K=%d) ' % self.n_fourier_terms if self.use_fourier else ''}"
            f"on {len(train_data)} observations (manual)"
        )

        model = SARIMAX(
            train_data,
            exog=exog,
            order=self.order,
            seasonal_order=seasonal_order,
            enforce_stationarity=enforce_stationarity,
            enforce_invertibility=enforce_invertibility,
        )
        self._sm_results = model.fit(disp=False, maxiter=500)
        self.results = self._sm_results
        self._finalize_fit(train_data, float(self._sm_results.aic), float(self._sm_results.bic), "manual")

    def _fit_auto_arima(
        self,
        train_data: pd.Series,
        enforce_stationarity: bool,
        enforce_invertibility: bool,
    ) -> None:
        try:
            from pmdarima import auto_arima
        except ImportError:
            raise ImportError("pmdarima required for auto_arima. pip install pmdarima")

        from pneumonia.config import SARIMA_MAX_ITERATIONS, SARIMA_STEPWISE, SARIMA_TRACE

        exog = _fourier_terms(train_data.index, self.n_fourier_terms) if self.use_fourier else None

        logger.info(
            f"auto_arima search — {len(train_data)} obs, "
            f"{'Fourier(K=%d)' % self.n_fourier_terms if self.use_fourier else 'seasonal ARIMA(s=52)'}, "
            f"stepwise={SARIMA_STEPWISE}"
        )

        if self.use_fourier:
            # With Fourier regressors, seasonality is captured by exog — disable seasonal ARIMA
            auto_model = auto_arima(
                train_data,
                X=exog,
                seasonal=False,
                d=None,               # auto-detect differencing order via KPSS
                max_p=3,
                max_q=3,
                max_d=2,
                information_criterion="aicc",
                stepwise=SARIMA_STEPWISE,
                trace=SARIMA_TRACE,
                max_iter=SARIMA_MAX_ITERATIONS,
                error_action="ignore",
                suppress_warnings=True,
                test="kpss",
            )
        else:
            auto_model = auto_arima(
                train_data,
                seasonal=True,
                m=52,
                d=None,
                D=None,
                max_p=2, max_q=2, max_d=2,
                max_P=1, max_Q=1, max_D=1,
                information_criterion="aicc",
                stepwise=SARIMA_STEPWISE,
                trace=SARIMA_TRACE,
                max_iter=SARIMA_MAX_ITERATIONS,
                error_action="ignore",
                suppress_warnings=True,
                test="kpss",
            )

        self.order = auto_model.order
        self.seasonal_order = auto_model.seasonal_order
        self.results = auto_model

        # Store the underlying statsmodels result for predict/apply
        self._sm_results = auto_model.arima_res_
        self._finalize_fit(
            train_data,
            float(auto_model.aic()),
            float(auto_model.bic()),
            "auto_arima",
        )

    def _finalize_fit(
        self, train_data: pd.Series, aic: float, bic: float, fit_method: str
    ) -> None:
        self.is_fitted = True
        self.fitted_date = datetime.now().isoformat()
        self._train_len = len(train_data)

        seasonal_desc = (
            f"Fourier(K={self.n_fourier_terms})" if self.use_fourier
            else str(self.seasonal_order)
        )
        logger.info(
            f"SARIMA fit complete — order={self.order}, seasonal={seasonal_desc}, "
            f"AIC={aic:.2f}, BIC={bic:.2f}"
        )

        self.metadata.update({
            "order": self.order,
            "seasonal_order": self.seasonal_order if not self.use_fourier else None,
            "use_fourier": self.use_fourier,
            "n_fourier_terms": self.n_fourier_terms if self.use_fourier else None,
            "aic": aic,
            "bic": bic,
            "n_obs": len(train_data),
            "fit_method": fit_method,
        })

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        fitted = "fitted" if self.is_fitted else "unfitted"
        seasonal = (
            f"Fourier(K={self.n_fourier_terms})" if self.use_fourier
            else str(self.seasonal_order)
        )
        return (
            f"SARIMAModel(dept={self.department}, age={self.age_group}, "
            f"order={self.order}, seasonal={seasonal}, {fitted})"
        )
