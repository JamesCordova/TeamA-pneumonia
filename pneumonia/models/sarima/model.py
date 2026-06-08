"""
SARIMA Model for forecasting pneumonia cases

Implements SARIMA (Seasonal ARIMA) for weekly forecasting of pneumonia cases.
"""

from typing import Optional, Tuple, Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime
import logging

from statsmodels.tsa.statespace.sarimax import SARIMAX

from pneumonia.models.base import BaseForecaster
from pneumonia.models.sarima.config import (
    DEFAULT_SARIMA_ORDER,
    DEFAULT_SARIMA_SEASONAL_ORDER,
    DEPARTMENTAL_CONFIGS,
)
from pneumonia.utils import setup_logger

logger = setup_logger(__name__)


class SARIMAModel(BaseForecaster):
    """
    Seasonal ARIMA (SARIMA) forecasting model for pneumonia cases.
    
    Inherits from BaseForecaster and implements SARIMA using statsmodels.
    """
    
    def __init__(
        self,
        department: str,
        age_group: str,
        order: Optional[Tuple[int, int, int]] = None,
        seasonal_order: Optional[Tuple[int, int, int, int]] = None,
    ):
        """
        Initialize SARIMA model.
        
        Args:
            department: Department name
            age_group: 'under5' or '60plus'
            order: (p, d, q) tuple. Uses default if None.
            seasonal_order: (P, D, Q, s) tuple. Uses default if None.
        """
        super().__init__(
            name="SARIMA",
            department=department,
            age_group=age_group,
        )
        
        # Get order from config or use defaults
        if order is None:
            order = DEPARTMENTAL_CONFIGS.get(
                self.department, {}
            ).get("order", DEFAULT_SARIMA_ORDER)
        
        if seasonal_order is None:
            seasonal_order = DEPARTMENTAL_CONFIGS.get(
                self.department, {}
            ).get("seasonal_order", DEFAULT_SARIMA_SEASONAL_ORDER)
        
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.results = None
        
        logger.info(
            f"Initialized SARIMA({order})×({seasonal_order}) "
            f"for {self.department} ({self.age_group})"
        )
    
    def fit(
        self,
        train_data: pd.Series,
        enforce_stationarity: bool = False,
        enforce_invertibility: bool = False,
        use_auto_arima: bool = False,
        **kwargs
    ) -> None:
        """
        Fit SARIMA model to training data.
        
        Args:
            train_data: Training time series (pandas Series with DatetimeIndex)
            enforce_stationarity: Whether to enforce stationarity constraints
            enforce_invertibility: Whether to enforce invertibility constraints
            use_auto_arima: If True, use pmdarima.auto_arima to find optimal order
            **kwargs: Additional arguments passed to SARIMAX
            
        Raises:
            ImportError: If use_auto_arima=True but pmdarima not installed
            TypeError: If train_data is not pd.Series
            ValueError: If training data is too short
        """
        if not isinstance(train_data, pd.Series):
            raise TypeError(f"Expected pd.Series, got {type(train_data)}")
        
        if len(train_data) < 52:
            logger.warning(f"Small training set: {len(train_data)} < 52 weeks. Consider more data.")
        
        # Auto ARIMA search if requested
        if use_auto_arima:
            try:
                self._fit_with_auto_arima(train_data, enforce_stationarity, enforce_invertibility, **kwargs)
            except ImportError:
                logger.warning("pmdarima not installed. Falling back to manual order parameters.")
                self._fit_with_manual_order(train_data, enforce_stationarity, enforce_invertibility, **kwargs)
        else:
            self._fit_with_manual_order(train_data, enforce_stationarity, enforce_invertibility, **kwargs)
    
    def _fit_with_manual_order(
        self,
        train_data: pd.Series,
        enforce_stationarity: bool,
        enforce_invertibility: bool,
        **kwargs
    ) -> None:
        """
        Fit SARIMA with manually specified order.
        
        Args:
            train_data: Training time series
            enforce_stationarity: Stationarity constraint
            enforce_invertibility: Invertibility constraint
            **kwargs: Additional SARIMAX arguments
        """
        logger.info(
            f"Fitting SARIMA{self.order}x{self.seasonal_order} "
            f"with {len(train_data)} observations (manual order)"
        )
        
        try:
            self.model = SARIMAX(
                train_data,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=enforce_stationarity,
                enforce_invertibility=enforce_invertibility,
                **kwargs
            )
            
            self.results = self.model.fit(disp=False, maxiter=400)
            
            self.is_fitted = True
            self.fitted_date = datetime.now().isoformat()
            
            logger.info(f"SARIMA fit completed. AIC: {self.results.aic:.2f}, BIC: {self.results.bic:.2f}")
            
            self.metadata.update({
                "order": self.order,
                "seasonal_order": self.seasonal_order,
                "aic": float(self.results.aic),
                "bic": float(self.results.bic),
                "n_obs": len(train_data),
                "fit_method": "manual",
            })
            
        except Exception as e:
            logger.error(f"SARIMA fitting failed: {str(e)}")
            raise
    
    def _fit_with_auto_arima(
        self,
        train_data: pd.Series,
        enforce_stationarity: bool,
        enforce_invertibility: bool,
        **kwargs
    ) -> None:
        """
        Fit SARIMA with auto_arima parameter search.
        
        Uses pmdarima.auto_arima to find optimal (p,d,q)×(P,D,Q,s).
        
        Args:
            train_data: Training time series
            enforce_stationarity: Stationarity constraint
            enforce_invertibility: Invertibility constraint
            **kwargs: Additional arguments for auto_arima
            
        Raises:
            ImportError: If pmdarima not installed
        """
        try:
            from pmdarima import auto_arima
        except ImportError:
            raise ImportError(
                "pmdarima is required for auto_arima. Install with: pip install pmdarima"
            )
        
        logger.info(f"Running auto_arima search with {len(train_data)} observations...")
        
        try:
            # Get config from main config.py if available
            from pneumonia.config import (
                SARIMA_MAX_ITERATIONS,
                SARIMA_STEPWISE,
                SARIMA_TRACE,
            )
            
            auto_model = auto_arima(
                train_data,
                seasonal=True,
                m=52,  # Weekly seasonality
                max_p=2,
                max_d=2,
                max_q=2,
                max_P=1,
                max_D=1,
                max_Q=1,
                stepwise=SARIMA_STEPWISE,
                trace=SARIMA_TRACE,
                max_iter=SARIMA_MAX_ITERATIONS,
                error_action="ignore",
                suppress_warnings=True,
                **kwargs
            )
            
            # Extract found order
            self.order = auto_model.order
            self.seasonal_order = auto_model.seasonal_order
            self.results = auto_model
            
            self.is_fitted = True
            self.fitted_date = datetime.now().isoformat()
            
            logger.info(
                f"auto_arima found: SARIMA{str(self.order)}x{str(self.seasonal_order)}. "
                f"AIC: {self.results.aic():.2f}, BIC: {self.results.bic():.2f}"
            )
            
            self.metadata.update({
                "order": self.order,
                "seasonal_order": self.seasonal_order,
                "aic": float(self.results.aic()),
                "bic": float(self.results.bic()),
                "n_obs": len(train_data),
                "fit_method": "auto_arima",
            })
            
        except Exception as e:
            logger.error(f"auto_arima fitting failed: {str(e)}")
            raise
    
    def predict(self, data: pd.Series, steps: int = 52) -> np.ndarray:
        """
        Generate forecasts anchored to the last real observations in data.

        The fitted SARIMA parameters are applied to data without refitting,
        so the state space is initialised at the end of data before forecasting.

        Args:
            data: Real observations available at prediction time.
            steps: Number of steps to forecast ahead (default: 52 weeks).

        Returns:
            Array of point forecasts of length steps.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        logger.info(f"Generating {steps}-step SARIMA forecast from {len(data)} observations")

        try:
            # Unwrap pmdarima to get the underlying statsmodels SARIMAXResults
            sm_results = self.results.arima_res_ if hasattr(self.results, 'arima_res_') else self.results

            updated = sm_results.apply(data, refit=False)
            predictions = updated.get_forecast(steps=steps).predicted_mean.values

            return np.array(predictions)

        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise
    
    def get_forecast_interval(
        self,
        data: pd.Series,
        steps: int = 52,
        alpha: float = 0.05,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate point forecasts and confidence intervals anchored to data.

        Args:
            data: Real observations available at prediction time.
            steps: Number of steps to forecast ahead (default: 52 weeks).
            alpha: Significance level (0.05 = 95% CI).

        Returns:
            Tuple of (predictions, lower_bound, upper_bound).
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        logger.info(f"Generating {steps}-step forecast with {(1-alpha)*100:.0f}% CI")

        try:
            sm_results = self.results.arima_res_ if hasattr(self.results, 'arima_res_') else self.results

            updated = sm_results.apply(data, refit=False)
            forecast = updated.get_forecast(steps=steps)
            predictions = forecast.predicted_mean.values
            ci = forecast.conf_int(alpha=alpha)
            lower = ci.iloc[:, 0].values
            upper = ci.iloc[:, 1].values

            return predictions, lower, upper

        except Exception as e:
            logger.error(f"Forecast interval generation failed: {str(e)}")
            raise
    
    def get_residuals(self) -> np.ndarray:
        """
        Get model residuals.
        
        Returns:
            Array of residuals
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted")
        
        # Handle both pmdarima (resid is method) and statsmodels (resid is attribute)
        resid = self.results.resid
        if callable(resid):
            return np.array(resid())
        else:
            return np.array(resid)
    
    def diagnostics(self) -> Dict[str, Any]:
        """
        Generate model diagnostics and quality metrics.
        
        Returns:
            Dictionary with diagnostic information:
            - Information criteria (AIC, BIC)
            - Residual statistics (mean, std, autocorrelation)
            - Stationarity test (ADF test on residuals)
            - Ljung-Box test for autocorrelation
            
        Raises:
            ValueError: If model not fitted
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before diagnostics")
        
        try:
            from statsmodels.tsa.stattools import adfuller, acf
            from scipy import stats
        except ImportError:
            logger.warning("Required modules for diagnostics not available")
            return {}
        
        # Get residuals, handling both pmdarima (method) and statsmodels (attribute)
        resid_raw = self.results.resid
        residuals = np.array(resid_raw()) if callable(resid_raw) else np.array(resid_raw)
        
        # Handle both SARIMAX (statsmodels) and ARIMA (pmdarima) results
        # pmdarima returns methods, statsmodels returns attributes
        try:
            aic_val = self.results.aic() if callable(self.results.aic) else self.results.aic
            bic_val = self.results.bic() if callable(self.results.bic) else self.results.bic
        except Exception:
            aic_val = np.nan
            bic_val = np.nan
        
        diagnostics_dict = {
            "aic": float(aic_val),
            "bic": float(bic_val),
            "rmse": float(np.sqrt(np.mean(residuals**2))),
            "residuals": {
                "mean": float(residuals.mean()),
                "std": float(residuals.std()),
                "min": float(residuals.min()),
                "max": float(residuals.max()),
            }
        }
        
        # Try to add SSR if available
        try:
            ssr_val = self.results.ssr() if callable(self.results.ssr) else self.results.ssr
            diagnostics_dict["ssr"] = float(ssr_val)
        except (AttributeError, TypeError):
            # Calculate SSR manually from residuals if not available
            diagnostics_dict["ssr"] = float(np.sum(residuals**2))
        
        # ADF test for stationarity
        try:
            adf_result = adfuller(residuals, autolag='AIC')
            diagnostics_dict["adf_test"] = {
                "statistic": float(adf_result[0]),
                "p_value": float(adf_result[1]),
                "stationary": adf_result[1] < 0.05,
            }
        except Exception as e:
            logger.warning(f"ADF test failed: {e}")
        
        # ACF (autocorrelation)
        try:
            acf_values = acf(residuals, nlags=min(40, len(residuals)//2))
            diagnostics_dict["acf"] = {
                "lag_1": float(acf_values[1]),
                "significant_lags": int((np.abs(acf_values) > 1.96/np.sqrt(len(residuals))).sum()),
            }
        except Exception as e:
            logger.warning(f"ACF calculation failed: {e}")
        
        logger.info(f"Diagnostics summary - AIC: {diagnostics_dict['aic']:.2f}, "
                   f"Residual std: {diagnostics_dict['residuals']['std']:.4f}")
        
        return diagnostics_dict
    
    def __repr__(self) -> str:
        """String representation."""
        fitted = "fitted" if self.is_fitted else "unfitted"
        return (
            f"SARIMAModel(dept={self.department}, age={self.age_group}, "
            f"order={self.order}, seasonal={self.seasonal_order}, {fitted})"
        )
