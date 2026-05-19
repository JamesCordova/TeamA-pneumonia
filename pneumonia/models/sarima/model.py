"""
SARIMA Model for forecasting pneumonia cases

Implements SARIMA (Seasonal ARIMA) for weekly forecasting of pneumonia cases.
"""

from typing import Optional, Tuple
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
        **kwargs
    ) -> None:
        """
        Fit SARIMA model to training data.
        
        Args:
            train_data: Training time series (pandas Series with DatetimeIndex)
            enforce_stationarity: Whether to enforce stationarity
            enforce_invertibility: Whether to enforce invertibility
            **kwargs: Additional arguments passed to SARIMAX
        """
        if not isinstance(train_data, pd.Series):
            raise TypeError(f"Expected pd.Series, got {type(train_data)}")
        
        if len(train_data) < 52:
            logger.warning(f"Small training set: {len(train_data)} < 52 weeks")
        
        logger.info(
            f"Fitting SARIMA{self.order}x{self.seasonal_order} "
            f"with {len(train_data)} observations"
        )
        
        try:
            # Create and fit SARIMAX model
            self.model = SARIMAX(
                train_data,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=enforce_stationarity,
                enforce_invertibility=enforce_invertibility,
                **kwargs
            )
            
            self.results = self.model.fit(
                disp=False,
                maxiter=400,
            )
            
            self.is_fitted = True
            self.fitted_date = datetime.now().isoformat()
            
            # Log summary statistics
            logger.info(f"SARIMA fit completed. AIC: {self.results.aic:.2f}")
            
            self.metadata.update({
                "order": self.order,
                "seasonal_order": self.seasonal_order,
                "aic": self.results.aic,
                "bic": self.results.bic,
                "n_obs": len(train_data),
            })
            
        except Exception as e:
            logger.error(f"SARIMA fitting failed: {str(e)}")
            raise
    
    def predict(self, steps: int = 52) -> np.ndarray:
        """
        Generate forecasts for specified steps ahead.
        
        Args:
            steps: Number of forecast steps (default: 52 weeks = 1 year)
            
        Returns:
            Array of point forecasts
            
        Raises:
            ValueError: If model not fitted
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        logger.info(f"Generating {steps}-step forecast")
        
        try:
            forecast = self.results.get_forecast(steps=steps)
            predictions = forecast.predicted_mean.values
            
            logger.info(f"Forecast completed: {predictions.shape}")
            return predictions
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise
    
    def get_forecast_interval(
        self,
        steps: int = 52,
        alpha: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate point forecasts and confidence intervals.
        
        Args:
            steps: Number of forecast steps
            alpha: Significance level (0.05 = 95% CI)
            
        Returns:
            Tuple of (predictions, lower_bound, upper_bound)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        logger.info(f"Generating {steps}-step forecast with {(1-alpha)*100:.0f}% CI")
        
        try:
            forecast = self.results.get_forecast(steps=steps)
            predictions = forecast.predicted_mean.values
            ci = forecast.conf_int(alpha=alpha)
            lower = ci.iloc[:, 0].values
            upper = ci.iloc[:, 1].values
            
            return predictions, lower, upper
            
        except Exception as e:
            logger.error(f"Forecast interval generation failed: {str(e)}")
            raise
    
    def get_residuals(self) -> pd.Series:
        """
        Get model residuals.
        
        Returns:
            Series of residuals
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted")
        
        return self.results.resid
    
    def diagnostics(self) -> None:
        """
        Print diagnostic summary and plot diagnostics.
        
        Should be called after fitting to assess model quality.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before diagnostics")
        
        logger.info("Model Diagnostics:")
        logger.info(f"AIC: {self.results.aic:.2f}")
        logger.info(f"BIC: {self.results.bic:.2f}")
        logger.info(f"SSR: {self.results.ssr:.2f}")
        
        # Test for autocorrelation in residuals
        residuals = self.results.resid
        if len(residuals) > 1:
            # Simple Ljung-Box test would be here
            logger.info(f"Residual std: {residuals.std():.4f}")
            logger.info(f"Residual mean: {residuals.mean():.4f}")
    
    def __repr__(self) -> str:
        """String representation."""
        fitted = "fitted" if self.is_fitted else "unfitted"
        return (
            f"SARIMAModel(dept={self.department}, age={self.age_group}, "
            f"order={self.order}, seasonal={self.seasonal_order}, {fitted})"
        )
