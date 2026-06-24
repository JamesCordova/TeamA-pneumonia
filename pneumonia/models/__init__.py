"""
Forecasting models module for pneumonia case prediction

Models:
    BaseForecaster: Abstract base class for all forecasting models
    NaiveForecaster: Repeats last observed value (baseline)
    SeasonalNaiveForecaster: Repeats last-season values (baseline)
    SARIMAModel: SARIMA (Seasonal ARIMA) implementation
    XGBoostModel: Gradient boosting tree-based forecasting
    RandomForestModel: Random forest ensemble forecasting
    EnsembleModel: Combined ensemble forecasting
"""

from pneumonia.models.base import BaseForecaster
from pneumonia.models.baselines.naive import NaiveForecaster
from pneumonia.models.baselines.seasonal_naive import SeasonalNaiveForecaster
from pneumonia.models.sarima.model import SARIMAModel
from pneumonia.models.ml.xgboost import XGBoostModel
from pneumonia.models.ml.random_forest import RandomForestModel
from pneumonia.models.ml.ensemble import EnsembleModel
from pneumonia.models.prophet.model import ProphetModel

__all__ = [
    "BaseForecaster",
    "NaiveForecaster",
    "SeasonalNaiveForecaster",
    "SARIMAModel",
    "XGBoostModel",
    "RandomForestModel",
    "EnsembleModel",
    "ProphetModel",
]
