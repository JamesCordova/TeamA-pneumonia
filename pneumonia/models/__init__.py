"""
Forecasting models module for pneumonia case prediction

This module provides classes and utilities for building, training, and evaluating
forecasting models to predict weekly pneumonia cases by department and age group.

Models:
    BaseForecaster: Abstract base class for all forecasting models
    SARIMAModel: SARIMA (Seasonal ARIMA) implementation
    XGBoostModel: Gradient boosting tree-based forecasting
    RandomForestModel: Random forest ensemble forecasting
    EnsembleModel: Combined ensemble forecasting

Utilities:
    utils: Data loading, splits, and preprocessing functions
    ml: Machine learning models submodule
"""

from pneumonia.models.base import BaseForecaster
from pneumonia.models.sarima.model import SARIMAModel
from pneumonia.models.ml.xgboost import XGBoostModel
from pneumonia.models.ml.random_forest import RandomForestModel
from pneumonia.models.ml.ensemble import EnsembleModel

__all__ = [
    "BaseForecaster",
    "SARIMAModel",
    "XGBoostModel",
    "RandomForestModel",
    "EnsembleModel",
]
