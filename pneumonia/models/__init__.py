"""
Forecasting models module for pneumonia case prediction

This module provides classes and utilities for building, training, and evaluating
forecasting models to predict weekly pneumonia cases by department and age group.

Models:
    BaseForecaster: Abstract base class for all forecasting models
    SARIMAModel: SARIMA (Seasonal ARIMA) implementation

Utilities:
    utils: Data loading, splits, and preprocessing functions
"""

from pneumonia.models.base import BaseForecaster
from pneumonia.models.sarima.model import SARIMAModel

__all__ = [
    "BaseForecaster",
    "SARIMAModel",
]
