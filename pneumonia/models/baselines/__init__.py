"""
Baseline forecasting models.

These deterministic models serve as lower-bound benchmarks that any
learned model should outperform.
"""

from pneumonia.models.baselines.naive import NaiveForecaster
from pneumonia.models.baselines.seasonal_naive import SeasonalNaiveForecaster
from pneumonia.models.baselines.holt_winters import HoltWintersForecaster

__all__ = ["NaiveForecaster", "SeasonalNaiveForecaster", "HoltWintersForecaster"]
