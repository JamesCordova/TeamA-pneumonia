"""
Visualization utilities for forecast comparison and reporting.

Modules:
  persistence       — save_predictions, save_walkforward_predictions, save_step_metrics
  classic_plot      — plot_classic (train/val/test comparison)
  backtest_plot     — plot_backtest (walk-forward backtest)
  comparison_plot   — plot_model_comparison (metric vs horizon, across models)
  step_metrics_plot — plot_step_metrics (per-step metric distribution + time evolution)
  _utils            — shared helpers (internal)
"""

from pneumonia.visualization.persistence import (
    load_step_metrics,
    save_predictions,
    save_step_metrics,
    save_walkforward_predictions,
)
from pneumonia.visualization.classic_plot import plot_classic
from pneumonia.visualization.backtest_plot import plot_backtest

__all__ = [
    "load_step_metrics",
    "save_predictions",
    "save_step_metrics",
    "save_walkforward_predictions",
    "plot_classic",
    "plot_backtest",
]
