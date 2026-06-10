# Compatibility shim — import directly from the new modules instead.
from pneumonia.visualization.persistence import save_predictions, save_walkforward_predictions
from pneumonia.visualization.classic_plot import plot_classic
from pneumonia.visualization.backtest_plot import plot_backtest

plot_forecasts = plot_classic  # legacy alias
