"""
Evaluation module initialization

Exports metrics and comparison utilities.
"""

from pneumonia.evaluation.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    mean_absolute_percentage_error,
    symmetric_mean_absolute_percentage_error,
    mean_directional_accuracy,
    compute_all_metrics,
    baseline_metrics,
)

from pneumonia.evaluation.compare_models import (
    compare_models,
    rank_models,
    generate_comparison_report,
    create_metrics_dataframe,
)

__all__ = [
    # Metrics
    "mean_absolute_error",
    "root_mean_squared_error",
    "mean_absolute_percentage_error",
    "symmetric_mean_absolute_percentage_error",
    "mean_directional_accuracy",
    "compute_all_metrics",
    "baseline_metrics",
    # Comparison
    "compare_models",
    "rank_models",
    "generate_comparison_report",
    "create_metrics_dataframe",
]
