"""
Model comparison and evaluation utilities

Functions for comparing multiple forecasting models and generating reports.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

from pneumonia.evaluation.metrics import compute_all_metrics, baseline_metrics
from pneumonia.utils import setup_logger

logger = setup_logger(__name__)


def compare_models(
    actual: np.ndarray,
    predictions: Dict[str, np.ndarray],
    include_baseline: bool = True
) -> pd.DataFrame:
    """
    Compare performance of multiple models on the same test data.
    
    Args:
        actual: Actual test values
        predictions: Dictionary with {model_name: predictions_array}
        include_baseline: Whether to include historical mean baseline
        
    Returns:
        DataFrame with model comparison results
    """
    results = []
    
    # Baseline
    if include_baseline:
        baseline = baseline_metrics(actual)
        results.append({
            "model": "baseline",
            **baseline
        })
    
    # Model predictions
    for model_name, pred in predictions.items():
        metrics = compute_all_metrics(actual, pred)
        results.append({
            "model": model_name,
            **metrics
        })
    
    df = pd.DataFrame(results)
    df = df.set_index("model")
    
    logger.info(f"Model comparison:\n{df}")
    
    return df


def rank_models(
    comparison_df: pd.DataFrame,
    metric: str = "rmse"
) -> pd.DataFrame:
    """
    Rank models by specified metric.
    
    Args:
        comparison_df: Output from compare_models()
        metric: Metric to rank by (lower is better)
        
    Returns:
        DataFrame sorted by metric
    """
    if metric not in comparison_df.columns:
        raise ValueError(f"Metric '{metric}' not found in results")
    
    ranked = comparison_df.sort_values(metric)
    ranked["rank"] = range(1, len(ranked) + 1)
    
    return ranked


def generate_comparison_report(
    actual: np.ndarray,
    predictions: Dict[str, np.ndarray],
    department: str = None,
    age_group: str = None
) -> str:
    """
    Generate human-readable comparison report.
    
    Args:
        actual: Actual test values
        predictions: Dictionary with model predictions
        department: Department name (for context)
        age_group: Age group (for context)
        
    Returns:
        Formatted report string
    """
    comparison = compare_models(actual, predictions)
    ranked = rank_models(comparison, metric="rmse")
    
    # Header
    report = "=" * 80 + "\n"
    report += "MODEL COMPARISON REPORT\n"
    if department:
        report += f"Department: {department}\n"
    if age_group:
        report += f"Age Group: {age_group}\n"
    report += "=" * 80 + "\n\n"
    
    # Metrics table
    report += "PERFORMANCE METRICS\n"
    report += "-" * 80 + "\n"
    report += ranked.to_string()
    report += "\n\n"
    
    # Winner
    best_model = ranked.index[0]
    best_rmse = ranked["rmse"].iloc[0]
    report += f"🏆 Best Model: {best_model} (RMSE: {best_rmse:.2f})\n"
    report += "=" * 80 + "\n"
    
    return report


def create_metrics_dataframe(
    test_sets: Dict[str, tuple],  # {name: (actual, predicted)}
    models: List[str],
    metrics: List[str] = None
) -> pd.DataFrame:
    """
    Create comprehensive metrics DataFrame across multiple test sets and models.
    
    Args:
        test_sets: Dictionary mapping test set names to (actual, predicted) tuples
        models: List of model names
        metrics: Metrics to compute. Default: ['mae', 'rmse', 'mape']
        
    Returns:
        DataFrame with shape (test_sets × models, metrics)
    """
    if metrics is None:
        metrics = ["mae", "rmse", "mape"]
    
    results = []
    
    for test_name, (actual, predicted) in test_sets.items():
        for model_name in models:
            row_metrics = compute_all_metrics(actual, predicted)
            
            row = {
                "test_set": test_name,
                "model": model_name,
            }
            row.update({m: row_metrics[m] for m in metrics if m in row_metrics})
            
            results.append(row)
    
    df = pd.DataFrame(results)
    
    return df
