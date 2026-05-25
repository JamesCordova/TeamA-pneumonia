#!/usr/bin/env python
"""
Plot SARIMA forecasting results

Generates visualization comparing actual vs predicted values on test set.

Usage:
    python scripts/plot_forecasting.py --department AMAZONAS --age_group under5
    python scripts/plot_forecasting.py --all  # Plot all departments
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pneumonia.models.utils import (
    get_available_departments,
    get_departmental_data,
    temporal_split,
    handle_missing_values,
    validate_time_series,
)
from pneumonia.models.sarima.model import SARIMAModel
from pneumonia.config import TEMPORAL_SPLIT_STRATEGY, MODEL_STORAGE_PATH, REPORTS_PATH
from pneumonia.utils import setup_logger

logger = setup_logger(__name__)


def plot_single_forecast(department: str, age_group: str, output_path: Path = None) -> None:
    """
    Generate forecasting plot for a single department/age group.
    
    Args:
        department: Department name (e.g., 'AMAZONAS')
        age_group: Age group ('under5' or '60plus')
        output_path: Path to save plot (optional)
    """
    logger.info(f"Plotting forecasts for {department} ({age_group})")
    
    try:
        # Stage 1: Load data
        logger.info("Loading data...")
        data = get_departmental_data(department, age_group)
        data = handle_missing_values(data)
        validate_time_series(data)
        
        # Stage 2: Split data
        logger.info("Splitting data...")
        train_data, val_data, test_data = temporal_split(
            data, 
            strategy=TEMPORAL_SPLIT_STRATEGY
        )
        
        # Stage 3: Load trained model
        logger.info("Loading trained model...")
        model_path = MODEL_STORAGE_PATH / department / age_group / "SARIMA.pkl"
        metadata_path = MODEL_STORAGE_PATH / department / age_group / "SARIMA_metadata.json"
        
        if not model_path.exists():
            logger.error(f"Model not found: {model_path}")
            logger.error(f"Train model first with: python scripts/train_sarima.py --department {department} --age_group {age_group}")
            return
        
        # Load the model (this is a static method that returns a new instance)
        model = SARIMAModel.load(str(model_path))
        
        # Stage 4: Generate predictions for test set
        logger.info("Generating predictions...")
        test_predictions = model.predict(steps=len(test_data))
        test_pred_lower, test_pred_upper = model.get_forecast_interval(steps=len(test_data))[1:]
        
        # Create time index
        test_dates = pd.date_range(
            start=test_data.index[0],
            periods=len(test_data),
            freq='W'
        )
        
        # Stage 5: Create visualization
        logger.info("Creating visualization...")
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        fig.suptitle(
            f'SARIMA Forecasting: {department} ({age_group})\nModel: {model.metadata.get("order", "?")} × {model.metadata.get("seasonal_order", "?")}',
            fontsize=14,
            fontweight='bold'
        )
        
        # ===== SUBPLOT 1: Full time series with all splits =====
        ax1 = axes[0]
        
        train_dates = pd.date_range(
            start=train_data.index[0],
            periods=len(train_data),
            freq='W'
        )
        val_dates = pd.date_range(
            start=val_data.index[0],
            periods=len(val_data),
            freq='W'
        )
        
        # Plot all data
        ax1.plot(train_dates, train_data.values, label='Training Data', color='#2E86AB', linewidth=1.5, alpha=0.8)
        ax1.plot(val_dates, val_data.values, label='Validation Data', color='#A23B72', linewidth=1.5, alpha=0.8)
        ax1.plot(test_dates, test_data.values, label='Test Data (Actual)', color='#F18F01', linewidth=2, marker='o', markersize=3)
        ax1.plot(test_dates, test_predictions, label='Predictions', color='#C73E1D', linewidth=2, linestyle='--', marker='s', markersize=3)
        
        # Add confidence interval
        ax1.fill_between(test_dates, test_pred_lower, test_pred_upper, alpha=0.2, color='#C73E1D', label='95% Confidence Interval')
        
        # Styling
        ax1.set_xlabel('Date', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Cases', fontsize=11, fontweight='bold')
        ax1.set_title('Full Time Series: Training → Validation → Test (with Predictions)', fontsize=12, fontweight='bold')
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.tick_params(axis='x', rotation=45)
        
        # ===== SUBPLOT 2: Zoomed-in test set comparison =====
        ax2 = axes[1]
        
        # Plot test data and predictions side by side
        x_pos = np.arange(len(test_data))
        width = 0.35
        
        ax2.bar(x_pos - width/2, test_data.values, width, label='Actual', color='#F18F01', alpha=0.8)
        ax2.bar(x_pos + width/2, test_predictions, width, label='Predicted', color='#C73E1D', alpha=0.8)
        
        # Add error bands
        ax2.fill_between(x_pos + width/2, test_pred_lower, test_pred_upper, alpha=0.2, color='#C73E1D')
        
        # Calculate metrics
        mae = np.mean(np.abs(test_data.values - test_predictions))
        rmse = np.sqrt(np.mean((test_data.values - test_predictions)**2))
        mape = np.mean(np.abs((test_data.values - test_predictions) / (np.abs(test_data.values) + 1e-10))) * 100
        
        # Styling
        ax2.set_xlabel('Test Week', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Cases', fontsize=11, fontweight='bold')
        ax2.set_title(f'Test Set Comparison (MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.1f}%)', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([f'W{i+1}' for i in range(len(test_data))], rotation=45, fontsize=8)
        
        plt.tight_layout()
        
        # Save plot
        if output_path is None:
            output_path = REPORTS_PATH / department / age_group / f"forecast_plot_{department}_{age_group}.png"
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to: {output_path}")
        
        plt.show()
        
    except Exception as e:
        logger.error(f"Plotting failed: {str(e)}", exc_info=True)
        raise


def plot_all_departments(age_group: str = 'under5') -> None:
    """
    Generate plots for all available departments.
    
    Args:
        age_group: Age group to plot
    """
    departments = get_available_departments()
    logger.info(f"Plotting {len(departments)} departments for {age_group} group...")
    
    for i, dept in enumerate(departments, 1):
        logger.info(f"[{i}/{len(departments)}] Processing {dept}...")
        try:
            plot_single_forecast(dept, age_group)
        except Exception as e:
            logger.warning(f"Failed to plot {dept}: {str(e)}")
            continue


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description='Plot SARIMA forecasting results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot single department
  python scripts/plot_forecasting.py --department AMAZONAS --age_group under5
  
  # Plot all departments
  python scripts/plot_forecasting.py --all --age_group under5
        """
    )
    
    dept_group = parser.add_mutually_exclusive_group(required=True)
    dept_group.add_argument(
        '--department', '-d',
        type=str,
        help='Department name (e.g., AMAZONAS)'
    )
    dept_group.add_argument(
        '--all', '-a',
        action='store_true',
        help='Plot all departments'
    )
    
    parser.add_argument(
        '--age_group', '-g',
        type=str,
        choices=['under5', '60plus'],
        default='under5',
        help='Age group (default: under5)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output file path for plot'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    
    return parser


def main():
    """Main execution."""
    parser = create_parser()
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    try:
        if args.all:
            plot_all_departments(age_group=args.age_group)
        else:
            output_path = Path(args.output) if args.output else None
            plot_single_forecast(args.department, args.age_group, output_path)
        
        logger.info("Done!")
        
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
