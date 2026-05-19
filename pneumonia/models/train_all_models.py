"""
Training script for all SARIMA models

Trains SARIMA models for all available departments and age groups.
Saves models and generates comparison reports.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
import logging

from pneumonia.models.utils import (
    get_available_departments,
    get_departmental_data,
    temporal_split,
    validate_time_series,
)
from pneumonia.models.sarima import SARIMAModel
from pneumonia.evaluation import compute_all_metrics, generate_comparison_report
from pneumonia.utils import setup_logger

# Auto ARIMA for parameter optimization
try:
    from pmdarima.auto_arima import auto_arima
    HAVE_AUTO_ARIMA = True
except ImportError:
    HAVE_AUTO_ARIMA = False

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logger = setup_logger(__name__)


def find_best_sarima_order(train_data: pd.Series, max_p: int = 2, max_d: int = 1, max_q: int = 2) -> tuple:
    """
    Find best SARIMA order using auto_arima if available.
    
    Args:
        train_data: Training time series
        max_p, max_d, max_q: Search ranges
        
    Returns:
        Tuple of (order, seasonal_order) or default values
    """
    if not HAVE_AUTO_ARIMA:
        return (1, 1, 1), (1, 1, 1, 52)
    
    try:
        logger.info("Finding optimal SARIMA parameters...")
        
        # Use auto_arima with seasonal=True and weekly period
        model = auto_arima(
            train_data,
            start_p=0, max_p=max_p,
            start_d=0, max_d=max_d,
            start_q=0, max_q=max_q,
            start_P=0, max_P=1,
            start_D=0, max_D=1,
            start_Q=0, max_Q=1,
            m=52,  # Weekly seasonal period
            seasonal=True,
            stepwise=True,
            trace=False,
            error_action="ignore",
            suppress_warnings=True,
            maxiter=100,
        )
        
        order = model.order
        seasonal_order = model.seasonal_order
        
        logger.info(f"✓ Optimal order found: SARIMA{order}×{seasonal_order}")
        return order, seasonal_order
        
    except Exception as e:
        logger.warning(f"auto_arima failed, using defaults: {str(e)}")
        return (1, 1, 1), (1, 1, 1, 52)


def train_single_model(
    department: str,
    age_group: str,
    save_models: bool = True,
    use_auto_arima: bool = False,
) -> dict:
    """
    Train SARIMA model for a single department-age group combination.
    
    Args:
        department: Department name
        age_group: 'under5' or '60plus'
        save_models: Whether to save fitted model
        use_auto_arima: Whether to use auto_arima for parameter optimization
        
    Returns:
        Dictionary with training results and metrics
    """
    result = {
        "department": department,
        "age_group": age_group,
        "status": "pending",
        "error": None,
        "order": None,
        "seasonal_order": None,
        "train_samples": None,
        "val_samples": None,
        "test_samples": None,
        "aic": None,
        "mae": None,
        "rmse": None,
        "mape": None,
        "smape": None,
        "mda": None,
    }
    
    try:
        logger.info(f"\n{'='*60}")
        logger.info(f"Training: {department} ({age_group})")
        logger.info(f"{'='*60}")
        
        # Load data
        ts = get_departmental_data(department, age_group=age_group)
        validate_time_series(ts, min_length=104)
        
        # Temporal split
        train, val, test = temporal_split(ts)
        
        result["train_samples"] = len(train)
        result["val_samples"] = len(val)
        result["test_samples"] = len(test)
        
        logger.info(f"Data loaded: Train={len(train)}, Val={len(val)}, Test={len(test)}")
        
        # Find optimal order if requested
        if use_auto_arima:
            order, seasonal_order = find_best_sarima_order(train)
        else:
            order, seasonal_order = None, None
        
        # Create and fit model
        model = SARIMAModel(
            department=department,
            age_group=age_group,
            order=order,
            seasonal_order=seasonal_order,
        )
        logger.info(f"Fitting SARIMA{model.order}×{model.seasonal_order}...")
        
        model.fit(train)
        result["aic"] = model.results.aic
        result["order"] = str(model.order)
        result["seasonal_order"] = str(model.seasonal_order)
        
        logger.info(f"✓ Model fitted (AIC: {model.results.aic:.2f})")
        
        # Predict on validation set
        val_forecast = model.predict(steps=len(val))
        val_metrics = compute_all_metrics(val.values, val_forecast)
        
        # Predict on test set
        test_forecast = model.predict(steps=len(test))
        test_metrics = compute_all_metrics(test.values, test_forecast)
        
        # Use test metrics for overall result
        result.update({
            "mae": test_metrics["mae"],
            "rmse": test_metrics["rmse"],
            "mape": test_metrics["mape"],
            "smape": test_metrics["smape"],
            "mda": test_metrics["mda"],
            "val_rmse": val_metrics["rmse"],
            "test_rmse": test_metrics["rmse"],
            "status": "success",
        })
        
        logger.info(
            f"✓ Evaluation complete - Test RMSE: {test_metrics['rmse']:.2f}, "
            f"MAPE: {test_metrics['mape']:.2f}%"
        )
        
        # Save model
        if save_models:
            model_dir = Path(f"models/{department}/{age_group}")
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / "sarima_model.pkl"
            
            model.save(str(model_path))
            logger.info(f"✓ Model saved to {model_path}")
        
        return result
        
    except Exception as e:
        logger.error(f"✗ Training failed: {str(e)}")
        result["status"] = "failed"
        result["error"] = str(e)
        return result


def train_all_models(
    departments: list = None,
    age_groups: list = None,
    save_models: bool = True,
    use_auto_arima: bool = False,
) -> pd.DataFrame:
    """
    Train SARIMA models for all departments and age groups.
    
    Args:
        departments: List of departments. Uses all available if None.
        age_groups: List of age groups. Defaults to ['under5', '60plus']
        save_models: Whether to save fitted models
        use_auto_arima: Whether to optimize parameters with auto_arima
        
    Returns:
        DataFrame with results for all models
    """
    if departments is None:
        departments = get_available_departments()
    
    if age_groups is None:
        age_groups = ["under5", "60plus"]
    
    logger.info(f"\n🚀 Starting training for {len(departments)} departments × {len(age_groups)} age groups")
    logger.info(f"Total models to train: {len(departments) * len(age_groups)}")
    logger.info(f"Auto ARIMA optimization: {'enabled' if use_auto_arima else 'disabled'}")
    
    results = []
    
    for i, department in enumerate(departments, 1):
        for j, age_group in enumerate(age_groups, 1):
            logger.info(f"\n[{i}/{len(departments)}] [{j}/{len(age_groups)}] Processing {department} ({age_group})...")
            
            result = train_single_model(
                department=department,
                age_group=age_group,
                save_models=save_models,
                use_auto_arima=use_auto_arima,
            )
            results.append(result)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Summary statistics
    logger.info(f"\n{'='*60}")
    logger.info("📊 TRAINING SUMMARY")
    logger.info(f"{'='*60}")
    
    successful = (results_df["status"] == "success").sum()
    failed = (results_df["status"] == "failed").sum()
    
    logger.info(f"✓ Successful: {successful}")
    logger.info(f"✗ Failed: {failed}")
    
    if successful > 0:
        logger.info(f"\nTest Set Performance:")
        logger.info(f"  Average RMSE:  {results_df[results_df['status']=='success']['rmse'].mean():.2f}")
        logger.info(f"  Average MAPE:  {results_df[results_df['status']=='success']['mape'].mean():.2f}%")
        logger.info(f"  Average MDA:   {results_df[results_df['status']=='success']['mda'].mean():.2f}%")
        
        logger.info(f"\nTop 5 Best Performers (by Test RMSE):")
        top5 = results_df[results_df['status']=='success'].nsmallest(5, 'rmse')[
            ['department', 'age_group', 'rmse', 'mape']
        ]
        for idx, row in top5.iterrows():
            logger.info(
                f"  {row['department']:12} {row['age_group']:8} "
                f"RMSE: {row['rmse']:7.2f}  MAPE: {row['mape']:6.2f}%"
            )
    
    return results_df


def save_training_results(results_df: pd.DataFrame) -> None:
    """Save training results to CSV and JSON."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # CSV report
    csv_path = f"reports/training_results_{timestamp}.csv"
    Path("reports").mkdir(exist_ok=True)
    results_df.to_csv(csv_path, index=False)
    logger.info(f"\n✓ Results saved to {csv_path}")
    
    # Summary stats
    summary_path = f"reports/training_summary_{timestamp}.txt"
    with open(summary_path, "w") as f:
        f.write("SARIMA MODEL TRAINING SUMMARY\n")
        f.write(f"Timestamp: {datetime.now()}\n")
        f.write("="*60 + "\n\n")
        
        successful = (results_df["status"] == "success").sum()
        f.write(f"Total Models: {len(results_df)}\n")
        f.write(f"Successful: {successful}\n")
        f.write(f"Failed: {len(results_df) - successful}\n\n")
        
        if successful > 0:
            f.write("Test Set Metrics (Average):\n")
            f.write(f"  RMSE:  {results_df[results_df['status']=='success']['rmse'].mean():.2f}\n")
            f.write(f"  MAPE:  {results_df[results_df['status']=='success']['mape'].mean():.2f}%\n")
            f.write(f"  SMAPE: {results_df[results_df['status']=='success']['smape'].mean():.2f}%\n")
            f.write(f"  MDA:   {results_df[results_df['status']=='success']['mda'].mean():.2f}%\n")
    
    logger.info(f"✓ Summary saved to {summary_path}")


if __name__ == "__main__":
    # Train all models with auto_arima optimization
    # Set use_auto_arima=False to use default parameters (faster)
    results_df = train_all_models(save_models=True, use_auto_arima=False)
    
    # Save results
    save_training_results(results_df)
    
    # Display final results
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    successful = results_df[results_df['status']=='success']
    if len(successful) > 0:
        print(successful.sort_values('rmse')[
            ['department', 'age_group', 'order', 'seasonal_order', 'rmse', 'mape', 'mda']
        ].to_string(index=False))
    else:
        print("No successful trainings")
