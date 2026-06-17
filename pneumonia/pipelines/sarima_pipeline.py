"""
SARIMA Training Pipeline

Orchestrates the complete workflow for training, validating, and testing SARIMA models:
1. Data Loading & Validation
2. Temporal Split (Dynamic or Year-based)
3. SARIMA Model Training (Manual or Auto ARIMA)
4. Validation & Diagnostics
5. Test Set Evaluation
6. Results Reporting
"""

from pathlib import Path
from typing import Optional, Dict, Tuple, List, Any
import pandas as pd
import numpy as np
import json
from datetime import datetime
import logging

from pneumonia.config import (
    TEMPORAL_SPLIT_STRATEGY,
    SARIMA_USE_AUTO_ARIMA,
    MODEL_STORAGE_PATH,
    REPORTS_PATH,
)
from pneumonia.models.utils import (
    get_available_departments,
    get_departmental_data,
    temporal_split,
    handle_missing_values,
    validate_time_series,
)
from pneumonia.models.sarima.model import SARIMAModel
from pneumonia.evaluation.metrics import compute_all_metrics, baseline_metrics
from pneumonia.utils import setup_logger

logger = setup_logger(__name__)


class SARIMAPipeline:
    """
    End-to-end SARIMA training and evaluation pipeline.
    
    Usage:
        pipeline = SARIMAPipeline(department='AMAZONAS', age_group='under5')
        results = pipeline.run()
        print(results)
    """
    
    def __init__(
        self,
        department: str,
        age_group: str = "under5",
        use_auto_arima: Optional[bool] = None,
        forecast_steps: int = 52,
        split_strategy: Optional[str] = None,
        order: Optional[Tuple[int, int, int]] = None,
        seasonal_order: Optional[Tuple[int, int, int, int]] = None,
        use_fourier: Optional[bool] = None,
        n_fourier_terms: Optional[int] = None,
    ):
        """
        Initialize pipeline.
        
        Args:
            department: Department name
            age_group: 'under5' or '60plus'
            use_auto_arima: Override config setting for auto_arima search
            forecast_steps: Number of steps to forecast (default: 52 weeks = 1 year)
            split_strategy: Override config 'dynamic' or 'years' splitting
        """
        self.department = department.upper()
        self.age_group = age_group.lower()
        self.use_auto_arima = use_auto_arima if use_auto_arima is not None else SARIMA_USE_AUTO_ARIMA
        self.forecast_steps = forecast_steps
        self.split_strategy = split_strategy or TEMPORAL_SPLIT_STRATEGY
        self.order = order
        self.seasonal_order = seasonal_order
        self.use_fourier = use_fourier
        self.n_fourier_terms = n_fourier_terms
        
        # Storage for results
        self.data = None
        self.train = None
        self.val = None
        self.test = None
        self.model = None
        self.val_forecasts: dict = {}
        self.test_forecasts: dict = {}
        self.results = {
            "department": self.department,
            "age_group": self.age_group,
            "timestamp": datetime.now().isoformat(),
            "stages": {},
        }
        
        logger.info(f"Pipeline initialized: {self.department} ({self.age_group})")
    
    def run(self) -> Dict[str, Any]:
        """
        Execute complete pipeline: load → split → train → validate → test → report.
        
        Returns:
            Dictionary with all results
        """
        try:
            logger.info(f"Starting SARIMA pipeline for {self.department}/{self.age_group}")
            
            self._stage_load_data()
            self._stage_temporal_split()
            self._stage_train_model()
            self._stage_validate_model()
            self._stage_test_model()
            self._stage_generate_report()
            
            logger.info("Pipeline completed successfully")
            return self.results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            self.results["error"] = str(e)
            raise
    
    def _stage_load_data(self) -> None:
        """Load and validate data for the department."""
        logger.info(f"Stage 1: Loading data for {self.department} ({self.age_group})")
        
        try:
            # Load data
            self.data = get_departmental_data(
                self.department,
                age_group=self.age_group
            )
            
            logger.info(f"Loaded {len(self.data)} observations ({self.data.index.min().date()} to {self.data.index.max().date()})")
            
            # Handle missing values
            nan_count = self.data.isna().sum()
            if nan_count > 0:
                logger.warning(f"Found {nan_count} missing values ({nan_count/len(self.data)*100:.1f}%)")
                self.data = handle_missing_values(self.data, method="interpolate")
            
            # Validate
            validate_time_series(self.data)
            
            self.results["stages"]["data_loading"] = {
                "n_observations": len(self.data),
                "date_range": {
                    "start": str(self.data.index.min()),
                    "end": str(self.data.index.max()),
                },
                "missing_values": nan_count,
                "status": "success",
            }
            
            logger.info("Stage 1 completed successfully")
            
        except Exception as e:
            logger.error(f"Data loading failed: {str(e)}")
            self.results["stages"]["data_loading"] = {"status": "failed", "error": str(e)}
            raise
    
    def _stage_temporal_split(self) -> None:
        """Split data into train, validation, and test sets."""
        logger.info(f"Stage 2: Temporal split (strategy: {self.split_strategy})")
        
        try:
            self.train, self.val, self.test = temporal_split(
                self.data,
                strategy=self.split_strategy
            )
            
            self.results["stages"]["temporal_split"] = {
                "strategy": self.split_strategy,
                "train": {"n_weeks": len(self.train), "date_range": [str(self.train.index.min()), str(self.train.index.max())]},
                "val": {"n_weeks": len(self.val), "date_range": [str(self.val.index.min()), str(self.val.index.max())]},
                "test": {"n_weeks": len(self.test), "date_range": [str(self.test.index.min()), str(self.test.index.max())]},
                "status": "success",
            }
            
            logger.info("Stage 2 completed successfully")
            
        except Exception as e:
            logger.error(f"Temporal split failed: {str(e)}")
            self.results["stages"]["temporal_split"] = {"status": "failed", "error": str(e)}
            raise
    
    def _stage_train_model(self) -> None:
        """Train SARIMA model on training set."""
        logger.info(f"Stage 3: Training SARIMA model (use_auto_arima={self.use_auto_arima})")
        
        try:
            # Create model
            self.model = SARIMAModel(
                department=self.department,
                age_group=self.age_group,
                order=self.order,
                seasonal_order=self.seasonal_order,
                use_fourier=self.use_fourier,
                n_fourier_terms=self.n_fourier_terms,
            )
            
            # Fit model
            self.model.fit(
                self.train,
                use_auto_arima=self.use_auto_arima
            )
            
            self.results["stages"]["training"] = {
                "n_training_obs": len(self.train),
                "order": self.model.order,
                "seasonal_order": self.model.seasonal_order,
                "fit_method": self.model.metadata.get("fit_method", "unknown"),
                "aic": self.model.metadata.get("aic"),
                "bic": self.model.metadata.get("bic"),
                "status": "success",
            }
            
            logger.info(f"Stage 3 completed - Model: SARIMA{self.model.order}x{self.model.seasonal_order}")
            
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            self.results["stages"]["training"] = {"status": "failed", "error": str(e)}
            raise
    
    def _stage_validate_model(self) -> None:
        """Validate model on validation set."""
        logger.info(f"Stage 4: Validating model on validation set ({len(self.val)} weeks)")
        
        try:
            if len(self.val) == 0:
                logger.warning("Validation set is empty, skipping validation stage")
                self.results["stages"]["validation"] = {"status": "skipped", "reason": "empty_val_set"}
                return
            
            # Predict on validation set (anchor to training data)
            val_forecast = self.model.predict(self.train, steps=len(self.val))
            self.val_forecasts['SARIMA'] = val_forecast
            
            # Compute metrics
            val_metrics = compute_all_metrics(self.val.values, val_forecast, training_actual=self.train.values)
            
            # Compute baseline for comparison
            baseline = baseline_metrics(self.val.values)
            
            # Diagnostics
            diagnostics = self.model.diagnostics()
            
            self.results["stages"]["validation"] = {
                "n_val_obs": len(self.val),
                "metrics": {k: float(v) if not np.isnan(v) else None for k, v in val_metrics.items()},
                "baseline": {k: float(v) if not np.isnan(v) else None for k, v in baseline.items() if k != "method"},
                "diagnostics": diagnostics,
                "status": "success",
            }
            
            logger.info(f"Validation metrics: MAE={val_metrics['mae']:.2f}, RMSE={val_metrics['rmse']:.2f}")
            logger.info("Stage 4 completed successfully")
            
        except Exception as e:
            logger.error(f"Validation failed: {str(e)}")
            self.results["stages"]["validation"] = {"status": "failed", "error": str(e)}
            raise
    
    def _stage_test_model(self) -> None:
        """Test model on test set (final evaluation)."""
        logger.info(f"Stage 5: Testing model on test set ({len(self.test)} weeks)")
        
        try:
            if len(self.test) == 0:
                logger.warning("Test set is empty, skipping test stage")
                self.results["stages"]["testing"] = {"status": "skipped", "reason": "empty_test_set"}
                return
            
            # Predict on test set (anchor to train + val so model sees all observed data)
            test_forecast = self.model.predict(
                pd.concat([self.train, self.val]), steps=len(self.test)
            )
            self.test_forecasts['SARIMA'] = test_forecast
            
            # Compute metrics
            test_metrics = compute_all_metrics(self.test.values, test_forecast, training_actual=pd.concat([self.train, self.val]).values)
            
            # Compute baseline for comparison
            baseline = baseline_metrics(self.test.values)
            
            self.results["stages"]["testing"] = {
                "n_test_obs": len(self.test),
                "metrics": {k: float(v) if not np.isnan(v) else None for k, v in test_metrics.items()},
                "baseline": {k: float(v) if not np.isnan(v) else None for k, v in baseline.items() if k != "method"},
                "status": "success",
            }
            
            logger.info(f"Test metrics: MAE={test_metrics['mae']:.2f}, RMSE={test_metrics['rmse']:.2f}")
            logger.info("Stage 5 completed successfully")
            
        except Exception as e:
            logger.error(f"Testing failed: {str(e)}")
            self.results["stages"]["testing"] = {"status": "failed", "error": str(e)}
            raise
    
    def _stage_generate_report(self) -> None:
        """Generate and save results report."""
        logger.info("Stage 6: Generating report and saving model")
        
        try:
            # Save model
            model_path = self.model.save()
            self.results["model_path"] = str(model_path)
            
            # Save results JSON
            results_dir = REPORTS_PATH / self.department / self.age_group
            results_dir.mkdir(parents=True, exist_ok=True)

            results_file = results_dir / "results.json"
            with open(results_file, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)

            from pneumonia.visualization.persistence import save_predictions
            pred_csv = save_predictions(
                reports_dir=REPORTS_PATH,
                department=self.department,
                age_group=self.age_group,
                train=self.train,
                val=self.val,
                test=self.test,
                model_name="SARIMA",
                val_forecast=self.val_forecasts.get("SARIMA"),
                test_forecast=self.test_forecasts.get("SARIMA"),
            )

            self.results["stages"]["reporting"] = {
                "model_path": str(model_path),
                "results_file": str(results_file),
                "predictions_csv": str(pred_csv),
                "status": "success",
            }

            logger.info(f"Results saved to {results_file}")
            logger.info("Stage 6 completed successfully")
            
        except Exception as e:
            logger.error(f"Report generation failed: {str(e)}")
            self.results["stages"]["reporting"] = {"status": "failed", "error": str(e)}
            raise
    
    def summary(self) -> str:
        """Generate summary of results."""
        lines = [
            f"\n{'='*80}",
            f"SARIMA PIPELINE RESULTS - {self.department}/{self.age_group}",
            f"{'='*80}",
        ]
        
        if "training" in self.results["stages"]:
            training = self.results["stages"]["training"]
            if training["status"] == "success":
                lines.append(f"Model: SARIMA{training['order']}x{training['seasonal_order']}")
                lines.append(f"AIC: {training['aic']:.2f}")
        
        if "validation" in self.results["stages"]:
            val = self.results["stages"]["validation"]
            if val["status"] == "success":
                lines.append(f"\nValidation Metrics:")
                for metric, value in val["metrics"].items():
                    if value is not None:
                        lines.append(f"  {metric}: {value:.4f}")
        
        if "testing" in self.results["stages"]:
            test = self.results["stages"]["testing"]
            if test["status"] == "success":
                lines.append(f"\nTest Metrics:")
                for metric, value in test["metrics"].items():
                    if value is not None:
                        lines.append(f"  {metric}: {value:.4f}")
        
        lines.append(f"{'='*80}\n")
        return "\n".join(lines)


def run_pipeline_for_all_departments(
    age_group: str = "under5",
    use_auto_arima: Optional[bool] = None,
    split_strategy: Optional[str] = None,
    forecast_steps: int = 52,
    order: Optional[Tuple[int, int, int]] = None,
    seasonal_order: Optional[Tuple[int, int, int, int]] = None,
    use_fourier: Optional[bool] = None,
    n_fourier_terms: Optional[int] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Run SARIMA pipeline for all available departments.
    
    Args:
        age_group: 'under5' or '60plus'
        use_auto_arima: Override config for auto_arima
        
    Returns:
        Dictionary with results for each department
    """
    logger.info(f"Running pipelines for all departments ({age_group})")
    
    departments = get_available_departments()
    all_results = {}
    
    for dept in departments:
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing {dept}")
        logger.info(f"{'='*80}")
        
        try:
            pipeline = SARIMAPipeline(
                department=dept,
                age_group=age_group,
                use_auto_arima=use_auto_arima,
                split_strategy=split_strategy,
                forecast_steps=forecast_steps,
                order=order,
                seasonal_order=seasonal_order,
                use_fourier=use_fourier,
                n_fourier_terms=n_fourier_terms,
            )
            
            results = pipeline.run()
            print(pipeline.summary())
            
            all_results[dept] = {
                "status": "success",
                "results": results,
            }
            
        except Exception as e:
            logger.error(f"Pipeline failed for {dept}: {str(e)}")
            all_results[dept] = {
                "status": "failed",
                "error": str(e),
            }
    
    logger.info(f"\n{'='*80}")
    logger.info("BATCH PROCESSING COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(f"Successfully trained: {sum(1 for r in all_results.values() if r['status'] == 'success')}/{len(departments)}")
    
    return all_results
