"""
Prophet Training Pipeline

Orchestrates the complete workflow for training, validating, and testing Prophet models:
1. Data Loading & Validation
2. Temporal Split (Dynamic or Year-based)
3. Prophet Model Training
4. Validation & Diagnostics
5. Test Set Evaluation
6. Results Reporting
"""

import json
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from pneumonia.config import (
    REPORTS_PATH,
    TEMPORAL_SPLIT_STRATEGY,
)
from pneumonia.evaluation.metrics import baseline_metrics, compute_all_metrics
from pneumonia.models.prophet.model import ProphetModel
from pneumonia.models.utils import (
    get_available_departments,
    get_departmental_data,
    handle_missing_values,
    temporal_split,
    validate_time_series,
)
from pneumonia.utils import setup_logger

logger = setup_logger(__name__)


class ProphetPipeline:
    """
    End-to-end Prophet training and evaluation pipeline.

    Usage:
        pipeline = ProphetPipeline(department='AMAZONAS', age_group='under5')
        results = pipeline.run()
        print(pipeline.summary())
    """

    def __init__(
        self,
        department: str,
        age_group: str = "under5",
        split_strategy: Optional[str] = None,
        forecast_steps: int = 52,
        growth: Optional[str] = None,
        yearly_seasonality: Optional[Any] = None,
        weekly_seasonality: Optional[Any] = None,
        daily_seasonality: Optional[Any] = None,
        seasonality_mode: Optional[str] = None,
        changepoint_prior_scale: Optional[float] = None,
        seasonality_prior_scale: Optional[float] = None,
        start_year: Optional[int] = None,
        **kwargs,
    ):
        self.department = department.upper()
        self.age_group = age_group.lower()
        self.split_strategy = split_strategy or TEMPORAL_SPLIT_STRATEGY
        self.forecast_steps = forecast_steps
        self.start_year = start_year

        # Prophet specific parameters
        self.model_params = {
            "growth": growth,
            "yearly_seasonality": yearly_seasonality,
            "weekly_seasonality": weekly_seasonality,
            "daily_seasonality": daily_seasonality,
            "seasonality_mode": seasonality_mode,
            "changepoint_prior_scale": changepoint_prior_scale,
            "seasonality_prior_scale": seasonality_prior_scale,
            **kwargs,
        }

        # Remove None values so that defaults are used instead
        self.model_params = {k: v for k, v in self.model_params.items() if v is not None}

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

        logger.info(f"ProphetPipeline initialized: {self.department} ({self.age_group})")

    def run(self) -> Dict[str, Any]:
        """
        Execute complete pipeline.
        """
        try:
            logger.info(f"Starting Prophet pipeline for {self.department}/{self.age_group}")

            self._stage_load_data()
            self._stage_temporal_split()
            self._stage_train_model()
            self._stage_validate_model()
            self._stage_test_model()
            self._stage_generate_report()

            logger.info("Prophet pipeline completed successfully")
            return self.results

        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            self.results["error"] = str(e)
            raise

    def _stage_load_data(self) -> None:
        """Load and validate data for the department."""
        logger.info(f"Stage 1: Loading data for {self.department} ({self.age_group})")

        try:
            self.data = get_departmental_data(
                self.department,
                age_group=self.age_group,
                start_year=self.start_year
            )
            logger.info(
                f"Loaded {len(self.data)} observations ({self.data.index.min().date()} to {self.data.index.max().date()})"
            )

            # Handle missing values
            nan_count = int(self.data.isna().sum())
            if nan_count > 0:
                logger.warning(f"Found {nan_count} missing values — interpolating")
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
            self.results["stages"]["data_loading"] = {"status": "failed", "error": str(e)}
            raise

    def _stage_temporal_split(self) -> None:
        """Split data into train, validation, and test sets."""
        logger.info(f"Stage 2: Temporal split (strategy: {self.split_strategy})")

        try:
            self.train, self.val, self.test = temporal_split(
                self.data, strategy=self.split_strategy
            )

            self.results["stages"]["temporal_split"] = {
                "strategy": self.split_strategy,
                "train": {
                    "n_weeks": len(self.train),
                    "date_range": [str(self.train.index.min()), str(self.train.index.max())],
                },
                "val": {
                    "n_weeks": len(self.val),
                    "date_range": [str(self.val.index.min()), str(self.val.index.max())],
                },
                "test": {
                    "n_weeks": len(self.test),
                    "date_range": [str(self.test.index.min()), str(self.test.index.max())],
                },
                "status": "success",
            }
            logger.info("Stage 2 completed successfully")

        except Exception as e:
            self.results["stages"]["temporal_split"] = {"status": "failed", "error": str(e)}
            raise

    def _stage_train_model(self) -> None:
        """Train Prophet model on training set."""
        logger.info("Stage 3: Training Prophet model")

        try:
            # Create model
            self.model = ProphetModel(
                department=self.department,
                age_group=self.age_group,
                **self.model_params,
            )

            # Fit model
            self.model.fit(self.train)

            self.results["stages"]["training"] = {
                "n_training_obs": len(self.train),
                "params": self.model._params,
                "status": "success",
            }
            logger.info("Stage 3 completed successfully")

        except Exception as e:
            self.results["stages"]["training"] = {"status": "failed", "error": str(e)}
            raise

    def _stage_validate_model(self) -> None:
        """Validate model on validation set."""
        logger.info(f"Stage 4: Validating model on validation set ({len(self.val)} weeks)")

        try:
            if len(self.val) == 0:
                logger.warning("Validation set is empty, skipping validation stage")
                self.results["stages"]["validation"] = {
                    "status": "skipped",
                    "reason": "empty_val_set",
                }
                return

            # Predict on validation set (anchored to training data)
            val_forecast = self.model.predict(self.train, steps=len(self.val))
            self.val_forecasts["Prophet"] = val_forecast

            # Compute metrics
            val_metrics = compute_all_metrics(self.val.values, val_forecast)

            # Compute baseline for comparison
            baseline = baseline_metrics(self.val.values)

            self.results["stages"]["validation"] = {
                "n_val_obs": len(self.val),
                "metrics": {
                    k: float(v) if not np.isnan(v) else None for k, v in val_metrics.items()
                },
                "baseline": {
                    k: float(v) if not np.isnan(v) else None
                    for k, v in baseline.items()
                    if k != "method"
                },
                "status": "success",
            }

            logger.info(f"Validation metrics: MAE={val_metrics['mae']:.2f}, RMSE={val_metrics['rmse']:.2f}")
            logger.info("Stage 4 completed successfully")

        except Exception as e:
            self.results["stages"]["validation"] = {"status": "failed", "error": str(e)}
            raise

    def _stage_test_model(self) -> None:
        """Test model on test set (final evaluation)."""
        logger.info(f"Stage 5: Testing model on test set ({len(self.test)} weeks)")

        try:
            if len(self.test) == 0:
                logger.warning("Test set is empty, skipping test stage")
                self.results["stages"]["testing"] = {
                    "status": "skipped",
                    "reason": "empty_test_set",
                }
                return

            # Predict on test set (anchored to train + val)
            test_forecast = self.model.predict(
                pd.concat([self.train, self.val]), steps=len(self.test)
            )
            self.test_forecasts["Prophet"] = test_forecast

            # Compute metrics
            test_metrics = compute_all_metrics(self.test.values, test_forecast)

            # Compute baseline
            baseline = baseline_metrics(self.test.values)

            self.results["stages"]["testing"] = {
                "n_test_obs": len(self.test),
                "metrics": {
                    k: float(v) if not np.isnan(v) else None for k, v in test_metrics.items()
                },
                "baseline": {
                    k: float(v) if not np.isnan(v) else None
                    for k, v in baseline.items()
                    if k != "method"
                },
                "status": "success",
            }

            logger.info(f"Test metrics: MAE={test_metrics['mae']:.2f}, RMSE={test_metrics['rmse']:.2f}")
            logger.info("Stage 5 completed successfully")

        except Exception as e:
            self.results["stages"]["testing"] = {"status": "failed", "error": str(e)}
            raise

    def _stage_generate_report(self) -> None:
        """Generate and save results report."""
        logger.info("Stage 6: Generating report and saving model")

        try:
            # Save model
            model_path = self.model.save()

            # Save results JSON
            results_dir = REPORTS_PATH / self.department / self.age_group
            results_dir.mkdir(parents=True, exist_ok=True)
            results_file = results_dir / "prophet_results.json"
            with open(results_file, "w") as f:
                json.dump(self.results, f, indent=2, default=str)

            # Save predictions CSV
            from pneumonia.visualization.persistence import save_predictions

            pred_csv = save_predictions(
                reports_dir=REPORTS_PATH,
                department=self.department,
                age_group=self.age_group,
                train=self.train,
                val=self.val,
                test=self.test,
                model_name="Prophet",
                val_forecast=self.val_forecasts.get("Prophet"),
                test_forecast=self.test_forecasts.get("Prophet"),
            )

            self.results["stages"]["reporting"] = {
                "model_path":      str(model_path),
                "results_file":    str(results_file),
                "predictions_csv": str(pred_csv),
                "status": "success",
            }

            logger.info(f"Results saved to {results_file}")
            logger.info("Stage 6 completed successfully")

        except Exception as e:
            self.results["stages"]["reporting"] = {"status": "failed", "error": str(e)}
            raise

    def summary(self) -> str:
        """Generate summary of results."""
        lines = [
            f"\n{'='*80}",
            f"PROPHET PIPELINE RESULTS - {self.department}/{self.age_group}",
            f"{'='*80}",
        ]

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


def run_prophet_for_all_departments(
    age_group: str = "under5",
    split_strategy: Optional[str] = None,
    forecast_steps: int = 52,
    start_year: Optional[int] = None,
    **kwargs,
) -> Dict[str, Dict[str, Any]]:
    """Run Prophet pipeline for all available departments."""
    logger.info(f"Running Prophet pipelines for all departments ({age_group})")

    departments = get_available_departments()
    all_results = {}

    for dept in departments:
        logger.info(f"\n{'='*80}\nProcessing {dept}\n{'='*80}")

        try:
            pipeline = ProphetPipeline(
                department=dept,
                age_group=age_group,
                split_strategy=split_strategy,
                forecast_steps=forecast_steps,
                start_year=start_year,
                **kwargs,
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

    successes = sum(1 for r in all_results.values() if r["status"] == "success")
    logger.info(
        f"\n{'='*80}\nPROPHET BATCH COMPLETE — {successes}/{len(departments)} succeeded\n{'='*80}"
    )

    return all_results
