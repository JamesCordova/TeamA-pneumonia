"""
RandomForest Forecasting Pipeline

Stages:
  1. Data loading & validation
  2. Temporal split
  3. Fit RandomForestModel on training set
  4. Validate on validation set
  5. Evaluate on test set
  6. Save model, results JSON, predictions CSV, and forecast plot
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import json
import numpy as np
import pandas as pd

from pneumonia.config import (
    TEMPORAL_SPLIT_STRATEGY,
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
from pneumonia.models.ml.random_forest import RandomForestModel
from pneumonia.evaluation.metrics import compute_all_metrics
from pneumonia.utils import setup_logger

logger = setup_logger(__name__)


class RandomForestPipeline:
    """
    End-to-end RandomForest training and evaluation pipeline.

    Usage:
        pipeline = RandomForestPipeline(department="AMAZONAS", age_group="under5")
        results  = pipeline.run()
        print(pipeline.summary())
    """

    def __init__(
        self,
        department: str,
        age_group: str = "under5",
        split_strategy: Optional[str] = None,
        forecast_steps: int = 52,
        rf_params: Optional[Dict] = None,
        lags: Optional[List[int]] = None,
        windows: Optional[List[int]] = None,
    ):
        """
        Args:
            department:      Department name.
            age_group:       'under5' or '60plus'.
            split_strategy:  'dynamic' or 'years' (overrides config).
            forecast_steps:  Weeks to forecast ahead (default 52).
            rf_params:       RandomForestRegressor hyperparameter overrides.
            lags:            Lag periods for feature engineering.
            windows:         Rolling window sizes for feature engineering.
        """
        self.department = department.upper()
        self.age_group = age_group.lower()
        self.split_strategy = split_strategy or TEMPORAL_SPLIT_STRATEGY
        self.forecast_steps = forecast_steps
        self.rf_params = rf_params
        self.lags = lags
        self.windows = windows

        self.data = None
        self.train = None
        self.val = None
        self.test = None
        self.model: Optional[RandomForestModel] = None
        self.val_forecasts: Dict[str, np.ndarray] = {}
        self.test_forecasts: Dict[str, np.ndarray] = {}

        self.results = {
            "department": self.department,
            "age_group": self.age_group,
            "timestamp": datetime.now().isoformat(),
            "stages": {},
        }

        logger.info(f"RandomForestPipeline initialised: {self.department} ({self.age_group})")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        """Execute all stages and return the results dict."""
        try:
            logger.info(f"Starting RandomForest pipeline for {self.department}/{self.age_group}")
            self._stage_load_data()
            self._stage_temporal_split()
            self._stage_train_model()
            self._stage_validate_model()
            self._stage_test_model()
            self._stage_generate_report()
            logger.info("RandomForest pipeline completed successfully")
            return self.results
        except Exception as exc:
            logger.error(f"Pipeline failed: {exc}")
            self.results["error"] = str(exc)
            raise

    def summary(self) -> str:
        lines = [
            f"\n{'='*80}",
            f"RANDOM FOREST PIPELINE RESULTS — {self.department}/{self.age_group}",
            f"{'='*80}",
        ]
        for stage_name in ("validation", "testing"):
            stage = self.results["stages"].get(stage_name, {})
            if stage.get("status") != "success":
                continue
            lines.append(f"\n{stage_name.title()} metrics:")
            for metric, value in stage.get("metrics", {}).items():
                if value is not None:
                    lines.append(f"  {metric}: {value:.4f}")
        if "training" in self.results["stages"]:
            top3 = self.results["stages"]["training"].get("top3_features", [])
            if top3:
                lines.append("\nTop 3 features:")
                for name, imp in top3:
                    lines.append(f"  {name}: {imp:.3f}")
        lines.append(f"{'='*80}\n")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Stages
    # ------------------------------------------------------------------

    def _stage_load_data(self) -> None:
        logger.info(f"Stage 1: Loading data for {self.department} ({self.age_group})")
        try:
            self.data = get_departmental_data(self.department, age_group=self.age_group)

            nan_count = int(self.data.isna().sum())
            if nan_count > 0:
                logger.warning(f"Found {nan_count} missing values — interpolating")
                self.data = handle_missing_values(self.data, method="interpolate")

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
            logger.info(f"Loaded {len(self.data)} observations")
        except Exception as exc:
            self.results["stages"]["data_loading"] = {"status": "failed", "error": str(exc)}
            raise

    def _stage_temporal_split(self) -> None:
        logger.info(f"Stage 2: Temporal split (strategy: {self.split_strategy})")
        try:
            self.train, self.val, self.test = temporal_split(
                self.data, strategy=self.split_strategy
            )
            self.results["stages"]["temporal_split"] = {
                "strategy": self.split_strategy,
                "train": {"n_weeks": len(self.train), "date_range": [str(self.train.index.min()), str(self.train.index.max())]},
                "val":   {"n_weeks": len(self.val),   "date_range": [str(self.val.index.min()),   str(self.val.index.max())]},
                "test":  {"n_weeks": len(self.test),  "date_range": [str(self.test.index.min()),  str(self.test.index.max())]},
                "status": "success",
            }
            logger.info(
                f"Split — train: {len(self.train)}, val: {len(self.val)}, test: {len(self.test)}"
            )
        except Exception as exc:
            self.results["stages"]["temporal_split"] = {"status": "failed", "error": str(exc)}
            raise

    def _stage_train_model(self) -> None:
        logger.info("Stage 3: Training RandomForest model")
        try:
            self.model = RandomForestModel(
                department=self.department,
                age_group=self.age_group,
                rf_params=self.rf_params,
                lags=self.lags,
                windows=self.windows,
            )
            self.model.fit(self.train)

            top3 = self.model.metadata.get("top3_features", [])
            self.results["stages"]["training"] = {
                "n_training_obs": len(self.train),
                "n_features":     self.model.metadata.get("n_features"),
                "lags":           self.model.lags,
                "windows":        self.model.windows,
                "n_estimators":   self.model.metadata.get("n_estimators"),
                "max_depth":      self.model.metadata.get("max_depth"),
                "top3_features":  top3,
                "status": "success",
            }
            logger.info("Stage 3 completed")
        except Exception as exc:
            self.results["stages"]["training"] = {"status": "failed", "error": str(exc)}
            raise

    def _stage_validate_model(self) -> None:
        logger.info(f"Stage 4: Validating on val set ({len(self.val)} weeks)")
        if len(self.val) == 0:
            self.results["stages"]["validation"] = {"status": "skipped", "reason": "empty_val_set"}
            return
        try:
            val_forecast = self.model.predict(self.train, steps=len(self.val))
            self.val_forecasts["RandomForest"] = val_forecast

            metrics = compute_all_metrics(self.val.values, val_forecast, training_actual=self.train.values)
            self.results["stages"]["validation"] = {
                "n_val_obs": len(self.val),
                "metrics": {k: float(v) if not np.isnan(v) else None for k, v in metrics.items()},
                "status": "success",
            }
            logger.info(f"Validation — MAE={metrics['mae']:.2f}, RMSE={metrics['rmse']:.2f}")
        except Exception as exc:
            self.results["stages"]["validation"] = {"status": "failed", "error": str(exc)}
            raise

    def _stage_test_model(self) -> None:
        logger.info(f"Stage 5: Testing on test set ({len(self.test)} weeks)")
        if len(self.test) == 0:
            self.results["stages"]["testing"] = {"status": "skipped", "reason": "empty_test_set"}
            return
        try:
            test_forecast = self.model.predict(
                pd.concat([self.train, self.val]), steps=len(self.test)
            )
            self.test_forecasts["RandomForest"] = test_forecast

            metrics = compute_all_metrics(self.test.values, test_forecast, training_actual=pd.concat([self.train, self.val]).values)
            self.results["stages"]["testing"] = {
                "n_test_obs": len(self.test),
                "metrics": {k: float(v) if not np.isnan(v) else None for k, v in metrics.items()},
                "status": "success",
            }
            logger.info(f"Test — MAE={metrics['mae']:.2f}, RMSE={metrics['rmse']:.2f}")
        except Exception as exc:
            self.results["stages"]["testing"] = {"status": "failed", "error": str(exc)}
            raise

    def _stage_generate_report(self) -> None:
        logger.info("Stage 6: Saving model and report")
        try:
            model_path = self.model.save()

            results_dir = REPORTS_PATH / self.department / self.age_group
            results_dir.mkdir(parents=True, exist_ok=True)
            results_file = results_dir / "random_forest_results.json"
            with open(results_file, "w") as f:
                json.dump(self.results, f, indent=2, default=str)

            from pneumonia.visualization.persistence import save_predictions
            pred_csv = save_predictions(
                reports_dir=REPORTS_PATH,
                department=self.department,
                age_group=self.age_group,
                train=self.train,
                val=self.val,
                test=self.test,
                model_name="RandomForest",
                val_forecast=self.val_forecasts.get("RandomForest"),
                test_forecast=self.test_forecasts.get("RandomForest"),
            )

            self.results["stages"]["reporting"] = {
                "model_path":     str(model_path),
                "results_file":   str(results_file),
                "predictions_csv": str(pred_csv),
                "status": "success",
            }
            logger.info(f"Results saved to {results_file}")
        except Exception as exc:
            self.results["stages"]["reporting"] = {"status": "failed", "error": str(exc)}
            raise


def run_rf_for_all_departments(
    age_group: str = "under5",
    split_strategy: Optional[str] = None,
    rf_params: Optional[Dict] = None,
    lags: Optional[List[int]] = None,
    windows: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Run RandomForest pipeline for every available department."""
    logger.info(f"Running RandomForest for all departments ({age_group})")
    departments = get_available_departments()
    all_results: Dict[str, Any] = {}

    for dept in departments:
        logger.info(f"\n{'='*80}\nProcessing {dept}\n{'='*80}")
        try:
            pipeline = RandomForestPipeline(
                department=dept,
                age_group=age_group,
                split_strategy=split_strategy,
                rf_params=rf_params,
                lags=lags,
                windows=windows,
            )
            pipeline.run()
            print(pipeline.summary())
            all_results[dept] = {"status": "success", "results": pipeline.results}
        except Exception as exc:
            logger.error(f"Pipeline failed for {dept}: {exc}")
            all_results[dept] = {"status": "failed", "error": str(exc)}

    successes = sum(1 for r in all_results.values() if r["status"] == "success")
    logger.info(
        f"\n{'='*80}\nRF BATCH COMPLETE — {successes}/{len(departments)} succeeded\n{'='*80}"
    )
    return all_results
