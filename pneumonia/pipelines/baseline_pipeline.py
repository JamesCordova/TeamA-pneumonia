"""
Baseline Forecasting Pipeline

Trains and evaluates both naive baselines (Naive and SeasonalNaive) for a
given department / age group using the same train/val/test split as the
SARIMA pipeline, so results are directly comparable.

Stages:
  1. Data loading & validation
  2. Temporal split
  3. Fit both baseline models on train set
  4. Validate on val set
  5. Evaluate on test set
  6. Save results report
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import json
from datetime import datetime

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
from pneumonia.models.baselines import NaiveForecaster, SeasonalNaiveForecaster, HoltWintersForecaster
from pneumonia.evaluation.metrics import compute_all_metrics
from pneumonia.utils import setup_logger

logger = setup_logger(__name__)

BASELINE_MODELS = {
    "Naive":        NaiveForecaster,
    "SeasonalNaive": SeasonalNaiveForecaster,
    "HoltWinters":  HoltWintersForecaster,
}


class BaselinePipeline:
    """
    End-to-end baseline training and evaluation pipeline.

    Usage:
        pipeline = BaselinePipeline(department="AMAZONAS", age_group="under5")
        results = pipeline.run()
    """

    def __init__(
        self,
        department: str,
        age_group: str = "under5",
        split_strategy: Optional[str] = None,
        season_length: int = 52,
        start_year: Optional[int] = None,
    ):
        self.department = department.upper()
        self.age_group = age_group.lower()
        self.split_strategy = split_strategy or TEMPORAL_SPLIT_STRATEGY
        self.season_length = season_length
        self.start_year = start_year

        self.data = None
        self.train = None
        self.val = None
        self.test = None
        self.models: Dict[str, Any] = {}
        self.val_forecasts: Dict[str, np.ndarray] = {}
        self.test_forecasts: Dict[str, np.ndarray] = {}

        self.results = {
            "department": self.department,
            "age_group": self.age_group,
            "timestamp": datetime.now().isoformat(),
            "stages": {},
        }

        logger.info(f"BaselinePipeline initialised: {self.department} ({self.age_group})")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        """Execute all pipeline stages and return the results dict."""
        try:
            logger.info(f"Starting baseline pipeline for {self.department}/{self.age_group}")
            self._stage_load_data()
            self._stage_temporal_split()
            self._stage_train_models()
            self._stage_validate_models()
            self._stage_test_models()
            self._stage_generate_report()
            logger.info("Baseline pipeline completed successfully")
            return self.results
        except Exception as exc:
            logger.error(f"Pipeline failed: {exc}")
            self.results["error"] = str(exc)
            raise

    def summary(self) -> str:
        lines = [
            f"\n{'='*80}",
            f"BASELINE PIPELINE RESULTS — {self.department}/{self.age_group}",
            f"{'='*80}",
        ]
        for stage_name in ("validation", "testing"):
            stage = self.results["stages"].get(stage_name, {})
            if stage.get("status") != "success":
                continue
            lines.append(f"\n{stage_name.title()} metrics:")
            for model_name, metrics in stage.get("models", {}).items():
                lines.append(f"  {model_name}:")
                for metric, value in metrics.items():
                    if value is not None:
                        lines.append(f"    {metric}: {value:.4f}")
        lines.append(f"{'='*80}\n")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Stages
    # ------------------------------------------------------------------

    def _stage_load_data(self) -> None:
        logger.info(f"Stage 1: Loading data for {self.department} ({self.age_group})")
        try:
            self.data = get_departmental_data(
                self.department,
                age_group=self.age_group,
                start_year=self.start_year
            )

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
                "train": {"n_weeks": len(self.train)},
                "val": {"n_weeks": len(self.val)},
                "test": {"n_weeks": len(self.test)},
                "status": "success",
            }
            logger.info(
                f"Split — train: {len(self.train)}, val: {len(self.val)}, test: {len(self.test)}"
            )
        except Exception as exc:
            self.results["stages"]["temporal_split"] = {"status": "failed", "error": str(exc)}
            raise

    def _stage_train_models(self) -> None:
        logger.info("Stage 3: Fitting baseline models")
        fitted_info = {}
        try:
            for name, ModelClass in BASELINE_MODELS.items():
                kwargs = {}
                if name == "SeasonalNaive":
                    kwargs["season_length"] = self.season_length

                model = ModelClass(
                    department=self.department,
                    age_group=self.age_group,
                    **kwargs,
                )
                model.fit(self.train)
                self.models[name] = model
                fitted_info[name] = {
                    "params": model.metadata,
                    "train_size": len(self.train),
                }
                logger.info(f"  {name}: fitted on {len(self.train)} obs")

            self.results["stages"]["training"] = {
                "models": fitted_info,
                "status": "success",
            }
        except Exception as exc:
            self.results["stages"]["training"] = {"status": "failed", "error": str(exc)}
            raise

    def _stage_validate_models(self) -> None:
        logger.info(f"Stage 4: Validating on val set ({len(self.val)} weeks)")
        if len(self.val) == 0:
            self.results["stages"]["validation"] = {"status": "skipped", "reason": "empty_val_set"}
            return
        try:
            model_metrics: Dict[str, Dict[str, float]] = {}
            for name, model in self.models.items():
                forecast = model.predict(self.train, steps=len(self.val))
                self.val_forecasts[name] = forecast
                metrics = compute_all_metrics(self.val.values, forecast)
                model_metrics[name] = {
                    k: float(v) if not np.isnan(v) else None
                    for k, v in metrics.items()
                }
                logger.info(
                    f"  {name} val — MAE={metrics['mae']:.2f}, RMSE={metrics['rmse']:.2f}"
                )

            self.results["stages"]["validation"] = {
                "n_val_obs": len(self.val),
                "models": model_metrics,
                "status": "success",
            }
        except Exception as exc:
            self.results["stages"]["validation"] = {"status": "failed", "error": str(exc)}
            raise

    def _stage_test_models(self) -> None:
        logger.info(f"Stage 5: Testing on test set ({len(self.test)} weeks)")
        if len(self.test) == 0:
            self.results["stages"]["testing"] = {"status": "skipped", "reason": "empty_test_set"}
            return
        try:
            model_metrics: Dict[str, Dict[str, float]] = {}
            for name, model in self.models.items():
                forecast = model.predict(
                    pd.concat([self.train, self.val]), steps=len(self.test)
                )
                self.test_forecasts[name] = forecast
                metrics = compute_all_metrics(self.test.values, forecast)
                model_metrics[name] = {
                    k: float(v) if not np.isnan(v) else None
                    for k, v in metrics.items()
                }
                logger.info(
                    f"  {name} test — MAE={metrics['mae']:.2f}, RMSE={metrics['rmse']:.2f}"
                )

            self.results["stages"]["testing"] = {
                "n_test_obs": len(self.test),
                "models": model_metrics,
                "status": "success",
            }
        except Exception as exc:
            self.results["stages"]["testing"] = {"status": "failed", "error": str(exc)}
            raise

    def _stage_generate_report(self) -> None:
        logger.info("Stage 6: Saving models and report")
        try:
            # Save models
            saved_paths = {}
            for name, model in self.models.items():
                path = model.save()
                saved_paths[name] = str(path)

            # Save results JSON
            results_dir = REPORTS_PATH / self.department / self.age_group
            results_dir.mkdir(parents=True, exist_ok=True)
            results_file = results_dir / "baselines_results.json"
            with open(results_file, "w") as f:
                json.dump(self.results, f, indent=2, default=str)

            from pneumonia.visualization.persistence import save_predictions
            pred_csvs = []
            for name in self.models:
                pred_csvs.append(save_predictions(
                    reports_dir=REPORTS_PATH,
                    department=self.department,
                    age_group=self.age_group,
                    train=self.train,
                    val=self.val,
                    test=self.test,
                    model_name=name,
                    val_forecast=self.val_forecasts.get(name),
                    test_forecast=self.test_forecasts.get(name),
                ))

            self.results["stages"]["reporting"] = {
                "model_paths": saved_paths,
                "results_file": str(results_file),
                "predictions_csvs": [str(p) for p in pred_csvs],
                "status": "success",
            }
            logger.info(f"Results saved to {results_file}")
        except Exception as exc:
            self.results["stages"]["reporting"] = {"status": "failed", "error": str(exc)}
            raise


def run_baselines_for_all_departments(
    age_group: str = "under5",
    split_strategy: Optional[str] = None,
    start_year: Optional[int] = None,
) -> Dict[str, Dict[str, Any]]:
    """Run baseline pipeline for every available department."""
    logger.info(f"Running baselines for all departments ({age_group})")
    departments = get_available_departments()
    all_results: Dict[str, Any] = {}

    for dept in departments:
        logger.info(f"\n{'='*80}\nProcessing {dept}\n{'='*80}")
        try:
            pipeline = BaselinePipeline(
                department=dept,
                age_group=age_group,
                split_strategy=split_strategy,
                start_year=start_year,
            )
            results = pipeline.run()
            print(pipeline.summary())
            all_results[dept] = {"status": "success", "results": results}
        except Exception as exc:
            logger.error(f"Baseline pipeline failed for {dept}: {exc}")
            all_results[dept] = {"status": "failed", "error": str(exc)}

    successes = sum(1 for r in all_results.values() if r["status"] == "success")
    logger.info(
        f"\n{'='*80}\nBASELINE BATCH COMPLETE — "
        f"{successes}/{len(departments)} succeeded\n{'='*80}"
    )
    return all_results
