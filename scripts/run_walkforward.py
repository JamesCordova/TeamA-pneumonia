#!/usr/bin/env python
"""
Walk-forward (rolling-origin) backtesting for any trained model.

Runs WalkForwardValidator over the full series for a department/age-group and
saves horizon-1 predictions to predictions.csv with split='backtest', which
plot_forecasting.py can then render as a dotted trace.

Usage:
    python scripts/run_walkforward.py --department AMAZONAS --model SARIMA
    python scripts/run_walkforward.py --department LIMA --age_group 60plus --model XGBoost \\
        --horizon 8 --step 4 --window_type expanding --train_size 260
    python scripts/run_walkforward.py --all --model RandomForest --horizon 4

Available models: SARIMA, RandomForest, XGBoost, SeasonalNaive, Naive
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pneumonia.config import REPORTS_PATH
from pneumonia.evaluation.walkforward import WalkForwardValidator
from pneumonia.models.utils import (
    get_available_departments,
    get_departmental_data,
    handle_missing_values,
    validate_time_series,
)
from pneumonia.utils import setup_logger
from pneumonia.visualization.persistence import save_walkforward_predictions

logger = setup_logger(__name__)

_MODEL_REGISTRY = {
    "sarima":        ("pneumonia.models.sarima.model",       "SARIMAModel"),
    "randomforest":  ("pneumonia.models.ml.random_forest",   "RandomForestModel"),
    "xgboost":       ("pneumonia.models.ml.xgboost",         "XGBoostModel"),
    "seasonalnaive": ("pneumonia.models.baselines.seasonal_naive", "SeasonalNaiveForecaster"),
    "naive":         ("pneumonia.models.baselines.naive",    "NaiveForecaster"),
}


def _resolve_model_class(model_name: str):
    key = model_name.lower().replace("_", "").replace("-", "")
    if key not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available: {', '.join(k.title() for k in _MODEL_REGISTRY)}"
        )
    module_path, class_name = _MODEL_REGISTRY[key]
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


def _print_metrics(label: str, metrics: dict) -> None:
    if not metrics:
        print(f"  {label}: no metrics computed")
        return
    parts = []
    for k in ("mae", "rmse", "mape", "r2"):
        v = metrics.get(k)
        if v is not None and not (isinstance(v, float) and v != v):
            parts.append(f"{k.upper()}={v:.4f}")
    print(f"  {label}: {', '.join(parts)}")


def run_walkforward_for(
    department: str,
    age_group: str,
    model_name: str,
    train_size: int,
    horizon: int,
    step: int,
    window_type: str,
    refit_every: int,
    save_horizon: int,
    extra_model_params: dict,
) -> int:
    logger.info(f"Walk-forward: {department}/{age_group} model={model_name}")

    data = get_departmental_data(department, age_group=age_group)
    if data.isna().any():
        data = handle_missing_values(data, method="interpolate")
    validate_time_series(data)

    model_class = _resolve_model_class(model_name)
    model_params = {"department": department, "age_group": age_group, **extra_model_params}

    validator = WalkForwardValidator(
        model_class=model_class,
        model_params=model_params,
        initial_train_size=train_size,
        horizon=horizon,
        step=step,
        window_type=window_type,
        refit_every=refit_every,
    )

    results = validator.run(data)

    print(f"\n{'='*70}")
    print(f"Walk-forward results — {department}/{age_group}  model={model_name}")
    print(f"  steps={results['n_steps']}  horizon={horizon}  step={step}  window={window_type}")
    print(f"  train_size={results['config']['train_size']}  refit_every={refit_every}")
    for h in range(1, horizon + 1):
        m = results["metrics_by_horizon"].get(h, {})
        _print_metrics(f"h={h}", m)
    print(f"{'='*70}\n")

    csv_path = save_walkforward_predictions(
        reports_dir=REPORTS_PATH,
        department=department,
        age_group=age_group,
        model_name=model_name,
        predictions_df=results["predictions"],
        horizon=save_horizon,
    )
    print(f"Predictions saved → {csv_path}")

    # Save metrics JSON alongside other model JSONs
    out_dir = REPORTS_PATH / department / age_group
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = out_dir / f"{model_name.lower()}_walkforward_metrics.json"
    payload = {
        "department": department,
        "age_group": age_group,
        "model": model_name,
        "config": results["config"],
        "metrics_by_horizon": {
            str(h): m for h, m in results["metrics_by_horizon"].items()
        },
        "n_steps": results["n_steps"],
    }
    with open(metrics_file, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    logger.info(f"Metrics JSON saved: {metrics_file}")
    return 0


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Walk-forward backtesting for pneumonia forecasting models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_walkforward.py --department AMAZONAS --model SARIMA
  python scripts/run_walkforward.py --all --model RandomForest --horizon 4 --step 4
  python scripts/run_walkforward.py --department LIMA --age_group 60plus \\
      --model XGBoost --window_type expanding --train_size 260 --refit_every 4
        """,
    )

    dept_group = parser.add_mutually_exclusive_group(required=True)
    dept_group.add_argument("--department", "-d", type=str,
                            help="Department name (e.g. AMAZONAS)")
    dept_group.add_argument("--all", "-a", action="store_true",
                            help="Run for all departments")

    parser.add_argument("--age_group", "-g", type=str,
                        choices=["under5", "60plus"], default="under5",
                        help="Age group (default: under5)")
    parser.add_argument("--model", "-m", type=str, required=True,
                        help="Model to evaluate: SARIMA, RandomForest, XGBoost, SeasonalNaive, Naive")

    # Walk-forward parameters
    parser.add_argument("--train_size", type=int, default=520,
                        help="Initial training window in weeks (default: 520 ≈ 10 years)")
    parser.add_argument("--horizon", type=int, default=4,
                        help="Forecast horizon in weeks per origin (default: 4)")
    parser.add_argument("--step", type=int, default=4,
                        help="Weeks to advance the origin between evaluations (default: 4)")
    parser.add_argument("--window_type", type=str, default="sliding",
                        choices=["sliding", "expanding"],
                        help="Window type: 'sliding' (fixed) or 'expanding' (default: sliding)")
    parser.add_argument("--refit_every", type=int, default=1,
                        help="Re-train every N steps; 0=fit once only (default: 1). "
                             "For SARIMA use 13 or 52 to keep runtime tractable.")
    parser.add_argument("--save_horizon", type=int, default=1,
                        help="Which horizon step to persist in predictions.csv (default: 1)")

    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose logging")
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger("pneumonia").setLevel(logging.DEBUG)

    departments = (
        get_available_departments() if args.all
        else [args.department.upper()]
    )

    failed = []
    for dept in departments:
        try:
            run_walkforward_for(
                department=dept,
                age_group=args.age_group,
                model_name=args.model,
                train_size=args.train_size,
                horizon=args.horizon,
                step=args.step,
                window_type=args.window_type,
                refit_every=args.refit_every,
                save_horizon=args.save_horizon,
                extra_model_params={},
            )
        except Exception as exc:
            logger.error(f"Failed for {dept}: {exc}")
            failed.append(dept)

    if failed:
        print(f"\nFailed departments: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
