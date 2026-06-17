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

    # RandomForest con hiperparámetros personalizados
    python scripts/run_walkforward.py --department AMAZONAS --model RandomForest \\
        --n_estimators 300 --max_depth 15
    python scripts/run_walkforward.py --department AMAZONAS --model RandomForest \\
        --lags 1 2 4 8 26 52 --windows 4 13 26 52

    # XGBoost con hiperparámetros personalizados
    python scripts/run_walkforward.py --department AMAZONAS --model XGBoost \\
        --n_estimators 500 --learning_rate 0.01 --max_depth 3 --colsample_bytree 0.7

    # SARIMA: orden fijo y más términos Fourier
    python scripts/run_walkforward.py --department AMAZONAS --model SARIMA \\
        --sarima_order 2 1 1 --n_fourier_terms 10
    # SARIMA clásico (sin Fourier)
    python scripts/run_walkforward.py --department AMAZONAS --model SARIMA --no_fourier

Available models: SARIMA, RandomForest, XGBoost, SeasonalNaive, Naive, HoltWinters
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from pneumonia.config import REPORTS_PATH
from pneumonia.evaluation.walkforward import WalkForwardValidator
from pneumonia.models.utils import (
    get_available_departments,
    get_departmental_data,
    validate_time_series,
)
from pneumonia.utils import setup_logger
from pneumonia.visualization.persistence import save_walkforward_predictions

logger = setup_logger(__name__)

_MODEL_REGISTRY = {
    "sarima":        ("pneumonia.models.sarima.model",             "SARIMAModel"),
    "randomforest":  ("pneumonia.models.ml.random_forest",         "RandomForestModel"),
    "xgboost":       ("pneumonia.models.ml.xgboost",               "XGBoostModel"),
    "seasonalnaive": ("pneumonia.models.baselines.seasonal_naive",  "SeasonalNaiveForecaster"),
    "naive":         ("pneumonia.models.baselines.naive",           "NaiveForecaster"),
    "holtwinters":   ("pneumonia.models.baselines.holt_winters",    "HoltWintersForecaster"),
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
    for k in ("mae", "rmse", "me", "r2", "mase", "smape", "mda"):
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
    extra_model_params: dict,
    start_year: Optional[int] = None,
) -> int:
    logger.info(f"Walk-forward: {department}/{age_group} model={model_name}")

    data = get_departmental_data(department, age_group=age_group, start_year=start_year)
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
    )
    print(f"Predictions saved -> {csv_path}")

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
    dept_group.add_argument("--department", "-d", type=str, nargs="+",
                            help="Department name(s) (e.g. AMAZONAS LIMA, or comma-separated: AMAZONAS,LIMA)")
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
    parser.add_argument("--start_year", type=int, default=None,
                        help="Drop all data before this year (e.g. 2007 for TACNA/TUMBES, "
                             "2008 for MOQUEGUA to skip early under-reporting).")
    # RandomForest / XGBoost hyperparameters (shared names where applicable)
    ml_group = parser.add_argument_group("ML model hyperparameters (RandomForest / XGBoost)")
    ml_group.add_argument("--n_estimators", type=int, default=None,
                          help="Number of trees (RF default: 100, XGB default: 300)")
    ml_group.add_argument("--max_depth", type=int, default=None,
                          help="Max tree depth (RF default: 10, XGB default: 4)")
    ml_group.add_argument("--min_samples_leaf", type=int, default=None,
                          help="[RF only] Min samples per leaf (default: 2)")
    ml_group.add_argument("--min_samples_split", type=int, default=None,
                          help="[RF only] Min samples to split a node (default: 5)")
    ml_group.add_argument("--max_features", type=str, default=None,
                          help="[RF only] Features per split: sqrt, log2 (default: sqrt)")
    ml_group.add_argument("--learning_rate", type=float, default=None,
                          help="[XGB only] Learning rate (default: 0.05)")
    ml_group.add_argument("--subsample", type=float, default=None,
                          help="[XGB only] Row subsample ratio (default: 0.9)")
    ml_group.add_argument("--colsample_bytree", type=float, default=None,
                          help="[XGB only] Feature subsample ratio per tree (default: 0.9)")
    ml_group.add_argument("--lags", type=int, nargs="+", default=None,
                          help="Lag periods as features, e.g. --lags 1 2 4 8 52 (default: 1 2 4 8 13)")
    ml_group.add_argument("--windows", type=int, nargs="+", default=None,
                          help="Rolling window sizes, e.g. --windows 4 13 26 52 (default: 4 13 26)")

    # SARIMA hyperparameters
    sarima_group = parser.add_argument_group("SARIMA hyperparameters")
    sarima_group.add_argument(
        "--sarima_order", type=int, nargs=3, default=None,
        metavar=("P", "D", "Q"),
        help="[SARIMA] Non-seasonal order (p d q), e.g. --sarima_order 2 1 1 "
             "(default: auto_arima or config fallback)",
    )
    sarima_group.add_argument(
        "--n_fourier_terms", type=int, default=None,
        help="[SARIMA] Fourier sin/cos pairs for seasonality (default: 6)",
    )
    fourier_group = sarima_group.add_mutually_exclusive_group()
    fourier_group.add_argument(
        "--no_fourier", action="store_true", default=False,
        help="[SARIMA] Use classical SAR/SMA instead of Fourier seasonality",
    )
    fourier_group.add_argument(
        "--fourier", action="store_true", default=False,
        help="[SARIMA] Force Fourier seasonality (overrides config if disabled)",
    )

    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose logging")
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger("pneumonia").setLevel(logging.DEBUG)

    departments = []
    if args.all:
        departments = get_available_departments()
    elif args.department:
        for d in args.department:
            departments.extend([x.strip().upper() for x in d.split(",") if x.strip()])

    # Build model-specific hyperparameter overrides from named CLI args
    model_key = args.model.lower().replace("_", "").replace("-", "")
    extra_model_params = {}

    if model_key == "randomforest":
        hp = {}
        for key in ("n_estimators", "max_depth", "min_samples_leaf",
                    "min_samples_split", "max_features"):
            val = getattr(args, key, None)
            if val is not None:
                hp[key] = val
        if hp:
            extra_model_params["rf_params"] = hp

    elif model_key == "xgboost":
        hp = {}
        for key in ("n_estimators", "max_depth", "learning_rate",
                    "subsample", "colsample_bytree"):
            val = getattr(args, key, None)
            if val is not None:
                hp[key] = val
        if hp:
            extra_model_params["xgb_params"] = hp

    elif model_key == "sarima":
        if args.sarima_order:
            extra_model_params["order"] = tuple(args.sarima_order)
        if args.n_fourier_terms is not None:
            extra_model_params["n_fourier_terms"] = args.n_fourier_terms
        if args.no_fourier:
            extra_model_params["use_fourier"] = False
        elif args.fourier:
            extra_model_params["use_fourier"] = True

    if args.lags:
        extra_model_params["lags"] = args.lags
    if args.windows:
        extra_model_params["windows"] = args.windows

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
                extra_model_params=extra_model_params,
                start_year=args.start_year,
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
