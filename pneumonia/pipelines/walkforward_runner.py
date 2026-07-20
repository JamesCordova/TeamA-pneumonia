"""
Generic walk-forward (rolling-origin) evaluation runner — any model in
_MODEL_REGISTRY, one department/age_group at a time.

Unlike the per-model classic pipelines in this package (SARIMAPipeline,
ProphetPipeline, ...), walk-forward evaluation doesn't need model-specific
orchestration: WalkForwardValidator already takes any BaseForecaster
subclass, so one function covers every model. This is what
scripts/run_walkforward.py and scripts/tune_models.py both call — the
scripts stay thin CLI layers (argparse + calling this + printing results).
"""

import importlib
import json
from typing import Optional

from pneumonia.config import REPORTS_PATH
from pneumonia.evaluation.results_db import find_existing_run, resolve_run_name, save_walkforward_run
from pneumonia.evaluation.results_db_query import reconstruct_results_from_run
from pneumonia.evaluation.walkforward import WalkForwardValidator
from pneumonia.models.utils import get_departmental_data, validate_time_series
from pneumonia.utils import setup_logger
from pneumonia.visualization.persistence import (
    resolve_local_run_name,
    save_step_metrics,
    save_walkforward_predictions,
)

logger = setup_logger(__name__)

MODEL_REGISTRY = {
    "sarima":        ("pneumonia.models.sarima.model",             "SARIMAModel"),
    "randomforest":  ("pneumonia.models.ml.random_forest",         "RandomForestModel"),
    "xgboost":       ("pneumonia.models.ml.xgboost",               "XGBoostModel"),
    "lstm":          ("pneumonia.models.rnn.lstm",                   "LSTMModel"),
    "gru":           ("pneumonia.models.rnn.gru",                    "GRUModel"),
    "seasonalnaive": ("pneumonia.models.baselines.seasonal_naive",  "SeasonalNaiveForecaster"),
    "naive":         ("pneumonia.models.baselines.naive",           "NaiveForecaster"),
    "holtwinters":   ("pneumonia.models.baselines.holt_winters",    "HoltWintersForecaster"),
    "prophet":       ("pneumonia.models.prophet.model",            "ProphetModel"),
}


def resolve_model_class(model_name: str):
    key = model_name.lower().replace("_", "").replace("-", "")
    if key not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available: {', '.join(k.title() for k in MODEL_REGISTRY)}"
        )
    module_path, class_name = MODEL_REGISTRY[key]
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


def _print_metrics(label: str, metrics: dict) -> None:
    if not metrics:
        print(f"  {label}: no metrics computed")
        return
    parts = []
    for k in ("mae", "rmse", "me", "r2", "smape", "mda"):
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
    run_name: Optional[str] = None,
) -> dict:
    """
    Run (or reuse, if the config already exists) one walk-forward evaluation
    and persist it locally + to results_unsa_ira.

    Returns {"run_name": the resolved name actually used for local/DB storage
    (may differ from the `run_name` argument — see resolve_run_name()),
    "metrics_by_horizon": {horizon_int: {metric: value}}} — the latter is what
    scripts/tune_models.py reads to score each trial.
    """
    run_name = run_name or model_name
    logger.info(f"Walk-forward: {department}/{age_group} model={model_name} run={run_name}")

    data = get_departmental_data(department, age_group=age_group, start_year=start_year)
    validate_time_series(data)

    model_class = resolve_model_class(model_name)
    model_params = {"department": department, "age_group": age_group, **extra_model_params}

    # Skip retraining if results_unsa_ira already has an identical configuration
    # (department, age_group, model, horizon, step, window_type, train_size,
    # refit_every, model_params) saved from a previous run: reconstruct the
    # local files from its stored predictions instead. Reliable here because
    # get_params() on a freshly-constructed (unfitted) instance already
    # matches what it would be after fit() for every model this can invoke —
    # including SARIMA, whose only exception (auto_arima re-resolving (p,d,q)
    # post-fit) is never triggered from the CLI callers (no --use_auto_arima
    # flag or fit_kwargs wiring there).
    config_probe = {
        "horizon": horizon, "step": step, "window_type": window_type,
        "train_size": train_size, "refit_every": refit_every,
    }
    reused_run_id = None
    probe_params = model_params
    try:
        probe = model_class(**model_params)
        probe_params = probe.get_params()
        reused_run_id = find_existing_run(
            department=department, age_group=age_group, model=model_name,
            config=config_probe, model_params=probe_params,
        )
        # Preview the name results_unsa_ira would resolve this run to (it may
        # auto-suffix if run_name is already used locally by a different
        # config), so the local files below are named consistently with it.
        run_name = resolve_run_name(
            department=department, age_group=age_group, model=model_name,
            run_name=run_name, config=config_probe, model_params=probe_params,
        )
    except Exception as exc:
        logger.warning(f"Could not check results_unsa_ira for an existing run: {exc}")
        run_name = resolve_local_run_name(
            reports_dir=REPORTS_PATH, department=department, age_group=age_group,
            run_name=run_name, model=model_name, config=config_probe,
            model_params=probe_params,
        )

    if reused_run_id is not None:
        print(f"Identical configuration already exists as run_id={reused_run_id} "
              f"— reconstructing local files without retraining.")
        results = reconstruct_results_from_run(reused_run_id)
    else:
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
    print(f"Walk-forward results — {department}/{age_group}  run={run_name} (model={model_name})")
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
        model_name=run_name,
        predictions_df=results["predictions"],
    )
    print(f"Predictions saved -> {csv_path}")

    step_csv_path = save_step_metrics(
        reports_dir=REPORTS_PATH,
        department=department,
        age_group=age_group,
        model_name=run_name,
        step_results=results["step_results"],
    )
    print(f"Step metrics saved -> {step_csv_path}")

    # Save metrics JSON alongside other model JSONs
    out_dir = REPORTS_PATH / department / age_group
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = out_dir / f"{run_name.lower()}_walkforward_metrics.json"
    payload = {
        "department": department,
        "age_group": age_group,
        "model": model_name,
        "run_name": run_name,
        "config": results["config"],
        "model_params": results["model_params"],
        "metrics_by_horizon": {
            str(h): m for h, m in results["metrics_by_horizon"].items()
        },
        "n_steps": results["n_steps"],
    }
    with open(metrics_file, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    logger.info(f"Metrics JSON saved: {metrics_file}")

    # Additive: also persist to results_unsa_ira (db/migrations/0001_walkforward_results).
    # Never blocks the file-based outputs above if the database is unreachable.
    # Skipped when reused_run_id is set — that data already exists there.
    if reused_run_id is not None:
        print(f"Database: reusing existing run_id={reused_run_id} (no new insert)")
    else:
        try:
            run_id, _ = save_walkforward_run(
                department=department,
                age_group=age_group,
                model=model_name,
                run_name=run_name,
                config=results["config"],
                model_params=results["model_params"],
                n_steps=results["n_steps"],
                step_results=results["step_results"],
            )
            print(f"Saved to database: run_id={run_id}")
        except Exception as exc:
            logger.warning(f"Could not save results to database: {exc}")

    return {"run_name": run_name, "metrics_by_horizon": results["metrics_by_horizon"]}
