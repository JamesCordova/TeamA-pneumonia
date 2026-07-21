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

import pandas as pd

from pneumonia.config import COVID_EXCLUDE_END, COVID_EXCLUDE_PERIODS, COVID_EXCLUDE_START, REPORTS_PATH
from pneumonia.evaluation.results_db import (
    append_walkforward_predictions,
    find_existing_run,
    get_run_coverage,
    resolve_run_name,
    save_walkforward_run,
)
from pneumonia.evaluation.results_db_query import reconstruct_results_from_run
from pneumonia.evaluation.metrics import recompute_metrics
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


def _n_steps_total(n: int, train_size: int, step: int) -> int:
    """How many walk-forward steps a continuous run(step_range=None) over a
    series of length n would produce, given train_size/step — the same
    termination condition as WalkForwardValidator.run()'s loop
    (`while train_end < n`), computed without running anything. Used to know
    where the 'holdout' range ends (the series' end) and whether a
    holdout_start leaves any steps at all."""
    if train_size >= n:
        return 0
    return (n - train_size - 1) // step + 1


def _holdout_boundary_step(y: pd.Series, train_size: int, step: int, holdout_start) -> int:
    """The smallest step_idx whose evaluation window starts at-or-after
    holdout_start — i.e. steps [0, k) are entirely before the holdout and
    steps [k, ...) fall at-or-after it. Pure date arithmetic (no model
    involved), so 'search' and 'holdout' pieces can be defined consistently
    regardless of which is computed first.

    holdout_start need not be a date present in y's index — it's compared
    positionally against how many of y's weeks fall strictly before it.
    """
    holdout_pos = int((y.index < pd.Timestamp(holdout_start)).sum())
    if holdout_pos <= train_size:
        return 0
    return -(-(holdout_pos - train_size) // step)  # ceil division, step > 0


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
    exclude_covid: bool = True,
    holdout_start: Optional[str] = None,
    mode: str = "full",
) -> dict:
    """
    Run (or reuse, if the config already exists) one walk-forward evaluation
    and persist it locally + to results_unsa_ira.

    Returns {"run_name": the resolved name actually used for local/DB storage
    (may differ from the `run_name` argument — see resolve_run_name()),
    "metrics_by_horizon": {horizon_int: {metric: value}}, "step_results": list
    of per-step dicts with 'metrics' (that step's own compute_all_metrics) and
    raw 'actuals'/'predictions'}. scripts/tune_models.py reads these to score
    each trial — metrics_by_horizon for --mode horizon, step_results for
    --mode macroaverage (mean of step metrics) or microaverage (metrics
    pooled from every step's raw actuals/predictions).

    exclude_covid: if True, dates in [config.COVID_EXCLUDE_START,
    COVID_EXCLUDE_END] are dropped from metrics_by_horizon and every step's
    own 'metrics' — not from training, not from the raw predictions saved
    locally/to the DB. The model still trains and forecasts through those
    dates (keeps the weekly series contiguous); they just don't count toward
    any reported score. See pneumonia.evaluation.metrics.keep_mask.

    holdout_start / mode: guard against model-selection bias when the same
    walk-forward span is used both to pick a configuration (hyperparameter
    search) and to report its performance. When holdout_start is given,
    dates before it are the "search" range (safe to select a model against)
    and dates at/after it are the protected "holdout" range (only ever
    scored once, after a configuration is already chosen — see
    scripts/tune_models.py). mode picks which range THIS call needs:
      'search'  — only compute/return the pre-holdout range. Physically
                  never forecasts holdout dates, so a losing hyperparameter
                  trial never spends time on dates its score can't use.
      'holdout' — only compute/return the holdout range, training on all
                  history up to it (including the search range) but scoring
                  only the holdout dates.
      'full'    — both ranges (holdout_start still required, so the run is
                  tagged/identified the same way 'search'/'holdout' runs of
                  the same hyperparameters are — see below).
    holdout_start=None (default) skips all of this and behaves exactly as
    before: one continuous, unprotected run over the whole series.

    Every (department, age_group, model, horizon, step, window_type,
    train_size, refit_every, model_params, holdout_start) combination maps
    to at most one row in results_unsa_ira.walkforward_runs — 'search',
    'holdout', and 'full' requests for the SAME hyperparameters all resolve
    to that same row, which accumulates whichever step ranges have been
    requested of it so far (in any order). A request only ever computes the
    steps it needs that aren't already stored; already-covered steps are
    read back, never recomputed. See pneumonia.evaluation.results_db's
    get_run_coverage()/append_walkforward_predictions() and
    WalkForwardValidator.run()'s step_range parameter (the bridge-fit that
    keeps a step range's refit schedule identical to a continuous run's).
    """
    if mode not in ("full", "search", "holdout"):
        raise ValueError(f"mode must be 'full', 'search', or 'holdout', got {mode!r}")
    if mode != "full" and holdout_start is None:
        raise ValueError(f"mode={mode!r} requires holdout_start")

    run_name = run_name or model_name
    logger.info(
        f"Walk-forward: {department}/{age_group} model={model_name} run={run_name} "
        f"mode={mode}" + (f" holdout_start={holdout_start}" if holdout_start else "")
    )

    exclude_periods = COVID_EXCLUDE_PERIODS if exclude_covid else None

    data = get_departmental_data(department, age_group=age_group, start_year=start_year)
    validate_time_series(data)

    model_class = resolve_model_class(model_name)
    model_params = {"department": department, "age_group": age_group, **extra_model_params}

    # Skip retraining if results_unsa_ira already has an identical configuration
    # (department, age_group, model, horizon, step, window_type, train_size,
    # refit_every, model_params, holdout_start) saved from a previous run:
    # reconstruct the local files from its stored predictions instead of
    # recomputing anything. Reliable here because get_params() on a
    # freshly-constructed (unfitted) instance already matches what it would
    # be after fit() for every model this can invoke — including SARIMA,
    # whose only exception (auto_arima re-resolving (p,d,q) post-fit) is
    # never triggered from the CLI callers (no --use_auto_arima flag or
    # fit_kwargs wiring there).
    config_probe = {
        "horizon": horizon, "step": step, "window_type": window_type,
        "train_size": train_size, "refit_every": refit_every,
    }
    existing_run_id = None
    db_available = True
    probe_params = model_params
    try:
        probe = model_class(**model_params)
        probe_params = probe.get_params()
        existing_run_id = find_existing_run(
            department=department, age_group=age_group, model=model_name,
            config=config_probe, model_params=probe_params, holdout_start=holdout_start,
        )
        # Preview the name results_unsa_ira would resolve this run to (it may
        # auto-suffix if run_name is already used locally by a different
        # config), so the local files below are named consistently with it.
        run_name = resolve_run_name(
            department=department, age_group=age_group, model=model_name,
            run_name=run_name, config=config_probe, model_params=probe_params,
            holdout_start=holdout_start,
        )
    except Exception as exc:
        logger.warning(f"Could not check results_unsa_ira for an existing run: {exc}")
        db_available = False
        run_name = resolve_local_run_name(
            reports_dir=REPORTS_PATH, department=department, age_group=age_group,
            run_name=run_name, model=model_name, config=config_probe,
            model_params=probe_params,
        )

    validator = WalkForwardValidator(
        model_class=model_class,
        model_params=model_params,
        initial_train_size=train_size,
        horizon=horizon,
        step=step,
        window_type=window_type,
        refit_every=refit_every,
    )

    if holdout_start is None:
        # Legacy, unprotected path — unchanged from before this feature.
        if existing_run_id is not None:
            print(f"Identical configuration already exists as run_id={existing_run_id} "
                  f"— reconstructing local files without retraining.")
            results = reconstruct_results_from_run(existing_run_id)
        else:
            results = validator.run(data)
        run_id_for_save = existing_run_id
    else:
        train_size_int = validator.resolve_train_size(data)
        n = len(data)
        n_steps_total = _n_steps_total(n, train_size_int, step)
        k = _holdout_boundary_step(data, train_size_int, step, holdout_start)

        if mode == "search" and k == 0:
            raise ValueError(
                f"holdout_start={holdout_start} leaves no steps before the holdout for "
                f"{department}/{age_group} (train_size={train_size} weeks already reaches "
                f"it) — nothing to search over. Lower --train_size or move --holdout_start later."
            )
        if mode == "holdout" and k >= n_steps_total:
            raise ValueError(
                f"holdout_start={holdout_start} is at/after the end of available data for "
                f"{department}/{age_group} ({data.index[-1].date()}) — no holdout steps to evaluate."
            )

        needed_ranges = []
        if mode in ("search", "full") and k > 0:
            needed_ranges.append((0, k))
        if mode in ("holdout", "full") and k < n_steps_total:
            needed_ranges.append((k, None))

        def _range_covered(coverage, rng):
            if coverage is None or coverage["min_step_idx"] is None:
                return False
            start, end = rng
            end_incl = (n_steps_total - 1) if end is None else (end - 1)
            return coverage["min_step_idx"] <= start and coverage["max_step_idx"] >= end_incl

        run_id_for_save = existing_run_id
        coverage = get_run_coverage(existing_run_id) if (existing_run_id and db_available) else None

        for rng in needed_ranges:
            if run_id_for_save is not None and _range_covered(coverage, rng):
                logger.info(f"step_range={rng} already covered by run_id={run_id_for_save} — reusing.")
                continue
            logger.info(f"Computing step_range={rng} for {run_name} (mode={mode})...")
            piece = validator.run(data, step_range=rng)
            if not db_available:
                continue
            try:
                if run_id_for_save is None:
                    run_id_for_save, run_name = save_walkforward_run(
                        department=department, age_group=age_group, model=model_name,
                        run_name=run_name, config=piece["config"], model_params=piece["model_params"],
                        n_steps=piece["n_steps"], step_results=piece["step_results"],
                        holdout_start=holdout_start,
                    )
                else:
                    append_walkforward_predictions(
                        run_id=run_id_for_save, step_results=piece["step_results"],
                        global_n_steps=piece["n_steps"],
                    )
                coverage = get_run_coverage(run_id_for_save)
            except Exception as exc:
                logger.warning(f"Could not persist step_range={rng} to database: {exc}")
                db_available = False

        if run_id_for_save is not None and db_available:
            step_idx_range = None if mode == "full" else (
                (0, k) if mode == "search" else (k, None)
            )
            results = reconstruct_results_from_run(run_id_for_save, step_idx_range=step_idx_range)
        else:
            # DB unreachable (or unavailable from the start) — nothing to
            # reuse/persist incrementally; compute everything mode needs in
            # one local-only pass.
            combined_range = (
                (0, None) if mode == "full" else
                ((0, k) if mode == "search" else (k, None))
            )
            results = validator.run(data, step_range=combined_range)

    # Date-based scoring policy applied after the fact, never inside the
    # validator/reconstruction — see recompute_metrics()'s docstring.
    results = recompute_metrics(results, exclude_periods)

    print(f"\n{'='*70}")
    print(f"Walk-forward results — {department}/{age_group}  run={run_name} (model={model_name})")
    print(f"  steps={results['n_steps']}  horizon={horizon}  step={step}  window={window_type}")
    print(f"  train_size={results['config']['train_size']}  refit_every={refit_every}")
    if exclude_covid:
        print(f"  scoring excludes: {COVID_EXCLUDE_START} to {COVID_EXCLUDE_END} (COVID period)")
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
    # When holdout_start is set, persistence (create-or-extend) already
    # happened incrementally above, one step_range at a time — nothing left
    # to do here. Only the legacy holdout_start=None path still needs a
    # single save at the end, matching its behavior from before this feature.
    if holdout_start is not None:
        if run_id_for_save is not None:
            print(f"Database: run_id={run_id_for_save} (created/extended as needed above)")
        else:
            print("Database: unavailable — results were computed locally only.")
    elif existing_run_id is not None:
        print(f"Database: reusing existing run_id={existing_run_id} (no new insert)")
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

    return {
        "run_name": run_name,
        "metrics_by_horizon": results["metrics_by_horizon"],
        "step_results": results["step_results"],
    }
