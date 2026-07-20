#!/usr/bin/env python
"""
Hyperparameter tuning via walk-forward validation.

Every trial is delegated to pneumonia.pipelines.walkforward_runner's
run_walkforward_for(), so each one gets the exact same identical-configuration
dedup, run_name resolution, and local + results_unsa_ira persistence as a
normal walk-forward run — trials that turn out functionally identical (e.g.
two SARIMA combos that only differ in seasonal_order while use_fourier=True)
are detected and reused instead of retrained.

Only model_params is searched. horizon/step/window_type/train_size/refit_every
(the walk-forward protocol) are fixed CLI inputs, not part of the search space
— see scripts/compare_models.py's --same_config for why: letting the search
move them would let it "pick" an easier evaluation window instead of a
genuinely better model, and makes trials incomparable with each other.

Three search strategies, all driven by the same per-model *_SEARCH_RANGES
dict (a plain {param: [values]} mapping defined next to each model, e.g.
pneumonia/models/sarima/config.py's SARIMA_SEARCH_RANGES):
  grid    — exhaustive ParameterGrid over the full range.
  random  — ParameterSampler, --n_iter samples.
  optuna  — TPE-based Bayesian search (same ranges, sampled via
            trial.suggest_categorical), --n_iter trials. Worth it once the
            range is too large to cover with grid/random cheaply, since each
            trial is a full walk-forward run — not a quick single fit.

Each trial's --metric is aggregated per --mode (mirrors scripts/compare_models.py
--mode): 'horizon' (default, mean across evaluated horizons or just
--opt_horizon), 'macroaverage' (mean of each step's own metric), or
'microaverage' (metric computed once on every step's pooled raw predictions)
— see _score() for the full explanation of what each one weighs differently.

Usage:
    python scripts/tune_models.py --department LIMA --model XGBoost
    python scripts/tune_models.py --department LIMA --model SARIMA --search_method grid
    python scripts/tune_models.py --department LIMA AMAZONAS --model SARIMA \\
        --search_method optuna --n_iter 30
    python scripts/tune_models.py --department LIMA --model SARIMA --metric rmse \\
        --train_size 260 --window_type expanding
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from sklearn.model_selection import ParameterGrid, ParameterSampler

from pneumonia.config import RANDOM_SEED, REPORTS_PATH
from pneumonia.evaluation.metrics import compute_all_metrics
from pneumonia.models.baselines.holt_winters import HOLTWINTERS_SEARCH_RANGES
from pneumonia.models.ml.config import RANDOM_FOREST_SEARCH_RANGES, XGBOOST_SEARCH_RANGES
from pneumonia.models.prophet.config import PROPHET_SEARCH_RANGES
from pneumonia.models.rnn.config import RNN_SEARCH_RANGES
from pneumonia.models.sarima.config import SARIMA_SEARCH_RANGES
from pneumonia.pipelines.walkforward_runner import run_walkforward_for
from pneumonia.utils import setup_logger

logger = setup_logger(__name__)

# {model CLI name: (search ranges, param adapter)}. The adapter folds one
# sampled combo into the extra_model_params shape run_walkforward_for()
# expects — nested under rf_params/xgb_params for the two ML models (their
# constructors take one dict), flat for everyone else. Mirrors exactly how
# scripts/run_walkforward.py's own CLI wires each model's hyperparameters.
# Naive/SeasonalNaive have no hyperparameters, so they have no entry here.
_MODEL_SPECS = {
    "RandomForest": (RANDOM_FOREST_SEARCH_RANGES, lambda c: {"rf_params": c}),
    "XGBoost":      (XGBOOST_SEARCH_RANGES,       lambda c: {"xgb_params": c}),
    "SARIMA":       (SARIMA_SEARCH_RANGES,        dict),
    "Prophet":      (PROPHET_SEARCH_RANGES,       dict),
    "HoltWinters":  (HOLTWINTERS_SEARCH_RANGES,   dict),
    "LSTM":         (RNN_SEARCH_RANGES,           dict),
    "GRU":          (RNN_SEARCH_RANGES,           dict),
}

_LOWER_IS_BETTER = {"mae", "rmse", "smape", "mape"}


def _mean(values: list) -> float:
    values = [v for v in values if v == v]  # drop NaN
    return sum(values) / len(values) if values else float("nan")


def _score(result: dict, metric: str, mode: str, opt_horizon: int = None) -> float:
    """
    Reduce one trial's result to a single number so trials can be compared.

    mode="horizon"      — mean of `metric` across every evaluated horizon
                           (result["metrics_by_horizon"]), or just `opt_horizon`
                           if given. Each horizon weighs equally regardless of
                           how many observations landed in it.
    mode="macroaverage"  — mean of `metric` across each walk-forward step's
                           own metrics (result["step_results"][i]["metrics"]).
                           Each step weighs equally.
    mode="microaverage"  — `metric` computed once on every step's raw
                           actual/predicted values pooled together. Each
                           individual prediction weighs equally (so
                           over-represented horizons/steps dominate more).
    """
    if mode == "horizon":
        metrics_by_horizon = result["metrics_by_horizon"]
        if opt_horizon is not None:
            return metrics_by_horizon.get(opt_horizon, {}).get(metric, float("nan"))
        return _mean([m.get(metric, float("nan")) for m in metrics_by_horizon.values()])

    if mode == "macroaverage":
        return _mean([
            s["metrics"].get(metric, float("nan"))
            for s in result["step_results"] if s.get("metrics")
        ])

    if mode == "microaverage":
        actuals, predicted = [], []
        for s in result["step_results"]:
            actuals.extend(s["actuals"])
            predicted.extend(s["predictions"])
        if not actuals:
            return float("nan")
        m = compute_all_metrics(np.asarray(actuals), np.asarray(predicted), warn_on_nan=False)
        return m.get(metric, float("nan"))

    raise ValueError(f"Unknown mode: {mode}")


def _is_better(a: float, b: float, metric: str) -> bool:
    """True if score `a` beats score `b` for `metric`. NaN never wins."""
    if a != a:
        return False
    if b != b:
        return True
    return a < b if metric in _LOWER_IS_BETTER else a > b


def tune_model(
    department: str,
    age_group: str,
    model_name: str,
    search_method: str,
    n_iter: int,
    metric: str,
    mode: str,
    opt_horizon: int,
    walkforward_kwargs: dict,
) -> dict:
    """
    Search model_params for `model_name` on department/age_group, scoring
    each trial with `metric` aggregated per `mode` (see _score()). Returns
    {"score", "run_name", "combo", "metrics_by_horizon"} for the best trial.

    Every trial (not just the best) is written to
    reports/{department}/{age_group}/tune_{model}_{search_method}_{timestamp}.json
    after each one completes — so a search that gets interrupted partway
    through doesn't lose the trials it already ran. The timestamp is fixed
    for the whole call, so it's one file per tune_model() run, never
    overwritten by a later one.
    """
    department = department.upper()
    search_ranges, adapt = _MODEL_SPECS[model_name]

    out_dir = REPORTS_PATH / department / age_group
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / (
        f"tune_{model_name.lower()}_{search_method}_{datetime.now():%Y%m%d_%H%M%S}.json"
    )

    best = {"score": float("nan"), "run_name": None, "combo": None, "metrics_by_horizon": None}
    trials = []

    def _save_summary() -> None:
        payload = {
            "department": department, "age_group": age_group, "model": model_name,
            "search_method": search_method, "metric": metric, "mode": mode,
            "opt_horizon": opt_horizon,
            "walkforward_config": walkforward_kwargs,
            "best": {k: v for k, v in best.items() if k != "metrics_by_horizon"},
            "trials": trials,
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)

    def consider(combo: dict, idx: int, total) -> float:
        run_name = f"{model_name}_tune{idx}"
        logger.info(f"[{idx}/{total}] {combo}")
        result = run_walkforward_for(
            department=department, age_group=age_group, model_name=model_name,
            extra_model_params=adapt(combo), run_name=run_name, **walkforward_kwargs,
        )
        score = _score(result, metric, mode, opt_horizon)
        logger.info(f"  {metric}={score:.4f} (run_name={result['run_name']})")
        trials.append({"run_name": result["run_name"], "combo": combo, "score": score})
        if _is_better(score, best["score"], metric):
            best.update(score=score, run_name=result["run_name"], combo=combo,
                        metrics_by_horizon=result["metrics_by_horizon"])
        _save_summary()
        return score

    if search_method in ("grid", "random"):
        if search_method == "grid":
            combos = list(ParameterGrid(search_ranges))
            logger.info(f"Grid search: {len(combos)} combinations")
        else:
            total_unique = len(ParameterGrid(search_ranges))
            n = min(n_iter, total_unique)
            combos = list(ParameterSampler(search_ranges, n_iter=n, random_state=RANDOM_SEED))
            logger.info(f"Random search: sampling {len(combos)} of {total_unique} combinations")

        for idx, combo in enumerate(combos, 1):
            try:
                consider(combo, idx, len(combos))
            except Exception as exc:
                logger.warning(f"  [{idx}/{len(combos)}] trial failed ({combo}): {exc}")

    elif search_method == "optuna":
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial: "optuna.Trial") -> float:
            combo = {k: trial.suggest_categorical(k, v) for k, v in search_ranges.items()}
            return consider(combo, trial.number + 1, n_iter)

        direction = "minimize" if metric in _LOWER_IS_BETTER else "maximize"
        study = optuna.create_study(
            direction=direction, sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED)
        )
        logger.info(f"Optuna ({direction}): {n_iter} trials")
        study.optimize(objective, n_trials=n_iter, catch=(Exception,))

    else:
        raise ValueError(f"Unknown search_method: {search_method}")

    if best["combo"] is None:
        raise RuntimeError("No trial completed successfully.")

    logger.info(f"Best {metric}={best['score']:.4f} — run_name={best['run_name']} — {best['combo']}")
    logger.info(f"Summary saved: {summary_path}")
    best["summary_path"] = summary_path
    return best


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning via walk-forward validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/tune_models.py --department LIMA --model XGBoost
  python scripts/tune_models.py --department LIMA --model SARIMA --search_method grid
  python scripts/tune_models.py --department LIMA AMAZONAS --model SARIMA \\
      --search_method optuna --n_iter 30
  python scripts/tune_models.py --department LIMA --model SARIMA --metric rmse \\
      --train_size 260 --window_type expanding
        """,
    )

    parser.add_argument("--department", "-d", type=str, nargs="+", required=True,
                        help="Department name(s) (e.g. AMAZONAS LIMA, or comma-separated)")
    parser.add_argument("--model", "-m", type=str, required=True,
                        choices=sorted(_MODEL_SPECS),
                        help="Model to tune (only models with a defined search space)")
    parser.add_argument("--age_group", "-g", type=str,
                        choices=["under5", "60plus"], default="under5",
                        help="Age group (default: under5)")
    parser.add_argument("--search_method", type=str,
                        choices=["grid", "random", "optuna"], default="random",
                        help="Search strategy (default: random)")
    parser.add_argument("--n_iter", type=int, default=20,
                        help="Trials for 'random'/'optuna' (default: 20). Ignored for 'grid'.")
    parser.add_argument("--metric", type=str, default="mae",
                        choices=["mae", "rmse", "smape", "mape", "r2", "mda"],
                        help="Metric to optimize (default: mae)")
    parser.add_argument("--mode", type=str,
                        choices=["horizon", "macroaverage", "microaverage"], default="horizon",
                        help="How to aggregate --metric across a trial's walk-forward run "
                             "(default: horizon), matching scripts/compare_models.py --mode: "
                             "'horizon' — mean across evaluated horizons (or just --opt_horizon). "
                             "'macroaverage' — mean of each step's own metric (each step weighs "
                             "equally). 'microaverage' — metric computed once on every step's "
                             "raw predictions pooled together (each prediction weighs equally).")
    parser.add_argument("--opt_horizon", type=int, default=None,
                        help="[--mode horizon] Score on this single horizon instead of "
                             "averaging across all evaluated horizons (default: average all).")

    # Fixed walk-forward protocol — intentionally NOT part of the search space,
    # see module docstring.
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--step", type=int, default=4)
    parser.add_argument("--window_type", choices=["sliding", "expanding"], default="sliding")
    parser.add_argument("--train_size", type=int, default=520)
    parser.add_argument("--refit_every", type=int, default=1)
    parser.add_argument("--start_year", type=int, default=None,
                        help="Start year to truncate early sub-reported data (e.g. 2008 for Moquegua)")

    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress non-essential output")

    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    if args.quiet:
        logging.getLogger("pneumonia").setLevel(logging.ERROR)
    elif args.verbose:
        logging.getLogger("pneumonia").setLevel(logging.DEBUG)

    if args.opt_horizon is not None and args.mode != "horizon":
        parser.error("--opt_horizon only applies to --mode horizon")

    walkforward_kwargs = dict(
        train_size=args.train_size, horizon=args.horizon, step=args.step,
        window_type=args.window_type, refit_every=args.refit_every,
        start_year=args.start_year,
    )

    departments = []
    for d in args.department:
        departments.extend([x.strip().upper() for x in d.split(",") if x.strip()])

    failed = []
    for dept in departments:
        try:
            best = tune_model(
                department=dept, age_group=args.age_group, model_name=args.model,
                search_method=args.search_method, n_iter=args.n_iter,
                metric=args.metric, mode=args.mode, opt_horizon=args.opt_horizon,
                walkforward_kwargs=walkforward_kwargs,
            )
        except Exception as exc:
            logger.error(f"Tuning failed for {dept}: {exc}")
            failed.append(dept)
            continue

        print(f"\n{'=' * 70}")
        print(f"Best {args.model} — {dept}/{args.age_group}  "
              f"({args.search_method}, {args.mode}, {args.metric})")
        print(f"{'=' * 70}")
        print(f"  {args.metric} = {best['score']:.4f}")
        print(f"  run_name    = {best['run_name']}")
        print(f"  params      = {best['combo']}")
        print(f"  summary     = {best['summary_path']}")
        print(f"{'=' * 70}\n")

    if failed:
        logger.error(f"Failed departments: {failed}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
