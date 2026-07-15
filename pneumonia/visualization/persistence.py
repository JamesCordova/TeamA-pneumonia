"""
Prediction persistence — read/write per-model long-format prediction CSVs.

Each model gets its own file: {model_name}_predictions.csv
Columns: date, split, model, actual, predicted, horizon, department, age_group
  split   : 'train' | 'val' | 'test' | 'backtest'
  model   : model name (e.g. 'SARIMA', 'Naive') or 'actual' for train-only rows
  horizon : weeks-ahead lead time of the prediction; only set for 'backtest'
            rows (walk-forward), NaN for train/val/test rows

Having one file per model allows pipelines to run in parallel without
write conflicts.
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from pneumonia.utils import setup_logger

logger = setup_logger(__name__)

_COLUMNS = ["date", "split", "model", "actual", "predicted", "horizon", "department", "age_group"]
_CONFIG_KEYS = ("horizon", "step", "window_type", "train_size", "refit_every")


def _csv_path(reports_dir: Path, department: str, age_group: str, model_name: str) -> Path:
    out_dir = Path(reports_dir) / department / age_group
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{model_name}_predictions.csv"


def resolve_local_run_name(
    reports_dir: Path,
    department: str,
    age_group: str,
    run_name: str,
    model: str,
    config: dict,
    model_params: dict,
) -> str:
    """
    Fallback for pneumonia.evaluation.results_db.resolve_run_name() when
    results_unsa_ira is unreachable: auto-suffix run_name ('run_name(1)',
    'run_name(2)', ...) if it's already used locally by a *_walkforward_metrics.json
    with a different model/config/model_params, mirroring the database's collision
    logic. Best-effort — only looks at this machine's reports/ directory.
    """
    out_dir = Path(reports_dir) / department / age_group
    if not out_dir.exists():
        return run_name

    existing_names = set()
    collision = False
    for f in out_dir.glob("*_walkforward_metrics.json"):
        try:
            with open(f, encoding="utf-8-sig") as fh:
                j = json.load(fh)
        except Exception:
            continue
        name = j.get("run_name", j.get("model"))
        existing_names.add(name)
        if name != run_name:
            continue
        same_config = (
            j.get("model") == model
            and {k: j.get("config", {}).get(k) for k in _CONFIG_KEYS}
               == {k: config.get(k) for k in _CONFIG_KEYS}
            and j.get("model_params") == model_params
        )
        if not same_config:
            collision = True

    if not collision:
        return run_name

    n = 1
    while f"{run_name}({n})" in existing_names:
        n += 1
    resolved_name = f"{run_name}({n})"
    logger.info(
        f"run_name '{run_name}' already used locally by a different configuration — "
        f"saving this one as '{resolved_name}' (results_unsa_ira unreachable, "
        "best-effort local check)."
    )
    return resolved_name


def save_predictions(
    reports_dir: Path,
    department: str,
    age_group: str,
    train: pd.Series,
    val: pd.Series,
    test: pd.Series,
    model_name: str,
    val_forecast: Optional[np.ndarray],
    test_forecast: Optional[np.ndarray],
) -> Path:
    """
    Persist actual and predicted values for a single model.

    Each model is stored in its own file ({model_name}_predictions.csv).
    Existing train/val/test rows are replaced; backtest rows are preserved.

    Args:
        reports_dir:   Base reports directory.
        department:    Department name (uppercase).
        age_group:     'under5' or '60plus'.
        train:         Training split Series with DatetimeIndex.
        val:           Validation split Series.
        test:          Test split Series.
        model_name:    Model name (e.g. 'SARIMA', 'HoltWinters').
        val_forecast:  Predicted values for the val split (array).
        test_forecast: Predicted values for the test split (array).

    Returns:
        Path to the saved CSV file.
    """
    csv_path = _csv_path(reports_dir, department, age_group, model_name)

    rows = []

    for date, value in train.items():
        rows.append({
            "date": date, "split": "train", "model": "actual",
            "actual": float(value), "predicted": np.nan,
            "department": department, "age_group": age_group,
        })

    for split_name, series, arr in [("val", val, val_forecast), ("test", test, test_forecast)]:
        if len(series) == 0 or arr is None:
            continue
        for i, (date, actual) in enumerate(series.items()):
            rows.append({
                "date": date, "split": split_name, "model": model_name,
                "actual": float(actual),
                "predicted": float(arr[i]) if i < len(arr) else np.nan,
                "department": department, "age_group": age_group,
            })

    new_df = pd.DataFrame(rows, columns=_COLUMNS)

    if csv_path.exists():
        try:
            existing = pd.read_csv(csv_path, parse_dates=["date"])
            # Preserve only backtest rows; replace everything else
            keep = existing["split"] == "backtest"
            combined = pd.concat([existing[keep], new_df], ignore_index=True)
        except Exception as exc:
            logger.warning(f"Could not read existing CSV ({exc}) — overwriting.")
            combined = new_df
    else:
        combined = new_df

    combined.sort_values(["date", "split", "model"], inplace=True)
    combined.to_csv(csv_path, index=False)
    logger.info(f"Predictions saved: {csv_path} ({len(new_df)} rows)")
    return csv_path


def save_step_metrics(
    reports_dir: Path,
    department: str,
    age_group: str,
    model_name: str,
    step_results: list,
) -> Path:
    """
    Persist per-step (per walk-forward origin) metrics for diagnostic plots.

    One row per step: the forecast's anchor date, step index, number of
    horizon observations pooled into that step's metrics (n_obs — usually
    equal to `horizon`, but may be smaller for the last step(s) near the end
    of the series), and the metric values themselves (mae, rmse, me, mape,
    smape, mda, r2). The date column lets plots align steps across models
    even when they don't share the same number of steps.

    Args:
        reports_dir:  Base reports directory.
        department:   Department name (uppercase).
        age_group:    'under5' or '60plus'.
        model_name:   Model name (e.g. 'SARIMA', 'XGBoost').
        step_results: `WalkForwardValidator.run()["step_results"]` — a list of
                      per-step dicts with 'forecast_start', 'step', 'actuals',
                      'metrics'.

    Returns:
        Path to the saved CSV file.
    """
    out_dir = Path(reports_dir) / department / age_group
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{model_name}_step_metrics.csv"

    rows = []
    for step in step_results:
        row = {
            "date":  step["forecast_start"],
            "step":  step["step"],
            "n_obs": len(step["actuals"]),
        }
        row.update(step["metrics"])
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    logger.info(f"Step metrics saved: {csv_path} ({len(df)} rows)")
    return csv_path


def load_step_metrics(reports_dir: Path, department: str, age_group: str) -> dict:
    """
    Return {model_name: DataFrame(date, step, n_obs, mae, rmse, ...)} read back
    from the *_step_metrics.csv files written by `save_step_metrics`.
    """
    out_dir = Path(reports_dir) / department / age_group
    files   = sorted(out_dir.glob("*_step_metrics.csv"))
    if not files:
        raise FileNotFoundError(
            f"No step metrics found in {out_dir}.\n"
            "Run scripts/run_walkforward.py first."
        )
    data = {}
    for f in files:
        model = f.stem[: -len("_step_metrics")]
        data[model] = pd.read_csv(f, parse_dates=["date"])
    return data


def save_walkforward_predictions(
    reports_dir: Path,
    department: str,
    age_group: str,
    model_name: str,
    predictions_df: pd.DataFrame,
) -> Path:
    """
    Save walk-forward validation predictions with split='backtest'.

    Combines all pred_h* columns so every evaluated date gets one prediction
    (each date has exactly one non-NaN value across horizons with step==horizon),
    keeping track of which pred_h* column it came from in the 'horizon' column.
    Existing backtest rows for this model are replaced; train/val/test rows
    are preserved.

    Args:
        reports_dir:     Base reports directory.
        department:      Department name (uppercase).
        age_group:       'under5' or '60plus'.
        model_name:      Model name (e.g. 'Naive', 'SARIMA').
        predictions_df:  DataFrame from WalkForwardValidator.run() with columns
                         'actual' and 'pred_h1' … 'pred_hN'.

    Returns:
        Path to the saved CSV file.
    """
    csv_path = _csv_path(reports_dir, department, age_group, model_name)

    pred_cols = sorted(
        [c for c in predictions_df.columns if c.startswith("pred_h")],
        key=lambda c: int(c[6:]),
    )
    if not pred_cols:
        raise ValueError("predictions_df has no 'pred_h*' columns")

    # For each date, take the smallest-horizon non-NaN prediction across all
    # pred_h* columns (with step==horizon every date has exactly one filled
    # column), keeping the horizon it came from.
    molten = (
        predictions_df[pred_cols]
        .rename_axis("date")
        .reset_index()
        .melt(id_vars="date", value_vars=pred_cols, var_name="horizon_col", value_name="predicted")
        .dropna(subset=["predicted"])
    )
    molten["horizon"] = molten["horizon_col"].str[6:].astype(int)
    molten = molten.sort_values(["date", "horizon"]).drop_duplicates(subset="date", keep="first")

    rows = []
    for _, r in molten.iterrows():
        rows.append({
            "date": r["date"], "split": "backtest", "model": model_name,
            "actual": float(predictions_df.loc[r["date"], "actual"]),
            "predicted": float(r["predicted"]),
            "horizon": int(r["horizon"]),
            "department": department, "age_group": age_group,
        })
    new_df = pd.DataFrame(rows, columns=_COLUMNS)

    if csv_path.exists():
        try:
            existing = pd.read_csv(csv_path, parse_dates=["date"])
            # Preserve train/val/test rows; replace backtest
            keep = existing["split"] != "backtest"
            combined = pd.concat([existing[keep], new_df], ignore_index=True)
        except Exception as exc:
            logger.warning(f"Could not read existing CSV ({exc}) — overwriting.")
            combined = new_df
    else:
        combined = new_df

    combined.sort_values(["date", "split", "model"], inplace=True)
    combined.to_csv(csv_path, index=False)
    logger.info(f"Backtest predictions saved: {csv_path} ({len(new_df)} rows)")
    return csv_path
