"""
Prediction persistence — read/write per-model long-format prediction CSVs.

Each model gets its own file: {model_name}_predictions.csv
Columns: date, split, model, actual, predicted, department, age_group
  split : 'train' | 'val' | 'test' | 'backtest'
  model : model name (e.g. 'SARIMA', 'Naive') or 'actual' for train-only rows

Having one file per model allows pipelines to run in parallel without
write conflicts.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from pneumonia.utils import setup_logger

logger = setup_logger(__name__)

_COLUMNS = ["date", "split", "model", "actual", "predicted", "department", "age_group"]


def _csv_path(reports_dir: Path, department: str, age_group: str, model_name: str) -> Path:
    out_dir = Path(reports_dir) / department / age_group
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{model_name}_predictions.csv"


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
    (each date has exactly one non-NaN value across horizons with step==horizon).
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

    # For each date take the first non-NaN prediction across all horizons.
    # With step==horizon every date has exactly one filled column.
    blended = predictions_df[pred_cols].stack().groupby(level=0).first()
    blended = blended.reindex(predictions_df.index)

    rows = []
    for date, pred_val in blended.items():
        if pd.isna(pred_val):
            continue
        rows.append({
            "date": date, "split": "backtest", "model": model_name,
            "actual": float(predictions_df.loc[date, "actual"]),
            "predicted": float(pred_val),
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
