"""
Prediction persistence — read/write the long-format predictions CSV.

Columns: date, split, model, actual, predicted, department, age_group
  split : 'train' | 'val' | 'test' | 'backtest'
  model : model name (e.g. 'SARIMA', 'Naive') or 'actual' for train-only rows
"""

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from pneumonia.utils import setup_logger

logger = setup_logger(__name__)

_COLUMNS = ["date", "split", "model", "actual", "predicted", "department", "age_group"]


def save_predictions(
    reports_dir: Path,
    department: str,
    age_group: str,
    train: pd.Series,
    val: pd.Series,
    test: pd.Series,
    model_forecasts: Dict[str, Dict[str, Optional[np.ndarray]]],
) -> Path:
    """
    Persist actual and predicted values to a long-format CSV.

    Existing train/val/test rows for the given models are replaced.
    Backtest rows (split='backtest') are never touched by this function.

    Args:
        reports_dir:     Base reports directory.
        department:      Department name (uppercase).
        age_group:       'under5' or '60plus'.
        train:           Training split Series with DatetimeIndex.
        val:             Validation split Series.
        test:            Test split Series.
        model_forecasts: {model_name: {'val': array_or_None, 'test': array_or_None}}

    Returns:
        Path to the saved CSV file.
    """
    out_dir = Path(reports_dir) / department / age_group
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "predictions.csv"

    rows = []

    for date, value in train.items():
        rows.append({
            "date": date, "split": "train", "model": "actual",
            "actual": float(value), "predicted": np.nan,
            "department": department, "age_group": age_group,
        })

    for split_name, series, arr_key in [("val", val, "val"), ("test", test, "test")]:
        if len(series) == 0:
            continue
        for model_name, forecasts in model_forecasts.items():
            arr = forecasts.get(arr_key)
            if arr is None:
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
            being_replaced = set(new_df["model"].unique())
            keep = ~(
                (existing["model"].isin(being_replaced))
                & (existing["split"].isin(["train", "val", "test"]))
                & (existing["department"] == department)
                & (existing["age_group"] == age_group)
            )
            combined = pd.concat([existing[keep], new_df], ignore_index=True)
        except Exception as exc:
            logger.warning(f"Could not read existing CSV ({exc}) — overwriting.")
            combined = new_df
    else:
        combined = new_df

    combined.sort_values(
        ["department", "age_group", "date", "split", "model"], inplace=True
    )
    combined.to_csv(csv_path, index=False)
    logger.info(f"Predictions CSV updated: {csv_path} ({len(new_df)} rows written)")
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
    Only replaces existing backtest rows for this model; train/val/test rows
    are never touched.

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
    out_dir = Path(reports_dir) / department / age_group
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "predictions.csv"

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
            keep = ~(
                (existing["split"] == "backtest")
                & (existing["model"] == model_name)
                & (existing["department"] == department)
                & (existing["age_group"] == age_group)
            )
            combined = pd.concat([existing[keep], new_df], ignore_index=True)
        except Exception as exc:
            logger.warning(f"Could not read existing CSV ({exc}) — overwriting.")
            combined = new_df
    else:
        combined = new_df

    combined.sort_values(
        ["department", "age_group", "date", "split", "model"], inplace=True
    )
    combined.to_csv(csv_path, index=False)
    logger.info(
        f"Backtest predictions saved for {model_name}: {csv_path} ({len(new_df)} rows)"
    )
    return csv_path
