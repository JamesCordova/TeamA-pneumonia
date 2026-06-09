"""
Forecast visualization and prediction persistence.

save_predictions() writes (or upserts) a long-format CSV at:
  reports/{DEPT}/{AGE_GROUP}/predictions.csv

Columns:
  date, split, model, actual, predicted, department, age_group

  split : 'train' | 'val' | 'test'
  model : model name (e.g. 'SARIMA', 'Naive') or 'actual' for train-only rows

plot_forecasts() reads that CSV and produces a multi-model comparison figure
saved at reports/{DEPT}/{AGE_GROUP}/forecast_plot.png.

Running both SARIMA and Baseline pipelines writes to the same CSV so the plot
shows every trained model side-by-side.
"""

from pathlib import Path
from typing import Dict, Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
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

    Existing rows for the given models are replaced (upsert by model+split).
    Training actuals are stored once under model='actual' so the plot has
    historical context without repeating the same values for every model.

    Args:
        reports_dir:     Base reports directory (REPORTS_PATH).
        department:      Department name (uppercase).
        age_group:       'under5' or '60plus'.
        train:           Training split as a pd.Series with DatetimeIndex.
        val:             Validation split Series.
        test:            Test split Series.
        model_forecasts: {model_name: {'val': array_or_None, 'test': array_or_None}}
                         Arrays must align index-wise with val and test respectively.

    Returns:
        Path to the saved CSV file.
    """
    out_dir = Path(reports_dir) / department / age_group
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "predictions.csv"

    rows = []

    # --- training actuals stored once under model='actual' ---
    for date, value in train.items():
        rows.append({
            "date": date,
            "split": "train",
            "model": "actual",
            "actual": float(value),
            "predicted": np.nan,
            "department": department,
            "age_group": age_group,
        })

    # --- val and test: actual value + each model's prediction ---
    for split_name, series, arr_key in [("val", val, "val"), ("test", test, "test")]:
        if len(series) == 0:
            continue
        for model_name, forecasts in model_forecasts.items():
            arr = forecasts.get(arr_key)
            if arr is None:
                continue
            for i, (date, actual) in enumerate(series.items()):
                rows.append({
                    "date": date,
                    "split": split_name,
                    "model": model_name,
                    "actual": float(actual),
                    "predicted": float(arr[i]) if i < len(arr) else np.nan,
                    "department": department,
                    "age_group": age_group,
                })

    new_df = pd.DataFrame(rows, columns=_COLUMNS)

    # --- upsert: drop stale rows for the models we are rewriting ---
    if csv_path.exists():
        try:
            existing = pd.read_csv(csv_path, parse_dates=["date"])
            being_replaced = set(new_df["model"].unique())
            keep = ~(
                (existing["model"].isin(being_replaced))
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


def plot_forecasts(
    department: str,
    age_group: str,
    reports_dir: Path,
    save_path: Optional[Path] = None,
    show: bool = False,
    figsize: tuple = (15, 5),
    last_n_train_weeks: int = 104,
) -> Optional[Path]:
    """
    Generate a multi-model forecast comparison figure from predictions.csv.

    Plots:
      - Trailing train actuals (last_n_train_weeks) as a solid black line
      - Val + test actuals as a dashed black line
      - One coloured line per model over the val+test range
      - Vertical markers for the val and test split boundaries

    Args:
        department:          Department name (uppercase).
        age_group:           'under5' or '60plus'.
        reports_dir:         Base reports directory (REPORTS_PATH).
        save_path:           Output path. Defaults to
                             reports_dir/department/age_group/forecast_plot.png
        show:                Call plt.show() (useful for notebooks).
        figsize:             Matplotlib figure size.
        last_n_train_weeks:  Trailing training weeks to show for context.

    Returns:
        Path to the saved figure, or None if no data was found.
    """
    csv_path = Path(reports_dir) / department / age_group / "predictions.csv"
    if not csv_path.exists():
        logger.warning(f"No predictions CSV at {csv_path} — run a pipeline first.")
        return None

    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = df[(df["department"] == department) & (df["age_group"] == age_group)].copy()

    if df.empty:
        logger.warning(f"No rows for {department}/{age_group} in {csv_path}.")
        return None

    fig, ax = plt.subplots(figsize=figsize)

    # --- training actuals (trailing context) ---
    train_df = (
        df[(df["split"] == "train") & (df["model"] == "actual")]
        .sort_values("date")
        .tail(last_n_train_weeks)
    )
    if not train_df.empty:
        ax.plot(
            train_df["date"], train_df["actual"],
            color="black", lw=1.3, label="Actual (train)",
        )

    # --- val + test actuals (one unique actual per date) ---
    forecast_df = df[df["split"].isin(["val", "test"])].copy()
    if not forecast_df.empty:
        actuals = forecast_df.drop_duplicates("date").sort_values("date")
        ax.plot(
            actuals["date"], actuals["actual"],
            color="black", lw=1.5, ls="--", label="Actual (val/test)",
        )

    # --- model predictions ---
    model_names = sorted(m for m in forecast_df["model"].unique() if m != "actual")
    palette = plt.cm.tab10.colors

    for idx, model_name in enumerate(model_names):
        mdf = forecast_df[forecast_df["model"] == model_name].sort_values("date")
        ax.plot(
            mdf["date"], mdf["predicted"],
            color=palette[idx % len(palette)],
            lw=1.5,
            alpha=0.85,
            label=model_name,
        )

    # --- split boundary markers ---
    ax.relim()
    ax.autoscale_view()
    ymin, ymax = ax.get_ylim()
    label_y = ymax - (ymax - ymin) * 0.06

    for split_name, color in [("val", "steelblue"), ("test", "darkorange")]:
        boundary = df[df["split"] == split_name]["date"].min()
        if pd.notna(boundary):
            ax.axvline(boundary, color=color, ls=":", lw=1.2, alpha=0.7)
            ax.text(boundary, label_y, f" {split_name}", color=color, fontsize=8)

    # --- formatting ---
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    fig.autofmt_xdate()

    age_label = "Under 5" if age_group == "under5" else "60+"
    ax.set_title(
        f"{department} — {age_label} — Forecast comparison", fontsize=12, pad=10
    )
    ax.set_xlabel("Week")
    ax.set_ylabel("Pneumonia cases")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.8)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()

    if save_path is None:
        save_path = Path(reports_dir) / department / age_group / "forecast_plot.png"

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Forecast plot saved: {save_path}")

    if show:
        plt.show()

    return Path(save_path)
