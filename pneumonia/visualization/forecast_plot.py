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
from typing import Dict, List, Optional

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

    # --- upsert: drop stale train/val/test rows for the models we are rewriting ---
    # Backtest rows (split='backtest') are managed exclusively by save_walkforward_predictions
    # and must not be clobbered by a regular pipeline re-run.
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


def plot_forecasts(
    department: str,
    age_group: str,
    reports_dir: Path,
    models: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
    show: bool = False,
    figsize: tuple = (15, 5),
    last_n_train_weeks: int = 104,
    show_backtest: bool = True,
    backtest_only: bool = False,
    year: Optional[int] = None,
) -> Optional[Path]:
    """
    Generate a forecast comparison figure from predictions.csv.

    Plots:
      - Trailing train actuals as a solid black line
      - Val + test actuals as a dashed black line
      - One coloured line per model over the val+test range
      - Vertical markers for the val and test split boundaries

    Args:
        department:          Department name (uppercase).
        age_group:           'under5' or '60plus'.
        reports_dir:         Base reports directory (REPORTS_PATH).
        models:              List of model names to include (e.g. ['SARIMA', 'XGBoost']).
                             None means all models in the CSV.
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

    # --- determine x-axis bounds early so train actuals can be clipped correctly ---
    # backtest_only  → x-axis covers the full backtest range; val/test model lines hidden
    # show_backtest  → backtest trace shown on top of the regular val/test view
    # neither        → classic view: 2 years before val through test end
    backtest_df = df[df["split"] == "backtest"].copy() if (show_backtest or backtest_only) else pd.DataFrame(columns=df.columns)
    forecast_df = df[df["split"].isin(["val", "test"])].copy()

    val_start = df[df["split"] == "val"]["date"].min()
    test_end   = df[df["split"] == "test"]["date"].max()

    if (show_backtest or backtest_only) and not backtest_df.empty:
        plot_min = backtest_df["date"].min()
    elif pd.notna(val_start):
        plot_min = pd.Timestamp(f"{val_start.year - 2}-01-01")
    else:
        plot_min = df["date"].min()

    plot_max = test_end if pd.notna(test_end) else df["date"].max()

    if year is not None:
        plot_min = pd.Timestamp(f"{year}-01-01")
        plot_max = pd.Timestamp(f"{year}-12-31")

    fig, ax = plt.subplots(figsize=figsize)

    # --- training actuals clipped to the visible range ---
    train_df = (
        df[(df["split"] == "train") & (df["model"] == "actual") & (df["date"] >= plot_min)]
        .sort_values("date")
    )
    if not train_df.empty:
        ax.plot(
            train_df["date"], train_df["actual"],
            color="black", lw=1.3, label="Actual (train)",
        )

    palette = plt.cm.tab10.colors
    model_names: List[str] = []

    # --- val + test actuals: always shown as reference regardless of mode ---
    if not forecast_df.empty:
        actuals = forecast_df.drop_duplicates("date").sort_values("date")
        ax.plot(
            actuals["date"], actuals["actual"],
            color="black", lw=1.5, ls="--", label="Actual (val/test)",
        )

    # --- val + test model predictions: hidden in backtest_only mode ---
    if not backtest_only:
        available = sorted(m for m in forecast_df["model"].unique() if m != "actual")
        model_names = [m for m in available if m in models] if models else available
        if models:
            missing = [m for m in models if m not in available]
            if missing:
                logger.warning(f"Models not found in CSV: {missing}. Available: {available}")

        for idx, model_name in enumerate(model_names):
            mdf = forecast_df[forecast_df["model"] == model_name].sort_values("date")
            ax.plot(
                mdf["date"], mdf["predicted"],
                color=palette[idx % len(palette)],
                lw=1.5, ls="--", alpha=0.85,
                label=model_name,
            )

    # --- backtest trace (walk-forward, solid line) ---
    backtest_models = [m for m in backtest_df["model"].unique() if (models is None or m in models)]
    for model_name in sorted(backtest_models):
        color_idx = model_names.index(model_name) if model_name in model_names else len(model_names)
        bdf = backtest_df[backtest_df["model"] == model_name].sort_values("date")
        ax.plot(
            bdf["date"], bdf["predicted"],
            color=palette[color_idx % len(palette)],
            lw=1.5, alpha=0.85,
            label=f"{model_name} (backtest)",
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

    # --- apply pre-computed x-limits ---
    if pd.notna(plot_min) and pd.notna(plot_max):
        ax.set_xlim(plot_min, plot_max)

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


def save_walkforward_predictions(
    reports_dir: Path,
    department: str,
    age_group: str,
    model_name: str,
    predictions_df: pd.DataFrame,
    horizon: int = 1,
) -> Path:
    """
    Save walk-forward validation predictions to predictions.csv for visualization.

    Existing rows for the given model name are replaced (upsert).

    Args:
        reports_dir:     Base reports directory (REPORTS_PATH).
        department:      Department name (uppercase).
        age_group:       'under5' or '60plus'.
        model_name:      Name of the model to register (e.g. 'SARIMA_WF').
        predictions_df:  The predictions DataFrame returned by WalkForwardValidator.run(),
                         which must contain 'actual' and 'pred_h{horizon}' columns.
        horizon:         The forecast horizon step to persist (default: 1).

    Returns:
        Path to the saved predictions.csv file.
    """
    out_dir = Path(reports_dir) / department / age_group
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "predictions.csv"

    pred_col = f"pred_h{horizon}"
    if pred_col not in predictions_df.columns:
        raise ValueError(f"Column '{pred_col}' not found in predictions_df")

    # Only keep rows where the requested horizon was actually forecast.
    # With step > 1, many rows have NaN for pred_h1 — skip them for a clean line.
    valid = predictions_df[predictions_df[pred_col].notna()]
    rows = []
    for date, row in valid.iterrows():
        rows.append({
            "date": date,
            "split": "backtest",
            "model": model_name,
            "actual": float(row["actual"]),
            "predicted": float(row[pred_col]),
            "department": department,
            "age_group": age_group,
        })
    new_df = pd.DataFrame(rows, columns=_COLUMNS)

    # --- upsert: drop only the backtest rows for this model (preserve val/test) ---
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
    logger.info(f"Walk-Forward predictions saved for {model_name} to {csv_path} ({len(new_df)} rows written)")
    return csv_path

