"""
Walk-forward backtest plot.

Shows:
  - Training actuals from the earliest backtest origin (solid black)
  - Val + test actuals as a reference line (dashed black)
  - One solid coloured line per model from the backtest predictions

Optional shaded mode replaces the per-model lines with a model-range band
(min to max across the selected backtest models), while still showing the
actual series on top.

The x-axis extends to the earliest backtest date so the full rolling-origin
history is visible, unlike the classic plot which only shows the last few years.

Saved to: reports/{DEPT}/{AGE_GROUP}/forecast_backtest[_YYYY][_shaded].png
"""

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd

from pneumonia.evaluation.results_db_query import read_predictions_from_db
from pneumonia.utils import setup_logger
from pneumonia.visualization._utils import (
    clip_axes,
    configure_date_axis,
    get_output_path,
    read_predictions,
    save_figure,
)

logger = setup_logger(__name__)


def plot_backtest(
    department: str,
    age_group: str,
    reports_dir: Path,
    models: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
    show: bool = False,
    figsize: tuple = (15, 5),
    year: Optional[int] = None,
    shaded: bool = False,
    source: str = "local",
) -> Optional[Path]:
    """
    Generate the walk-forward backtest figure.

    Args:
        department:   Department name (uppercase).
        age_group:    'under5' or '60plus'.
        reports_dir:  Base reports directory.
        models:       Model names to include; None = all backtest models.
        save_path:    Output path; defaults to backtest_plot.png.
        show:         Call plt.show() after saving.
        figsize:      Matplotlib figure size.
        year:         Restrict x-axis to a single calendar year.
        shaded:       If True, draw a shaded min/max band across the selected
                  models instead of plotting each model as a separate line.
        source:       'local' (default) — *_predictions.csv under reports_dir.
                  'db' — results_unsa_ira, latest run per run_name. DB-sourced
                  data only has backtest rows (no classic train/val/test), so
                  the pre-backtest actual line segment won't be drawn.

    Returns:
        Path to the saved PNG, or None if no backtest data was found.
    """
    df = (
        read_predictions_from_db(department, age_group) if source == "db"
        else read_predictions(reports_dir, department, age_group)
    )
    if df is None:
        return None

    backtest_df = df[df["split"] == "backtest"].copy()
    if backtest_df.empty:
        logger.warning(
            f"No backtest data for {department}/{age_group}. "
            "Run scripts/run_walkforward.py first."
        )
        return None

    test_end = df[df["split"] == "test"]["date"].max()

    if year is not None:
        plot_min = pd.Timestamp(f"{year}-01-01")
        plot_max = pd.Timestamp(f"{year}-12-31")
    else:
        plot_min = backtest_df["date"].min()
        plot_max = test_end if pd.notna(test_end) else backtest_df["date"].max()

    fig, ax = plt.subplots(figsize=figsize)
    palette = plt.cm.tab10.colors

    # --- single continuous actual line ---
    # Sources (any combination may exist depending on which pipelines were run):
    #   1. train split rows (model='actual')  — from classic pipeline
    #   2. val/test split rows                — from classic pipeline
    #   3. backtest split rows                — always present after walk-forward
    train_actuals = (
        df[(df["split"] == "train") & (df["model"] == "actual") & (df["date"] >= plot_min)]
        [["date", "actual"]]
    )
    forecast_df = df[df["split"].isin(["val", "test"])].copy()
    val_test_actuals = forecast_df.drop_duplicates("date")[["date", "actual"]]
    backtest_actuals = backtest_df.drop_duplicates("date")[["date", "actual"]]
    all_actuals = (
        pd.concat([train_actuals, val_test_actuals, backtest_actuals])
        .drop_duplicates("date")
        .sort_values("date")
    )
    if not all_actuals.empty:
        ax.plot(
            all_actuals["date"],
            all_actuals["actual"],
            color="black",
            lw=1.3,
            label="Actual",
            zorder=4,
        )

    # --- backtest model predictions (solid coloured lines) ---
    available = sorted(backtest_df["model"].unique())
    bt_models = [m for m in available if m in models] if models else available
    if models:
        missing = [m for m in models if m not in available]
        if missing:
            logger.warning(
                f"Backtest models not found: {missing}. Available: {available}"
            )

    model_frames = {
        name: backtest_df[backtest_df["model"] == name].sort_values("date")
        for name in bt_models
    }

    if shaded:
        series_by_model = []
        for name, bdf in model_frames.items():
            series_by_model.append(
                bdf.set_index("date")["predicted"].rename(name)
            )
        if series_by_model:
            pred_matrix = pd.concat(series_by_model, axis=1).sort_index()
            range_lower = pred_matrix.min(axis=1, skipna=True)
            range_upper = pred_matrix.max(axis=1, skipna=True)
            range_mean = pred_matrix.mean(axis=1, skipna=True)
            ax.fill_between(
                pred_matrix.index,
                range_lower.to_numpy(),
                range_upper.to_numpy(),
                color=palette[0],
                alpha=0.18,
                zorder=1,
                label="Model range",
            )
            ax.plot(
                pred_matrix.index,
                range_mean.to_numpy(),
                color=palette[0],
                lw=1.4,
                alpha=0.9,
                zorder=2,
                label="Mean prediction",
            )
    else:
        for idx, name in enumerate(bt_models):
            bdf = model_frames[name]
            ax.plot(
                bdf["date"],
                bdf["predicted"],
                color=palette[idx % len(palette)],
                lw=1.5,
                alpha=0.85,
                label=f"{name} (backtest)",
            )

    configure_date_axis(ax, plot_min, plot_max)

    age_label = "Under 5" if age_group == "under5" else "60+"
    ax.set_title(
        f"{department} — {age_label} — Walk-forward backtest", fontsize=12, pad=10
    )
    ax.set_ylabel("Pneumonia cases")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.8)
    ax.grid(True, alpha=0.25)

    clip_axes(ax, df, plot_min, plot_max)

    if save_path is None:
        suffix = f"_{year}" if year is not None else ""
        suffix += "_shaded" if shaded else ""
        filename = f"forecast_backtest{suffix}.png"
        base_dir = Path(reports_dir) / department / age_group
        save_path = get_output_path(base_dir, filename, output_type="figures")

    path = save_figure(fig, save_path, show)
    logger.info(f"Backtest plot saved: {path}")
    return path
