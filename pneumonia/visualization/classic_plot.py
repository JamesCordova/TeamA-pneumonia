"""
Classic train/val/test forecast comparison plot.

Shows:
  - Trailing training actuals (solid black)
  - Val + test actuals (dashed black)
  - One dashed coloured line per model over the val + test range
  - Vertical markers at the val and test split boundaries

Saved to: reports/{DEPT}/{AGE_GROUP}/forecast_plot.png
"""

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd

from pneumonia.utils import setup_logger
from pneumonia.visualization._utils import (
    clip_axes,
    configure_date_axis,
    read_predictions,
    save_figure,
)

logger = setup_logger(__name__)


def plot_classic(
    department: str,
    age_group: str,
    reports_dir: Path,
    models: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
    show: bool = False,
    figsize: tuple = (15, 5),
    year: Optional[int] = None,
) -> Optional[Path]:
    """
    Generate the classic forecast comparison figure.

    Args:
        department:   Department name (uppercase).
        age_group:    'under5' or '60plus'.
        reports_dir:  Base reports directory.
        models:       Model names to include; None = all.
        save_path:    Output path; defaults to forecast_plot.png.
        show:         Call plt.show() after saving.
        figsize:      Matplotlib figure size.
        year:         Restrict x-axis to a single calendar year.

    Returns:
        Path to the saved PNG, or None if no data was found.
    """
    df = read_predictions(reports_dir, department, age_group)
    if df is None:
        return None

    val_start = df[df["split"] == "val"]["date"].min()
    test_end   = df[df["split"] == "test"]["date"].max()

    if year is not None:
        plot_min = pd.Timestamp(f"{year}-01-01")
        plot_max = pd.Timestamp(f"{year}-12-31")
    elif pd.notna(val_start):
        plot_min = pd.Timestamp(f"{val_start.year - 2}-01-01")
        plot_max = test_end if pd.notna(test_end) else df["date"].max()
    else:
        plot_min = df["date"].min()
        plot_max = df["date"].max()

    fig, ax = plt.subplots(figsize=figsize)
    palette = plt.cm.tab10.colors

    # --- single continuous actual line (train + val + test) ---
    train_actuals = (
        df[(df["split"] == "train") & (df["model"] == "actual") & (df["date"] >= plot_min)]
        [["date", "actual"]]
    )
    forecast_df = df[df["split"].isin(["val", "test"])].copy()
    val_test_actuals = forecast_df.drop_duplicates("date")[["date", "actual"]]
    all_actuals = pd.concat([train_actuals, val_test_actuals]).sort_values("date")
    if not all_actuals.empty:
        ax.plot(all_actuals["date"], all_actuals["actual"],
                color="black", lw=1.3, label="Actual")

    # --- model predictions (dashed coloured lines) ---
    available = sorted(m for m in forecast_df["model"].unique() if m != "actual")
    model_names = [m for m in available if m in models] if models else available
    if models:
        missing = [m for m in models if m not in available]
        if missing:
            logger.warning(f"Models not found in CSV: {missing}. Available: {available}")

    for idx, name in enumerate(model_names):
        mdf = forecast_df[forecast_df["model"] == name].sort_values("date")
        ax.plot(mdf["date"], mdf["predicted"],
                color=palette[idx % len(palette)],
                lw=1.5, ls="--", alpha=0.85, label=name)

    # --- split boundary markers ---
    ax.autoscale_view()
    ymin, ymax = ax.get_ylim()
    label_y = ymax - (ymax - ymin) * 0.06
    for split_name, color in [("val", "steelblue"), ("test", "darkorange")]:
        boundary = df[df["split"] == split_name]["date"].min()
        if pd.notna(boundary):
            ax.axvline(boundary, color=color, ls=":", lw=1.2, alpha=0.7)
            ax.text(boundary, label_y, f" {split_name}", color=color, fontsize=8)

    configure_date_axis(ax, plot_min, plot_max)

    age_label = "Under 5" if age_group == "under5" else "60+"
    ax.set_title(f"{department} — {age_label} — Forecast comparison", fontsize=12, pad=10)
    ax.set_ylabel("Pneumonia cases")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.8)
    ax.grid(True, alpha=0.25)

    clip_axes(ax, df, plot_min, plot_max)

    if save_path is None:
        suffix = f"_{year}" if year is not None else ""
        save_path = Path(reports_dir) / department / age_group / f"forecast_plot{suffix}.png"

    path = save_figure(fig, save_path, show)
    logger.info(f"Classic plot saved: {path}")
    return path
