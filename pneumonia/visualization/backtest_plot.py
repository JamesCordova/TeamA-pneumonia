"""
Walk-forward backtest plot with premium styling.

Shows:
  - Training actuals from the earliest backtest origin (solid dark charcoal)
  - Shaded backtest evaluation window
  - Val + test actuals as a reference line (dashed black)
  - One solid coloured line per model from the backtest predictions
  - Clean layout, custom fonts, grid, and despined axes.

Saved to: reports/{DEPT}/{AGE_GROUP}/backtest_plot.png
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

# Premium color palette matching modern design tokens
PALETTE = {
    "randomforest": "#2563eb",   # Royal Blue
    "sarima": "#059669",         # Emerald Green
    "xgboost": "#7c3aed",        # Purple
    "seasonalnaive": "#dc2626",  # Red
    "naive": "#d97706",          # Amber
    "holtwinters": "#db2777",    # Pink
}


def plot_backtest(
    department: str,
    age_group: str,
    reports_dir: Path,
    models: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
    show: bool = False,
    figsize: tuple = (15, 5.5),
    year: Optional[int] = None,
) -> Optional[Path]:
    """
    Generate a styled walk-forward backtest figure.

    Args:
        department:   Department name (uppercase).
        age_group:    'under5' or '60plus'.
        reports_dir:  Base reports directory.
        models:       Model names to include; None = all backtest models.
        save_path:    Output path; defaults to backtest_plot.png.
        show:         Call plt.show() after saving.
        figsize:      Matplotlib figure size.
        year:         Restrict x-axis to a single calendar year.

    Returns:
        Path to the saved PNG, or None if no backtest data was found.
    """
    df = read_predictions(reports_dir, department, age_group)
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

    fig, ax = plt.subplots(figsize=figsize, facecolor="white")

    # Style options
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "sans-serif"]
    plt.rcParams["font.family"] = "sans-serif"

    # --- backtest split shading in the background ---
    backtest_start = backtest_df["date"].min()
    backtest_end = backtest_df["date"].max()
    if pd.notna(backtest_start) and pd.notna(backtest_end):
        ax.axvspan(backtest_start, backtest_end, color="#f5f3ff", alpha=0.6, label="Backtest Period")

    # --- single continuous actual line ---
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
        ax.plot(all_actuals["date"], all_actuals["actual"],
                color="#1f2937", lw=1.8, label="Actual cases")

    # --- backtest model predictions (solid coloured lines) ---
    available = sorted(backtest_df["model"].unique())
    bt_models = [m for m in available if m in models] if models else available
    if models:
        missing = [m for m in models if m not in available]
        if missing:
            logger.warning(
                f"Backtest models not found: {missing}. Available: {available}"
            )

    for idx, name in enumerate(bt_models):
        bdf = backtest_df[backtest_df["model"] == name].sort_values("date")
        key = name.lower().replace("_", "").replace("-", "")
        color = PALETTE.get(key, plt.cm.tab10.colors[idx % 10])
        ax.plot(bdf["date"], bdf["predicted"],
                color=color,
                lw=2.0, alpha=0.9, label=f"{name} (backtest)")

    configure_date_axis(ax, plot_min, plot_max)

    age_label = "Under 5 Years" if age_group == "under5" else "60+ Years"
    ax.set_title(
        f"Walk-Forward Backtest — {department} ({age_label})", fontsize=14, fontweight="bold", color="#1f2937", pad=12
    )
    ax.set_ylabel("Pneumonia Cases", fontsize=10, fontweight="semibold", color="#374151")
    
    # Legend
    legend = ax.legend(
        loc="upper left", 
        fontsize=9, 
        frameon=True,
        facecolor="#f9fafb",
        edgecolor="#e5e7eb"
    )
    legend.get_frame().set_boxstyle("round,pad=0.4")
    
    # Grid & Spines
    ax.grid(color="#e5e7eb", linestyle="--", linewidth=0.7, alpha=0.8)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#d1d5db")
    ax.spines["bottom"].set_color("#d1d5db")
    ax.tick_params(colors="#4b5563")

    clip_axes(ax, df, plot_min, plot_max)

    if save_path is None:
        save_path = Path(reports_dir) / department / age_group / "backtest_plot.png"

    path = save_figure(fig, save_path, show)
    logger.info(f"Backtest plot saved: {path}")
    return path
