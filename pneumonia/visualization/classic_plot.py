"""
Classic train/val/test forecast comparison plot with premium styling.

Shows:
  - Trailing training actuals (solid dark charcoal)
  - Val + test actuals as reference
  - Shaded regions for validation and testing split boundaries
  - One dashed coloured line per model over the val + test range
  - Clean layout, custom fonts, grid, and despined axes.

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

# Premium color palette matching modern design tokens
PALETTE = {
    "randomforest": "#2563eb",   # Royal Blue
    "sarima": "#059669",         # Emerald Green
    "xgboost": "#7c3aed",        # Purple
    "seasonalnaive": "#dc2626",  # Red
    "naive": "#d97706",          # Amber
    "holtwinters": "#db2777",    # Pink
}


def plot_classic(
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
    Generate a styled classic forecast comparison figure.

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

    # Style options
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "sans-serif"]
    plt.rcParams["font.family"] = "sans-serif"

    val_start = df[df["split"] == "val"]["date"].min()
    test_start = df[df["split"] == "test"]["date"].min()
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

    fig, ax = plt.subplots(figsize=figsize, facecolor="white")

    # --- split shading first (so it stays in background) ---
    if pd.notna(val_start):
        val_shade_end = test_start if pd.notna(test_start) else (test_end if pd.notna(test_end) else df["date"].max())
        ax.axvspan(val_start, val_shade_end, color="#eff6ff", alpha=0.6, label="Validation Period")
    if pd.notna(test_start) and pd.notna(test_end):
        ax.axvspan(test_start, test_end, color="#fff7ed", alpha=0.6, label="Testing Period")

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
                color="#1f2937", lw=1.8, label="Actual cases")

    # --- model predictions (dashed coloured lines) ---
    available = sorted(m for m in forecast_df["model"].unique() if m != "actual")
    model_names = [m for m in available if m in models] if models else available
    if models:
        missing = [m for m in models if m not in available]
        if missing:
            logger.warning(f"Models not found in CSV: {missing}. Available: {available}")

    for idx, name in enumerate(model_names):
        mdf = forecast_df[forecast_df["model"] == name].sort_values("date")
        key = name.lower().replace("_", "").replace("-", "")
        color = PALETTE.get(key, plt.cm.tab10.colors[idx % 10])
        ax.plot(mdf["date"], mdf["predicted"],
                color=color,
                lw=2.0, ls="--", alpha=0.9, label=name)

    # --- split boundary markers ---
    ax.autoscale_view()
    ymin, ymax = ax.get_ylim()
    label_y = ymax - (ymax - ymin) * 0.08
    for split_name, color, label_text in [("val", "#1d4ed8", "Validation"), ("test", "#c2410c", "Testing")]:
        boundary = df[df["split"] == split_name]["date"].min()
        if pd.notna(boundary):
            ax.axvline(boundary, color=color, ls=":", lw=1.5, alpha=0.8)
            ax.text(boundary, label_y, f"  {label_text} Start", color=color, fontsize=8.5, fontweight="bold")

    configure_date_axis(ax, plot_min, plot_max)

    age_label = "Under 5 Years" if age_group == "under5" else "60+ Years"
    ax.set_title(f"Forecast Comparison — {department} ({age_label})", fontsize=14, fontweight="bold", color="#1f2937", pad=12)
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
        save_path = Path(reports_dir) / department / age_group / "forecast_plot.png"

    path = save_figure(fig, save_path, show)
    logger.info(f"Classic plot saved: {path}")
    return path
