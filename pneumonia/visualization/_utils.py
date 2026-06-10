"""Shared helpers for visualization modules."""

from pathlib import Path
from typing import Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from pneumonia.utils import setup_logger

logger = setup_logger(__name__)


def read_predictions(
    reports_dir: Path, department: str, age_group: str
) -> Optional[pd.DataFrame]:
    csv_path = Path(reports_dir) / department / age_group / "predictions.csv"
    if not csv_path.exists():
        logger.warning(f"No predictions CSV at {csv_path} — run a pipeline first.")
        return None
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = df[(df["department"] == department) & (df["age_group"] == age_group)].copy()
    if df.empty:
        logger.warning(f"No rows for {department}/{age_group} in {csv_path}.")
        return None
    return df


def configure_date_axis(ax, plot_min: pd.Timestamp, plot_max: pd.Timestamp) -> None:
    if not (pd.notna(plot_min) and pd.notna(plot_max)):
        return
    range_days = (plot_max - plot_min).days
    if range_days <= 366:
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=0))
        ax.tick_params(axis="x", which="minor", length=3, color="gray")
        ax.set_xlabel("Month / Week")
    elif range_days <= 366 * 3:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.set_xlabel("Quarter")
    else:
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.set_xlabel("Year")
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(45)
        lbl.set_ha("right")


def clip_axes(
    ax, df: pd.DataFrame, plot_min: pd.Timestamp, plot_max: pd.Timestamp
) -> None:
    """Set x-limits and re-scale y to only data visible in [plot_min, plot_max]."""
    if not (pd.notna(plot_min) and pd.notna(plot_max)):
        return
    ax.set_xlim(plot_min, plot_max)
    vis = df[(df["date"] >= plot_min) & (df["date"] <= plot_max)]
    if not vis.empty:
        all_vals = pd.concat([vis["actual"].dropna(), vis["predicted"].dropna()])
        if not all_vals.empty:
            v_min, v_max = all_vals.min(), all_vals.max()
            pad = max((v_max - v_min) * 0.10, 1.0)
            ax.set_ylim(max(0.0, v_min - pad), v_max + pad)


def save_figure(fig, save_path: Path, show: bool = False) -> Path:
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    if show:
        plt.show()
    return Path(save_path)
