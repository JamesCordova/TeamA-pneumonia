"""
Per-step walk-forward diagnostic figure.

Two-panel figure per metric:
  Left  — boxplot of the metric's distribution across all steps, with a
      marker for the mean.
  Right — heatmap of the metric by forecast date (rows=models, columns=date),
      so each cell shows the metric for that step/date directly.
"""

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pneumonia.visualization.comparison_plot import (
    METRIC_LABELS,
    VALID_METRICS,
    better_label,
    _draw_model_date_heatmap,
)


def plot_step_metrics(
    step_data: Dict[str, pd.DataFrame],
    metric: str,
    department: Optional[str] = None,
    save_path: Optional[Path] = None,
    trend_window: int = 13,
    show: bool = False,
) -> Optional[Path]:
    """
    Generate a two-panel per-step diagnostic figure for one metric.

    Args:
        step_data: {model_name: DataFrame} — each DataFrame has a 'date'
                   column and one column per metric (as saved by
                   `pneumonia.visualization.persistence.save_step_metrics`).
                   Row counts may differ across models.
        metric:    Metric key, e.g. 'mae', 'rmse', 'r2'.
        department: Optional label (e.g. department name) shown in the figure title.
        save_path: Where to save the PNG. Returns None if not provided.
        trend_window: Retained for backward compatibility. The right panel is now a
                       heatmap and does not use rolling smoothing.
        show:      Call plt.show() after saving.

    Returns:
        Path to saved PNG, or None.
    """
    if metric not in VALID_METRICS:
        raise ValueError(f"metric must be one of {sorted(VALID_METRICS)}, got '{metric}'")

    models = [
        m for m, df in step_data.items()
        if metric in df.columns and df[metric].notna().any()
    ]
    if not models:
        print(
            f"[step_metrics_plot] No data for metric '{metric}' in any model — "
            "re-run scripts/run_walkforward.py to regenerate step metrics."
        )
        return None

    palette = plt.cm.tab10.colors
    colors  = {m: palette[i % len(palette)] for i, m in enumerate(models)}
    ylabel  = METRIC_LABELS[metric]
    better  = better_label(metric)

    fig, (ax_box, ax_time) = plt.subplots(1, 2, figsize=(14, 5.5))
    title = f"Per-step Walk-forward Diagnostics — {ylabel}"
    if department:
        title = f"{department} — {title}"
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # ------------------------------------------------------------------ #
    # Panel 1: distribution across steps (boxplot + mean marker)
    # ------------------------------------------------------------------ #
    box_values = [step_data[m][metric].dropna().values for m in models]
    bp = ax_box.boxplot(box_values, tick_labels=models, patch_artist=True)
    for patch, m in zip(bp["boxes"], models):
        patch.set_facecolor(colors[m])
        patch.set_alpha(0.35)

    means   = [np.mean(v) if len(v) else np.nan for v in box_values]
    medians = [line.get_ydata()[0] for line in bp["medians"]]
    ax_box.scatter(
        range(1, len(models) + 1), means,
        marker="D", s=45, color="black", zorder=3, label="mean",
    )

    fmt = (lambda v: f"{v:.2f}") if metric == "r2" else (lambda v: f"{v:.1f}")
    for i, (mean_val, median_val) in enumerate(zip(means, medians), start=1):
        if not np.isnan(mean_val):
            ax_box.text(i + 0.12, mean_val, fmt(mean_val),
                        fontsize=7.5, va="center", ha="left", color="black")
        ax_box.text(i - 0.12, median_val, fmt(median_val),
                    fontsize=7.5, va="center", ha="right", color="firebrick")

    plt.setp(ax_box.get_xticklabels(), rotation=25, ha="right", fontsize=9)
    ax_box.set_ylabel(ylabel)
    ax_box.set_title(f"Distribution across steps — {better}")
    ax_box.grid(axis="y", alpha=0.25)
    ax_box.legend(fontsize=9)
    if metric in {"r2", "me"}:
        ax_box.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)

    # ------------------------------------------------------------------ #
    # Panel 2: metric heatmap over time, anchored by forecast date
    # ------------------------------------------------------------------ #
    step_frames = {m: step_data[m].sort_values("date") for m in models}
    _draw_model_date_heatmap(fig, ax_time, step_frames, metric, models, fill_method=None)
    ax_time.set_title(f"{ylabel} by forecast date — {better}")
    ax_time.set_xlabel("Forecast date")
    ax_time.set_ylabel("Model")
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved: {save_path}")

    if show:
        plt.show()
    plt.close(fig)
    return save_path
