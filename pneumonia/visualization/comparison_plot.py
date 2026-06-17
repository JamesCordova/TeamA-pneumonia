"""
Walk-forward model comparison figure with premium styling.

Two-panel figure:
  Left  — grouped bar chart: selected metric at h_short vs h_long per model
  Right — line chart: metric values across all horizons per model
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# Premium color palette matching modern design tokens
PALETTE = {
    "randomforest": "#2563eb",   # Royal Blue
    "sarima": "#059669",         # Emerald Green
    "xgboost": "#7c3aed",        # Purple
    "seasonalnaive": "#dc2626",  # Red
    "naive": "#d97706",          # Amber
    "holtwinters": "#db2777",    # Pink
}
DEFAULT_COLOR = "#4b5563"        # Cool Gray

METRIC_LABELS = {
    "mae":   "MAE (cases)",
    "rmse":  "RMSE (cases)",
    "me":    "ME (bias)",
    "r2":    "R2 Score",
    "mase":  "MASE",
    "smape": "SMAPE (%)",
    "mda":   "MDA (%)",
}

VALID_METRICS = set(METRIC_LABELS)


def plot_model_comparison(
    metrics: dict,
    horizons: list,
    h_short: int,
    h_long: int,
    metric: str = "mae",
    save_path: Optional[Path] = None,
    show: bool = False,
) -> Optional[Path]:
    """
    Generate a styled two-panel model comparison figure.

    Args:
        metrics:   {model_name: {horizon_int: {metric_key: value}}}
        horizons:  All available horizons.
        h_short:   Short horizon for bar chart (e.g. 1).
        h_long:    Long horizon for bar chart (e.g. 4).
        metric:    Metric to plot.
        save_path: Where to save the PNG. Returns None if not provided.
        show:      Call plt.show() after saving.

    Returns:
        Path to saved PNG, or None.
    """
    if metric not in VALID_METRICS:
        raise ValueError(f"metric must be one of {sorted(VALID_METRICS)}, got '{metric}'")

    # Clean matplotlib style configuration locally
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "sans-serif"]
    plt.rcParams["font.family"] = "sans-serif"

    models = list(metrics.keys())
    
    # Assign premium colors to models
    colors = {}
    for m in models:
        key = m.lower().replace("_", "").replace("-", "")
        colors[m] = PALETTE.get(key, DEFAULT_COLOR)

    ylabel = METRIC_LABELS[metric]

    fig, (ax_bar, ax_line) = plt.subplots(1, 2, figsize=(14, 5.5), facecolor="white")
    
    # Elegant title badge styling
    fig.suptitle(
        f"Model Evaluation Comparison — {ylabel}", 
        fontsize=14, 
        fontweight="bold", 
        color="#1f2937",
        y=0.98
    )

    # ------------------------------------------------------------------ #
    # Panel 1: Grouped bar chart at h_short and h_long
    # ------------------------------------------------------------------ #
    x = np.arange(len(models))
    width = 0.32
    vals_s = [metrics[m].get(h_short, {}).get(metric, np.nan) for m in models]
    vals_l = [metrics[m].get(h_long,  {}).get(metric, np.nan) for m in models]

    # Create bars with round edges using alpha or edge colors
    bars_s = ax_bar.bar(
        x - width / 2, vals_s, width,
        label=f"Horizon h={h_short}",
        color=[colors[m] for m in models], 
        alpha=0.9,
        edgecolor="white",
        linewidth=0.7
    )
    bars_l = ax_bar.bar(
        x + width / 2, vals_l, width,
        label=f"Horizon h={h_long}",
        color=[colors[m] for m in models], 
        alpha=0.45,
        edgecolor=[colors[m] for m in models], 
        linewidth=1.2
    )

    # Add data values on top of the bars
    for bar in list(bars_s) + list(bars_l):
        h = bar.get_height()
        if not np.isnan(h):
            # Formatter depends on whether metric is percentage or decimal
            label_text = f"{h:.2f}" if abs(h) < 10 else f"{h:.1f}"
            ax_bar.text(
                bar.get_x() + bar.get_width() / 2, h + (0.01 * (h if h > 0 else -h)),
                label_text, ha="center", va="bottom", fontsize=8, fontweight="bold",
                color="#374151"
            )

    # Styling Axis 1
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(models, rotation=15, ha="right", fontsize=9, color="#4b5563", fontweight="semibold")
    ax_bar.set_ylabel(ylabel, fontsize=10, fontweight="semibold", color="#374151")
    ax_bar.set_title(f"Comparison: h={h_short} (solid) vs h={h_long} (faded)", fontsize=11, fontweight="semibold", color="#4b5563", pad=12)
    ax_bar.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax_bar.grid(axis="y", color="#e5e7eb", linestyle="--", linewidth=0.7, alpha=0.8)
    ax_bar.grid(axis="y", which="minor", color="#f3f4f6", linestyle=":", linewidth=0.5, alpha=0.5)
    
    # Despine
    for spine in ["top", "right"]:
        ax_bar.spines[spine].set_visible(False)
    ax_bar.spines["left"].set_color("#d1d5db")
    ax_bar.spines["bottom"].set_color("#d1d5db")
    ax_bar.tick_params(colors="#4b5563")

    # ------------------------------------------------------------------ #
    # Panel 2: Degradation lines across all horizons
    # ------------------------------------------------------------------ #
    for model in models:
        hs = sorted(h for h in horizons if h in metrics[model])
        vals = [metrics[model][h].get(metric, np.nan) for h in hs]
        
        # Plot styled line with matching marker
        ax_line.plot(
            hs, vals, 
            marker="o", 
            markersize=6,
            markeredgecolor="white",
            markeredgewidth=1.0,
            lw=2.2,
            label=model, 
            color=colors[model]
        )

    # Styling Axis 2
    ax_line.set_xticks(horizons)
    ax_line.set_xlabel("Forecast Horizon (weeks ahead)", fontsize=10, fontweight="semibold", color="#374151")
    ax_line.set_ylabel(ylabel, fontsize=10, fontweight="semibold", color="#374151")
    ax_line.set_title(f"Degradation Analysis over Horizons", fontsize=11, fontweight="semibold", color="#4b5563", pad=12)
    
    # Legend
    legend = ax_line.legend(
        fontsize=9, 
        loc="upper left" if metric != "r2" else "lower left", 
        frameon=True,
        facecolor="#f9fafb",
        edgecolor="#e5e7eb"
    )
    legend.get_frame().set_boxstyle("round,pad=0.5")
    
    ax_line.grid(color="#e5e7eb", linestyle="--", linewidth=0.7, alpha=0.8)
    
    # Despine
    for spine in ["top", "right"]:
        ax_line.spines[spine].set_visible(False)
    ax_line.spines["left"].set_color("#d1d5db")
    ax_line.spines["bottom"].set_color("#d1d5db")
    ax_line.tick_params(colors="#4b5563")

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        if not show:
            plt.close(fig)
        print(f"Figure saved: {save_path}")
        return save_path

    if show:
        plt.show()
    plt.close(fig)
    return None
