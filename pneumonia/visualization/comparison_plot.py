"""
Walk-forward model comparison figure.

Two-panel figure:
  Left  — grouped bar chart: selected metric at h_short vs h_long per model
  Right — line chart: metric degradation from h_short to h_long per model
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

METRIC_LABELS = {
    "mae":   "MAE (cases)",
    "rmse":  "RMSE (cases)",
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
    Generate a two-panel model comparison figure.

    Args:
        metrics:   {model_name: {horizon_int: {metric_key: value}}}
        horizons:  All available horizons (used for the degradation line).
        h_short:   Short horizon for bar chart (e.g. 1).
        h_long:    Long horizon for bar chart (e.g. 4).
        metric:    Metric to plot: 'mae', 'rmse', 'smape', or 'mda'.
        save_path: Where to save the PNG. Returns None if not provided.
        show:      Call plt.show() after saving.

    Returns:
        Path to saved PNG, or None.
    """
    if metric not in VALID_METRICS:
        raise ValueError(f"metric must be one of {sorted(VALID_METRICS)}, got '{metric}'")

    models  = list(metrics.keys())
    palette = plt.cm.tab10.colors
    colors  = {m: palette[i % len(palette)] for i, m in enumerate(models)}
    ylabel  = METRIC_LABELS[metric]

    fig, (ax_bar, ax_line) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Walk-forward model comparison — {ylabel}", fontsize=13, y=1.01
    )

    # ------------------------------------------------------------------ #
    # Panel 1: grouped bar chart at h_short and h_long
    # ------------------------------------------------------------------ #
    x      = np.arange(len(models))
    width  = 0.35
    vals_s = [metrics[m].get(h_short, {}).get(metric, np.nan) for m in models]
    vals_l = [metrics[m].get(h_long,  {}).get(metric, np.nan) for m in models]

    bars_s = ax_bar.bar(
        x - width / 2, vals_s, width,
        label=f"h={h_short}",
        color=[colors[m] for m in models], alpha=0.9,
    )
    bars_l = ax_bar.bar(
        x + width / 2, vals_l, width,
        label=f"h={h_long}",
        color=[colors[m] for m in models], alpha=0.45,
        edgecolor=[colors[m] for m in models], linewidth=1.2,
    )

    for bar in list(bars_s) + list(bars_l):
        h = bar.get_height()
        if not np.isnan(h):
            ax_bar.text(
                bar.get_x() + bar.get_width() / 2, h + 0.05,
                f"{h:.1f}", ha="center", va="bottom", fontsize=7.5,
            )

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(models, rotation=25, ha="right", fontsize=9)
    ax_bar.set_ylabel(ylabel)
    ax_bar.set_title(f"h={h_short} (solid) vs h={h_long} (faded)")
    ax_bar.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax_bar.grid(axis="y", alpha=0.25)
    ax_bar.grid(axis="y", which="minor", alpha=0.12)

    # ------------------------------------------------------------------ #
    # Panel 2: degradation lines across all horizons
    # ------------------------------------------------------------------ #
    for model in models:
        hs   = sorted(h for h in horizons if h in metrics[model])
        vals = [metrics[model][h].get(metric, np.nan) for h in hs]
        ax_line.plot(hs, vals, marker="o", lw=1.8,
                     label=model, color=colors[model])

    ax_line.set_xticks(horizons)
    ax_line.set_xlabel("Forecast horizon (weeks ahead)")
    ax_line.set_ylabel(ylabel)
    ax_line.set_title(f"{ylabel} across horizons")
    ax_line.legend(fontsize=9, loc="upper left")
    ax_line.grid(alpha=0.25)

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
