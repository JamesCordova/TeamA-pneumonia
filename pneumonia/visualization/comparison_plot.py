"""
Walk-forward model comparison figure.

Two-panel figure:
  Left  — grouped bar chart: selected metric at h_short vs h_long per model
  Right — line chart: metric across all horizons per model
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import Normalize, TwoSlopeNorm
import numpy as np
import pandas as pd

from pneumonia.visualization._utils import enable_legend_picking
from pneumonia.visualization.theme import DIVERGING_CMAP, NO_DATA_COLOR, SEQUENTIAL_CMAP

# Color ramps are centralized in pneumonia.visualization.theme so both panels
# use the same visual language. Sequential: one hue (blue), light→dark, step
# 100→700. Diverging: blue↔red poles with a near-white neutral midpoint
# (fades into the figure background at 0 — only meaningful bias stands out).
METRIC_LABELS = {
    "mae":   "MAE (cases)",
    "rmse":  "RMSE (cases)",
    "me":    "ME (bias)",
    "mape":  "MAPE (%)",
    "smape": "SMAPE (%)",
    "mda":   "MDA (%)",
    "r2":    "R²",
}

HIGHER_IS_BETTER = {"mda", "r2"}
ZERO_IS_BETTER   = {"me"}
VALID_METRICS = set(METRIC_LABELS)

# Ratio metrics computed relative to a per-window baseline (e.g. r2's SS_tot from
# that window's own mean). Averaging per-step values of these is not equivalent to
# computing them on pooled data, and is unstable when a window's baseline ≈ 0.
# Excluded by default from macroaverage mode; extend this set if similar ratio
# metrics (e.g. MASE, RMSSE) are added later.
NON_ADDITIVE_METRICS = {"r2"}


def better_label(metric: str) -> str:
    """Direction hint for a metric: higher/lower-is-better, or closer-to-0 for bias metrics."""
    if metric in HIGHER_IS_BETTER:
        return "↑ higher is better"
    if metric in ZERO_IS_BETTER:
        return "→ 0 is ideal (over/under-prediction)"
    return "↓ lower is better"


def plot_model_comparison(
    metrics: dict,
    horizons: list,
    h_short: int,
    h_long: int,
    metric: str = "mae",
    department: Optional[str] = None,
    save_path: Optional[Path] = None,
    show: bool = False,
) -> Optional[Path]:
    """
    Generate a two-panel model comparison figure.

    Args:
        metrics:    {model_name: {horizon_int: {metric_key: value}}}
        horizons:   All available horizons (used for the line chart).
        h_short:    Short horizon for bar chart (e.g. 1).
        h_long:     Long horizon for bar chart (e.g. 4).
        metric:     Metric to plot: 'mae', 'rmse', 'smape', 'mda', 'me', or 'r2'.
        department: Optional label (e.g. department name) shown in the figure title.
        save_path:  Where to save the PNG. Returns None if not provided.
        show:       Call plt.show() after saving.

    Returns:
        Path to saved PNG, or None.
    """
    if metric not in VALID_METRICS:
        raise ValueError(f"metric must be one of {sorted(VALID_METRICS)}, got '{metric}'")

    has_data = any(
        not np.isnan(v)
        for by_h in metrics.values()
        for h_data in by_h.values()
        for v in [h_data.get(metric, np.nan)]
    )
    if not has_data:
        print(
            f"[comparison_plot] No data for metric '{metric}' in any model — "
            "re-run scripts/run_walkforward.py to regenerate metrics."
        )
        return None

    models = list(metrics.keys())
    palette = plt.cm.tab10.colors
    colors  = {m: palette[i % len(palette)] for i, m in enumerate(models)}
    ylabel  = METRIC_LABELS[metric]

    fig, (ax_bar, ax_line) = plt.subplots(1, 2, figsize=(14, 5.5))
    title = f"Model Evaluation Comparison — {ylabel}"
    if department:
        title = f"{department} — {title}"
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # ------------------------------------------------------------------ #
    # Panel 1: Grouped bar chart at h_short and h_long
    # ------------------------------------------------------------------ #
    x = np.arange(len(models))
    width = 0.32
    vals_s = [metrics[m].get(h_short, {}).get(metric, np.nan) for m in models]
    vals_l = [metrics[m].get(h_long,  {}).get(metric, np.nan) for m in models]

    bars_s = ax_bar.bar(
        x - width / 2, vals_s, width,
        label=f"Horizon h={h_short}",
        color=[colors[m] for m in models],
        alpha=0.9,
        edgecolor="white",
        linewidth=0.7,
    )
    bars_l = ax_bar.bar(
        x + width / 2, vals_l, width,
        label=f"Horizon h={h_long}",
        color=[colors[m] for m in models],
        alpha=0.45,
        edgecolor=[colors[m] for m in models],
        linewidth=1.2,
    )

    # Value labels above (or below for negative R²) each bar
    all_vals = [v for v in vals_s + vals_l if not np.isnan(v)]
    val_range = (max(all_vals) - min(all_vals)) if all_vals else 1
    offset = val_range * 0.02 or 0.05

    for bar in list(bars_s) + list(bars_l):
        h = bar.get_height()
        if not np.isnan(h):
            ax_bar.text(
                bar.get_x() + bar.get_width() / 2,
                h + offset if h >= 0 else h - offset * 3,
                f"{h:.2f}" if metric == "r2" else f"{h:.1f}",
                ha="center",
                va="bottom" if h >= 0 else "top",
                fontsize=7.5,
            )

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(models, rotation=25, ha="right", fontsize=9)
    ax_bar.set_ylabel(ylabel)
    better = better_label(metric)
    ax_bar.set_title(f"h={h_short} (solid) vs h={h_long} (faded) — {better}")
    ax_bar.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax_bar.grid(axis="y", alpha=0.25)
    ax_bar.grid(axis="y", which="minor", alpha=0.12)
    if metric in {"r2", "me"}:
        ax_bar.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)

    # ------------------------------------------------------------------ #
    # Panel 2: metric across all horizons
    # ------------------------------------------------------------------ #
    line_title = f"{ylabel} across horizons — {better}"
    lines_by_model = {}
    for model in models:
        hs = sorted(h for h in horizons if h in metrics[model])
        vals = [metrics[model][h].get(metric, np.nan) for h in hs]
        (line,) = ax_line.plot(
            hs, vals,
            marker="o",
            markersize=6,
            markeredgecolor="white",
            markeredgewidth=1.0,
            lw=2.2,
            label=model,
            color=colors[model],
        )
        lines_by_model[model] = line

    ax_line.set_xticks(horizons)
    ax_line.set_xlabel("Forecast horizon (weeks ahead)")
    ax_line.set_ylabel(ylabel)
    ax_line.set_title(line_title)
    legend = ax_line.legend(
        fontsize=9, loc="upper left" if metric not in HIGHER_IS_BETTER else "lower left"
    )
    enable_legend_picking(fig, legend, lines_by_model)
    ax_line.grid(alpha=0.25)
    if metric in {"r2", "me"}:
        ax_line.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved: {save_path}")

    if show:
        plt.show()
    plt.close(fig)
    return save_path


def _draw_micro_heatmap(fig, ax, cumulative_metrics: dict, metric: str, models_order: list) -> bool:
    """
    Draw the cumulative (expanding-window) heatmap onto `ax`: rows are models
    (in `models_order`), columns are forecast dates, color is the metric's
    pooled value computed on every backtest prediction from the start of the
    backtest up to that date. Returns False (and leaves `ax` with just a note)
    if no model has data for `metric`.

    Color encodes magnitude, not "goodness": the module's color ramps are
    defined once at the top of the file, and this heatmap maps values onto them
    through numeric normalization. Lower/higher-is-better metrics use a
    light→dark sequential ramp reversed so light always means "better" and
    dark always means "worse" (a consistent reading rule across metrics); the
    zero-is-ideal bias metric (me) uses a diverging ramp centered on 0, which
    fades into the figure background near zero.
    """
    models = [
        m for m in models_order
        if m in cumulative_metrics
        and metric in cumulative_metrics[m].columns
        and cumulative_metrics[m][metric].notna().any()
    ]
    if not models:
        ax.text(0.5, 0.5, "No cumulative data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return False

    all_dates = sorted(set().union(*(set(cumulative_metrics[m]["date"]) for m in models)))
    date_index = pd.DatetimeIndex(all_dates)

    matrix = np.full((len(models), len(date_index)), np.nan)
    for i, m in enumerate(models):
        s = cumulative_metrics[m].set_index("date")[metric].reindex(date_index)
        matrix[i] = s.ffill().to_numpy()

    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0:
        ax.text(0.5, 0.5, "No cumulative data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return False

    higher_is_better = metric in HIGHER_IS_BETTER
    zero_is_better   = metric in ZERO_IS_BETTER

    if zero_is_better:
        max_abs = float(np.max(np.abs(finite)))
        max_abs = max(max_abs, 1.0)
        norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)
        cmap = DIVERGING_CMAP
    else:
        vmin, vmax = float(np.min(finite)), float(np.max(finite))
        if np.isclose(vmin, vmax):
            pad = max(abs(vmin) * 0.05, 1.0)
            vmin -= pad
            vmax += pad
        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap = SEQUENTIAL_CMAP.reversed() if higher_is_better else SEQUENTIAL_CMAP
    cmap = cmap.copy()
    cmap.set_bad(color=NO_DATA_COLOR)

    masked = np.ma.masked_invalid(matrix)
    im = ax.imshow(masked, cmap=cmap, norm=norm, aspect="auto", interpolation="none")

    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=9)

    n_ticks = min(8, len(date_index))
    tick_pos = np.linspace(0, len(date_index) - 1, n_ticks).astype(int)
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(
        [date_index[i].strftime("%Y-%m") for i in tick_pos],
        rotation=30, ha="right", fontsize=8,
    )

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(METRIC_LABELS[metric], fontsize=9)
    return True


def plot_micro_comparison(
    metrics: dict,
    cumulative_metrics: dict,
    metric: str,
    department: Optional[str] = None,
    save_path: Optional[Path] = None,
    show: bool = False,
) -> Optional[Path]:
    """
    Two-panel microaverage figure: the metric is computed once over every
    backtest prediction pooled across all steps/horizons (no per-horizon split,
    unlike plot_model_comparison).

      Left  — bar chart: final pooled value per model (one snapshot number).
      Right — heatmap: same pooled computation but as an expanding window over
              time (rows=models, columns=date), showing how the score evolves
              and stabilizes rather than just its final value.

    Args:
        metrics:            {model_name: {metric_key: value}} — final snapshot,
                             flat, no horizon axis.
        cumulative_metrics: {model_name: DataFrame(date, metric_key, ...)} — one
                             row per evaluated date (expanding window), as
                             returned by compute_cumulative_micro_metrics.
        metric:     Metric to plot: 'mae', 'rmse', 'smape', 'mda', 'me', or 'r2'.
        department: Optional label (e.g. department name) shown in the figure title.
        save_path:  Where to save the PNG. Returns None if not provided.
        show:       Call plt.show() after saving.

    Returns:
        Path to saved PNG, or None.
    """
    if metric not in VALID_METRICS:
        raise ValueError(f"metric must be one of {sorted(VALID_METRICS)}, got '{metric}'")

    values = {m: v.get(metric, np.nan) for m, v in metrics.items()}
    values = {m: v for m, v in values.items() if not np.isnan(v)}
    if not values:
        print(
            f"[comparison_plot] No data for metric '{metric}' in any model — "
            "re-run scripts/run_walkforward.py to regenerate predictions."
        )
        return None

    higher_is_better = metric in HIGHER_IS_BETTER
    models  = sorted(values, key=lambda m: values[m], reverse=higher_is_better)
    palette = plt.cm.tab10.colors
    colors  = [palette[i % len(palette)] for i in range(len(models))]
    ylabel  = METRIC_LABELS[metric]
    better  = better_label(metric)

    fig, (ax_bar, ax_heat) = plt.subplots(
        1, 2, figsize=(15, 5.5), gridspec_kw={"width_ratios": [1, 1.6]}
    )
    title = f"Model Evaluation Comparison (microaverage) — {ylabel}"
    if department:
        title = f"{department} — {title}"
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # ------------------------------------------------------------------ #
    # Panel 1: bar chart — final pooled snapshot
    # ------------------------------------------------------------------ #
    bars = ax_bar.bar(models, [values[m] for m in models], color=colors, alpha=0.85)
    fmt = "%.2f" if metric == "r2" else "%.1f"
    ax_bar.bar_label(bars, fmt=fmt, fontsize=8.5, padding=2)

    ax_bar.set_ylabel(ylabel)
    ax_bar.set_title("Final pooled snapshot — sorted best → worst")
    ax_bar.set_xlabel(better)
    plt.setp(ax_bar.get_xticklabels(), rotation=20, ha="right", fontsize=9)
    ax_bar.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax_bar.grid(axis="y", alpha=0.25)
    ax_bar.grid(axis="y", which="minor", alpha=0.12)
    if metric in {"r2", "me"}:
        ax_bar.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)

    # ------------------------------------------------------------------ #
    # Panel 2: cumulative heatmap — same models, ordered to match the bars
    # ------------------------------------------------------------------ #
    _draw_micro_heatmap(fig, ax_heat, cumulative_metrics, metric, models)
    heat_title = "Cumulative over time (light = better, dark = worse)"
    if metric == "me":
        heat_title = "Cumulative over time (neutral = 0, darker = farther from 0)"
    ax_heat.set_title(heat_title)
    ax_heat.set_xlabel("Forecast date")

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved: {save_path}")

    if show:
        plt.show()
    plt.close(fig)
    return save_path
