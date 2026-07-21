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
    top_n: Optional[int] = None,
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
        top_n:      If given, keep only the top_n best models (ranked by `metric`
                    at h_short) — with many models (e.g. every tune_models.py
                    trial), the full set overlaps and becomes unreadable. The
                    line chart (all horizons) is restricted to the same subset,
                    so both panels always show the same models.

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

    higher_is_better = metric in HIGHER_IS_BETTER
    models = list(metrics.keys())
    if top_n is not None and len(models) > top_n:
        ranked = sorted(
            models,
            key=lambda m: metrics[m].get(h_short, {}).get(metric, np.nan),
        )
        ranked = [m for m in ranked if not np.isnan(metrics[m].get(h_short, {}).get(metric, np.nan))]
        if higher_is_better:
            ranked = ranked[::-1]
        models = ranked[:top_n]

    # Best → worst top-to-bottom on a horizontal bar chart reads naturally
    # (best at top); reversed here because barh plots first-to-last bottom-to-top.
    models = sorted(
        models,
        key=lambda m: metrics[m].get(h_short, {}).get(metric, np.nan),
        reverse=not higher_is_better,
    )[::-1]

    palette = plt.cm.tab10.colors
    colors  = {m: palette[i % len(palette)] for i, m in enumerate(models)}
    ylabel  = METRIC_LABELS[metric]

    # Figure height must include room for the legend below the axes, not
    # just the axes themselves — otherwise reserving that room later via
    # tight_layout(rect=...) just squeezes an unchanged-size figure instead
    # of actually making space, and the axes/labels end up cramped.
    n_legend_cols = 1 if len(models) <= 6 else (2 if len(models) <= 14 else 3)
    n_legend_rows = -(-len(models) // n_legend_cols)  # ceil
    legend_height = 0.45 + 0.24 * n_legend_rows
    fig_height = max(5.5, 0.32 * len(models) + 2) + legend_height
    fig, (ax_bar, ax_line) = plt.subplots(1, 2, figsize=(14, fig_height))
    title = f"Model Evaluation Comparison — {ylabel}"
    if department:
        title = f"{department} — {title}"
    if top_n is not None and len(metrics) > len(models):
        title += f"  (top {len(models)} of {len(metrics)})"
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # ------------------------------------------------------------------ #
    # Panel 1: Grouped horizontal bar chart at h_short and h_long — models
    # stack down the y-axis (comfortably fits many rows without label
    # rotation/overlap) instead of packing names into a fixed-width x-axis.
    # ------------------------------------------------------------------ #
    y = np.arange(len(models))
    width = 0.32
    vals_s = [metrics[m].get(h_short, {}).get(metric, np.nan) for m in models]
    vals_l = [metrics[m].get(h_long,  {}).get(metric, np.nan) for m in models]

    bars_s = ax_bar.barh(
        y + width / 2, vals_s, width,
        label=f"Horizon h={h_short}",
        color=[colors[m] for m in models],
        alpha=0.9,
        edgecolor="white",
        linewidth=0.7,
    )
    bars_l = ax_bar.barh(
        y - width / 2, vals_l, width,
        label=f"Horizon h={h_long}",
        color=[colors[m] for m in models],
        alpha=0.45,
        edgecolor=[colors[m] for m in models],
        linewidth=1.2,
    )

    # Value labels beside (or before, for negative R²) each bar
    all_vals = [v for v in vals_s + vals_l if not np.isnan(v)]
    val_range = (max(all_vals) - min(all_vals)) if all_vals else 1
    offset = val_range * 0.02 or 0.05

    for bar in list(bars_s) + list(bars_l):
        w = bar.get_width()
        if not np.isnan(w):
            ax_bar.text(
                w + offset if w >= 0 else w - offset * 3,
                bar.get_y() + bar.get_height() / 2,
                f"{w:.2f}" if metric == "r2" else f"{w:.1f}",
                ha="left" if w >= 0 else "right",
                va="center",
                fontsize=8,
            )

    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels(models, fontsize=9)
    ax_bar.set_xlabel(ylabel)
    better = better_label(metric)
    ax_bar.set_title(f"h={h_short} (solid) vs h={h_long} (faded) — {better}")
    ax_bar.xaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax_bar.grid(axis="x", alpha=0.25)
    ax_bar.grid(axis="x", which="minor", alpha=0.12)
    ax_bar.margins(y=0.02)
    if metric in {"r2", "me"}:
        ax_bar.axvline(0, color="black", lw=0.8, ls="--", alpha=0.5)
    # No legend here — the title above already states "h=short (solid) vs
    # h=long (faded)", so a legend would just repeat it while eating into
    # the same space the line chart's (real, per-model) legend needs below.

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
    ax_line.grid(alpha=0.25)
    if metric in {"r2", "me"}:
        ax_line.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)

    # Figure-level legend (not attached to either axes) — its position is in
    # figure fractions, so it stays clear of both panels' xlabels regardless
    # of axes size, unlike an axes-anchored legend whose "below the axes"
    # offset has to be re-tuned any time the axes' own height/labels change.
    handles = [lines_by_model[m] for m in models]
    legend = fig.legend(
        handles, models, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 0.0),
        ncol=n_legend_cols,
    )
    enable_legend_picking(fig, legend, lines_by_model)

    # Reserve exactly the figure fraction the legend needs (computed in
    # inches above, converted to a fraction of this specific fig_height) —
    # tight_layout() has no way to know a figure-level legend needs room.
    fig.tight_layout(rect=[0, legend_height / fig_height, 1, 1])

    if save_path is not None:
        save_path = Path(save_path)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved: {save_path}")

    if show:
        plt.show()
    plt.close(fig)
    return save_path


def _draw_model_date_heatmap(
    fig,
    ax,
    model_frames: dict,
    metric: str,
    models_order: list,
    fill_method: Optional[str] = "ffill",
) -> bool:
    """
    Draw a model-by-date heatmap onto `ax`: rows are models (in `models_order`),
    columns are dates, color is the selected metric value for that date. When
    `fill_method` is 'ffill', values are forward-filled after reindexing so the
    heatmap can represent cumulative or expanding-window metrics.
    Returns False (and leaves `ax` with just a note) if no model has data for
    `metric`.

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
        if m in model_frames
        and metric in model_frames[m].columns
        and model_frames[m][metric].notna().any()
    ]
    if not models:
        ax.text(0.5, 0.5, "No cumulative data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return False

    all_dates = sorted(set().union(*(set(model_frames[m]["date"]) for m in models)))
    date_index = pd.DatetimeIndex(all_dates)

    matrix = np.full((len(models), len(date_index)), np.nan)
    for i, m in enumerate(models):
        # groupby(...).last() instead of set_index(...) — a date can repeat
        # within one model's cumulative frame when horizon > step (overlapping
        # walk-forward windows put more than one evaluation on the same
        # calendar date); take the last (most complete) expanding-window
        # value for that date so reindex() has a unique index to align to.
        s = model_frames[m].groupby("date")[metric].last().reindex(date_index)
        real_dates = model_frames[m]["date"]
        if fill_method == "ffill":
            s = s.ffill()
            # Don't let ffill extend past this model's own last real
            # evaluation — models with different coverage (e.g. a
            # tune_models.py trial that only has a pre-holdout range, next
            # to one that also has its holdout confirmed) share this date
            # axis; without this, a shorter model's last known value would
            # visually repeat all the way to the longest model's end date,
            # making it look like it has data (and "is fine") in a range it
            # was never actually evaluated on.
            s[date_index > real_dates.max()] = np.nan
        elif fill_method == "bfill":
            s = s.bfill()
            s[date_index < real_dates.min()] = np.nan
        matrix[i] = s.to_numpy()

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
    top_n: Optional[int] = None,
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
        top_n:      If given, keep only the top_n best models (ranked by
                    `metric`) — with many models, the full set overlaps and
                    becomes unreadable. Both panels are restricted to the
                    same subset.

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
    n_total = len(values)
    models  = sorted(values, key=lambda m: values[m], reverse=higher_is_better)
    if top_n is not None and len(models) > top_n:
        models = models[:top_n]
    # Best → worst top-to-bottom on a horizontal bar chart reads naturally
    # (best at top); reversed here because barh plots first-to-last bottom-to-top.
    models = models[::-1]
    palette = plt.cm.tab10.colors
    colors  = [palette[i % len(palette)] for i in range(len(models))]
    ylabel  = METRIC_LABELS[metric]
    better  = better_label(metric)

    fig_height = max(5.5, 0.32 * len(models) + 2)
    fig, (ax_bar, ax_heat) = plt.subplots(
        1, 2, figsize=(15, fig_height), gridspec_kw={"width_ratios": [1, 1.6]}
    )
    title = f"Model Evaluation Comparison (microaverage) — {ylabel}"
    if department:
        title = f"{department} — {title}"
    if top_n is not None and n_total > len(models):
        title += f"  (top {len(models)} of {n_total})"
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # ------------------------------------------------------------------ #
    # Panel 1: horizontal bar chart — final pooled snapshot. Models stack
    # down the y-axis instead of packing names into a fixed-width x-axis.
    # ------------------------------------------------------------------ #
    bars = ax_bar.barh(models, [values[m] for m in models], color=colors, alpha=0.85)
    fmt = "%.2f" if metric == "r2" else "%.1f"
    ax_bar.bar_label(bars, fmt=fmt, fontsize=8.5, padding=3)

    ax_bar.set_xlabel(ylabel)
    ax_bar.set_title("Final pooled snapshot — best → worst, top to bottom")
    ax_bar.tick_params(axis="y", labelsize=9)
    ax_bar.xaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax_bar.grid(axis="x", alpha=0.25)
    ax_bar.grid(axis="x", which="minor", alpha=0.12)
    ax_bar.margins(y=0.02)
    if metric in {"r2", "me"}:
        ax_bar.axvline(0, color="black", lw=0.8, ls="--", alpha=0.5)
    ax_bar.set_xlabel(f"{ylabel}  ({better})")

    # ------------------------------------------------------------------ #
    # Panel 2: cumulative heatmap — same models, ordered to match the bars
    # ------------------------------------------------------------------ #
    _draw_model_date_heatmap(fig, ax_heat, cumulative_metrics, metric, models, fill_method="ffill")
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
