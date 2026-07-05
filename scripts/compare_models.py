#!/usr/bin/env python
"""
Walk-forward model comparison across horizons.

Reads all *_walkforward_metrics.json files for a department/age_group and
produces:
  1. A printed + saved comparison table (model × horizon for MAE, RMSE, ME, R², SMAPE, MDA)
  2. A two-panel figure via pneumonia.visualization.comparison_plot:
       - Grouped bar chart: selected metric at h_short vs h_long
       - Line chart: metric across all horizons per model

Output filenames follow the project convention:
    - Horizon comparisons: `comparison_horizon_{metric}[_YYYY].png` and `comparison_horizon.csv`
    - Microaverage:         `comparison_microaverage_{metric}[_YYYY].png` and `comparison_microaverage[_YYYY].csv`

Usage:
    python scripts/compare_models.py --department AMAZONAS
    python scripts/compare_models.py --department AMAZONAS --age_group 60plus
    python scripts/compare_models.py --department AMAZONAS --metric rmse
    python scripts/compare_models.py --department AMAZONAS --horizons 1 4

Use --mode to pick the aggregation strategy:
  horizon      (default) — table + bar/line charts from metrics_by_horizon, where
               each metric is computed once over all steps pooled per horizon.
    macroaverage — per-step diagnostic figures (boxplot + mean, and metric heatmap
                             by forecast date) via pneumonia.visualization.step_metrics_plot.
                             Non-additive ratio metrics (e.g. r2 — see NON_ADDITIVE_METRICS in comparison_plot.py)
                             are excluded from the default metric list in this mode, since averaging
               per-step ratios is not equivalent to computing them on pooled data.
               Pass --metric explicitly to include them anyway.
    microaverage — table + a two-panel figure per metric from every backtest
               prediction pooled across ALL steps/horizons (like 'horizon' but
               without splitting by h): a bar chart of the final pooled snapshot,
               plus a heatmap (rows=models, columns=date) of the same pooled
               metric as an expanding window over time. r2 is valid here (not
               excluded) because it's computed once on the full pool, not
               averaged per group. --year restricts to one calendar year.
    python scripts/compare_models.py --department AMAZONAS --mode macroaverage
    python scripts/compare_models.py --department AMAZONAS --mode macroaverage --year 2020
    python scripts/compare_models.py --department AMAZONAS --mode macroaverage --metric r2
    python scripts/compare_models.py --department AMAZONAS --mode microaverage
    python scripts/compare_models.py --department AMAZONAS --mode microaverage --year 2020

Add --interactive to open the plot in a window (plt.show()) instead of just
saving it to disk. Requires exactly one --metric, so you only ever have one
window to close:
    python scripts/compare_models.py --department AMAZONAS --metric r2 --interactive
    python scripts/compare_models.py --department AMAZONAS --mode microaverage \\
        --metric r2 --interactive
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from pneumonia.config import REPORTS_PATH
from pneumonia.evaluation.metrics import compute_all_metrics
from pneumonia.utils import setup_logger
from pneumonia.visualization._utils import read_predictions
from pneumonia.visualization.comparison_plot import (
    METRIC_LABELS,
    NON_ADDITIVE_METRICS,
    VALID_METRICS,
    plot_micro_comparison,
    plot_model_comparison,
)
from pneumonia.visualization.persistence import load_step_metrics
from pneumonia.visualization.step_metrics_plot import plot_step_metrics

logger = setup_logger(__name__)

TABLE_METRICS = ["mae", "rmse", "me", "r2", "smape", "mda"]
TABLE_HEADERS = {
    "mae":   "MAE",
    "rmse":  "RMSE",
    "me":    "ME",
    "mape":  "MAPE (%)",
    "smape": "SMAPE (%)",
    "mda":   "MDA (%)",
    "r2":    "R²",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_metrics(reports_dir: Path, department: str, age_group: str) -> dict:
    """Return {run_name: {horizon_int: {metric: value}}} from walkforward JSONs."""
    out_dir = Path(reports_dir) / department / age_group
    files   = sorted(out_dir.glob("*_walkforward_metrics.json"))
    if not files:
        raise FileNotFoundError(
            f"No walkforward metrics found in {out_dir}.\n"
            "Run scripts/run_walkforward.py first."
        )
    data = {}
    for f in files:
        with open(f, encoding="utf-8-sig") as fh:
            j = json.load(fh)
        run_name       = j.get("run_name", j["model"])
        data[run_name] = {int(h): v for h, v in j["metrics_by_horizon"].items()}
    return data


def _load_backtest_predictions(
    reports_dir: Path, department: str, age_group: str, year: int = None
) -> pd.DataFrame:
    """Return backtest-split rows from *_predictions.csv, or raise FileNotFoundError."""
    df = read_predictions(reports_dir, department, age_group)
    if df is None:
        raise FileNotFoundError(
            f"No predictions found for {department}/{age_group}.\n"
            "Run scripts/run_walkforward.py first."
        )
    backtest = df[df["split"] == "backtest"]
    if year is not None:
        backtest = backtest[backtest["date"].dt.year == year]
    if backtest.empty:
        suffix = f" in year={year}" if year is not None else ""
        raise FileNotFoundError(
            f"No backtest predictions found for {department}/{age_group}{suffix}.\n"
            "Run scripts/run_walkforward.py first."
        )
    return backtest


def load_micro_metrics(
    reports_dir: Path, department: str, age_group: str, year: int = None
) -> dict:
    """
    Return {run_name: {metric: value}} computed once over every backtest
    prediction pooled across all steps/horizons (no per-horizon split) —
    same underlying computation as metrics_by_horizon, just not divided by h.
    If `year` is given, restricts to backtest predictions from that calendar year.
    """
    backtest = _load_backtest_predictions(reports_dir, department, age_group, year)
    data = {}
    for run_name, g in backtest.groupby("model"):
        data[run_name] = compute_all_metrics(
            g["actual"].values, g["predicted"].values, warn_on_nan=False
        )
    return data


def compute_cumulative_micro_metrics(
    reports_dir: Path, department: str, age_group: str, min_obs: int = 8, year: int = None
) -> dict:
    """
    Return {run_name: DataFrame(date, mae, rmse, me, mape, smape, mda, r2)} — one
    row per evaluated backtest date from the min_obs'th evaluation onward, each
    computed once on every prediction pooled from the start of the backtest up
    to and including that date (expanding window). Shows how the pooled metric
    evolves/stabilizes over time, unlike load_micro_metrics's single final
    snapshot. If `year` is given, the window starts fresh at that year (not
    blended with prior years' history), matching macroaverage's --year semantics.

    min_obs skips the earliest, tiniest windows (e.g. r2's SS_tot from just 1-2
    points can be near zero, sending it to wild outliers that would otherwise
    dominate a shared color scale).
    """
    backtest = _load_backtest_predictions(reports_dir, department, age_group, year)
    data = {}
    for run_name, g in backtest.groupby("model"):
        g = g.sort_values("date")
        actual    = g["actual"].to_numpy()
        predicted = g["predicted"].to_numpy()
        start = min(min_obs, len(g))
        rows = []
        for i in range(start, len(g) + 1):
            m = compute_all_metrics(actual[:i], predicted[:i], warn_on_nan=False)
            m["date"] = g["date"].iloc[i - 1]
            rows.append(m)
        data[run_name] = pd.DataFrame(rows)
    return data


def build_table(metrics: dict, horizons: list) -> pd.DataFrame:
    """Build model × (horizon × metric) DataFrame for all TABLE_METRICS."""
    rows = []
    for model, by_h in metrics.items():
        row = {"Model": model}
        for h in horizons:
            if h not in by_h:
                continue
            for key in TABLE_METRICS:
                val = by_h[h].get(key, np.nan)
                row[f"h={h} {TABLE_HEADERS[key]}"] = (
                    round(val, 2) if not np.isnan(val) else np.nan
                )
        rows.append(row)
    return pd.DataFrame(rows).set_index("Model")


def build_micro_table(micro_metrics: dict) -> pd.DataFrame:
    """Build model × metric DataFrame for all TABLE_METRICS, pooled (no horizon split)."""
    rows = []
    for model, m in micro_metrics.items():
        row = {"Model": model}
        for key in TABLE_METRICS:
            val = m.get(key, np.nan)
            row[TABLE_HEADERS[key]] = round(val, 2) if not np.isnan(val) else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("Model")


def degradation_pct(metrics: dict, h_short: int, h_long: int, metric: str) -> pd.Series:
    """
    Metric change from h_short to h_long per model.

    For error metrics (mae, rmse, smape, mape): (v_long/v_short - 1) × 100 % — positive = worse.
    For score metrics (mda, r2):                v_long - v_short (absolute) — negative = worse.
    For bias (me): ideal value is 0, not min/max, so direction doesn't indicate "better" —
        only |ME| growing means the model's over/under-prediction is getting worse:
        |v_long| - |v_short| (absolute) — positive = bias grew.
    """
    result = {}
    label  = TABLE_HEADERS.get(metric, metric.upper())
    higher_is_better = metric in {"mda", "r2"}
    is_bias = metric == "me"

    for model, by_h in metrics.items():
        v_s = by_h.get(h_short, {}).get(metric, np.nan)
        v_l = by_h.get(h_long,  {}).get(metric, np.nan)
        if np.isnan(v_s) or np.isnan(v_l):
            result[model] = np.nan
        elif is_bias:
            result[model] = round(abs(v_l) - abs(v_s), 3)
        elif higher_is_better:
            result[model] = round(v_l - v_s, 3)
        elif v_s > 0:
            result[model] = round((v_l / v_s - 1) * 100, 1)
        else:
            result[model] = np.nan

    if is_bias:
        label, unit = f"{label} magnitude", ""
    else:
        unit = "" if higher_is_better else " (%)"
    return pd.Series(result, name=f"{label} change h{h_short}→h{h_long}{unit}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def compare_macroaverage(
    department: str,
    age_group: str,
    metric_names: list,
    trend_window: int = 13,
    year: int = None,
    interactive: bool = False,
) -> None:
    """Per-step diagnostic figures (boxplot + mean, and metric evolution over time)."""
    department = department.upper()
    out_dir    = Path(REPORTS_PATH) / department / age_group

    try:
        step_data = load_step_metrics(REPORTS_PATH, department, age_group)
    except FileNotFoundError as exc:
        logger.warning(f"Skipping step metrics for {department}: {exc}")
        return

    if year is not None:
        step_data = {m: df[df["date"].dt.year == year] for m, df in step_data.items()}
        step_data = {m: df for m, df in step_data.items() if not df.empty}
        if not step_data:
            logger.warning(f"No step metrics found for year={year} in {department}")
            return

    suffix = f"_{year}" if year is not None else ""
    for metric in metric_names:
        fig_path = out_dir / f"step_metrics_{metric}{suffix}.png"
        plot_step_metrics(
            step_data    = step_data,
            metric       = metric,
            department   = department,
            save_path    = fig_path,
            trend_window = trend_window,
            show         = interactive,
        )


def compare_microaverage(
    department: str,
    age_group: str,
    metric_names: list,
    interactive: bool = False,
    year: int = None,
) -> None:
    """
    Model comparison pooling every backtest prediction across all steps/horizons —
    each metric computed once per model (same philosophy as horizon mode, just not
    divided by horizon). If `year` is given, restricted to that calendar year only
    (same semantics as macroaverage's --year).
    """
    department = department.upper()
    out_dir    = Path(REPORTS_PATH) / department / age_group

    try:
        micro_metrics = load_micro_metrics(REPORTS_PATH, department, age_group, year=year)
        cumulative_metrics = compute_cumulative_micro_metrics(
            REPORTS_PATH, department, age_group, year=year
        )
    except FileNotFoundError as exc:
        logger.warning(f"Skipping microaverage for {department}: {exc}")
        return

    table = build_micro_table(micro_metrics)
    label = f"{department} / {age_group}" + (f" ({year})" if year is not None else "")
    print(f"\n{'='*70}")
    print(f"Model comparison (microaverage) — {label}")
    print(f"{'='*70}")
    print(table.to_string())
    print(f"{'='*70}\n")

    suffix = f"_{year}" if year is not None else ""
    csv_path = out_dir / f"comparison_microaverage{suffix}.csv"
    table.to_csv(csv_path)
    print(f"Table saved: {csv_path}")

    for metric in metric_names:
        fig_path = out_dir / f"comparison_microaverage_{metric}{suffix}.png"
        plot_micro_comparison(
            metrics             = micro_metrics,
            cumulative_metrics  = cumulative_metrics,
            metric              = metric,
            department          = department,
            save_path           = fig_path,
            show                = interactive,
        )


def compare(
    department: str,
    age_group: str,
    horizons: list,
    metric_names: list,
    mode: str = "horizon",
    trend_window: int = 13,
    year: int = None,
    interactive: bool = False,
) -> None:
    department = department.upper()

    if mode == "macroaverage":
        compare_macroaverage(
            department, age_group, metric_names, trend_window, year, interactive
        )
        return
    if mode == "microaverage":
        compare_microaverage(department, age_group, metric_names, interactive, year)
        return

    metrics = load_metrics(REPORTS_PATH, department, age_group)

    available_horizons = sorted({h for m in metrics.values() for h in m})
    horizons = [h for h in horizons if h in available_horizons]
    if not horizons:
        raise ValueError(f"No requested horizons found. Available: {available_horizons}")

    h_short = horizons[0]
    h_long  = horizons[-1]

    # --- Table ---
    table = build_table(metrics, horizons)

    print(f"\n{'='*70}")
    print(f"Model comparison — {department} / {age_group}")
    print(f"{'='*70}")
    print(table.to_string())

    out_dir    = Path(REPORTS_PATH) / department / age_group
    full_table = table.copy()

    for metric in metric_names:
        degr = degradation_pct(metrics, h_short, h_long, metric)
        change_label = "change" if metric in {"mda", "r2", "me"} else "degradation"
        print(f"\n--- {METRIC_LABELS[metric]} {change_label} h={h_short}→h={h_long} ---")
        print(degr.sort_values().to_string())
        full_table[degr.name] = degr

        suffix = f"_{year}" if year is not None else ""
        fig_path = out_dir / f"comparison_horizon_{metric}{suffix}.png"
        plot_model_comparison(
            metrics    = metrics,
            horizons   = available_horizons,
            h_short    = h_short,
            h_long     = h_long,
            metric     = metric,
            department = department,
            save_path  = fig_path,
            show       = interactive,
        )
    print(f"{'='*70}\n")

    csv_path = out_dir / "comparison_horizon.csv"
    full_table.to_csv(csv_path)
    print(f"Table saved: {csv_path}")

    # --- Plain-language summary ---
    mae_s  = {m: metrics[m].get(h_short, {}).get("mae", np.nan) for m in metrics}
    mae_l  = {m: metrics[m].get(h_long,  {}).get("mae", np.nan) for m in metrics}
    mda_s  = {m: metrics[m].get(h_short, {}).get("mda", np.nan) for m in metrics}
    r2_s   = {m: metrics[m].get(h_short, {}).get("r2",  np.nan) for m in metrics}
    mae_degr = degradation_pct(metrics, h_short, h_long, "mae")

    mae_s_valid = {m: v for m, v in mae_s.items() if not np.isnan(v)}
    mae_l_valid = {m: v for m, v in mae_l.items() if not np.isnan(v)}
    mda_s_valid = {m: v for m, v in mda_s.items() if not np.isnan(v)}
    r2_s_valid  = {m: v for m, v in r2_s.items()  if not np.isnan(v)}
    stable_degr = mae_degr.dropna()

    print("=== Conclusiones automáticas ===")
    if mae_s_valid:
        best_short = min(mae_s_valid, key=mae_s_valid.__getitem__)
        print(f"  Mejor modelo a corto plazo (h={h_short}, MAE): "
              f"{best_short} ({mae_s_valid[best_short]:.2f})")
    if mae_l_valid:
        best_long = min(mae_l_valid, key=mae_l_valid.__getitem__)
        print(f"  Mejor modelo a largo plazo  (h={h_long},  MAE): "
              f"{best_long} ({mae_l_valid[best_long]:.2f})")
    if not stable_degr.empty:
        most_stable = stable_degr.idxmin()
        print(f"  Más estable ante horizonte largo (menor degradación MAE): "
              f"{most_stable} ({stable_degr[most_stable]:+.1f}%)")
    if mda_s_valid:
        best_mda = max(mda_s_valid, key=mda_s_valid.__getitem__)
        print(f"  Mejor dirección de cambio (MDA h={h_short}): "
              f"{best_mda} ({mda_s_valid[best_mda]:.1f}%)")
    if r2_s_valid:
        best_r2 = max(r2_s_valid, key=r2_s_valid.__getitem__)
        print(f"  Mejor ajuste general (R² h={h_short}): "
              f"{best_r2} ({r2_s_valid[best_r2]:.3f})")
    print()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--department", "-d", required=True, nargs="+",
                        help="Department name(s) (e.g. AMAZONAS LIMA, or comma-separated: AMAZONAS,LIMA)")
    parser.add_argument("--age_group",  "-g", default="under5",
                        choices=["under5", "60plus"],
                        help="Age group (default: under5)")
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 3, 4],
                        help="Horizons to include in the table (default: 1 2 3 4)")
    parser.add_argument("--metric", "-m", nargs="+", default=None,
                        choices=sorted(VALID_METRICS) + ["all"],
                        help="Metric(s) for the bar/line charts. Default depends on --mode: "
                             "all metrics for 'horizon', or all except "
                             f"{sorted(NON_ADDITIVE_METRICS)} for 'macroaverage' (pass this "
                             "explicitly to include them anyway).")
    parser.add_argument("--mode", choices=["horizon", "macroaverage", "microaverage"],
                        default="horizon",
                        help="Aggregation strategy: 'horizon' (default) — table + bar/line "
                             "charts from metrics_by_horizon. 'macroaverage' — per-step "
                            "diagnostic figures (boxplot + mean, and metric heatmap by "
                            "forecast date) from *_step_metrics.csv. 'microaverage' — "
                            "table + bar/heatmap comparison from every backtest prediction "
                            "pooled across all steps/horizons, each metric computed once "
                            "per model (like 'horizon' but without splitting by h).")
    parser.add_argument("--trend_window", type=int, default=None,
                        help="[--mode macroaverage] Retained for backward compatibility; "
                            "the current macroaverage right panel is a heatmap and does not use "
                            "rolling smoothing. Default: 13 normally, or 4 when --year is set.")
    parser.add_argument("--year", type=int, default=None,
                        help="[--mode macroaverage/microaverage] Restrict to a single "
                             "calendar year (default: all years). macroaverage: filters "
                            "step metrics (boxplot + heatmap). microaverage: pools "
                             "only that year's backtest predictions, and the cumulative "
                             "heatmap window starts fresh at that year instead of the full "
                             "history.")
    parser.add_argument("--interactive", action="store_true",
                        help="Open the plot in a window (plt.show()) instead of just saving "
                             "it to disk. Requires exactly one --metric, so only one window "
                             "needs to be closed.")
    args = parser.parse_args()

    departments = []
    for d in args.department:
        departments.extend([x.strip().upper() for x in d.split(",") if x.strip()])

    if args.metric is not None:
        metric_names = sorted(VALID_METRICS) if "all" in args.metric else args.metric
    elif args.mode == "macroaverage":
        metric_names = sorted(VALID_METRICS - NON_ADDITIVE_METRICS)
        print(
            f"[macroaverage] Excluding non-additive metrics by default: "
            f"{sorted(NON_ADDITIVE_METRICS)} (pass --metric to include them explicitly)"
        )
    else:
        metric_names = sorted(VALID_METRICS)

    if args.interactive and len(metric_names) != 1:
        parser.error(
            f"--interactive requires exactly one metric, got {metric_names}. "
            "Pass a single --metric <name>."
        )

    if args.trend_window is not None:
        trend_window = args.trend_window
    else:
        trend_window = 4 if args.year is not None else 13

    for dept in departments:
        try:
            compare(
                dept, args.age_group, args.horizons, metric_names,
                mode=args.mode, trend_window=trend_window,
                year=args.year, interactive=args.interactive,
            )
        except Exception as exc:
            logger.error(f"Failed to compare models for {dept}: {exc}")


if __name__ == "__main__":
    main()
