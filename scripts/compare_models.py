#!/usr/bin/env python
"""
Walk-forward model comparison across horizons.

Reads all *_walkforward_metrics.json files for a department/age_group and
produces:
  1. A printed + saved comparison table (model × horizon for MAE, RMSE, ME, R², SMAPE, MDA)
  2. A two-panel figure via pneumonia.visualization.comparison_plot:
       - Grouped bar chart: selected metric at h_short vs h_long
       - Line chart: metric across all horizons per model

Usage:
    python scripts/compare_models.py --department AMAZONAS
    python scripts/compare_models.py --department AMAZONAS --age_group 60plus
    python scripts/compare_models.py --department AMAZONAS --metric rmse
    python scripts/compare_models.py --department AMAZONAS --horizons 1 4

Add --step_metrics to also render the per-step diagnostic figures (boxplot +
mean, and metric evolution over time) via pneumonia.visualization.step_metrics_plot
— same data source used standalone by scripts/plot_step_metrics.py:
    python scripts/compare_models.py --department AMAZONAS --step_metrics
    python scripts/compare_models.py --department AMAZONAS --step_metrics --year 2020
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from pneumonia.config import REPORTS_PATH
from pneumonia.utils import setup_logger
from pneumonia.visualization.comparison_plot import (
    METRIC_LABELS,
    VALID_METRICS,
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

def compare(
    department: str,
    age_group: str,
    horizons: list,
    metric_names: list,
    step_metrics: bool = False,
    trend_window: int = 13,
    year: int = None,
) -> None:
    department = department.upper()
    metrics    = load_metrics(REPORTS_PATH, department, age_group)

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

        fig_path = out_dir / f"model_comparison_{metric}.png"
        plot_model_comparison(
            metrics    = metrics,
            horizons   = available_horizons,
            h_short    = h_short,
            h_long     = h_long,
            metric     = metric,
            department = department,
            save_path  = fig_path,
        )
    print(f"{'='*70}\n")

    csv_path = out_dir / "model_comparison.csv"
    full_table.to_csv(csv_path)
    print(f"Table saved: {csv_path}")

    # --- Per-step diagnostic figures (optional) ---
    if step_metrics:
        try:
            step_data = load_step_metrics(REPORTS_PATH, department, age_group)
        except FileNotFoundError as exc:
            logger.warning(f"Skipping step metrics for {department}: {exc}")
        else:
            if year is not None:
                step_data = {
                    m: df[df["date"].dt.year == year] for m, df in step_data.items()
                }
                step_data = {m: df for m, df in step_data.items() if not df.empty}
                if not step_data:
                    logger.warning(f"No step metrics found for year={year} in {department}")

            suffix = f"_{year}" if year is not None else ""
            for metric in metric_names:
                fig_path = out_dir / f"step_metrics_{metric}{suffix}.png"
                plot_step_metrics(
                    step_data        = step_data,
                    metric           = metric,
                    department       = department,
                    save_path        = fig_path,
                    trend_window = trend_window,
                )

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
    parser.add_argument("--metric", "-m", nargs="+", default=["all"],
                        choices=sorted(VALID_METRICS) + ["all"],
                        help="Metric(s) for the bar/line charts (default: all)")
    parser.add_argument("--step_metrics", action="store_true",
                        help="Also render per-step diagnostic figures (boxplot + mean, "
                             "and metric evolution over time) from *_step_metrics.csv")
    parser.add_argument("--trend_window", type=int, default=None,
                        help="[--step_metrics] Steps to average over for the time-series "
                             "overlay. Default: 13 normally, or 4 when --year is set (a "
                             "single year has too few steps for a 13-step window). Not "
                             "related to run_walkforward.py's --window_type — this only "
                             "smooths the diagnostic plot.")
    parser.add_argument("--year", type=int, default=None,
                        help="[--step_metrics] Restrict step metrics (boxplot + time "
                             "evolution) to a single calendar year (default: all years)")
    args = parser.parse_args()

    departments = []
    for d in args.department:
        departments.extend([x.strip().upper() for x in d.split(",") if x.strip()])

    metric_names = sorted(VALID_METRICS) if "all" in args.metric else args.metric

    if args.trend_window is not None:
        trend_window = args.trend_window
    else:
        trend_window = 4 if args.year is not None else 13

    for dept in departments:
        try:
            compare(
                dept, args.age_group, args.horizons, metric_names,
                step_metrics=args.step_metrics, trend_window=trend_window,
                year=args.year,
            )
        except Exception as exc:
            logger.error(f"Failed to compare models for {dept}: {exc}")


if __name__ == "__main__":
    main()
