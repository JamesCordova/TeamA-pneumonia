#!/usr/bin/env python
"""
Walk-forward model comparison across horizons.

Reads all *_walkforward_metrics.json files for a department/age_group and
produces:
  1. A printed + saved comparison table (model × horizon for MAE, RMSE, SMAPE, MDA)
  2. A two-panel figure via pneumonia.visualization.comparison_plot:
       - Grouped bar chart: selected metric at h_short vs h_long
       - Line chart: metric across all horizons per model

Usage:
    python scripts/compare_models.py --department AMAZONAS
    python scripts/compare_models.py --department AMAZONAS --age_group 60plus
    python scripts/compare_models.py --department AMAZONAS --metric rmse
    python scripts/compare_models.py --department AMAZONAS --horizons 1 4
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

logger = setup_logger(__name__)

TABLE_METRICS = ["mae", "rmse", "me", "r2", "mase", "smape", "mda"]
TABLE_HEADERS = {
    "mae":   "MAE",
    "rmse":  "RMSE",
    "me":    "ME",
    "r2":    "R2",
    "mase":  "MASE",
    "smape": "SMAPE (%)",
    "mda":   "MDA (%)",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_metrics(reports_dir: Path, department: str, age_group: str) -> dict:
    """Return {model_name: {horizon_int: {metric: value}}} from walkforward JSONs."""
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
        model      = j["model"]
        data[model] = {int(h): v for h, v in j["metrics_by_horizon"].items()}
    return data


def build_table(metrics: dict, horizons: list) -> pd.DataFrame:
    """Build model × (horizon × metric) DataFrame with MAE, RMSE, SMAPE, MDA."""
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
    """(metric_hlong / metric_hshort - 1) × 100 per model for the given metric."""
    result = {}
    label  = TABLE_HEADERS.get(metric, metric.upper())
    for model, by_h in metrics.items():
        v_s = by_h.get(h_short, {}).get(metric, np.nan)
        v_l = by_h.get(h_long,  {}).get(metric, np.nan)
        if v_s and v_s > 0:
            result[model] = round((v_l / v_s - 1) * 100, 1)
        else:
            result[model] = np.nan
    return pd.Series(result, name=f"{label} degradation h{h_short}->h{h_long} (%)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def compare(department: str, age_group: str, horizons: list, metric: str) -> None:
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
    degr  = degradation_pct(metrics, h_short, h_long, metric)

    print(f"\n{'='*70}")
    print(f"Model comparison — {department} / {age_group}")
    print(f"{'='*70}")
    print(table.to_string())
    print(f"\n--- {METRIC_LABELS[metric]} degradation h={h_short}->h={h_long} ---")
    print(degr.sort_values().to_string())
    print(f"{'='*70}\n")

    out_dir  = Path(REPORTS_PATH) / department / age_group
    csv_path = out_dir / "model_comparison.csv"
    full_table = table.copy()
    full_table[degr.name] = degr
    full_table.to_csv(csv_path)
    print(f"Table saved: {csv_path}")

    # --- Figure ---
    fig_path = out_dir / f"model_comparison_{metric}.png"
    plot_model_comparison(
        metrics    = metrics,
        horizons   = available_horizons,
        h_short    = h_short,
        h_long     = h_long,
        metric     = metric,
        save_path  = fig_path,
    )

    # --- Plain-language summary (always in MAE for interpretability) ---
    mae_s  = {m: metrics[m].get(h_short, {}).get("mae", np.nan) for m in metrics}
    mae_l  = {m: metrics[m].get(h_long,  {}).get("mae", np.nan) for m in metrics}
    mda_s  = {m: metrics[m].get(h_short, {}).get("mda", np.nan) for m in metrics}
    mae_degr = degradation_pct(metrics, h_short, h_long, "mae")

    best_short  = min(mae_s, key=lambda m: mae_s[m] if not np.isnan(mae_s[m]) else np.inf)
    best_long   = min(mae_l, key=lambda m: mae_l[m] if not np.isnan(mae_l[m]) else np.inf)
    best_mda    = max(mda_s, key=lambda m: mda_s[m] if not np.isnan(mda_s[m]) else -np.inf)
    most_stable = mae_degr.dropna().idxmin()

    print("=== Conclusiones automáticas ===")
    print(f"  Mejor modelo a corto plazo (h={h_short}, MAE): "
          f"{best_short} ({mae_s[best_short]:.2f})")
    print(f"  Mejor modelo a largo plazo  (h={h_long},  MAE): "
          f"{best_long} ({mae_l[best_long]:.2f})")
    degr_val = mae_degr[most_stable]
    degr_str = f"{degr_val:+.1f}%"
    print(f"  Más estable ante horizonte largo (menor degradación MAE): "
          f"{most_stable} ({degr_str})")
    print(f"  Mejor dirección de cambio (MDA h={h_short}): "
          f"{best_mda} ({mda_s[best_mda]:.1f}%)")
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
    parser.add_argument("--metric", "-m", default="mae",
                        choices=sorted(VALID_METRICS),
                        help="Metric for the bar/line chart (default: mae)")
    args = parser.parse_args()

    departments = []
    for d in args.department:
        departments.extend([x.strip().upper() for x in d.split(",") if x.strip()])

    for dept in departments:
        try:
            compare(dept, args.age_group, args.horizons, args.metric)
        except Exception as exc:
            logger.error(f"Failed to compare models for {dept}: {exc}")


if __name__ == "__main__":
    main()
