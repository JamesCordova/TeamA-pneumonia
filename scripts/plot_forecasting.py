#!/usr/bin/env python
"""
Plot forecast comparison from stored predictions CSV.

Each pipeline run (SARIMA, baselines, RandomForest) writes its predictions to:
  reports/{DEPT}/{AGE_GROUP}/predictions.csv

This script reads that CSV and generates a multi-model comparison figure.
Running multiple pipelines before plotting will show all models together.

Usage:
    python scripts/plot_forecasting.py --department AMAZONAS --age_group under5
    python scripts/plot_forecasting.py --all --age_group 60plus
    python scripts/plot_forecasting.py --department LIMA --output my_plot.png
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pneumonia.config import REPORTS_PATH
from pneumonia.models.utils import get_available_departments
from pneumonia.utils import setup_logger
from pneumonia.visualization.forecast_plot import plot_forecasts

logger = setup_logger(__name__)


def plot_one(department: str, age_group: str, models=None, output: Path = None) -> None:
    path = plot_forecasts(
        department=department.upper(),
        age_group=age_group.lower(),
        reports_dir=REPORTS_PATH,
        models=models,
        save_path=output,
        show=False,
    )
    if path:
        print(f"Plot saved: {path}")
    else:
        print(
            f"No predictions CSV found for {department}/{age_group}.\n"
            f"Run at least one pipeline first:\n"
            f"  python scripts/train_sarima.py --department {department} --age_group {age_group}\n"
            f"  python scripts/train_baselines.py --department {department} --age_group {age_group}"
        )


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot forecast comparison from stored predictions CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/plot_forecasting.py --department AMAZONAS --age_group under5
  python scripts/plot_forecasting.py --all --age_group 60plus
  python scripts/plot_forecasting.py --department LIMA --output reports/lima_plot.png
        """,
    )

    dept_group = parser.add_mutually_exclusive_group(required=True)
    dept_group.add_argument("--department", "-d", type=str, help="Department name")
    dept_group.add_argument("--all", "-a", action="store_true", help="Plot all departments")

    parser.add_argument(
        "--age_group", "-g",
        type=str,
        choices=["under5", "60plus"],
        default="under5",
        help="Age group (default: under5)",
    )
    parser.add_argument("--output", "-o", type=str, help="Output file path for plot")
    parser.add_argument(
        "--models", "-m", type=str, nargs="+",
        help="Models to include (e.g. --models SARIMA XGBoost). Default: all models.",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    output = Path(args.output) if args.output else None

    if args.all:
        departments = get_available_departments()
        logger.info(f"Plotting {len(departments)} departments for {args.age_group}")
        for dept in departments:
            try:
                plot_one(dept, args.age_group, models=args.models)
            except Exception as exc:
                logger.warning(f"Failed for {dept}: {exc}")
    else:
        plot_one(args.department, args.age_group, models=args.models, output=output)


if __name__ == "__main__":
    main()
