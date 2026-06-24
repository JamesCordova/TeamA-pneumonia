#!/usr/bin/env python
"""
Plot forecast results from stored predictions CSV.

Two plot types:
  classic   — train/val/test comparison  → forecast_plot.png
  backtest  — walk-forward backtest      → backtest_plot.png
  both      — generate both files

Usage:
    python scripts/plot_forecasting.py --department AMAZONAS --age_group under5
    python scripts/plot_forecasting.py --department LIMA --plot backtest
    python scripts/plot_forecasting.py --all --age_group 60plus --plot both
    python scripts/plot_forecasting.py --department AMAZONAS --models SARIMA --year 2022
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pneumonia.config import REPORTS_PATH
from pneumonia.models.utils import get_available_departments
from pneumonia.utils import setup_logger
from pneumonia.visualization.backtest_plot import plot_backtest
from pneumonia.visualization.classic_plot import plot_classic

logger = setup_logger(__name__)


def plot_one(
    department: str,
    age_group: str,
    plot_type: str = "classic",
    models=None,
    output: Path = None,
    year: int = None,
) -> None:
    dept  = department.upper()
    group = age_group.lower()

    if plot_type in ("classic", "both"):
        path = plot_classic(
            department=dept,
            age_group=group,
            reports_dir=REPORTS_PATH,
            models=models,
            save_path=output if plot_type == "classic" else None,
            year=year,
        )
        if path:
            print(f"Classic plot saved: {path}")
        else:
            print(
                f"No predictions CSV for {dept}/{group}. "
                f"Run a pipeline first (e.g. scripts/train_sarima.py)."
            )

    if plot_type in ("backtest", "both"):
        path = plot_backtest(
            department=dept,
            age_group=group,
            reports_dir=REPORTS_PATH,
            models=models,
            save_path=output if plot_type == "backtest" else None,
            year=year,
        )
        if path:
            print(f"Backtest plot saved: {path}")
        else:
            print(
                f"No backtest data for {dept}/{group}. "
                f"Run scripts/run_walkforward.py first."
            )


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot forecast results from stored predictions CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/plot_forecasting.py --department AMAZONAS --age_group under5
  python scripts/plot_forecasting.py --department LIMA --plot backtest
  python scripts/plot_forecasting.py --all --age_group 60plus --plot both
  python scripts/plot_forecasting.py --department AMAZONAS --models SARIMA --year 2022
        """,
    )

    dept_group = parser.add_mutually_exclusive_group(required=True)
    dept_group.add_argument("--department", "-d", type=str, nargs="+",
                            help="Department name(s) (e.g. AMAZONAS LIMA, or comma-separated: AMAZONAS,LIMA)")
    dept_group.add_argument("--all", "-a", action="store_true", help="Plot all departments")

    parser.add_argument(
        "--age_group", "-g",
        type=str, choices=["under5", "60plus"], default="under5",
        help="Age group (default: under5)",
    )
    parser.add_argument(
        "--plot", "-p",
        type=str, choices=["classic", "backtest", "both"], default="classic",
        help="Plot type: classic (val/test), backtest (walk-forward), or both (default: classic)",
    )
    parser.add_argument("--output", "-o", type=str,
                        help="Output file path (only for --plot classic or backtest, not both)")
    parser.add_argument(
        "--models", "-m", type=str, nargs="+",
        help="Models to include (e.g. --models SARIMA XGBoost). Default: all.",
    )
    parser.add_argument(
        "--year", type=int,
        help="Restrict plot to a single year (e.g. --year 2022).",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger("pneumonia").setLevel(logging.DEBUG)

    output = Path(args.output) if args.output else None

    departments = []
    if args.all:
        departments = get_available_departments()
    elif args.department:
        for d in args.department:
            departments.extend([x.strip().upper() for x in d.split(",") if x.strip()])

    if len(departments) > 1 and output:
        logger.warning("Multiple departments specified with --output. Ignoring --output to avoid overwriting files.")
        output = None

    logger.info(f"Plotting {len(departments)} departments ({args.age_group}, {args.plot})")
    for dept in departments:
        try:
            plot_one(dept, args.age_group, plot_type=args.plot,
                     models=args.models, output=output, year=args.year)
        except Exception as exc:
            logger.warning(f"Failed for {dept}: {exc}")


if __name__ == "__main__":
    main()
