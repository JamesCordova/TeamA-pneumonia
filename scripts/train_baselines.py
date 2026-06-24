#!/usr/bin/env python
"""
Command-line interface for training naive baseline models.

Usage:
    python scripts/train_baselines.py --department AMAZONAS --age_group under5
    python scripts/train_baselines.py --all --age_group 60plus
    python scripts/train_baselines.py --help
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pneumonia.models.utils import get_available_departments
from pneumonia.pipelines.baseline_pipeline import (
    BaselinePipeline,
    run_baselines_for_all_departments,
)
from pneumonia.utils import setup_logger

logger = setup_logger(__name__)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train naive baseline forecasters for pneumonia cases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/train_baselines.py --department AMAZONAS --age_group under5
  python scripts/train_baselines.py --all --age_group 60plus
  python scripts/train_baselines.py --department LIMA --split_strategy dynamic
        """,
    )

    dept_group = parser.add_mutually_exclusive_group(required=True)
    dept_group.add_argument("--department", "-d", type=str, nargs="+",
                            help="Department name(s) (e.g. AMAZONAS LIMA, or comma-separated: AMAZONAS,LIMA)")
    dept_group.add_argument("--all", "-a", action="store_true",
                            help="Run baselines for all departments")

    parser.add_argument("--age_group", "-g", type=str,
                        choices=["under5", "60plus"], default="under5",
                        help="Age group (default: under5)")
    parser.add_argument("--split_strategy", "-s", type=str,
                        choices=["dynamic", "years"],
                        help="Temporal split strategy (overrides config)")
    parser.add_argument("--season_length", type=int, default=52,
                        help="Season length for SeasonalNaive (default: 52 weeks)")
    parser.add_argument("--start_year", type=int, default=None,
                        help="Start year to truncate early sub-reported data")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose logging")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Suppress non-essential output")

    return parser


def train_single(
    department: str,
    age_group: str = "under5",
    split_strategy: str = None,
    season_length: int = 52,
    start_year: int = None,
) -> int:
    try:
        logger.info(f"\n{'='*80}\nBaselines for {department} ({age_group})\n{'='*80}\n")
        pipeline = BaselinePipeline(
            department=department,
            age_group=age_group,
            split_strategy=split_strategy,
            season_length=season_length,
            start_year=start_year,
        )
        pipeline.run()
        print(pipeline.summary())
        return 0
    except Exception as exc:
        logger.error(f"Failed: {exc}")
        return 1


def train_all(
    age_group: str = "under5",
    split_strategy: str = None,
    start_year: int = None,
) -> int:
    try:
        departments = get_available_departments()
        logger.info(f"\n{'='*80}\nBaselines for all departments ({age_group})\n{'='*80}\n")
        logger.info(f"Departments: {', '.join(departments)}\n")
        results = run_baselines_for_all_departments(
            age_group=age_group,
            split_strategy=split_strategy,
            start_year=start_year,
        )
        successes = sum(1 for r in results.values() if r["status"] == "success")
        failures = len(results) - successes
        if failures > 0:
            for dept, r in results.items():
                if r["status"] == "failed":
                    logger.error(f"  - {dept}: {r.get('error', 'unknown')}")
        return 0 if failures == 0 else 1
    except Exception as exc:
        logger.error(f"Batch failed: {exc}")
        return 1


def main():
    parser = create_parser()
    args = parser.parse_args()

    if args.quiet:
        logging.getLogger("pneumonia").setLevel(logging.ERROR)
    elif args.verbose:
        logging.getLogger("pneumonia").setLevel(logging.DEBUG)

    try:
        if args.all:
            code = train_all(
                age_group=args.age_group,
                split_strategy=args.split_strategy,
                start_year=args.start_year,
            )
        else:
            # Parse list of departments
            departments = []
            for d in args.department:
                departments.extend([x.strip().upper() for x in d.split(",") if x.strip()])
            
            # Loop through departments
            code = 0
            for dept in departments:
                c = train_single(
                    department=dept,
                    age_group=args.age_group,
                    split_strategy=args.split_strategy,
                    season_length=args.season_length,
                    start_year=args.start_year,
                )
                if c != 0:
                    code = c
        return code
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        return 130
    except Exception as exc:
        logger.error(f"Fatal error: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
