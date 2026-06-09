#!/usr/bin/env python
"""
Train RandomForest forecasting model for pneumonia cases.

Usage:
    python scripts/train_random_forest.py --department AMAZONAS --age_group under5
    python scripts/train_random_forest.py --all --age_group 60plus
    python scripts/train_random_forest.py --department LIMA --n_estimators 200
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pneumonia.models.utils import get_available_departments
from pneumonia.pipelines.random_forest_pipeline import (
    RandomForestPipeline,
    run_rf_for_all_departments,
)
from pneumonia.utils import setup_logger

logger = setup_logger(__name__)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train RandomForest forecaster for pneumonia cases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/train_random_forest.py --department AMAZONAS --age_group under5
  python scripts/train_random_forest.py --all --age_group 60plus
  python scripts/train_random_forest.py --department LIMA --n_estimators 200 --max_depth 15
        """,
    )

    dept_group = parser.add_mutually_exclusive_group(required=True)
    dept_group.add_argument("--department", "-d", type=str,
                            help="Department name (e.g. AMAZONAS, LIMA)")
    dept_group.add_argument("--all", "-a", action="store_true",
                            help="Train for all departments")

    parser.add_argument("--age_group", "-g", type=str,
                        choices=["under5", "60plus"], default="under5",
                        help="Age group (default: under5)")
    parser.add_argument("--split_strategy", "-s", type=str,
                        choices=["dynamic", "years"],
                        help="Temporal split strategy (overrides config)")

    # RandomForest hyperparameter overrides
    parser.add_argument("--n_estimators", type=int,
                        help="Number of trees (default from config)")
    parser.add_argument("--max_depth", type=int,
                        help="Max tree depth (default from config)")

    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose logging")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Suppress non-essential output")

    return parser


def train_single(
    department: str,
    age_group: str = "under5",
    split_strategy: str = None,
    rf_params: dict = None,
) -> int:
    try:
        logger.info(f"\n{'='*80}\nRandomForest for {department} ({age_group})\n{'='*80}\n")
        pipeline = RandomForestPipeline(
            department=department,
            age_group=age_group,
            split_strategy=split_strategy,
            rf_params=rf_params,
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
    rf_params: dict = None,
) -> int:
    try:
        departments = get_available_departments()
        logger.info(f"\n{'='*80}\nRandomForest for all departments ({age_group})\n{'='*80}\n")
        logger.info(f"Departments: {', '.join(departments)}\n")
        results = run_rf_for_all_departments(
            age_group=age_group,
            split_strategy=split_strategy,
            rf_params=rf_params,
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

    # Build optional RF param overrides from CLI args
    rf_params = {}
    if args.n_estimators:
        rf_params["n_estimators"] = args.n_estimators
    if args.max_depth:
        rf_params["max_depth"] = args.max_depth
    rf_params = rf_params or None

    try:
        if args.all:
            code = train_all(
                age_group=args.age_group,
                split_strategy=args.split_strategy,
                rf_params=rf_params,
            )
        else:
            code = train_single(
                department=args.department,
                age_group=args.age_group,
                split_strategy=args.split_strategy,
                rf_params=rf_params,
            )
        return code
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        return 130
    except Exception as exc:
        logger.error(f"Fatal error: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
