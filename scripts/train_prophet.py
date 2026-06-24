#!/usr/bin/env python
"""
Command-line interface for training Prophet models

Usage:
    python scripts/train_prophet.py --department AMAZONAS --age_group under5
    python scripts/train_prophet.py --all --age_group 60plus
    python scripts/train_prophet.py --help
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pneumonia.models.utils import get_available_departments
from pneumonia.pipelines.prophet_pipeline import (
    ProphetPipeline,
    run_prophet_for_all_departments,
)
from pneumonia.utils import setup_logger

logger = setup_logger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Train Prophet forecasting models for pneumonia cases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train single department
  python scripts/train_prophet.py --department AMAZONAS --age_group under5
  
  # Train with custom trend flexibility
  python scripts/train_prophet.py --department LIMA --age_group 60plus --changepoint_prior_scale 0.1
  
  # Train all departments
  python scripts/train_prophet.py --all --age_group under5
        """,
    )

    # Department selection
    dept_group = parser.add_mutually_exclusive_group(required=True)
    dept_group.add_argument(
        "--department", "-d",
        type=str,
        nargs="+",
        help="Department name(s) (e.g., AMAZONAS LIMA, or comma-separated: AMAZONAS,LIMA)",
    )
    dept_group.add_argument(
        "--all", "-a",
        action="store_true",
        help="Train models for all departments",
    )

    # Age group
    parser.add_argument(
        "--age_group", "-g",
        type=str,
        choices=["under5", "60plus"],
        default="under5",
        help="Age group to train for (default: under5)",
    )

    # Split strategy
    parser.add_argument(
        "--split_strategy", "-s",
        type=str,
        choices=["dynamic", "years"],
        help="Temporal split strategy (dynamic=percentage-based, years=year-range-based)",
    )

    # Prophet parameters overrides
    prophet_group = parser.add_argument_group("Prophet hyperparameter overrides")
    prophet_group.add_argument(
        "--growth",
        type=str,
        choices=["linear", "flat"],
        help="Prophet growth type (linear or flat)",
    )
    prophet_group.add_argument(
        "--changepoint_prior_scale",
        type=float,
        help="Flexibility of the trend changepoints (default: 0.05)",
    )
    prophet_group.add_argument(
        "--seasonality_prior_scale",
        type=float,
        help="Flexibility of the seasonality component (default: 10.0)",
    )

    parser.add_argument(
        "--start_year",
        type=int,
        help="Start year to truncate early sub-reported data",
    )
    # Verbosity
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    # Quiet mode
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress non-essential output",
    )

    return parser


def train_single_department(
    department: str,
    age_group: str = "under5",
    split_strategy: str = None,
    **kwargs,
) -> int:
    """
    Train Prophet model for a single department.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        logger.info(f"\n{'='*80}")
        logger.info(f"Training Prophet for {department} ({age_group})")
        logger.info(f"{'='*80}\n")

        pipeline = ProphetPipeline(
            department=department,
            age_group=age_group,
            split_strategy=split_strategy,
            **kwargs,
        )

        pipeline.run()
        print(pipeline.summary())

        return 0

    except Exception as e:
        logger.error(f"Training failed for {department}: {str(e)}")
        return 1


def train_all_departments(
    age_group: str = "under5",
    split_strategy: str = None,
    **kwargs,
) -> int:
    """
    Train Prophet models for all departments.

    Returns:
        Exit code (0 if all successful, 1 if any failed)
    """
    try:
        logger.info(f"\n{'='*80}")
        logger.info(f"Training Prophet for all departments ({age_group})")
        logger.info(f"{'='*80}\n")

        departments = get_available_departments()
        logger.info(f"Found {len(departments)} departments: {', '.join(departments)}\n")

        results = run_prophet_for_all_departments(
            age_group=age_group,
            split_strategy=split_strategy,
            **kwargs,
        )

        successes = sum(1 for r in results.values() if r["status"] == "success")
        failures = len(results) - successes

        logger.info(f"\n{'='*80}")
        logger.info(f"BATCH TRAINING COMPLETE")
        logger.info(f"{'='*80}")
        logger.info(f"✓ Successful: {successes}/{len(results)}")
        if failures > 0:
            logger.error(f"✗ Failed: {failures}/{len(results)}")
            for dept, result in results.items():
                if result["status"] == "failed":
                    logger.error(f"  - {dept}: {result.get('error', 'unknown error')}")
        logger.info(f"{'='*80}\n")

        return 0 if failures == 0 else 1

    except Exception as e:
        logger.error(f"Batch training failed: {str(e)}")
        return 1


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Configure logging
    if args.quiet:
        logging.getLogger("pneumonia").setLevel(logging.ERROR)
    elif args.verbose:
        logging.getLogger("pneumonia").setLevel(logging.DEBUG)

    # Collect prophet overrides
    prophet_overrides = {}
    if args.start_year is not None:
        prophet_overrides["start_year"] = args.start_year
    if args.growth is not None:
        prophet_overrides["growth"] = args.growth
    if args.changepoint_prior_scale is not None:
        prophet_overrides["changepoint_prior_scale"] = args.changepoint_prior_scale
    if args.seasonality_prior_scale is not None:
        prophet_overrides["seasonality_prior_scale"] = args.seasonality_prior_scale

    try:
        if args.all:
            exit_code = train_all_departments(
                age_group=args.age_group,
                split_strategy=args.split_strategy,
                **prophet_overrides,
            )
        else:
            # Parse list of departments
            departments = []
            for d in args.department:
                departments.extend([x.strip().upper() for x in d.split(",") if x.strip()])

            exit_code = 0
            for dept in departments:
                code = train_single_department(
                    department=dept,
                    age_group=args.age_group,
                    split_strategy=args.split_strategy,
                    **prophet_overrides,
                )
                if code != 0:
                    exit_code = code

        return exit_code

    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
