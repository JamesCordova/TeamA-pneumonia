#!/usr/bin/env python
"""
Command-line interface for training SARIMA models

Usage:
    python scripts/train_sarima.py --department AMAZONAS --age_group under5
    python scripts/train_sarima.py --all --age_group 60plus --use_auto_arima
    python scripts/train_sarima.py --help
"""

import argparse
import sys
from pathlib import Path
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pneumonia.pipelines.sarima_pipeline import SARIMAPipeline, run_pipeline_for_all_departments
from pneumonia.models.utils import get_available_departments
from pneumonia.utils import setup_logger

logger = setup_logger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Train SARIMA forecasting models for pneumonia cases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train single department
  python scripts/train_sarima.py --department AMAZONAS --age_group under5
  
  # Train with auto_arima parameter search
  python scripts/train_sarima.py --department LIMA --age_group 60plus --use_auto_arima
  
  # Train all departments
  python scripts/train_sarima.py --all --age_group under5
  
  # Use dynamic temporal split instead of year-based
  python scripts/train_sarima.py --department AMAZONAS --split_strategy dynamic
        """
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
    
    # Model options
    parser.add_argument(
        "--use_auto_arima",
        action="store_true",
        help="Use auto_arima parameter search (overrides config)",
    )
    
    parser.add_argument(
        "--no_auto_arima",
        action="store_true",
        help="Disable auto_arima search (overrides config)",
    )
    
    # SARIMA manual parameter customization
    sarima_group = parser.add_argument_group("SARIMA manual parameters")
    sarima_group.add_argument(
        "--sarima_order", type=int, nargs=3, default=None,
        metavar=("P", "D", "Q"),
        help="[SARIMA] Non-seasonal order (p d q), e.g. --sarima_order 2 1 1",
    )
    sarima_group.add_argument(
        "--n_fourier_terms", type=int, default=None,
        help="[SARIMA] Fourier sin/cos pairs for seasonality (default: 6)",
    )
    fourier_group = sarima_group.add_mutually_exclusive_group()
    fourier_group.add_argument(
        "--no_fourier", action="store_true", default=False,
        help="[SARIMA] Use classical SAR/SMA instead of Fourier seasonality",
    )
    fourier_group.add_argument(
        "--fourier", action="store_true", default=False,
        help="[SARIMA] Force Fourier seasonality (overrides config if disabled)",
    )
    
    # Split strategy
    parser.add_argument(
        "--split_strategy", "-s",
        type=str,
        choices=["dynamic", "years"],
        help="Temporal split strategy (dynamic=percentage-based, years=year-range-based)",
    )
    
    # Forecast steps
    parser.add_argument(
        "--forecast_steps", "-f",
        type=int,
        default=52,
        help="Number of weeks to forecast (default: 52)",
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
    use_auto_arima: bool = None,
    split_strategy: str = None,
    forecast_steps: int = 52,
    order: tuple = None,
    n_fourier_terms: int = None,
    use_fourier: bool = None,
) -> int:
    """
    Train SARIMA model for a single department.
    
    Args:
        department: Department name
        age_group: 'under5' or '60plus'
        use_auto_arima: Override config setting
        split_strategy: Override config setting
        forecast_steps: Number of steps to forecast
        
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        logger.info(f"\n{'='*80}")
        logger.info(f"Training SARIMA for {department} ({age_group})")
        logger.info(f"{'='*80}\n")
        
        pipeline = SARIMAPipeline(
            department=department,
            age_group=age_group,
            use_auto_arima=use_auto_arima,
            forecast_steps=forecast_steps,
            split_strategy=split_strategy,
            order=order,
            n_fourier_terms=n_fourier_terms,
            use_fourier=use_fourier,
        )
        
        results = pipeline.run()
        print(pipeline.summary())
        
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        return 1


def train_all_departments(
    age_group: str = "under5",
    use_auto_arima: bool = None,
    split_strategy: str = None,
    forecast_steps: int = 52,
    order: tuple = None,
    n_fourier_terms: int = None,
    use_fourier: bool = None,
) -> int:
    """
    Train SARIMA models for all departments.
    
    Args:
        age_group: 'under5' or '60plus'
        use_auto_arima: Override config setting
        split_strategy: Override config setting
        forecast_steps: Number of steps to forecast
        
    Returns:
        Exit code (0 if all successful, 1 if any failed)
    """
    try:
        logger.info(f"\n{'='*80}")
        logger.info(f"Training SARIMA for all departments ({age_group})")
        logger.info(f"{'='*80}\n")
        
        # Get available departments
        departments = get_available_departments()
        logger.info(f"Found {len(departments)} departments: {', '.join(departments)}\n")
        
        results = run_pipeline_for_all_departments(
            age_group=age_group,
            use_auto_arima=use_auto_arima,
            split_strategy=split_strategy,
            forecast_steps=forecast_steps,
            order=order,
            n_fourier_terms=n_fourier_terms,
            use_fourier=use_fourier,
        )
        
        # Summary
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
        # Suppress non-critical logs
        logging.getLogger("pneumonia").setLevel(logging.ERROR)
    elif args.verbose:
        # Enable debug logging
        logging.getLogger("pneumonia").setLevel(logging.DEBUG)
    
    # Determine auto_arima setting
    use_auto_arima = None
    if args.use_auto_arima:
        use_auto_arima = True
    elif args.no_auto_arima:
        use_auto_arima = False

    # Build SARIMA manual parameters
    order = tuple(args.sarima_order) if args.sarima_order else None
    use_fourier = None
    if args.no_fourier:
        use_fourier = False
    elif args.fourier:
        use_fourier = True
    
    try:
        if args.all:
            # Train all departments
            exit_code = train_all_departments(
                age_group=args.age_group,
                use_auto_arima=use_auto_arima,
                split_strategy=args.split_strategy,
                forecast_steps=args.forecast_steps,
                order=order,
                n_fourier_terms=args.n_fourier_terms,
                use_fourier=use_fourier,
            )
        else:
            # Parse list of departments
            departments = []
            for d in args.department:
                departments.extend([x.strip().upper() for x in d.split(",") if x.strip()])
            
            # Loop through specified departments
            exit_code = 0
            for dept in departments:
                code = train_single_department(
                    department=dept,
                    age_group=args.age_group,
                    use_auto_arima=use_auto_arima,
                    split_strategy=args.split_strategy,
                    forecast_steps=args.forecast_steps,
                    order=order,
                    n_fourier_terms=args.n_fourier_terms,
                    use_fourier=use_fourier,
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
    exit_code = main()
    sys.exit(exit_code)
