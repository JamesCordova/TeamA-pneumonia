#!/usr/bin/env python
"""
Automated Hyperparameter Tuning Script for ML Models.

Optimizes hyperparameters for XGBoost or RandomForest models on a specific department
and age group by evaluating parameter combinations on the validation split.

Usage:
    # Random search (default, fast)
    python scripts/tune_models.py --department AMAZONAS --model RandomForest --n_iter 15
    
    # Grid search (exhaustive)
    python scripts/tune_models.py --department AMAZONAS --model XGBoost --search_method grid
    
    # Override age group or search metric
    python scripts/tune_models.py --department LIMA --model XGBoost --age_group 60plus --metric smape
"""

import argparse
import json
import logging
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid, ParameterSampler

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pneumonia.config import REPORTS_PATH, RANDOM_SEED
from pneumonia.models.utils import get_departmental_data, temporal_split
from pneumonia.models.ml.config import (
    XGBOOST_SEARCH_RANGES,
    RANDOM_FOREST_SEARCH_RANGES,
)
from pneumonia.models.ml.random_forest import RandomForestModel
from pneumonia.models.ml.xgboost import XGBoostModel
from pneumonia.evaluation.metrics import compute_all_metrics
from pneumonia.utils import setup_logger

logger = setup_logger(__name__)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Optimize hyperparameters for RandomForest or XGBoost forecasters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/tune_models.py --department AMAZONAS --model RandomForest --n_iter 15
  python scripts/tune_models.py --department LIMA --model XGBoost --search_method grid
  python scripts/tune_models.py --department AMAZONAS --model XGBoost --metric smape
        """,
    )

    parser.add_argument("--department", "-d", type=str, required=True,
                        help="Department name (e.g. AMAZONAS, LIMA)")
    parser.add_argument("--model", "-m", type=str, required=True,
                        choices=["RandomForest", "XGBoost"],
                        help="Model class to tune")
    parser.add_argument("--age_group", "-g", type=str,
                        choices=["under5", "60plus"], default="under5",
                        help="Age group (default: under5)")
    parser.add_argument("--search_method", type=str,
                        choices=["grid", "random"], default="random",
                        help="Search strategy: 'grid' (exhaustive) or 'random' (sampled)")
    parser.add_argument("--n_iter", type=int, default=20,
                        help="Number of parameter combinations to sample for random search (default: 20)")
    parser.add_argument("--metric", type=str, default="mae",
                        choices=["mae", "rmse", "smape"],
                        help="Validation metric to minimize (default: mae)")
    parser.add_argument("--split_strategy", "-s", type=str,
                        choices=["dynamic", "years"], default="dynamic",
                        help="Temporal split strategy (default: dynamic)")
    parser.add_argument("--start_year", type=int,
                        help="Start year to truncate early sub-reported data (e.g. 2008 for Moquegua)")
    
    # Optional CLI overrides for feature engineering parameters
    parser.add_argument("--lags", type=int, nargs="+",
                        help="Feature lag periods, e.g. --lags 1 2 4 8 (overrides config defaults)")
    parser.add_argument("--windows", type=int, nargs="+",
                        help="Feature rolling window sizes, e.g. --windows 4 13 (overrides config defaults)")

    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose logging")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Suppress non-essential output")

    return parser


def tune_model(
    department: str,
    model_name: str,
    age_group: str = "under5",
    search_method: str = "random",
    n_iter: int = 20,
    metric_to_optimize: str = "mae",
    split_strategy: str = "dynamic",
    start_year: int = None,
    lags: list = None,
    windows: list = None,
) -> dict:
    department = department.upper()
    
    # 1. Load Data
    logger.info(f"Loading data for {department} ({age_group})")
    data = get_departmental_data(department, age_group=age_group, start_year=start_year)
    
    # 2. Temporal Split
    train, val, test = temporal_split(data, strategy=split_strategy)
    if len(val) == 0:
        raise ValueError("Validation set is empty. Tuning requires a validation split.")
        
    logger.info(f"Data split - Train: {len(train)} weeks, Val: {len(val)} weeks")

    # 3. Setup Parameter Space
    if model_name == "RandomForest":
        search_ranges = RANDOM_FOREST_SEARCH_RANGES
        model_class = RandomForestModel
        param_override_key = "rf_params"
    elif model_name == "XGBoost":
        search_ranges = XGBOOST_SEARCH_RANGES
        model_class = XGBoostModel
        param_override_key = "xgb_params"
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    # 4. Generate Parameter Combinations
    if search_method == "grid":
        param_list = list(ParameterGrid(search_ranges))
        logger.info(f"Starting Grid Search. Total combinations: {len(param_list)}")
    else:
        param_list = list(ParameterSampler(search_ranges, n_iter=n_iter, random_state=RANDOM_SEED))
        # Ensure we don't try to sample more than the unique combination space size
        total_unique = len(ParameterGrid(search_ranges))
        if n_iter >= total_unique:
            param_list = list(ParameterGrid(search_ranges))
            logger.info(f"n_iter ({n_iter}) >= total unique combinations ({total_unique}). Running full Grid Search.")
        else:
            logger.info(f"Starting Random Search. Sampling {len(param_list)} of {total_unique} combinations.")

    best_score = float("inf")
    best_params = None
    best_metrics = None
    
    # 5. Run Search
    for idx, params in enumerate(param_list, 1):
        try:
            logger.info(f"[{idx}/{len(param_list)}] Evaluating params: {params}")
            
            # Map parameters to fit method
            model_kwargs = {
                "department": department,
                "age_group": age_group,
                "lags": lags,
                "windows": windows,
                param_override_key: params
            }
            
            model = model_class(**model_kwargs)
            
            # Train model on training set
            model.fit(train)
            
            # Generate multi-step forecast on validation set
            val_forecast = model.predict(train, steps=len(val))
            
            # Calculate evaluation metrics
            metrics = compute_all_metrics(val.values, val_forecast, warn_on_nan=False)
            
            score = metrics[metric_to_optimize]
            logger.info(f"  Result -> {metric_to_optimize.upper()}: {score:.4f} | MAE: {metrics['mae']:.4f} | SMAPE: {metrics['smape']:.4f}")
            
            if score < best_score:
                best_score = score
                best_params = params
                best_metrics = metrics
                logger.info(f"  New best combination found!")
                
        except Exception as exc:
            logger.warning(f"  Skipping combination due to error: {exc}")
            continue

    if best_params is None:
        raise RuntimeError("No parameter combinations successfully completed evaluation.")

    # 6. Report and Save Results
    results = {
        "department": department,
        "age_group": age_group,
        "model": model_name,
        "search_method": search_method,
        "metric_optimized": metric_to_optimize,
        "best_score": best_score,
        "best_params": best_params,
        "best_validation_metrics": best_metrics,
    }

    # Save to reports folder
    output_dir = REPORTS_PATH / department / age_group
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"best_{model_name.lower()}_params.json"
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    logger.info(f"Tuning complete. Best validation {metric_to_optimize.upper()}: {best_score:.4f}")
    logger.info(f"Best parameters: {best_params}")
    logger.info(f"Results saved to: {output_file}")
    
    return results


def main():
    parser = create_parser()
    args = parser.parse_args()

    if args.quiet:
        logging.getLogger("pneumonia").setLevel(logging.ERROR)
    elif args.verbose:
        logging.getLogger("pneumonia").setLevel(logging.DEBUG)

    try:
        results = tune_model(
            department=args.department,
            model_name=args.model,
            age_group=args.age_group,
            search_method=args.search_method,
            n_iter=args.n_iter,
            metric_to_optimize=args.metric,
            split_strategy=args.split_strategy,
            start_year=args.start_year,
            lags=args.lags,
            windows=args.windows,
        )
        
        # Display recommendations
        print("\n" + "=" * 80)
        print(f"RECOMMENDED BEST PARAMETERS FOR {results['model']} ({results['department']}/{results['age_group']})")
        print("=" * 80)
        print(f"Validation {results['metric_optimized'].upper()}: {results['best_score']:.4f}")
        print("\nTo use these parameters, you can add them to the 'DEPARTMENTAL_CONFIGS' dictionary")
        print("inside pneumonia/models/ml/config.py:")
        print("-" * 80)
        
        dept_key = results['department']
        param_dict_name = "random_forest_params" if results['model'] == "RandomForest" else "xgboost_params"
        
        print(f'    "{dept_key}": {{')
        print(f'        "{param_dict_name}": {json.dumps(results["best_params"])}')
        print(f'    }},')
        print("-" * 80)
        print("=" * 80 + "\n")
        
        return 0
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        return 130
    except Exception as exc:
        logger.error(f"Fatal error: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
