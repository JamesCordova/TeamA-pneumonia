"""
Utilities for forecasting models

Provides functions for data loading, preprocessing, and train/validation/test splits.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Union, Literal
import logging

from pneumonia.config import (
    DATA_RAW_PATH,
    DATA_PROCESSED_PATH,
    TEMPORAL_SPLIT_STRATEGY,
    DEFAULT_TRAIN_RATIO,
    DEFAULT_VAL_RATIO,
    DEFAULT_TEST_RATIO,
    DEFAULT_TRAIN_YEARS,
    DEFAULT_VAL_YEARS,
    DEFAULT_TEST_YEARS,
)
from pneumonia.utils import setup_logger

logger = setup_logger(__name__)

# Data paths
WEEKLY_DATA_PATH = Path(DATA_RAW_PATH) / "iras_data_raw.csv"
ANNUAL_DATA_PATH = Path(DATA_PROCESSED_PATH) / "annual_disease_measures_by_department.csv"

# Minimum weeks for model training
MIN_WEEKS_FOR_TRAINING = 104  # 2 years

# Departments with known systematic under-reporting in early years.
# Values are the first year with reliable continuous surveillance.
_SUBREGISTRO_DEPTS: dict = {
    "MOQUEGUA": 2008,
    "TACNA":    2007,
    "TUMBES":   2007,
}


def load_weekly_cases(
    filepath: Optional[Path] = None,
    department: Optional[str] = None,
    age_group: str = "under5"
) -> pd.DataFrame:
    """
    Load weekly pneumonia case data.
    
    Args:
        filepath: Path to weekly data CSV. Uses default if None.
        department: Filter by department. Returns all if None.
        age_group: 'under5' or '60plus'
        
    Returns:
        DataFrame with columns: week_start, year, department, cases
        
    Raises:
        ValueError: If age_group is invalid or data structure is wrong
    """
    if filepath is None:
        filepath = WEEKLY_DATA_PATH
    
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    if age_group not in ["under5", "60plus"]:
        raise ValueError(f"age_group must be 'under5' or '60plus', got {age_group}")
    
    logger.info(f"Loading weekly data from {filepath}")
    
    # Load data
    df = pd.read_csv(filepath)
    df.columns = [c.strip().lower() for c in df.columns]
    
    # Map column names
    case_col = "pneumonia_under5" if age_group == "under5" else "pneumonia_60plus"
    
    # Validate required columns
    required_cols = ["week_start", "department", case_col]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Available: {list(df.columns)}")
    
    # Select and prepare data
    df = df[["week_start", "department", case_col]].copy()
    df.columns = ["week_start", "department", "cases"]
    
    # Standardize department names
    df["department"] = df["department"].astype(str).str.strip().str.upper()
    
    # Parse dates
    df["week_start"] = pd.to_datetime(df["week_start"], errors="coerce")
    df = df.dropna(subset=["week_start"]).copy()
    
    # Extract year and week number
    df["year"] = df["week_start"].dt.year
    df["week"] = df["week_start"].dt.isocalendar().week
    
    # Convert cases to numeric
    df["cases"] = pd.to_numeric(df["cases"], errors="coerce").fillna(0)
    
    # Filter by department if specified
    if department is not None:
        department = department.upper()
        df = df[df["department"] == department].copy()
        logger.info(f"Filtered to {department}: {len(df)} records")
    
    # Sort by week_start
    df = df.sort_values("week_start").reset_index(drop=True)
    
    logger.info(f"Loaded {len(df)} records ({df['department'].nunique()} departments, {df['year'].min()}-{df['year'].max()})")
    
    return df


def get_departmental_data(
    department: str,
    age_group: str = "under5",
    filepath: Optional[Path] = None,
    start_year: Optional[int] = None,
) -> pd.Series:
    """
    Get time series of cases for a specific department and age group.

    Aggregates cases across all districts for each week, enforces a regular
    7-day frequency (filling the 3 ISO week-53 gaps via linear interpolation),
    and optionally truncates the series to a given start year.

    Args:
        department:  Department name.
        age_group:   'under5' or '60plus'.
        filepath:    Path to data file; uses default if None.
        start_year:  If provided, drop all observations before this year.
                     Useful for departments with early-year under-reporting.

    Returns:
        pandas Series indexed by week_start with aggregated case counts
        and a regular 7-day frequency.
    """
    df = load_weekly_cases(filepath=filepath, department=department, age_group=age_group)

    if df.empty:
        raise ValueError(f"No data found for {department} ({age_group})")

    # Aggregate cases across all districts per week
    ts = df.groupby("week_start")["cases"].sum().sort_index()

    # Enforce regular 7-day frequency.
    # The raw CSV skips ISO week-53 in 2004, 2009 and 2015, leaving 3 gaps of
    # 14 days that cause statsmodels to warn about irregular frequency.
    ts = ts.asfreq("7D")
    if ts.isna().any():
        ts = handle_missing_values(ts, method="interpolate")

    logger.info(
        f"Aggregated {len(df)} district records to {len(ts)} weeks "
        f"for {department} ({age_group})"
    )

    # Warn about departments with known early-year under-reporting
    dept_upper = department.upper()
    if dept_upper in _SUBREGISTRO_DEPTS and start_year is None:
        rec = _SUBREGISTRO_DEPTS[dept_upper]
        zeros_early = int((ts[ts.index.year < rec] == 0).sum())
        print(
            f"\n[ADVERTENCIA] {dept_upper}: se detectaron {zeros_early} semanas con cero "
            f"casos antes de {rec}, lo que indica subregistro sistemático.\n"
            f"  → Se recomienda usar start_year={rec} para excluir ese período.\n"
            f"  → Ejemplo: get_departmental_data('{dept_upper}', start_year={rec})\n"
        )
        logger.warning(
            f"{dept_upper}: {zeros_early} zero-case weeks before {rec} suggest "
            f"under-reporting. Consider start_year={rec}."
        )

    if start_year is not None:
        ts = ts[ts.index.year >= start_year]
        logger.info(f"Series truncated to start_year={start_year}: {len(ts)} weeks remaining")

    return ts


def temporal_split(
    data: pd.Series,
    strategy: Optional[str] = None,
    train_ratio: Optional[float] = None,
    val_ratio: Optional[float] = None,
    test_ratio: Optional[float] = None,
    train_years: Optional[Tuple[int, int]] = None,
    val_years: Optional[Tuple[int, int]] = None,
    test_years: Optional[Tuple[int, int]] = None,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Split time series into train, validation, and test sets.
    
    Supports two strategies:
    1. 'dynamic': Splits based on ratios (train_ratio, val_ratio, test_ratio)
       - Automatically calculates indices based on data length
       - Useful for datasets with varying time ranges
       
    2. 'years': Splits based on year ranges (train_years, val_years, test_years)
       - Uses fixed year boundaries (e.g., 2000-2019 for training)
       - Better for consistent, reproducible splits
    
    Args:
        data: Time series data with DatetimeIndex
        strategy: 'dynamic' or 'years'. Uses TEMPORAL_SPLIT_STRATEGY from config if None.
        train_ratio: Fraction for training (0-1). Only used if strategy='dynamic'.
        val_ratio: Fraction for validation (0-1). Only used if strategy='dynamic'.
        test_ratio: Fraction for testing (0-1). Only used if strategy='dynamic'.
        train_years: (start_year, end_year) for training. Only used if strategy='years'.
        val_years: (start_year, end_year) for validation. Only used if strategy='years'.
        test_years: (start_year, end_year) for testing. Only used if strategy='years'.
        
    Returns:
        Tuple of (train_data, val_data, test_data)
        
    Raises:
        ValueError: If ratios don't sum to 1.0, years are invalid, or training set is too small
    """
    # Use config defaults if not provided
    if strategy is None:
        strategy = TEMPORAL_SPLIT_STRATEGY
    
    if strategy == "dynamic":
        return _temporal_split_dynamic(
            data,
            train_ratio or DEFAULT_TRAIN_RATIO,
            val_ratio or DEFAULT_VAL_RATIO,
            test_ratio or DEFAULT_TEST_RATIO,
        )
    elif strategy == "years":
        return _temporal_split_years(
            data,
            train_years or DEFAULT_TRAIN_YEARS,
            val_years or DEFAULT_VAL_YEARS,
            test_years or DEFAULT_TEST_YEARS,
        )
    else:
        raise ValueError(f"strategy must be 'dynamic' or 'years', got {strategy}")


def _temporal_split_dynamic(
    data: pd.Series,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Split time series dynamically based on ratio of observations.
    
    Args:
        data: Time series with DatetimeIndex
        train_ratio: Fraction for training (e.g., 0.8 = 80%)
        val_ratio: Fraction for validation (e.g., 0.1 = 10%)
        test_ratio: Fraction for testing (e.g., 0.1 = 10%)
        
    Returns:
        Tuple of (train, val, test) series
        
    Raises:
        ValueError: If ratios don't sum to ~1.0 or training set is too small
    """
    # Validate ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if not abs(total_ratio - 1.0) < 0.01:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
    
    n = len(data)
    train_end = int(np.ceil(n * train_ratio))
    val_end = train_end + int(np.ceil(n * val_ratio))
    
    train = data.iloc[:train_end]
    val = data.iloc[train_end:val_end]
    test = data.iloc[val_end:]
    
    # Validation
    if len(train) < MIN_WEEKS_FOR_TRAINING:
        raise ValueError(
            f"Insufficient training data: {len(train)} weeks < {MIN_WEEKS_FOR_TRAINING} required. "
            f"Consider increasing train_ratio (current: {train_ratio})"
        )
    
    logger.info(
        f"Temporal split (dynamic): "
        f"Train={len(train)} weeks ({train_ratio*100:.0f}%), "
        f"Val={len(val)} weeks ({val_ratio*100:.0f}%), "
        f"Test={len(test)} weeks ({test_ratio*100:.0f}%)"
    )
    
    return train, val, test


def _temporal_split_years(
    data: pd.Series,
    train_years: Tuple[int, int] = (2000, 2019),
    val_years: Tuple[int, int] = (2020, 2021),
    test_years: Tuple[int, int] = (2022, 2023),
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Split time series based on year ranges.
    
    Args:
        data: Time series with DatetimeIndex
        train_years: (start_year, end_year) inclusive for training
        val_years: (start_year, end_year) inclusive for validation
        test_years: (start_year, end_year) inclusive for testing
        
    Returns:
        Tuple of (train, val, test) series
        
    Raises:
        ValueError: If years are invalid or training set is too small
    """
    # Extract years from index
    data_years = data.index.year
    data_year_range = (data_years.min(), data_years.max())
    
    # Validate year ranges
    all_years = list(train_years) + list(val_years) + list(test_years)
    invalid_years = [y for y in all_years if y < data_year_range[0] or y > data_year_range[1]]
    if invalid_years:
        logger.warning(
            f"Year range {invalid_years} outside data range {data_year_range}. "
            f"Some sets may be empty."
        )
    
    # Create masks
    train_mask = (data_years >= train_years[0]) & (data_years <= train_years[1])
    val_mask = (data_years >= val_years[0]) & (data_years <= val_years[1])
    test_mask = (data_years >= test_years[0]) & (data_years <= test_years[1])
    
    train = data[train_mask]
    val = data[val_mask]
    test = data[test_mask]
    
    # Validation
    if len(train) < MIN_WEEKS_FOR_TRAINING:
        raise ValueError(
            f"Insufficient training data: {len(train)} weeks < {MIN_WEEKS_FOR_TRAINING} required. "
            f"Data available: {data_year_range}. "
            f"Consider extending train_years (current: {train_years})"
        )
    
    if len(val) == 0:
        logger.warning(f"Validation set is empty. Val years {val_years} not in data range {data_year_range}")
    
    if len(test) == 0:
        logger.warning(f"Test set is empty. Test years {test_years} not in data range {data_year_range}")
    
    logger.info(
        f"Temporal split (years): "
        f"Train={len(train)} weeks ({train_years[0]}-{train_years[1]}), "
        f"Val={len(val)} weeks ({val_years[0]}-{val_years[1]}), "
        f"Test={len(test)} weeks ({test_years[0]}-{test_years[1]})"
    )
    
    return train, val, test


def handle_missing_values(
    data: pd.Series,
    method: str = "interpolate",
) -> pd.Series:
    """
    Handle missing values in time series.
    
    Args:
        data: Input time series
        method: 'interpolate' (linear), 'forward_fill', or 'backward_fill'
        
    Returns:
        Time series with missing values handled
        
    Raises:
        ValueError: If method is unknown
    """
    if data.isna().sum() == 0:
        logger.debug("No missing values detected")
        return data
    
    nan_count = data.isna().sum()
    logger.info(f"Handling {nan_count} missing values ({nan_count/len(data)*100:.1f}%) with method={method}")
    
    data = data.copy()
    
    if method == "interpolate":
        # Linear interpolation for missing values in the middle
        data = data.interpolate(method="linear", limit_direction="both")
    elif method == "forward_fill":
        # Forward fill then backward fill for edge cases
        data = data.ffill().bfill()
    elif method == "backward_fill":
        # Backward fill then forward fill
        data = data.bfill().ffill()
    else:
        raise ValueError(f"Unknown method: {method}. Must be 'interpolate', 'forward_fill', or 'backward_fill'")
    
    # Fill any remaining NaNs at edges with 0
    remaining_nans = data.isna().sum()
    if remaining_nans > 0:
        logger.warning(f"Filling {remaining_nans} remaining NaNs with 0")
        data = data.fillna(0)
    
    return data


def validate_time_series(
    data: pd.Series,
    min_length: int = MIN_WEEKS_FOR_TRAINING
) -> bool:
    """
    Validate time series quality.
    
    Args:
        data: Time series to validate
        min_length: Minimum required length
        
    Returns:
        True if valid, raises exception otherwise
    """
    if not isinstance(data, pd.Series):
        raise TypeError(f"Expected pd.Series, got {type(data)}")
    
    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError(f"Expected DatetimeIndex, got {type(data.index)}")
    
    if len(data) < min_length:
        raise ValueError(
            f"Time series too short: {len(data)} < {min_length}"
        )
    
    if data.isna().sum() / len(data) > 0.2:
        logger.warning(f"High missing data ratio: {data.isna().sum() / len(data) * 100:.1f}%")
    
    if (data < 0).any():
        logger.warning("Negative values detected in time series")
    
    logger.info(f"Time series validation passed: {len(data)} observations")
    
    return True


def get_available_departments(filepath: Optional[Path] = None) -> list:
    """
    Get list of all departments with sufficient data.
    
    Args:
        filepath: Path to data file
        
    Returns:
        List of department names with at least MIN_WEEKS_FOR_TRAINING observations
    """
    df = load_weekly_cases(filepath=filepath)
    
    # Count observations per department
    dept_counts = df.groupby("department").size()
    
    # Filter by minimum weeks
    available = dept_counts[dept_counts >= MIN_WEEKS_FOR_TRAINING].index.tolist()
    
    logger.info(
        f"Found {len(available)} departments with {MIN_WEEKS_FOR_TRAINING}+ weeks"
    )
    
    return sorted(available)
