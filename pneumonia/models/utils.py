"""
Utilities for forecasting models

Provides functions for data loading, preprocessing, and train/validation/test splits.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from typing import Tuple, Optional
import logging

from pneumonia.config import DATA_RAW_PATH, DATA_PROCESSED_PATH
from pneumonia.utils import setup_logger

logger = setup_logger(__name__)

# Data paths
WEEKLY_DATA_PATH = Path(DATA_RAW_PATH) / "iras_data_raw.csv"
ANNUAL_DATA_PATH = Path(DATA_PROCESSED_PATH) / "annual_disease_measures_by_department.csv"

# Minimum weeks for model training
MIN_WEEKS_FOR_TRAINING = 104  # 2 years


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
    filepath: Optional[Path] = None
) -> pd.Series:
    """
    Get time series of cases for a specific department and age group.
    
    Aggregates cases across all districts for each week.
    
    Args:
        department: Department name
        age_group: 'under5' or '60plus'
        filepath: Path to data file
        
    Returns:
        pandas Series indexed by week_start with aggregated case counts
    """
    df = load_weekly_cases(filepath=filepath, department=department, age_group=age_group)
    
    if df.empty:
        raise ValueError(f"No data found for {department} ({age_group})")
    
    # Aggregate cases across all districts per week
    ts = df.groupby("week_start")["cases"].sum()
    ts = ts.sort_index()
    
    logger.info(f"Aggregated {len(df)} district records to {len(ts)} weeks for {department} ({age_group})")
    
    return ts


def temporal_split(
    data: pd.Series,
    train_years: Tuple[int, int] = (2000, 2019),
    val_years: Tuple[int, int] = (2020, 2021),
    test_years: Tuple[int, int] = (2022, 2023)
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Split time series into train, validation, and test sets based on years.
    
    Args:
        data: Time series data with DatetimeIndex
        train_years: (start_year, end_year) for training
        val_years: (start_year, end_year) for validation
        test_years: (start_year, end_year) for testing
        
    Returns:
        Tuple of (train_data, val_data, test_data)
        
    Raises:
        ValueError: If train set has fewer than MIN_WEEKS_FOR_TRAINING
    """
    # Extract years from index
    data_years = data.index.year
    
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
            f"Insufficient training data: {len(train)} weeks < {MIN_WEEKS_FOR_TRAINING} required"
        )
    
    logger.info(
        f"Temporal split: Train={len(train)} weeks, Val={len(val)} weeks, Test={len(test)} weeks"
    )
    
    return train, val, test


def handle_missing_values(
    data: pd.Series,
    method: str = "interpolate"
) -> pd.Series:
    """
    Handle missing values in time series.
    
    Args:
        data: Input time series
        method: 'interpolate' (linear), 'forward_fill', or 'backward_fill'
        
    Returns:
        Time series with missing values handled
    """
    if data.isna().sum() == 0:
        return data
    
    logger.info(f"Handling {data.isna().sum()} missing values with method={method}")
    
    data = data.copy()
    
    if method == "interpolate":
        data = data.interpolate(method="linear", limit_direction="both")
    elif method == "forward_fill":
        data = data.fillna(method="ffill").fillna(method="bfill")
    elif method == "backward_fill":
        data = data.fillna(method="bfill").fillna(method="ffill")
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Fill any remaining NaNs with 0
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
