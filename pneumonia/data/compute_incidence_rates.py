"""
Module for computing pneumonia disease measures (incidence, hospitalization, and mortality rates)
by department and year from IRAS (Sistema de Información de Infecciones Respiratorias Agudas) data.

This module aggregates weekly surveillance data to annual department-level measures,
computes rates per 100,000 population, and generates summary tables.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import logging
from typing import Optional

from pneumonia.config import (
    DATA_RAW_PATH,
    DATA_EXTERNAL_PATH,
    DATA_PROCESSED_PATH,
)
from pneumonia.utils import setup_logger

logger = setup_logger(__name__)

# Rate denominator for standardization
RATE_SCALE_PER_100K = 100_000

# Data file paths
POPULATION_DATA_PATH = Path(DATA_EXTERNAL_PATH) / "department_population_interpolated.csv"
IRAS_RAW_DATA_PATH = Path(DATA_RAW_PATH) / "iras_data_raw.csv"
OUTPUT_ANNUAL_MEASURES_PATH = Path(DATA_PROCESSED_PATH) / "annual_disease_measures_by_department.csv"


def compute_disease_rate(
    numerator: pd.Series,
    denominator: pd.Series,
    rate_scale: int = RATE_SCALE_PER_100K,
) -> pd.Series:
    """
    Compute disease rate with safe handling of zero denominators.

    Args:
        numerator: Count of cases/hospitalizations/deaths
        denominator: Population count
        rate_scale: Scale factor (default: 100,000 for rate per 100k population)

    Returns:
        Disease rate per rate_scale population
    """
    denominator_safe = denominator.replace(0, np.nan)
    rate = (numerator / denominator_safe) * rate_scale
    return rate


def load_and_prepare_population_data(population_path: Path) -> pd.DataFrame:
    """
    Load and standardize population data.

    Args:
        population_path: Path to population CSV file

    Returns:
        DataFrame with standardized columns
    """
    logger.info(f"Loading population data from: {population_path}")
    
    if not population_path.exists():
        logger.error(f"Population data file not found: {population_path}")
        raise FileNotFoundError(f"Population data file not found: {population_path}")
    
    population_df = pd.read_csv(population_path)
    
    # Standardize column names
    population_df.columns = population_df.columns.str.lower().str.strip()
    
    # Ensure department ID is zero-padded string for consistent merging
    population_df["iddpto"] = (
        population_df["iddpto"].astype(str).str.zfill(2)
    )
    
    logger.info(
        f"Loaded population data: {len(population_df)} rows, "
        f"Years: {population_df['year'].min()}-{population_df['year'].max()}"
    )
    
    return population_df


def load_and_prepare_iras_data(iras_path: Path) -> pd.DataFrame:
    """
    Load and standardize IRAS surveillance data.

    Args:
        iras_path: Path to IRAS CSV file

    Returns:
        DataFrame with standardized columns and valid year data
    """
    logger.info(f"Loading IRAS data from: {iras_path}")
    
    if not iras_path.exists():
        logger.error(f"IRAS data file not found: {iras_path}")
        raise FileNotFoundError(f"IRAS data file not found: {iras_path}")
    
    iras_df = pd.read_csv(iras_path)
    
    # Standardize column names
    iras_df.columns = iras_df.columns.str.lower().str.strip()
    
    # Rename 'ano' to 'year' for consistency
    if 'ano' in iras_df.columns:
        iras_df = iras_df.rename(columns={'ano': 'year'})
    
    # Ensure department ID is zero-padded string
    iras_df["iddpto"] = (
        iras_df["iddpto"].astype(str).str.zfill(2)
    )
    
    # Parse year column and remove invalid entries
    iras_df["year"] = pd.to_numeric(iras_df["year"], errors="coerce").astype("Int64")
    initial_rows = len(iras_df)
    iras_df = iras_df.dropna(subset=["year"])
    dropped_rows = initial_rows - len(iras_df)
    
    if dropped_rows > 0:
        logger.warning(f"Dropped {dropped_rows} rows with invalid year values")
    
    logger.info(
        f"Loaded IRAS data: {len(iras_df)} rows, "
        f"Years: {iras_df['year'].min()}-{iras_df['year'].max()}"
    )
    
    return iras_df


def aggregate_weekly_to_annual(iras_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate weekly IRAS surveillance data to annual department-level counts.

    Groups data by department and year, summing all case counts,
    hospitalizations, and deaths across the year.

    Args:
        iras_df: DataFrame with weekly IRAS data

    Returns:
        DataFrame with annual aggregated counts by department
    """
    logger.info("Aggregating weekly IRAS data to annual totals by department...")
    
    annual_aggregated = (
        iras_df.groupby(
            ["iddpto", "year"],
            as_index=False,
        )
        .agg(
            # Pneumonia cases by age group
            cases_men5=("pneumonia_under5", "sum"),
            cases_60mas=("pneumonia_60plus", "sum"),
            # Hospitalizations by age group
            hosp_men5=("hosp_under5", "sum"),
            hosp_60mas=("hosp_60plus", "sum"),
            # Deaths by age group
            death_men5=("deaths_under5", "sum"),
            death_60mas=("deaths_60plus", "sum"),
        )
    )
    
    logger.info(f"Aggregated to {len(annual_aggregated)} department-year combinations")
    
    return annual_aggregated


def compute_annual_disease_measures(
    annual_aggregated: pd.DataFrame,
    population_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute annual disease measures by merging aggregated IRAS data with population data.

    Computes incidence, hospitalization, and mortality rates per 100k population
    for each age group.

    Args:
        annual_aggregated: Annual aggregated counts by department and year
        population_df: Population data by department and year

    Returns:
        DataFrame with comprehensive disease measures and rates
    """
    logger.info("Computing disease rates and measures...")
    
    # Select relevant population columns
    population_subset = population_df[
        ["iddpto", "year", "department", "ubigeo", "population"]
    ]
    
    # Merge annual aggregates with population data
    disease_measures_df = annual_aggregated.merge(
        population_subset,
        on=["iddpto", "year"],
        how="inner",
    )
    
    initial_records = len(disease_measures_df)
    
    # Compute incidence rates per 100,000 population
    disease_measures_df["cases_rate_men5"] = compute_disease_rate(
        disease_measures_df["cases_men5"],
        disease_measures_df["population"],
    ).round(2)
    
    disease_measures_df["cases_rate_60mas"] = compute_disease_rate(
        disease_measures_df["cases_60mas"],
        disease_measures_df["population"],
    ).round(2)
    
    # Compute hospitalization rates per 100,000 population
    disease_measures_df["hosp_rate_men5"] = compute_disease_rate(
        disease_measures_df["hosp_men5"],
        disease_measures_df["population"],
    ).round(2)
    
    disease_measures_df["hosp_rate_60mas"] = compute_disease_rate(
        disease_measures_df["hosp_60mas"],
        disease_measures_df["population"],
    ).round(2)
    
    # Compute mortality rates per 100,000 population
    disease_measures_df["death_rate_men5"] = compute_disease_rate(
        disease_measures_df["death_men5"],
        disease_measures_df["population"],
    ).round(2)
    
    disease_measures_df["death_rate_60mas"] = compute_disease_rate(
        disease_measures_df["death_60mas"],
        disease_measures_df["population"],
    ).round(2)
    
    # Incidence rate is the same as cases rate
    disease_measures_df["inc_rate_men5"] = disease_measures_df["cases_rate_men5"]
    disease_measures_df["inc_rate_60mas"] = disease_measures_df["cases_rate_60mas"]
    
    logger.info(f"Computed disease measures for {initial_records} records. "
        f"Data completeness: {(1 - disease_measures_df.isna().sum().sum() / (initial_records * disease_measures_df.shape[1])) * 100:.2f}%"
    )
    
    return disease_measures_df


def reorder_output_columns(disease_measures_df: pd.DataFrame) -> pd.DataFrame:
    """
    Reorder columns in logical groups for better readability and analysis.

    Groups columns by: identifiers, population, case counts, hospitalization counts,
    death counts, incidence rates, hospitalization rates, and mortality rates.

    Args:
        disease_measures_df: DataFrame with computed disease measures

    Returns:
        DataFrame with reordered columns
    """
    output_columns = [
        # Geographic and temporal identifiers
        "department",
        "year",
        "ubigeo",
        "iddpto",
        "population",
        # Case counts
        "cases_men5",
        "cases_60mas",
        # Hospitalization counts
        "hosp_men5",
        "hosp_60mas",
        # Death counts
        "death_men5",
        "death_60mas",
        # Incidence rates per 100k
        "cases_rate_men5",
        "cases_rate_60mas",
        # Hospitalization rates per 100k
        "hosp_rate_men5",
        "hosp_rate_60mas",
        # Mortality rates per 100k
        "death_rate_men5",
        "death_rate_60mas",
        # Incidence rates (same as cases rate)
        "inc_rate_men5",
        "inc_rate_60mas",
    ]
    
    return disease_measures_df[output_columns]


def save_disease_measures(
    disease_measures_df: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> Path:
    """
    Save computed disease measures to CSV file.

    Args:
        disease_measures_df: DataFrame with disease measures
        output_path: Output file path (uses default if not provided)

    Returns:
        Path to saved file
    """
    if output_path is None:
        output_path = OUTPUT_ANNUAL_MEASURES_PATH
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving disease measures to: {output_path}")
    disease_measures_df.to_csv(output_path, index=False)
    logger.info(f"Successfully saved {len(disease_measures_df)} records")
    
    return output_path


def validate_output_data_quality(disease_measures_df: pd.DataFrame) -> None:
    """
    Validate data quality and log summary statistics of computed measures.

    Args:
        disease_measures_df: DataFrame with disease measures
    """
    logger.info("=" * 60)
    logger.info("DATA QUALITY VALIDATION SUMMARY")
    logger.info("=" * 60)
    
    # Year range
    year_range = (
        disease_measures_df["year"].min(),
        disease_measures_df["year"].max(),
    )
    logger.info(f"Year range: {year_range[0]} - {year_range[1]}")
    
    # Department coverage
    unique_departments = disease_measures_df["iddpto"].nunique()
    logger.info(f"Unique departments: {unique_departments}")
    
    # Population data coverage
    missing_population = disease_measures_df["population"].isna().sum()
    logger.info(f"Records with missing population: {missing_population}")
    
    # Check for invalid population values
    invalid_population = (disease_measures_df["population"] <= 0).sum()
    if invalid_population > 0:
        logger.warning(f"Records with population <= 0: {invalid_population}")
    
    # Completeness of key measures
    rate_columns = [
        'cases_rate_men5', 'cases_rate_60mas',
        'hosp_rate_men5', 'hosp_rate_60mas',
        'death_rate_men5', 'death_rate_60mas',
        'inc_rate_men5', 'inc_rate_60mas'
    ]
    for col in rate_columns:
        missing = disease_measures_df[col].isna().sum()
        if missing > 0:
            logger.warning(f"Missing values in {col}: {missing}")
    
    logger.info("=" * 60)
    logger.info("Sample of computed data (first 3 records):")
    logger.info("\n" + str(disease_measures_df.head(3)))
    logger.info("=" * 60)


def compute_and_save_pneumonia_incidence_rates(
    population_path: Optional[Path] = None,
    iras_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Main orchestration function for computing and saving pneumonia disease measures.

    Loads IRAS surveillance data and population data, aggregates weekly data to annual
    department-level totals, computes disease rates, and saves results.

    Args:
        population_path: Path to population data CSV (uses default if not provided)
        iras_path: Path to IRAS data CSV (uses default if not provided)
        output_path: Path for output CSV (uses default if not provided)

    Returns:
        DataFrame with computed annual disease measures
    """
    if population_path is None:
        population_path = POPULATION_DATA_PATH
    if iras_path is None:
        iras_path = IRAS_RAW_DATA_PATH
    
    population_path = Path(population_path)
    iras_path = Path(iras_path)
    
    logger.info("Starting pneumonia incidence rate computation pipeline...")
    
    try:
        # Load and prepare data
        population_data = load_and_prepare_population_data(population_path)
        iras_data = load_and_prepare_iras_data(iras_path)
        
        # Aggregate and compute measures
        annual_aggregated = aggregate_weekly_to_annual(iras_data)
        disease_measures = compute_annual_disease_measures(
            annual_aggregated,
            population_data,
        )
        
        # Reorder columns for readability
        disease_measures_ordered = reorder_output_columns(disease_measures)
        
        # Save results
        saved_path = save_disease_measures(disease_measures_ordered, output_path)
        
        # Validate and report
        validate_output_data_quality(disease_measures_ordered)
        
        logger.info(
            f"Pipeline completed successfully. "
            f"Output saved to: {saved_path}"
        )
        
        return disease_measures_ordered
        
    except Exception as e:
        logger.error(f"Error in incidence rate computation pipeline: {str(e)}")
        raise


if __name__ == "__main__":
    # Run pipeline when executed as script
    result = compute_and_save_pneumonia_incidence_rates()
    print("\nFirst few records of computed measures:")
    print(result.head())
