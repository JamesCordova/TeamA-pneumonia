"""
Module for interpolating population data across years for departments.

This module loads census data by department and year, creates a complete year grid,
and performs linear interpolation to fill missing years, generating a continuous
population dataset from 2000-2023.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import logging
from typing import Optional, List

from pneumonia.config import (
    DATA_INTERIM_PATH,
    DATA_EXTERNAL_PATH,
)
from pneumonia.utils import setup_logger

logger = setup_logger(__name__)

# Data file paths
CENSUS_INPUT_PATH = Path(DATA_INTERIM_PATH) / "department_census_by_year.csv"
POPULATION_OUTPUT_PATH = Path(DATA_EXTERNAL_PATH) / "department_population_interpolated.csv"

# Year range for interpolation
INTERPOLATION_START_YEAR = 1993
INTERPOLATION_END_YEAR = 2023
OUTPUT_START_YEAR = 2000
OUTPUT_END_YEAR = 2023


def load_census_data(census_file_path: Path) -> pd.DataFrame:
    """
    Load census data from CSV file.

    Reads CSV file containing department census data with years as columns.
    Normalizes column names to lowercase for consistent handling.

    Args:
        census_file_path: Path to census CSV file

    Returns:
        DataFrame with census data
    """
    logger.info(f"Loading census data from: {census_file_path}")
    
    if not census_file_path.exists():
        logger.error(f"Census data file not found: {census_file_path}")
        raise FileNotFoundError(f"Census data file not found: {census_file_path}")
    
    census_df = pd.read_csv(census_file_path)
    
    # Normalize column names to lowercase
    census_df.columns = census_df.columns.str.lower().str.strip()
    
    logger.info(f"Loaded census data: {census_df.shape[0]} departments, {census_df.shape[1]} columns")
    logger.info(f"Columns: {census_df.columns.tolist()}")
    
    return census_df


def extract_year_columns(df: pd.DataFrame) -> List[str]:
    """
    Identify and extract year columns from dataset.

    Detects columns that are purely numeric and represent years.

    Args:
        df: Input DataFrame

    Returns:
        List of column names that represent years
    """
    year_columns = [col for col in df.columns if str(col).isdigit()]
    logger.info(f"Identified {len(year_columns)} year columns: {min(map(int, year_columns))}-{max(map(int, year_columns))}")
    
    return year_columns


def prepare_long_format_data(
    df: pd.DataFrame,
    year_columns: List[str],
    id_columns: List[str],
) -> pd.DataFrame:
    """
    Reshape census data from wide to long format.

    Converts year columns to rows, creating department-year-population records.

    Args:
        df: Input DataFrame in wide format
        year_columns: Columns representing years
        id_columns: Columns containing department identifiers

    Returns:
        DataFrame in long format with columns: id_columns + ['year', 'population']
    """
    logger.info(f"Reshaping data from wide to long format...")
    
    # Ensure year columns are numeric
    df[year_columns] = df[year_columns].apply(pd.to_numeric, errors='coerce')
    
    # Reshape from wide to long
    df_long = df.melt(
        id_vars=id_columns,
        value_vars=year_columns,
        var_name="year",
        value_name="population",
    )
    
    # Convert year to integer
    df_long["year"] = df_long["year"].astype(int)
    
    logger.info(f"Reshaped to long format: {len(df_long)} records")
    
    return df_long


def create_complete_year_grid(
    df_long: pd.DataFrame,
    department_identifier_col: str,
    start_year: int = INTERPOLATION_START_YEAR,
    end_year: int = INTERPOLATION_END_YEAR,
) -> pd.DataFrame:
    """
    Create complete year grid for all departments.

    Generates a grid containing every department for every year in the range,
    to enable interpolation for missing years.

    Args:
        df_long: Long-format DataFrame with department and year columns
        department_identifier_col: Name of column identifying departments
        start_year: First year to include in grid
        end_year: Last year to include in grid

    Returns:
        DataFrame with complete year grid
    """
    logger.info(f"Creating complete year grid ({start_year}-{end_year})...")
    
    # Get unique department identifiers and their metadata
    department_info_cols = [col for col in df_long.columns if col not in ['year', 'population']]
    unique_departments = df_long[department_info_cols].drop_duplicates()
    
    logger.info(f"Processing {len(unique_departments)} unique departments")
    
    # Create full year range
    full_year_range = range(start_year, end_year + 1)
    
    # Create Cartesian product of departments and years
    full_grid = pd.MultiIndex.from_product(
        [unique_departments[department_identifier_col].unique(), full_year_range],
        names=[department_identifier_col, "year"],
    ).to_frame(index=False)
    
    # Merge department metadata back
    full_grid = full_grid.merge(
        unique_departments,
        on=department_identifier_col,
        how="left",
    )
    
    # Merge population values where available
    df_complete = full_grid.merge(
        df_long[[department_identifier_col, "year", "population"]],
        on=[department_identifier_col, "year"],
        how="left",
    )
    
    logger.info(f"Created complete grid: {len(df_complete)} records")
    
    return df_complete


def interpolate_population_values(
    df: pd.DataFrame,
    department_identifier_col: str,
) -> pd.DataFrame:
    """
    Perform linear interpolation on population values by department.

    Fills missing population values using linear interpolation within each department.

    Args:
        df: DataFrame with complete year grid
        department_identifier_col: Name of column identifying departments

    Returns:
        DataFrame with interpolated population values
    """
    logger.info("Performing linear interpolation of population values...")
    
    df_sorted = df.sort_values([department_identifier_col, "year"])
    
    # Interpolate within each department
    df_interpolated = df_sorted.copy()
    df_interpolated["population"] = (
        df_interpolated.groupby(department_identifier_col)["population"]
        .transform(lambda x: x.interpolate(method="linear"))
    )
    
    # Check for remaining missing values (at edges)
    missing_count = df_interpolated["population"].isna().sum()
    if missing_count > 0:
        logger.warning(f"After interpolation: {missing_count} remaining missing values (likely at edges)")
    
    logger.info("Interpolation completed")
    
    return df_interpolated


def filter_output_year_range(
    df: pd.DataFrame,
    start_year: int = OUTPUT_START_YEAR,
    end_year: int = OUTPUT_END_YEAR,
) -> pd.DataFrame:
    """
    Filter dataset to desired year range.

    Args:
        df: Input DataFrame
        start_year: First year to include
        end_year: Last year to include

    Returns:
        Filtered DataFrame
    """
    logger.info(f"Filtering to year range {start_year}-{end_year}")
    
    df_filtered = df[
        (df["year"] >= start_year) &
        (df["year"] <= end_year)
    ].copy()
    
    logger.info(f"After filtering: {len(df_filtered)} records")
    
    return df_filtered


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names to lowercase and consistent format.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with standardized column names
    """
    df.columns = df.columns.str.lower().str.strip()
    return df


def round_population_values(df: pd.DataFrame, column: str = "population") -> pd.DataFrame:
    """
    Round population values to nearest integer.

    Args:
        df: Input DataFrame
        column: Population column name

    Returns:
        DataFrame with rounded population values
    """
    df[column] = df[column].round(0).astype("Int64")
    return df


def reorder_and_select_columns(
    df: pd.DataFrame,
    output_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Reorder and select final output columns.

    Args:
        df: Input DataFrame
        output_columns: List of columns to include in output (uses all if not provided)

    Returns:
        DataFrame with selected and reordered columns
    """
    if output_columns is None:
        # Default column order
        output_columns = [col for col in df.columns if col in 
                         ['ubigeo', 'iddpto', 'department', 'year', 'population']]
    
    return df[output_columns]


def validate_interpolation_quality(df: pd.DataFrame) -> None:
    """
    Validate quality of interpolated dataset and log summary statistics.

    Args:
        df: Interpolated population DataFrame
    """
    logger.info("=" * 60)
    logger.info("INTERPOLATION QUALITY VALIDATION SUMMARY")
    logger.info("=" * 60)
    
    # Year range
    logger.info(
        f"Year range: {df['year'].min()}-{df['year'].max()} "
        f"({df['year'].max() - df['year'].min() + 1} years)"
    )
    
    # Department coverage
    unique_departments = df.groupby(['ubigeo', 'department']).size()
    logger.info(f"Unique departments: {len(unique_departments)}")
    
    # Year coverage per department
    years_per_dept = df.groupby('iddpto')['year'].count()
    expected_years = df['year'].nunique()
    logger.info(
        f"Years per department: min={years_per_dept.min()}, "
        f"max={years_per_dept.max()}, expected={expected_years}"
    )
    
    # Missing values
    missing_population = df['population'].isna().sum()
    logger.info(f"Missing population values: {missing_population}")
    
    # Population range
    logger.info(
        f"Population range: {df['population'].min():,.0f} - "
        f"{df['population'].max():,.0f}"
    )
    
    # Invalid values
    invalid_population = (df['population'] <= 0).sum()
    if invalid_population > 0:
        logger.warning(f"Records with population <= 0: {invalid_population}")
    
    logger.info("=" * 60)
    logger.info("Sample of interpolated data (first 5 records):")
    logger.info("\n" + str(df.head(5)))
    logger.info("=" * 60)


def save_interpolated_population_data(
    df: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> Path:
    """
    Save interpolated population data to CSV file.

    Args:
        df: DataFrame with interpolated population data
        output_path: Output file path (uses default if not provided)

    Returns:
        Path to saved file
    """
    if output_path is None:
        output_path = POPULATION_OUTPUT_PATH
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving interpolated population data to: {output_path}")
    df.to_csv(output_path, index=False)
    logger.info(f"Successfully saved {len(df)} records")
    
    return output_path


def interpolate_and_save_population_data(
    census_file_path: Optional[Path] = None,
    output_file_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Main orchestration function for interpolating and saving population data.

    Loads census data, reshapes to long format, creates complete year grid,
    performs linear interpolation, filters to desired year range, and saves results.

    Args:
        census_file_path: Path to census Excel file (uses default if not provided)
        output_file_path: Path for output CSV (uses default if not provided)

    Returns:
        DataFrame with interpolated population data
    """
    if census_file_path is None:
        census_file_path = CENSUS_INPUT_PATH
    
    census_file_path = Path(census_file_path)
    
    logger.info("Starting population data interpolation pipeline...")
    logger.info(f"Configuration: Interpolation years {INTERPOLATION_START_YEAR}-{INTERPOLATION_END_YEAR}, "
                f"Output years {OUTPUT_START_YEAR}-{OUTPUT_END_YEAR}")
    logger.info(f"Input source: {census_file_path.name}")
    
    try:
        # Load data
        census_data = load_census_data(census_file_path)
        
        # Identify year columns
        year_columns = extract_year_columns(census_data)
        
        # Reshape to long format
        id_columns = ['ubigeo', 'department', 'iddpto']
        df_long = prepare_long_format_data(census_data, year_columns, id_columns)
        
        # Create complete year grid
        df_complete = create_complete_year_grid(df_long, 'iddpto')
        
        # Interpolate population values
        df_interpolated = interpolate_population_values(df_complete, 'iddpto')
        
        # Filter to output year range
        df_filtered = filter_output_year_range(df_interpolated)
        
        # Standardize column names
        df_filtered = standardize_column_names(df_filtered)
        
        # Round population values
        df_filtered = round_population_values(df_filtered)
        
        # Select and reorder columns
        output_columns = ['ubigeo', 'iddpto', 'department', 'year', 'population']
        df_final = reorder_and_select_columns(df_filtered, output_columns)
        
        # Validate quality
        validate_interpolation_quality(df_final)
        
        # Save results
        saved_path = save_interpolated_population_data(df_final, output_file_path)
        
        logger.info(
            f"Pipeline completed successfully. "
            f"Output saved to: {saved_path}"
        )
        
        return df_final
        
    except Exception as e:
        logger.error(f"Error in population interpolation pipeline: {str(e)}")
        raise


if __name__ == "__main__":
    # Run pipeline when executed as script
    result = interpolate_and_save_population_data()
    print("\nFirst few records of interpolated population data:")
    print(result.head(10))
