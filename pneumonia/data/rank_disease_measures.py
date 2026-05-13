"""
Module for ranking pneumonia disease measures (cases, hospitalizations, deaths rates)
by department and year, and computing top-K frequency statistics.

This module processes annual disease measures to generate ranking grids (ranks per year)
and frequency tables showing how many years each department appears in the top-K highest rates.
"""

from pathlib import Path
import pandas as pd
import logging
from typing import Dict, Tuple

from pneumonia.config import DATA_PROCESSED_PATH
from pneumonia.utils import setup_logger

logger = setup_logger(__name__)

# Ranking configuration
TOP_K = 3

# Year filters for different population groups
CHILD_START_YEAR = 2000
OLDER_START_YEAR = 2006

# Data file paths
ANNUAL_MEASURES_PATH = Path(DATA_PROCESSED_PATH) / "annual_disease_measures_by_department.csv"
OUTPUT_RANKING_PATH = Path(DATA_PROCESSED_PATH) / "ranking"


def add_rank_each_year(
    data: pd.DataFrame,
    value_col: str
) -> pd.DataFrame:
    """
    Rank regions within each year (1 = highest rate).
    
    Uses method='min' so tied values receive the same best rank.
    
    Args:
        data: DataFrame with year and value columns
        value_col: Column name containing values to rank
        
    Returns:
        DataFrame with added 'rank' column
    """
    logger.debug(f"Ranking by column: {value_col}")
    out = data.copy()
    out["rank"] = out.groupby("year")[value_col].rank(
        ascending=False,
        method="min"
    )
    out["rank"] = out["rank"].astype(int)
    return out


def make_rank_grid(ranked: pd.DataFrame) -> pd.DataFrame:
    """
    Create a pivot table with regions as rows and years as columns.
    
    Each cell contains the rank for that region-year combination.
    
    Args:
        ranked: DataFrame with rank column
        
    Returns:
        Pivot table (region x year grid)
    """
    logger.debug("Creating rank grid pivot table")
    grid = ranked.pivot(index="region", columns="year", values="rank")
    return grid.sort_index()


def count_topk_years(
    ranked: pd.DataFrame,
    k: int
) -> pd.DataFrame:
    """
    Count how many years each department appears in top-K highest rates.
    
    Computes both absolute counts and percentage of total years.
    
    Args:
        ranked: DataFrame with rank column
        k: K value for top-K selection
        
    Returns:
        DataFrame with frequency statistics sorted by top-K appearances
    """
    logger.debug(f"Computing top-{k} frequency statistics")
    total_years = ranked["year"].nunique()
    logger.info(f"Total years in dataset: {total_years}")

    freq = (
        ranked.assign(in_topk=(ranked["rank"] <= k))
              .groupby(["iddpto", "region"], as_index=False)["in_topk"]
              .sum()
              .rename(columns={"in_topk": f"top{k}_appearances"})
              .sort_values(f"top{k}_appearances", ascending=False)
    )
    
    freq[f"percent_years_in_top{k}"] = (
        freq[f"top{k}_appearances"] / total_years * 100
    ).round(1)
    
    return freq


def standardize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names and data types.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Standardized DataFrame
    """
    logger.debug("Standardizing column names and types")
    
    # Standardize column names
    df.columns = df.columns.str.lower().str.strip()
    
    # Standardize iddpto as zero-padded string
    df["iddpto"] = df["iddpto"].astype(str).str.zfill(2)
    
    # Convert year to integer
    df["year"] = pd.to_numeric(df["year"], errors="coerce").fillna(0).astype(int)
    
    # Add region name (use department or departamento column if available)
    if "department" in df.columns:
        df["region"] = df["department"].astype(str).str.strip()
        logger.debug("Using 'department' column for region names")
    elif "departamento" in df.columns:
        df["region"] = df["departamento"].astype(str).str.strip()
        logger.debug("Using 'departamento' column for region names")
    else:
        df["region"] = df["iddpto"]
        logger.warning("No department/departamento column found; using iddpto as region")
    
    return df


def validate_measures(
    df: pd.DataFrame,
    requested_measures: Dict[str, str]
) -> Dict[str, str]:
    """
    Validate which requested measures exist in the DataFrame.
    
    Args:
        df: Input DataFrame
        requested_measures: Dictionary of label -> column name mappings
        
    Returns:
        Dictionary of available measures
    """
    logger.info("Validating available rate columns")
    
    # Keep only measures that exist
    measures = {
        label: col
        for label, col in requested_measures.items()
        if col in df.columns
    }
    
    logger.info(f"Found {len(measures)}/{len(requested_measures)} requested measures")
    for label, col in measures.items():
        logger.info(f"  ✓ {label}: {col}")
    
    # Warn about missing measures
    missing = [c for c in requested_measures.values() if c not in df.columns]
    if missing:
        logger.warning(f"Missing {len(missing)} expected rate columns: {missing}")
        logger.warning("Ensure incidence rates have been computed before ranking")
    
    if not measures:
        logger.error("No rate columns found for ranking")
        raise ValueError(
            "No rate columns found to rank. "
            "Run compute_incidence_rates.py first to create rate columns."
        )
    
    return measures


def rank_disease_measures(
    input_path: Path = ANNUAL_MEASURES_PATH,
    output_dir: Path = OUTPUT_RANKING_PATH,
    top_k: int = TOP_K,
    child_start_year: int = CHILD_START_YEAR,
    older_start_year: int = OLDER_START_YEAR
) -> Tuple[Dict, Dict]:
    """
    Rank disease measures by department and year.
    
    Generates ranking grids and top-K frequency tables for each measure.
    
    Args:
        input_path: Path to annual disease measures CSV
        output_dir: Directory to save ranking outputs
        top_k: Number of top regions to track (default: 3)
        child_start_year: Start year for children measures (default: 2000)
        older_start_year: Start year for older adults measures (default: 2006)
        
    Returns:
        Tuple of (ranking_grids_dict, frequency_stats_dict)
    """
    logger.info(f"Starting disease measure ranking from: {input_path}")
    
    # Load data
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} rows from input file")
    
    # Standardize
    df = standardize_dataframe(df)
    
    # Measures to rank (rates per 100k)
    requested_measures = {
        "children_cases_rate": "cases_rate_men5",
        "children_hosp_rate": "hosp_rate_men5",
        "children_death_rate": "death_rate_men5",
        "older_cases_rate": "cases_rate_60mas",
        "older_hosp_rate": "hosp_rate_60mas",
        "older_death_rate": "death_rate_60mas",
    }
    
    measures = validate_measures(df, requested_measures)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory prepared: {output_dir}")
    
    ranking_grids = {}
    frequency_stats = {}
    
    # Rank each measure
    for label, col in measures.items():
        logger.info(f"\nRanking measure: {label} (column: {col})")
        
        # Select appropriate year filter
        if label.startswith("older_"):
            df_use = df[df["year"] >= older_start_year].copy()
            logger.debug(f"Using years >= {older_start_year} for older adults measures")
        else:
            df_use = df[df["year"] >= child_start_year].copy()
            logger.debug(f"Using years >= {child_start_year} for children measures")
        
        # Prepare data
        small = df_use[["iddpto", "region", "year", col]].dropna().copy()
        logger.debug(f"Prepared {len(small)} non-null rows for ranking")
        
        # Rank
        ranked = add_rank_each_year(small, col)
        grid = make_rank_grid(ranked)
        freq = count_topk_years(ranked, top_k)
        
        # Save files
        grid_file = output_dir / f"rank_grid_{label}.csv"
        freq_file = output_dir / f"top{top_k}_frequency_{label}.csv"
        
        grid.to_csv(grid_file)
        freq.to_csv(freq_file, index=False)
        
        ranking_grids[label] = grid
        frequency_stats[label] = freq
        
        logger.info(f"✓ Saved: {grid_file}")
        logger.info(f"✓ Saved: {freq_file}")
        logger.debug(f"Top regions in top-{top_k}:\n{freq.head(5).to_string()}")
    
    logger.info(
        f"\n✓ Ranking complete! "
        f"Processed {len(measures)} measures. "
        f"Children: years {child_start_year}–2023, "
        f"Older adults: years {older_start_year}–2023"
    )
    
    return ranking_grids, frequency_stats


if __name__ == "__main__":
    try:
        logger.info("=" * 60)
        logger.info("DISEASE MEASURE RANKING")
        logger.info("=" * 60)
        
        grids, freqs = rank_disease_measures()
        
        logger.info("=" * 60)
        logger.info("Process completed successfully")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Fatal error during ranking: {str(e)}", exc_info=True)
        raise
