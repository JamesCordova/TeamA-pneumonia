"""
Regional Analysis of Pneumonia Disease Measures

This module provides functions to analyze disease trends at the regional (departmental)
level using weekly data, with seasonality decomposition and comparison of top-performing
regions identified from ranking analysis.

Functions:
    load_weekly_data: Load and validate raw weekly data
    load_top_regions: Extract top 3 regions from ranking files
    prepare_regional_data: Filter and aggregate weekly data by region
    compute_weekly_incidence: Calculate incidence rates per 100k
    compute_monthly_seasonality: Aggregate to monthly level for seasonality analysis
    decompose_time_series: Perform STL decomposition
    plot_regional_analysis_grid: Create 2x2 visualization grid
    run_regional_analysis: Orchestrate complete analysis workflow
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import STL
import warnings

warnings.filterwarnings("ignore")

from pneumonia.config import DATA_RAW_PATH, DATA_PROCESSED_PATH
from pneumonia.utils import setup_logger

logger = setup_logger(__name__)

# Configuration paths
DATA_RAW = Path(DATA_RAW_PATH)
DATA_PROCESSED = Path(DATA_PROCESSED_PATH)
REPORTS_FIGURES_PATH = Path("reports") / "figures"

WEEKLY_DATA_PATH = DATA_RAW / "iras_data_raw.csv"
RANKING_DIR = DATA_PROCESSED / "ranking"
ANNUAL_DATA_PATH = DATA_PROCESSED / "annual_disease_measures_by_department.csv"

# Style configuration
sns.set_style("whitegrid")

# Color palette
COLOR_PALETTE = {
    "trend": "#1f77b4",
    "seasonal": "#ff7f0e",
    "observed": "#7f7f7f",
}

# Year filters
CHILD_START_YEAR = 2000
ADULT_START_YEAR = 2006
END_YEAR = 2023

# STL decomposition period (weeks)
STL_PERIOD = 52

# Minimum cases for valid analysis
MIN_CASES_FOR_ANALYSIS = 10


# Ranking file configuration
RANKING_CONFIG = {
    "children_cases": {
        "file": RANKING_DIR / "top3_frequency_children_cases_rate.csv",
        "measure_col": "pneumonia_under5",
        "year_start": CHILD_START_YEAR,
        "year_end": END_YEAR,
        "label": "Cases",
        "group": "Children <5",
    },
    "children_hosp": {
        "file": RANKING_DIR / "top3_frequency_children_hosp_rate.csv",
        "measure_col": "hosp_under5",
        "year_start": CHILD_START_YEAR,
        "year_end": END_YEAR,
        "label": "Hospitalizations",
        "group": "Children <5",
    },
    "children_death": {
        "file": RANKING_DIR / "top3_frequency_children_death_rate.csv",
        "measure_col": "deaths_under5",
        "year_start": CHILD_START_YEAR,
        "year_end": END_YEAR,
        "label": "Deaths",
        "group": "Children <5",
    },
    "older_cases": {
        "file": RANKING_DIR / "top3_frequency_older_cases_rate.csv",
        "measure_col": "pneumonia_60plus",
        "year_start": ADULT_START_YEAR,
        "year_end": END_YEAR,
        "label": "Cases",
        "group": "Adults 60+",
    },
    "older_hosp": {
        "file": RANKING_DIR / "top3_frequency_older_hosp_rate.csv",
        "measure_col": "hosp_60plus",
        "year_start": ADULT_START_YEAR,
        "year_end": END_YEAR,
        "label": "Hospitalizations",
        "group": "Adults 60+",
    },
    "older_death": {
        "file": RANKING_DIR / "top3_frequency_older_death_rate.csv",
        "measure_col": "deaths_60plus",
        "year_start": ADULT_START_YEAR,
        "year_end": END_YEAR,
        "label": "Deaths",
        "group": "Adults 60+",
    },
}


def load_weekly_data(filepath=WEEKLY_DATA_PATH):
    """
    Load and validate raw weekly epidemiological data.
    
    Args:
        filepath: Path to raw IRAS CSV file
        
    Returns:
        DataFrame with validated columns and proper data types
        
    Raises:
        ValueError: If required columns are missing
    """
    logger.info(f"Loading weekly data from {filepath}")
    
    df = pd.read_csv(filepath)
    df.columns = [c.strip().lower() for c in df.columns]
    
    # Required columns
    required = [
        "ano", "semana", "department", "pneumonia_under5", "pneumonia_60plus",
        "hosp_under5", "hosp_60plus", "deaths_under5", "deaths_60plus",
        "week_start"
    ]
    
    missing = [c for c in required if c not in df.columns]
    if missing:
        error_msg = f"Missing columns: {missing}\nAvailable: {list(df.columns)}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Standardize department names
    df["department"] = df["department"].astype(str).str.strip().str.upper()
    
    # Convert numeric columns
    numeric_cols = [
        "pneumonia_under5", "pneumonia_60plus",
        "hosp_under5", "hosp_60plus",
        "deaths_under5", "deaths_60plus", "ano", "semana"
    ]
    
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    
    # Build date column
    df["week_start"] = pd.to_datetime(df["week_start"], errors="coerce")
    df = df.dropna(subset=["week_start"]).copy()
    
    # Extract year and month
    df["year"] = df["week_start"].dt.year.astype(int)
    df["month"] = df["week_start"].dt.month.astype(int)
    
    logger.info(f"Loaded {len(df)} rows")
    return df


def load_annual_population(filepath=ANNUAL_DATA_PATH):
    """
    Load population data from annual aggregates.
    
    Args:
        filepath: Path to annual disease measures CSV
        
    Returns:
        DataFrame with department, year, population
    """
    logger.info(f"Loading population data from {filepath}")
    
    df = pd.read_csv(filepath)
    df.columns = [c.strip().lower() for c in df.columns]
    
    # Extract relevant columns
    pop = df[["department", "year", "population"]].copy()
    pop["department"] = pop["department"].astype(str).str.strip().str.upper()
    pop["year"] = pop["year"].astype(int)
    pop["population"] = pd.to_numeric(pop["population"], errors="coerce")
    
    pop = pop.dropna(subset=["population"])
    
    logger.info(f"Loaded population for {pop['department'].nunique()} departments")
    return pop


def load_top_regions(ranking_file):
    """
    Extract top 3 regions from a ranking file.
    
    Args:
        ranking_file: Path to ranking CSV file
        
    Returns:
        List of top 3 department names (uppercase)
    """
    logger.info(f"Loading regions from {ranking_file}")
    
    df = pd.read_csv(ranking_file)
    df.columns = [c.strip().lower() for c in df.columns]
    
    # Find department column
    dept_col = None
    for col in df.columns:
        if "depart" in col.lower() or "region" in col.lower():
            dept_col = col
            break
    
    if dept_col is None:
        raise ValueError(f"No department/region column found in {ranking_file}")
    
    regions = df[dept_col].head(3).astype(str).str.strip().str.upper().tolist()
    
    logger.info(f"Found {len(regions)} top regions: {regions}")
    return regions


def prepare_regional_data(df, department, year_start, year_end):
    """
    Filter weekly data for a specific region and year range.
    
    Args:
        df: Full weekly DataFrame
        department: Department name (uppercase)
        year_start: Start year (inclusive)
        year_end: End year (inclusive)
        
    Returns:
        Filtered and sorted DataFrame
    """
    regional = df[
        (df["department"] == department) &
        (df["year"] >= year_start) &
        (df["year"] <= year_end)
    ].copy()
    
    regional = regional.sort_values("week_start").reset_index(drop=True)
    
    if regional.empty:
        logger.warning(f"No data found for {department} ({year_start}-{year_end})")
    else:
        logger.info(f"Prepared {len(regional)} weeks for {department}")
    
    return regional


def compute_weekly_incidence(df, measure_col, pop_df):
    """
    Calculate weekly incidence rate per 100,000 population.
    
    Args:
        df: Weekly regional data
        measure_col: Column name with case counts
        pop_df: Population data (department, year, population)
        
    Returns:
        DataFrame with incidence column added
    """
    result = df.copy()
    
    # Merge with population
    result = result.merge(
        pop_df[["year", "population"]],
        on="year",
        how="left"
    )
    
    result = result.dropna(subset=["population"]).copy()
    
    # Calculate incidence per 100k
    result["cases"] = result[measure_col].astype(float)
    result["incidence"] = (result["cases"] / result["population"]) * 100000
    
    logger.info(f"Computed incidence rates for {len(result)} weeks")
    return result


def compute_monthly_seasonality(df):
    """
    Aggregate weekly incidence to monthly level.
    
    Args:
        df: Weekly data with incidence column
        
    Returns:
        DataFrame aggregated by year and month
    """
    monthly = df.groupby(["year", "month"], as_index=False)["incidence"].mean()
    
    logger.info(f"Computed monthly aggregates: {len(monthly)} months")
    return monthly


def compute_annual_trend(df):
    """
    Aggregate weekly incidence to annual level.
    
    Args:
        df: Weekly data with incidence column
        
    Returns:
        DataFrame aggregated by year
    """
    annual = df.groupby("year", as_index=False)["incidence"].mean()
    
    logger.info(f"Computed annual averages: {len(annual)} years")
    return annual


def decompose_time_series(df, period=STL_PERIOD):
    """
    Perform STL (Seasonal and Trend decomposition using Loess).
    
    Args:
        df: Weekly data with week_start and incidence columns
        period: Period for seasonal component (52 for weeks)
        
    Returns:
        STL result object or None if insufficient data
    """
    logger.info(f"Decomposing time series (period={period} weeks)")
    
    try:
        # Aggregate by date (in case of duplicates) and sort
        ts_data = df.groupby("week_start")[["incidence"]].mean().sort_index()
        
        # Check if we have enough data
        n_points = len(ts_data)
        if n_points < period * 2:
            logger.warning(f"Insufficient data for STL (need at least {period*2}, have {n_points})")
            return None
        
        # Resample to weekly frequency and interpolate gaps
        ts_freq = ts_data.asfreq("W-MON")
        ts_interp = ts_freq["incidence"].interpolate(
            method="linear",
            limit_direction="both"
        )
        
        # Adaptive period: use smaller period if data is limited
        actual_period = min(period, n_points // 4)
        actual_period = max(4, actual_period)  # Minimum period of 4
        
        stl = STL(ts_interp, period=actual_period, seasonal=actual_period + 1).fit()
        logger.info(f"STL decomposition successful (period={actual_period})")
        return stl
    except Exception as e:
        logger.warning(f"STL decomposition failed: {str(e)}")
        return None


def plot_regional_analysis_grid(
    df_weekly, annual_df, monthly_df, stl_result,
    region, group_label, measure_label, output_path=None
):
    """
    Create 2x2 grid visualization of regional analysis.
    
    Subplots:
    1. Annual trend with year markers
    2. Monthly boxplot showing seasonality
    3. Year×Month heatmap
    4. STL decomposition (trend + seasonal components)
    
    Args:
        df_weekly: Weekly data with incidence column
        annual_df: Annual aggregated data
        monthly_df: Monthly aggregated data
        stl_result: STL decomposition result (or None)
        region: Department name
        group_label: Group descriptor (e.g., "Children <5")
        measure_label: Measure type (e.g., "Cases")
        output_path: Path to save figure (optional)
    """
    logger.info(f"Creating visualization grid for {region}")
    
    if output_path is None:
        safe_region = region.replace(" ", "_")
        safe_measure = measure_label.lower().replace(" ", "_")
        filename = f"{safe_region}_{safe_measure}_analysis.png"
        output_path = REPORTS_FIGURES_PATH / filename
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f"{region} — {group_label}: {measure_label}", fontsize=14, fontweight="bold")
    
    # 1. Annual trend
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(
        annual_df["year"], annual_df["incidence"],
        marker="o", linewidth=2, markersize=6,
        color=COLOR_PALETTE["trend"]
    )
    ax1.set_title("Annual Trend (Mean Incidence)", fontweight="bold")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Incidence per 100k")
    ax1.set_xticks(annual_df["year"].values[::2])  # Show every other year
    ax1.tick_params(axis="x", rotation=45)
    ax1.grid(alpha=0.3)
    
    # 2. Monthly seasonality boxplot
    ax2 = plt.subplot(2, 2, 2)
    sns.boxplot(
        data=monthly_df, x="month", y="incidence",
        ax=ax2, palette="Set2"
    )
    ax2.set_title("Monthly Seasonality Pattern", fontweight="bold")
    ax2.set_xlabel("Month")
    ax2.set_ylabel("Incidence per 100k")
    ax2.grid(axis="y", alpha=0.3)
    
    # 3. Year×Month heatmap
    ax3 = plt.subplot(2, 2, 3)
    heat_pivot = monthly_df.pivot(index="year", columns="month", values="incidence")
    sns.heatmap(
        heat_pivot, cmap="YlOrRd", ax=ax3,
        cbar_kws={"label": "Incidence per 100k"}
    )
    ax3.set_title("Year × Month Heatmap", fontweight="bold")
    ax3.set_xlabel("Month")
    ax3.set_ylabel("Year")
    
    # 4. STL Decomposition
    ax4 = plt.subplot(2, 2, 4)
    
    if stl_result is not None:
        # Plot observed and trend
        ax4.plot(
            stl_result.observed.index, stl_result.observed.values,
            color=COLOR_PALETTE["observed"], alpha=0.5,
            label="Observed", linewidth=1
        )
        ax4.plot(
            stl_result.trend.index, stl_result.trend.values,
            color=COLOR_PALETTE["trend"],
            label="Trend", linewidth=2
        )
        
        # Plot seasonal on secondary y-axis
        ax4_2 = ax4.twinx()
        ax4_2.plot(
            stl_result.seasonal.index, stl_result.seasonal.values,
            color=COLOR_PALETTE["seasonal"],
            label="Seasonal", linewidth=1.5, alpha=0.7
        )
        ax4_2.set_ylabel("Seasonal Component", color=COLOR_PALETTE["seasonal"])
        ax4_2.tick_params(axis="y", labelcolor=COLOR_PALETTE["seasonal"])
        
        ax4.set_title("STL Decomposition", fontweight="bold")
        ax4.set_xlabel("Date")
        ax4.set_ylabel("Incidence per 100k", color=COLOR_PALETTE["trend"])
        ax4.tick_params(axis="y", labelcolor=COLOR_PALETTE["trend"])
        
        # Combined legend
        h1, l1 = ax4.get_legend_handles_labels()
        h2, l2 = ax4_2.get_legend_handles_labels()
        ax4.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=9)
        
    else:
        ax4.text(
            0.5, 0.5, "Insufficient data for STL decomposition",
            ha="center", va="center",
            transform=ax4.transAxes,
            fontsize=11, color="red"
        )
        ax4.set_title("STL Decomposition (N/A)", fontweight="bold")
    
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Regional analysis grid saved to {output_path}")
    plt.close()


def run_regional_analysis():
    """
    Execute complete regional analysis workflow.
    
    Loads ranking files, processes each top region, and generates
    visualizations and summaries for all region × measure combinations.
    """
    logger.info("=" * 70)
    logger.info("Starting Regional Analysis")
    logger.info("=" * 70)
    
    try:
        # Load base data
        weekly_df = load_weekly_data()
        pop_df = load_annual_population()
        
        # Track results
        analysis_count = 0
        
        # Process each ranking configuration
        for config_key, config in RANKING_CONFIG.items():
            ranking_file = config["file"]
            measure_col = config["measure_col"]
            year_start = config["year_start"]
            year_end = config["year_end"]
            group_label = config["group"]
            measure_label = config["label"]
            
            logger.info(f"\nProcessing {config_key}: {group_label} - {measure_label}")
            
            if not ranking_file.exists():
                logger.warning(f"Ranking file not found: {ranking_file}")
                continue
            
            try:
                # Load top regions from ranking file
                top_regions = load_top_regions(ranking_file)
                
                # Analyze each top region
                for region in top_regions:
                    logger.info(f"  Analyzing {region}")
                    
                    # Prepare regional data
                    regional = prepare_regional_data(
                        weekly_df, region, year_start, year_end
                    )
                    
                    if regional.empty:
                        logger.warning(f"  Skipping {region} (no data)")
                        continue
                    
                    # Filter regional population
                    region_pop = pop_df[pop_df["department"] == region].copy()
                    
                    # Compute incidence
                    regional = compute_weekly_incidence(
                        regional, measure_col, region_pop
                    )
                    
                    regional = regional[regional["cases"] >= MIN_CASES_FOR_ANALYSIS].copy()
                    
                    if len(regional) < 10:
                        logger.warning(f"  Insufficient data for {region}")
                        continue
                    
                    # Compute aggregates
                    annual = compute_annual_trend(regional)
                    monthly = compute_monthly_seasonality(regional)
                    
                    # STL decomposition
                    stl_result = decompose_time_series(regional)
                    
                    # Create visualization
                    plot_regional_analysis_grid(
                        regional, annual, monthly, stl_result,
                        region, group_label, measure_label
                    )
                    
                    analysis_count += 1
                    
            except Exception as e:
                logger.error(f"Error processing {config_key}: {str(e)}", exc_info=True)
                continue
        
        logger.info("=" * 70)
        logger.info(f"Regional Analysis completed: {analysis_count} visualizations created")
        logger.info("=" * 70)
        
        print(f"\n✓ Analysis complete: {analysis_count} regional analyses generated")
        print(f"✓ Output saved to: {REPORTS_FIGURES_PATH}")
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    run_regional_analysis()
