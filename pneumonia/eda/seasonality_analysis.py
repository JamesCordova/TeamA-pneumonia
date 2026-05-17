"""
Seasonality Analysis of Pneumonia Disease Measures

This module provides functions to analyze seasonal patterns in disease incidence
at national and departmental levels using box plots, heatmaps, and STL decomposition.

Functions:
    load_and_prepare_weekly_data: Load and validate weekly epidemiological data
    load_population_data: Load annual population data
    compute_incidence: Calculate incidence rate per 100,000
    aggregate_to_monthly: Aggregate weekly data to monthly level
    compute_monthly_summary_stats: Calculate monthly statistics
    plot_national_seasonality_boxplots: Create boxplot visualization
    plot_national_seasonality_heatmaps: Create heatmap visualization
    plot_national_stl_decomposition: Create STL decomposition grid
    plot_departmental_boxplots_grid: Create departmental boxplots
    plot_departmental_seasonality_comparison: Compare departments
    run_seasonality_analysis: Orchestrate complete workflow
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
ANNUAL_DATA_PATH = DATA_PROCESSED / "annual_disease_measures_by_department.csv"

# Style configuration
sns.set_style("whitegrid")

# Year filters
CHILD_START_YEAR = 2000
ADULT_START_YEAR = 2006
END_YEAR = 2023

# STL decomposition period (weeks)
STL_PERIOD = 52

# Top departments for comparison
TOP_DEPTS_N = 6

# Month names for display
MONTH_NAMES = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
}


def load_and_prepare_weekly_data(filepath=WEEKLY_DATA_PATH):
    """
    Load and validate raw weekly epidemiological data.
    
    Args:
        filepath: Path to raw IRAS CSV file
        
    Returns:
        DataFrame with validated columns, date index, and computed date fields
        
    Raises:
        ValueError: If required columns are missing
    """
    logger.info(f"Loading weekly data from {filepath}")
    
    df = pd.read_csv(filepath)
    df.columns = [c.strip().lower() for c in df.columns]
    
    # Required columns
    required = [
        "ano", "semana", "department", "pneumonia_under5", "pneumonia_60plus",
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
    numeric_cols = ["pneumonia_under5", "pneumonia_60plus", "ano", "semana"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    
    # Build date column
    df["week_start"] = pd.to_datetime(df["week_start"], errors="coerce")
    df = df.dropna(subset=["week_start"]).copy()
    
    # Extract year and month
    df["year"] = df["week_start"].dt.year.astype(int)
    df["month"] = df["week_start"].dt.month.astype(int)
    
    df = df.sort_values("week_start").reset_index(drop=True)
    
    logger.info(f"Loaded {len(df)} rows")
    return df


def load_population_data(filepath=ANNUAL_DATA_PATH):
    """
    Load population data from annual aggregates.
    
    Args:
        filepath: Path to annual disease measures CSV
        
    Returns:
        DataFrame with department, year, population columns
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


def compute_incidence(cases, population, scale=100000):
    """
    Calculate incidence rate per 100,000 population.
    
    Args:
        cases: Number of cases
        population: Population denominator
        scale: Scale factor (default 100,000)
        
    Returns:
        Incidence rate per scale population
    """
    return (cases / population) * scale


def aggregate_to_monthly(weekly_df, pop_df):
    """
    Aggregate weekly data to monthly level with population merge.
    
    Args:
        weekly_df: Weekly data with year, month columns
        pop_df: Population data by year and department
        
    Returns:
        DataFrame aggregated to monthly level with incidence rates
    """
    logger.info("Aggregating weekly data to monthly level")
    
    # National aggregation (sum across all departments)
    monthly = (
        weekly_df.groupby(["year", "month"], as_index=False)
        .agg(
            cases_men5=("pneumonia_under5", "sum"),
            cases_60mas=("pneumonia_60plus", "sum")
        )
    )
    
    # Merge with national population
    nat_pop = pop_df.groupby("year", as_index=False)["population"].sum()
    monthly = monthly.merge(nat_pop, on="year", how="left")
    monthly = monthly.dropna(subset=["population"])
    
    # Compute incidence rates
    monthly["incidence_men5"] = compute_incidence(
        monthly["cases_men5"], monthly["population"]
    )
    monthly["incidence_60mas"] = compute_incidence(
        monthly["cases_60mas"], monthly["population"]
    )
    
    logger.info(f"Computed monthly aggregates: {len(monthly)} months")
    return monthly


def aggregate_to_monthly_by_department(weekly_df, pop_df):
    """
    Aggregate weekly data to monthly level by department.
    
    Args:
        weekly_df: Weekly data with year, month, department columns
        pop_df: Population data by year and department
        
    Returns:
        DataFrame aggregated to monthly level by department
    """
    logger.info("Aggregating weekly data to monthly level by department")
    
    # Departmental aggregation
    monthly_dept = (
        weekly_df.groupby(["department", "year", "month"], as_index=False)
        .agg(
            cases_men5=("pneumonia_under5", "sum"),
            cases_60mas=("pneumonia_60plus", "sum")
        )
    )
    
    # Merge with population
    monthly_dept = monthly_dept.merge(
        pop_df[["department", "year", "population"]],
        on=["department", "year"],
        how="left"
    )
    monthly_dept = monthly_dept.dropna(subset=["population"])
    
    # Compute incidence rates
    monthly_dept["incidence_men5"] = compute_incidence(
        monthly_dept["cases_men5"], monthly_dept["population"]
    )
    monthly_dept["incidence_60mas"] = compute_incidence(
        monthly_dept["cases_60mas"], monthly_dept["population"]
    )
    
    logger.info(f"Computed departmental monthly aggregates")
    return monthly_dept


def compute_monthly_summary_stats(monthly_df, age_group="men5"):
    """
    Calculate summary statistics by month.
    
    Args:
        monthly_df: Monthly aggregated data
        age_group: 'men5' or '60mas'
        
    Returns:
        DataFrame with month statistics (mean, median, std, min, max)
    """
    col = f"incidence_{age_group}"
    
    summary = (
        monthly_df.groupby("month")[col]
        .agg(["mean", "median", "std", "min", "max"])
        .round(3)
    )
    
    logger.info(f"Computed monthly summary stats for {age_group}")
    return summary


def plot_national_seasonality_boxplots(monthly_df, output_path=None):
    """
    Create boxplot visualization of national monthly seasonality.
    
    Shows distribution of incidence rates for each month across all years.
    Creates side-by-side plots for children and older adults.
    
    Args:
        monthly_df: Monthly aggregated data
        output_path: Path to save figure (optional)
    """
    logger.info("Creating national seasonality boxplots")
    
    if output_path is None:
        output_path = REPORTS_FIGURES_PATH / "national_seasonality_boxplots.png"
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
    fig.suptitle("National Monthly Seasonality of Pneumonia Incidence", fontsize=14, fontweight="bold")
    
    # Children <5
    sns.boxplot(
        data=monthly_df,
        x="month", y="incidence_men5",
        ax=axes[0], palette="Set2"
    )
    axes[0].set_title("Children <5 Years", fontweight="bold")
    axes[0].set_xlabel("Month")
    axes[0].set_ylabel("Incidence per 100,000")
    axes[0].set_xticklabels([MONTH_NAMES[i] for i in range(1, 13)])
    axes[0].grid(axis="y", alpha=0.3)
    
    # Adults 60+
    monthly_adults = monthly_df[monthly_df["year"] >= ADULT_START_YEAR].copy()
    sns.boxplot(
        data=monthly_adults,
        x="month", y="incidence_60mas",
        ax=axes[1], palette="Set2"
    )
    axes[1].set_title("Adults 60+ Years", fontweight="bold")
    axes[1].set_xlabel("Month")
    axes[1].set_ylabel("Incidence per 100,000")
    axes[1].set_xticklabels([MONTH_NAMES[i] for i in range(1, 13)])
    axes[1].grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"National seasonality boxplots saved to {output_path}")
    plt.close()


def plot_national_seasonality_heatmaps(monthly_df, output_path=None):
    """
    Create heatmap visualization of year×month seasonality.
    
    Shows incidence patterns across years and months for both age groups.
    
    Args:
        monthly_df: Monthly aggregated data
        output_path: Path to save figure (optional)
    """
    logger.info("Creating national seasonality heatmaps")
    
    if output_path is None:
        output_path = REPORTS_FIGURES_PATH / "national_seasonality_heatmaps.png"
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Pivot for heatmaps
    heat_men5 = monthly_df.pivot(index="year", columns="month", values="incidence_men5")
    
    monthly_adults = monthly_df[monthly_df["year"] >= ADULT_START_YEAR].copy()
    heat_60mas = monthly_adults.pivot(index="year", columns="month", values="incidence_60mas")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle("Year × Month Seasonality Heatmap", fontsize=14, fontweight="bold")
    
    # Children heatmap
    sns.heatmap(
        heat_men5, cmap="YlOrRd", ax=axes[0],
        cbar_kws={"label": "Incidence per 100k"}
    )
    axes[0].set_title("Children <5 Years", fontweight="bold")
    axes[0].set_xlabel("Month")
    axes[0].set_ylabel("Year")
    axes[0].set_xticklabels([MONTH_NAMES[i] for i in range(1, 13)])
    
    # Adults heatmap
    sns.heatmap(
        heat_60mas, cmap="YlOrRd", ax=axes[1],
        cbar_kws={"label": "Incidence per 100k"}
    )
    axes[1].set_title("Adults 60+ Years", fontweight="bold")
    axes[1].set_xlabel("Month")
    axes[1].set_ylabel("Year")
    axes[1].set_xticklabels([MONTH_NAMES[i] for i in range(1, 13)])
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"National seasonality heatmaps saved to {output_path}")
    plt.close()


def plot_national_stl_decomposition(weekly_df, pop_df, output_path=None):
    """
    Create STL decomposition visualization at national level.
    
    Shows observed, trend, seasonal, and residual components for both age groups.
    
    Args:
        weekly_df: Weekly data
        pop_df: Population data
        output_path: Path to save figure (optional)
    """
    logger.info("Creating national STL decomposition visualization")
    
    if output_path is None:
        output_path = REPORTS_FIGURES_PATH / "national_stl_decomposition.png"
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Aggregate to national weekly with population
    weekly_nat = (
        weekly_df.groupby(["week_start"], as_index=False)
        .agg(
            cases_men5=("pneumonia_under5", "sum"),
            cases_60mas=("pneumonia_60plus", "sum"),
            year=("year", "first")
        )
    )
    
    # Merge with national population
    nat_pop = pop_df.groupby("year", as_index=False)["population"].sum()
    weekly_nat = weekly_nat.merge(nat_pop, on="year", how="left")
    weekly_nat = weekly_nat.dropna(subset=["population"])
    
    # Compute incidence
    weekly_nat["incidence_men5"] = compute_incidence(
        weekly_nat["cases_men5"], weekly_nat["population"]
    )
    weekly_nat["incidence_60mas"] = compute_incidence(
        weekly_nat["cases_60mas"], weekly_nat["population"]
    )
    
    # Set index and sort
    weekly_nat = weekly_nat.set_index("week_start").sort_index()
    
    # STL decomposition for children
    series_men5 = weekly_nat["incidence_men5"]
    series_men5_freq = series_men5.asfreq("W-MON").interpolate(limit_direction="both")
    stl_men5 = STL(series_men5_freq, period=STL_PERIOD).fit()
    
    # STL decomposition for adults
    weekly_adults = weekly_nat[weekly_nat["year"] >= ADULT_START_YEAR].copy()
    series_60mas = weekly_adults["incidence_60mas"]
    series_60mas_freq = series_60mas.asfreq("W-MON").interpolate(limit_direction="both")
    stl_60mas = STL(series_60mas_freq, period=STL_PERIOD).fit()
    
    # Create visualization
    fig, axes = plt.subplots(4, 2, figsize=(16, 12), sharex="col")
    fig.suptitle("National STL Decomposition (52-week period)", fontsize=14, fontweight="bold")
    
    # Column titles
    axes[0, 0].set_title("Children <5 Years", fontweight="bold", fontsize=12)
    axes[0, 1].set_title("Adults 60+ Years", fontweight="bold", fontsize=12)
    
    # Row 1: Observed
    axes[0, 0].plot(series_men5_freq.index, series_men5_freq.values, color="#1f77b4", linewidth=1)
    axes[0, 0].set_ylabel("Observed", fontweight="bold")
    axes[0, 0].grid(alpha=0.3)
    
    axes[0, 1].plot(series_60mas_freq.index, series_60mas_freq.values, color="#1f77b4", linewidth=1)
    axes[0, 1].grid(alpha=0.3)
    
    # Row 2: Trend
    axes[1, 0].plot(stl_men5.trend.index, stl_men5.trend.values, color="#ff7f0e", linewidth=1.5)
    axes[1, 0].set_ylabel("Trend", fontweight="bold")
    axes[1, 0].grid(alpha=0.3)
    
    axes[1, 1].plot(stl_60mas.trend.index, stl_60mas.trend.values, color="#ff7f0e", linewidth=1.5)
    axes[1, 1].grid(alpha=0.3)
    
    # Row 3: Seasonal
    axes[2, 0].plot(stl_men5.seasonal.index, stl_men5.seasonal.values, color="#2ca02c", linewidth=1)
    axes[2, 0].set_ylabel("Seasonal", fontweight="bold")
    axes[2, 0].grid(alpha=0.3)
    
    axes[2, 1].plot(stl_60mas.seasonal.index, stl_60mas.seasonal.values, color="#2ca02c", linewidth=1)
    axes[2, 1].grid(alpha=0.3)
    
    # Row 4: Residual
    axes[3, 0].plot(stl_men5.resid.index, stl_men5.resid.values, color="#d62728", linewidth=0.8, alpha=0.7)
    axes[3, 0].set_ylabel("Residual", fontweight="bold")
    axes[3, 0].set_xlabel("Date")
    axes[3, 0].grid(alpha=0.3)
    
    axes[3, 1].plot(stl_60mas.resid.index, stl_60mas.resid.values, color="#d62728", linewidth=0.8, alpha=0.7)
    axes[3, 1].set_xlabel("Date")
    axes[3, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"National STL decomposition saved to {output_path}")
    plt.close()


def plot_departmental_boxplots_grid(monthly_by_dept, top_n=TOP_DEPTS_N, output_path=None):
    """
    Create grid of boxplots for top departments by average incidence.
    
    Args:
        monthly_by_dept: Monthly data aggregated by department
        top_n: Number of top departments to show
        output_path: Path to save figure (optional)
    """
    logger.info(f"Creating departmental boxplots grid (top {top_n})")
    
    if output_path is None:
        output_path = REPORTS_FIGURES_PATH / "departmental_boxplots_grid.png"
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get top departments by average incidence (children)
    top_depts = (
        monthly_by_dept.groupby("department")["incidence_men5"].mean()
        .nlargest(top_n)
        .index.tolist()
    )
    
    dept_filtered = monthly_by_dept[monthly_by_dept["department"].isin(top_depts)].copy()
    
    # Create grid
    n_cols = 3
    n_rows = (len(top_depts) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle(f"Top {top_n} Departments: Monthly Seasonality Pattern (Children <5)", fontsize=14, fontweight="bold")
    
    for idx, dept in enumerate(top_depts):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        dept_data = dept_filtered[dept_filtered["department"] == dept]
        
        sns.boxplot(
            data=dept_data,
            x="month", y="incidence_men5",
            ax=ax, palette="Set2"
        )
        
        avg_incidence = dept_data["incidence_men5"].mean()
        ax.set_title(f"{dept}\n(avg: {avg_incidence:.1f}/100k)", fontweight="bold")
        ax.set_xlabel("Month")
        ax.set_ylabel("Incidence per 100,000")
        ax.set_xticklabels([MONTH_NAMES[i] for i in range(1, 13)])
        ax.grid(axis="y", alpha=0.3)
    
    # Hide extra subplots
    for idx in range(len(top_depts), n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis("off")
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Departmental boxplots grid saved to {output_path}")
    plt.close()


def plot_departmental_seasonality_comparison(monthly_by_dept, top_n=TOP_DEPTS_N, output_path=None):
    """
    Create comparison heatmaps for top departments.
    
    Args:
        monthly_by_dept: Monthly data aggregated by department
        top_n: Number of top departments to show
        output_path: Path to save figure (optional)
    """
    logger.info(f"Creating departmental seasonality comparison (top {top_n})")
    
    if output_path is None:
        output_path = REPORTS_FIGURES_PATH / "departmental_seasonality_comparison.png"
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get top departments by average incidence
    top_depts = (
        monthly_by_dept.groupby("department")["incidence_men5"].mean()
        .nlargest(top_n)
        .index.tolist()
    )
    
    # Create heatmaps
    n_cols = 3
    n_rows = (len(top_depts) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle(f"Top {top_n} Departments: Year×Month Seasonality Heatmap (Children <5)", fontsize=14, fontweight="bold")
    
    for idx, dept in enumerate(top_depts):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        dept_data = monthly_by_dept[monthly_by_dept["department"] == dept]
        heat = dept_data.pivot(index="year", columns="month", values="incidence_men5")
        
        sns.heatmap(
            heat, cmap="YlOrRd", ax=ax,
            cbar_kws={"label": "Incidence per 100k"}
        )
        
        ax.set_title(f"{dept}", fontweight="bold")
        ax.set_xlabel("Month")
        ax.set_ylabel("Year")
        ax.set_xticklabels([MONTH_NAMES[i] for i in range(1, 13)])
    
    # Hide extra subplots
    for idx in range(len(top_depts), n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis("off")
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Departmental seasonality comparison saved to {output_path}")
    plt.close()


def run_seasonality_analysis():
    """
    Execute complete seasonality analysis workflow.
    
    Loads data, computes aggregates, and creates visualizations at both
    national and departmental levels.
    """
    logger.info("=" * 70)
    logger.info("Starting Seasonality Analysis")
    logger.info("=" * 70)
    
    try:
        # Load data
        weekly_df = load_and_prepare_weekly_data()
        pop_df = load_population_data()
        
        # NATIONAL LEVEL ANALYSIS
        logger.info("\n" + "=" * 70)
        logger.info("NATIONAL LEVEL ANALYSIS")
        logger.info("=" * 70)
        
        # Aggregate to monthly
        monthly_nat = aggregate_to_monthly(weekly_df, pop_df)
        
        # Print summary statistics
        logger.info("\nChildren <5 Monthly Incidence Summary (per 100k)")
        child_summary = compute_monthly_summary_stats(monthly_nat, "men5")
        logger.info(f"\n{child_summary}")
        
        monthly_adults = monthly_nat[monthly_nat["year"] >= ADULT_START_YEAR].copy()
        logger.info("\nAdults 60+ Monthly Incidence Summary (per 100k)")
        adult_summary = compute_monthly_summary_stats(monthly_adults, "60mas")
        logger.info(f"\n{adult_summary}")
        
        # National visualizations
        plot_national_seasonality_boxplots(monthly_nat)
        plot_national_seasonality_heatmaps(monthly_nat)
        plot_national_stl_decomposition(weekly_df, pop_df)
        
        # DEPARTMENTAL LEVEL ANALYSIS
        logger.info("\n" + "=" * 70)
        logger.info("DEPARTMENTAL LEVEL ANALYSIS")
        logger.info("=" * 70)
        
        # Aggregate to monthly by department
        monthly_by_dept = aggregate_to_monthly_by_department(weekly_df, pop_df)
        
        # Departmental visualizations
        plot_departmental_boxplots_grid(monthly_by_dept, top_n=TOP_DEPTS_N)
        plot_departmental_seasonality_comparison(monthly_by_dept, top_n=TOP_DEPTS_N)
        
        logger.info("=" * 70)
        logger.info("Seasonality Analysis completed successfully")
        logger.info("=" * 70)
        
        print("\n" + "=" * 70)
        print("SEASONALITY ANALYSIS SUMMARY")
        print("=" * 70)
        print(f"✓ Data loaded: {len(weekly_df)} weekly records")
        print(f"✓ National periods: {len(monthly_nat)} months (children), {len(monthly_adults)} months (adults)")
        print(f"✓ Departments analyzed: {monthly_by_dept['department'].nunique()}")
        print(f"✓ Visualizations generated: 5 national + departmental grids")
        print(f"✓ Output saved to: {REPORTS_FIGURES_PATH}")
        print("=" * 70 + "\n")
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    run_seasonality_analysis()
