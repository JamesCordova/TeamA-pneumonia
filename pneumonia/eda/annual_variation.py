"""
Annual Variation Analysis for Pneumonia Disease Measures

This module provides functions to analyze and visualize annual variation in
pneumonia disease measures (cases, hospitalizations, and deaths) at both
national and departmental levels.

Functions:
    load_disease_measures: Load and validate disease measures from CSV
    compute_national_rates: Compute national-level rates
    compute_departmental_rates: Compute departmental-level rates
    plot_national_variation_grid: Create 2x2 grid of national trends
    plot_departmental_variation: Visualize top departments by measure
    save_annual_tables: Save analysis results to CSV files
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from pneumonia.config import DATA_PROCESSED_PATH
from pneumonia.utils import setup_logger

logger = setup_logger(__name__)

# Configuration paths
DATA_PATH = Path(DATA_PROCESSED_PATH) / "annual_disease_measures_by_department.csv"
REPORTS_FIGURES_PATH = Path("reports") / "figures"
REPORTS_TABLES_PATH = Path("reports") / "tables"

# Color and style constants
COLOR_PALETTE = {
    "cases_child": "#8ecae6",
    "cases_old": "#1d4ed8",
    "hosp_child": "#ffd166",
    "hosp_old": "#f59e0b",
    "death_child": "#95d5b2",
    "death_old": "#2d6a4f",
}

MARKERS = {
    "cases": "^",
    "hosp": "s",
    "death": "*",
}

# Required columns for analysis
REQUIRED_COLUMNS = [
    "year", "department", "population",
    "cases_men5", "cases_60mas",
    "hosp_men5", "hosp_60mas",
    "death_men5", "death_60mas"
]

# Rate conversion constant
RATE_SCALE_PER_100K = 100_000


def _rate_per_100k(count, pop):
    """
    Calculate rate per 100,000 population.
    
    Args:
        count: Numerator (count of cases/hospitalizations/deaths)
        pop: Denominator (population)
        
    Returns:
        Rate per 100,000 population
    """
    return (count / pop) * RATE_SCALE_PER_100K


def load_disease_measures(filepath=DATA_PATH):
    """
    Load and validate disease measures from CSV.
    
    Args:
        filepath: Path to CSV file with disease measures
        
    Returns:
        DataFrame with validated columns and numeric types
        
    Raises:
        ValueError: If required columns are missing
    """
    logger.info(f"Loading disease measures from {filepath}")
    
    df = pd.read_csv(filepath)
    df.columns = [c.strip().lower() for c in df.columns]
    
    # Validate required columns
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        error_msg = f"Missing columns: {missing}\nAvailable: {list(df.columns)}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Convert to numeric types
    numeric_cols = [c for c in REQUIRED_COLUMNS if c != "year" and c != "department"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    
    df["year"] = df["year"].astype(int)
    df["department"] = df["department"].astype(str).str.strip()
    
    logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    return df


def compute_national_rates(df):
    """
    Compute national-level disease rates by year.
    
    Aggregates data across all departments and calculates
    rates per 100,000 population for all measure types.
    
    Args:
        df: DataFrame with disease measures
        
    Returns:
        DataFrame with national aggregated rates indexed by year
    """
    logger.info("Computing national-level rates")
    
    nat = (
        df.groupby("year", as_index=False)[
            [
                "population",
                "cases_men5", "cases_60mas",
                "hosp_men5", "hosp_60mas",
                "death_men5", "death_60mas"
            ]
        ]
        .sum()
        .sort_values("year")
        .reset_index(drop=True)
    )
    
    # Compute rates for children (<5 years)
    nat["cases_rate_men5"] = _rate_per_100k(nat["cases_men5"], nat["population"])
    nat["hosp_rate_men5"] = _rate_per_100k(nat["hosp_men5"], nat["population"])
    nat["death_rate_men5"] = _rate_per_100k(nat["death_men5"], nat["population"])
    
    # Compute rates for older adults (60+ years)
    nat["cases_rate_60mas"] = _rate_per_100k(nat["cases_60mas"], nat["population"])
    nat["hosp_rate_60mas"] = _rate_per_100k(nat["hosp_60mas"], nat["population"])
    nat["death_rate_60mas"] = _rate_per_100k(nat["death_60mas"], nat["population"])
    
    logger.info(f"National rates computed for {len(nat)} years")
    return nat


def compute_departmental_rates(df):
    """
    Compute departmental-level disease rates by year and department.
    
    Calculates rates per 100,000 population for each department
    across all years.
    
    Args:
        df: DataFrame with disease measures
        
    Returns:
        DataFrame with departmental rates
    """
    logger.info("Computing departmental-level rates")
    
    dept = df.copy()
    
    # Compute rates for children (<5 years)
    dept["cases_rate_men5"] = _rate_per_100k(dept["cases_men5"], dept["population"])
    dept["hosp_rate_men5"] = _rate_per_100k(dept["hosp_men5"], dept["population"])
    dept["death_rate_men5"] = _rate_per_100k(dept["death_men5"], dept["population"])
    
    # Compute rates for older adults (60+ years)
    dept["cases_rate_60mas"] = _rate_per_100k(dept["cases_60mas"], dept["population"])
    dept["hosp_rate_60mas"] = _rate_per_100k(dept["hosp_60mas"], dept["population"])
    dept["death_rate_60mas"] = _rate_per_100k(dept["death_60mas"], dept["population"])
    
    logger.info(f"Departmental rates computed")
    return dept


def plot_national_variation_grid(nat_df, output_path=None):
    """
    Create 2x2 grid visualization of national disease trends.
    
    Plots cases, hospitalizations, and deaths for both children (<5)
    and older adults (60+) with dual y-axes for measure comparison.
    
    Args:
        nat_df: DataFrame with national aggregated rates
        output_path: Path to save figure (optional)
    """
    logger.info("Creating national variation grid visualization")
    
    if output_path is None:
        output_path = REPORTS_FIGURES_PATH / "national_annual_variation_grid.png"
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    years = nat_df["year"].astype(int).tolist()
    
    # Determine y-axis limits
    max_cases_hosp = max(
        nat_df["cases_rate_men5"].max(), nat_df["hosp_rate_men5"].max(),
        nat_df["cases_rate_60mas"].max(), nat_df["hosp_rate_60mas"].max()
    )
    max_deaths = max(nat_df["death_rate_men5"].max(), nat_df["death_rate_60mas"].max())
    
    y_cases_hosp = (0, max_cases_hosp * 1.10 if max_cases_hosp > 0 else 1)
    y_deaths = (0, max_deaths * 1.15 if max_deaths > 0 else 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), sharex=True, constrained_layout=True)
    
    def add_combined_legend(ax, ax_r):
        """Combine legends from dual y-axes."""
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax_r.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="upper left")
    
    # Children: Cases vs Hospitalizations
    ax = axes[0, 0]
    ax_r = ax.twinx()
    ax.plot(
        years, nat_df["cases_rate_men5"],
        color=COLOR_PALETTE["cases_child"], marker=MARKERS["cases"], linewidth=2,
        label="Cases /100k"
    )
    ax_r.plot(
        years, nat_df["hosp_rate_men5"],
        color=COLOR_PALETTE["hosp_child"], marker=MARKERS["hosp"], linewidth=2,
        label="Hospitalizations /100k"
    )
    ax.set_title("Children <5: Cases vs Hospitalizations")
    ax.set_ylabel("Cases rate per 100k", labelpad=10)
    ax_r.set_ylabel("Hospitalizations rate per 100k", labelpad=12)
    ax.set_ylim(*y_cases_hosp)
    ax_r.set_ylim(*y_cases_hosp)
    ax.grid(True, alpha=0.3)
    add_combined_legend(ax, ax_r)
    
    # Children: Cases vs Deaths
    ax = axes[0, 1]
    ax_r = ax.twinx()
    ax.plot(
        years, nat_df["cases_rate_men5"],
        color=COLOR_PALETTE["cases_child"], marker=MARKERS["cases"], linewidth=2,
        label="Cases /100k"
    )
    ax_r.plot(
        years, nat_df["death_rate_men5"],
        color=COLOR_PALETTE["death_child"], marker=MARKERS["death"], linewidth=2,
        label="Deaths /100k"
    )
    ax.set_title("Children <5: Cases vs Deaths")
    ax.set_ylabel("Cases rate per 100k", labelpad=10)
    ax_r.set_ylabel("Deaths rate per 100k", labelpad=12)
    ax.set_ylim(*y_cases_hosp)
    ax_r.set_ylim(*y_deaths)
    ax.grid(True, alpha=0.3)
    add_combined_legend(ax, ax_r)
    
    # Older adults: Cases vs Hospitalizations
    ax = axes[1, 0]
    ax_r = ax.twinx()
    ax.plot(
        years, nat_df["cases_rate_60mas"],
        color=COLOR_PALETTE["cases_old"], marker=MARKERS["cases"], linewidth=2,
        label="Cases /100k"
    )
    ax_r.plot(
        years, nat_df["hosp_rate_60mas"],
        color=COLOR_PALETTE["hosp_old"], marker=MARKERS["hosp"], linewidth=2,
        label="Hospitalizations /100k"
    )
    ax.set_title("Adults 60+: Cases vs Hospitalizations")
    ax.set_ylabel("Cases rate per 100k", labelpad=10)
    ax_r.set_ylabel("Hospitalizations rate per 100k", labelpad=12)
    ax.set_ylim(*y_cases_hosp)
    ax_r.set_ylim(*y_cases_hosp)
    ax.grid(True, alpha=0.3)
    add_combined_legend(ax, ax_r)
    
    # Older adults: Cases vs Deaths
    ax = axes[1, 1]
    ax_r = ax.twinx()
    ax.plot(
        years, nat_df["cases_rate_60mas"],
        color=COLOR_PALETTE["cases_old"], marker=MARKERS["cases"], linewidth=2,
        label="Cases /100k"
    )
    ax_r.plot(
        years, nat_df["death_rate_60mas"],
        color=COLOR_PALETTE["death_old"], marker=MARKERS["death"], linewidth=2,
        label="Deaths /100k"
    )
    ax.set_title("Adults 60+: Cases vs Deaths")
    ax.set_ylabel("Cases rate per 100k", labelpad=10)
    ax_r.set_ylabel("Deaths rate per 100k", labelpad=12)
    ax.set_ylim(*y_cases_hosp)
    ax_r.set_ylim(*y_deaths)
    ax.grid(True, alpha=0.3)
    add_combined_legend(ax, ax_r)
    
    # Configure x-axis for all subplots
    for r in range(2):
        for c in range(2):
            axes[r, c].set_xticks(years)
            axes[r, c].tick_params(axis="x", rotation=45)
    
    axes[1, 0].set_xlabel("Year")
    axes[1, 1].set_xlabel("Year")
    
    plt.suptitle("National Annual Variation (Rates per 100k): Children <5 vs Adults 60+")
    
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"National variation grid saved to {output_path}")
    plt.close()


def plot_departmental_variation(dept_df, measure="cases_rate_men5", top_n=10, output_path=None):
    """
    Visualize top departments by disease measure over time.
    
    Creates line plot showing trends for the top N departments
    by average rate for a given measure.
    
    Args:
        dept_df: DataFrame with departmental rates
        measure: Rate column to analyze (e.g., 'cases_rate_men5')
        top_n: Number of top departments to plot
        output_path: Path to save figure (optional)
    """
    logger.info(f"Creating departmental variation plot for {measure} (top {top_n})")
    
    if output_path is None:
        measure_name = measure.replace("_rate_", "_").replace("_", " ").title()
        filename = f"departmental_variation_{measure}.png"
        output_path = REPORTS_FIGURES_PATH / filename
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get top departments by average rate
    top_depts = (
        dept_df.groupby("department")[measure].mean()
        .nlargest(top_n)
        .index.tolist()
    )
    
    dept_filtered = dept_df[dept_df["department"].isin(top_depts)]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for dept in top_depts:
        dept_data = dept_filtered[dept_filtered["department"] == dept].sort_values("year")
        ax.plot(
            dept_data["year"], dept_data[measure],
            marker="o", label=dept, linewidth=2
        )
    
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel(f"{measure.replace('_rate_', '').replace('_', ' ').title()} (per 100k)", fontsize=12)
    ax.set_title(f"Top {top_n} Departments: {measure.replace('_rate_', '').replace('_', ' ').title()} Trend")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Departmental variation plot saved to {output_path}")
    plt.close()


def save_annual_tables(nat_df, dept_df, output_dir=None):
    """
    Save analysis results to CSV files.
    
    Saves national aggregated rates and departmental rates
    to separate CSV files in reports/tables/.
    
    Args:
        nat_df: DataFrame with national rates
        dept_df: DataFrame with departmental rates
        output_dir: Directory to save files (optional)
    """
    if output_dir is None:
        output_dir = REPORTS_TABLES_PATH
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save national table
    nat_path = output_dir / "national_annual_variation_rates.csv"
    nat_df.to_csv(nat_path, index=False)
    logger.info(f"National rates saved to {nat_path}")
    
    # Save departmental table
    dept_path = output_dir / "departmental_annual_variation_rates.csv"
    dept_df.to_csv(dept_path, index=False)
    logger.info(f"Departmental rates saved to {dept_path}")


def run_analysis():
    """
    Execute complete annual variation analysis workflow.
    
    Loads data, computes national and departmental rates,
    creates visualizations, and saves results.
    """
    logger.info("=" * 60)
    logger.info("Starting Annual Variation Analysis")
    logger.info("=" * 60)
    
    try:
        # Load data
        df = load_disease_measures()
        
        # Compute national and departmental rates
        nat_df = compute_national_rates(df)
        dept_df = compute_departmental_rates(df)
        
        # Create visualizations
        plot_national_variation_grid(nat_df)
        
        # Plot top departments for key measures
        measures_to_plot = [
            "cases_rate_men5", "hosp_rate_men5", "death_rate_men5",
            "cases_rate_60mas", "hosp_rate_60mas", "death_rate_60mas"
        ]
        
        for measure in measures_to_plot:
            plot_departmental_variation(dept_df, measure=measure, top_n=10)
        
        # Save results
        save_annual_tables(nat_df, dept_df)
        
        logger.info("=" * 60)
        logger.info("Annual Variation Analysis completed successfully")
        logger.info("=" * 60)
        
        # Print summary statistics
        print("\n" + "=" * 60)
        print("NATIONAL ANNUAL VARIATION SUMMARY")
        print("=" * 60)
        print(f"\nData range: {nat_df['year'].min()} - {nat_df['year'].max()}")
        print(f"Total years: {len(nat_df)}")
        print("\nChildren <5 years:")
        print(f"  Cases rate:           {nat_df['cases_rate_men5'].mean():.2f} per 100k (avg)")
        print(f"  Hospitalizations rate: {nat_df['hosp_rate_men5'].mean():.2f} per 100k (avg)")
        print(f"  Deaths rate:          {nat_df['death_rate_men5'].mean():.2f} per 100k (avg)")
        print("\nAdults 60+ years:")
        print(f"  Cases rate:           {nat_df['cases_rate_60mas'].mean():.2f} per 100k (avg)")
        print(f"  Hospitalizations rate: {nat_df['hosp_rate_60mas'].mean():.2f} per 100k (avg)")
        print(f"  Deaths rate:          {nat_df['death_rate_60mas'].mean():.2f} per 100k (avg)")
        print("=" * 60 + "\n")
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    run_analysis()
