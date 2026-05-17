"""
Outlier Detection Analysis for Pneumonia Disease Measures

This module implements a hybrid outlier detection approach combining:
1. IQR (Interquartile Range) - detects global extremes
2. ARIMA Residuals - detects contextual deviations in time series

Functions:
    load_data: Load annual disease measures
    detect_iqr_outliers: Detect extreme values using IQR method
    fit_arima_departmental: Fit ARIMA model per department
    detect_arima_outliers: Detect anomalies using ARIMA residuals
    detect_hybrid_outliers: Combine both methods
    compute_outlier_severity: Quantify anomaly magnitude
    classify_outlier_type: Classify outlier type
    plot_outlier_detection: Visualize outliers on time series
    plot_outlier_comparison: Compare detection methods
    generate_outlier_report: Create summary report
    run_outlier_detection: Orchestrate complete workflow
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings("ignore")

from pneumonia.config import DATA_PROCESSED_PATH
from pneumonia.utils import setup_logger

logger = setup_logger(__name__)

# Configuration paths
DATA_PROCESSED = Path(DATA_PROCESSED_PATH)
REPORTS_FIGURES_PATH = Path("reports") / "figures"

ANNUAL_DATA_PATH = DATA_PROCESSED / "annual_disease_measures_by_department.csv"

# Style configuration
sns.set_style("whitegrid")

# IQR Configuration
IQR_MULTIPLIER = 1.5

# ARIMA Configuration
ARIMA_ORDER = (1, 1, 1)
ARIMA_THRESHOLD_STD = 2.5
ARIMA_MIN_OBSERVATIONS = 10

# Color mapping for outlier types
OUTLIER_COLORS = {
    "no_outlier": "#2ecc71",  # Green
    "confirmed": "#e74c3c",   # Red (both methods)
    "iqr_only": "#f39c12",    # Orange (IQR only)
    "arima_only": "#3498db",  # Blue (ARIMA only)
}

# Measure configuration
RATE_COLUMNS = [
    "cases_rate_men5", "cases_rate_60mas",
    "hosp_rate_men5", "hosp_rate_60mas",
    "death_rate_men5", "death_rate_60mas"
]

AGE_GROUPS = {"men5": "Children <5", "60mas": "Adults 60+"}


def load_data(filepath=ANNUAL_DATA_PATH):
    """
    Load annual disease measures data.
    
    Args:
        filepath: Path to annual disease CSV
        
    Returns:
        DataFrame with validated data
    """
    logger.info(f"Loading annual data from {filepath}")
    
    df = pd.read_csv(filepath)
    df.columns = [c.strip().lower() for c in df.columns]
    
    # Validate required columns
    required = ["department", "year"] + RATE_COLUMNS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    
    # Standardize department names
    df["department"] = df["department"].astype(str).str.strip().str.upper()
    
    # Convert to numeric
    for col in RATE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    
    logger.info(f"Loaded {len(df)} records from {df['department'].nunique()} departments")
    return df


def detect_iqr_outliers(df, column, multiplier=IQR_MULTIPLIER):
    """
    Detect outliers using Interquartile Range method.
    
    Values outside [Q1 - multiplier*IQR, Q3 + multiplier*IQR] are outliers.
    
    Args:
        df: DataFrame with column to analyze
        column: Column name to analyze
        multiplier: IQR multiplier (typically 1.5)
        
    Returns:
        DataFrame with detected outliers
    """
    logger.info(f"Detecting IQR outliers for {column}")
    
    data = df[column].dropna()
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    # Identify outliers
    outlier_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
    outliers = df[outlier_mask].copy()
    outliers["method"] = "iqr"
    outliers["iqr_lower"] = lower_bound
    outliers["iqr_upper"] = upper_bound
    
    logger.info(f"Found {len(outliers)} IQR outliers")
    return outliers


def fit_arima_departmental(df, column, order=ARIMA_ORDER):
    """
    Fit ARIMA model per department to get residuals.
    
    Args:
        df: DataFrame with time series data
        column: Column to fit
        order: ARIMA order tuple
        
    Returns:
        Dictionary with {department: ARIMA result object}
    """
    logger.info(f"Fitting ARIMA models for {column}")
    
    arima_results = {}
    
    for dept in df["department"].unique():
        dept_data = df[df["department"] == dept].sort_values("year")
        
        # Check minimum data requirement
        if len(dept_data) < ARIMA_MIN_OBSERVATIONS:
            logger.debug(f"Skipping {dept} - insufficient data ({len(dept_data)} < {ARIMA_MIN_OBSERVATIONS})")
            continue
        
        try:
            # Fit ARIMA
            model = ARIMA(dept_data[column].dropna(), order=order)
            result = model.fit()
            arima_results[dept] = result
        except Exception as e:
            logger.warning(f"ARIMA failed for {dept}: {str(e)}")
            continue
    
    logger.info(f"Successfully fitted {len(arima_results)} ARIMA models")
    return arima_results


def detect_arima_outliers(df, column, threshold=ARIMA_THRESHOLD_STD, order=ARIMA_ORDER):
    """
    Detect outliers using ARIMA residuals.
    
    Values with |residual| > threshold * std(residuals) are outliers.
    
    Args:
        df: DataFrame with time series data
        column: Column to analyze
        threshold: Standard deviation threshold
        order: ARIMA order tuple
        
    Returns:
        DataFrame with detected outliers
    """
    logger.info(f"Detecting ARIMA outliers for {column}")
    
    arima_results = fit_arima_departmental(df, column, order=order)
    
    outliers_list = []
    
    for dept, result in arima_results.items():
        residuals = result.resid
        residual_std = residuals.std()
        
        # Identify outlier residuals
        outlier_mask = np.abs(residuals) > threshold * residual_std
        
        if outlier_mask.any():
            # Get the corresponding rows
            outlier_dates = residuals.index[outlier_mask]
            dept_data = df[df["department"] == dept].sort_values("year")
            outlier_rows = dept_data[dept_data["year"].isin(outlier_dates)].copy()
            
            outlier_rows["method"] = "arima"
            outlier_rows["arima_residual"] = residuals[outlier_mask].values
            outlier_rows["arima_resid_std"] = residual_std
            
            outliers_list.append(outlier_rows)
    
    outliers = pd.concat(outliers_list, ignore_index=True) if outliers_list else pd.DataFrame()
    logger.info(f"Found {len(outliers)} ARIMA outliers")
    return outliers


def detect_hybrid_outliers(df, column):
    """
    Detect outliers using hybrid approach (IQR + ARIMA).
    
    Combines both methods to identify outliers with classification:
    - confirmed: Both methods detected the outlier
    - iqr_only: Only IQR detected
    - arima_only: Only ARIMA detected
    
    Args:
        df: DataFrame with data
        column: Column to analyze
        
    Returns:
        DataFrame with all outliers and type classification
    """
    logger.info(f"Running hybrid outlier detection for {column}")
    
    # Detect with both methods
    iqr_outliers = detect_iqr_outliers(df, column)
    arima_outliers = detect_arima_outliers(df, column)
    
    # Create index for comparison (department + year)
    iqr_set = set(zip(iqr_outliers["department"], iqr_outliers["year"]))
    arima_set = set(zip(arima_outliers["department"], arima_outliers["year"]))
    
    # Classify outliers
    hybrid_outliers = []
    
    # Confirmed outliers (both methods)
    confirmed_set = iqr_set & arima_set
    if confirmed_set:
        mask = df.apply(
            lambda row: (row["department"], row["year"]) in confirmed_set,
            axis=1
        )
        confirmed = df[mask].copy()
        confirmed["outlier_type"] = "confirmed"
        hybrid_outliers.append(confirmed)
    
    # IQR only
    iqr_only_set = iqr_set - arima_set
    if iqr_only_set:
        mask = df.apply(
            lambda row: (row["department"], row["year"]) in iqr_only_set,
            axis=1
        )
        iqr_only = df[mask].copy()
        iqr_only["outlier_type"] = "iqr_only"
        hybrid_outliers.append(iqr_only)
    
    # ARIMA only
    arima_only_set = arima_set - iqr_set
    if arima_only_set:
        mask = df.apply(
            lambda row: (row["department"], row["year"]) in arima_only_set,
            axis=1
        )
        arima_only = df[mask].copy()
        arima_only["outlier_type"] = "arima_only"
        hybrid_outliers.append(arima_only)
    
    result = pd.concat(hybrid_outliers, ignore_index=True) if hybrid_outliers else pd.DataFrame()
    logger.info(f"Hybrid detection: {len(confirmed_set)} confirmed, {len(iqr_only_set)} IQR-only, {len(arima_only_set)} ARIMA-only")
    
    return result


def compute_outlier_severity(df, column, outliers_df):
    """
    Compute severity metrics for each outlier.
    
    Args:
        df: Full dataset
        column: Column being analyzed
        outliers_df: DataFrame with detected outliers
        
    Returns:
        DataFrame with severity scores added
    """
    data = df[column].dropna()
    mean = data.mean()
    std = data.std()
    
    outliers_df = outliers_df.copy()
    outliers_df["deviation_std"] = (outliers_df[column] - mean) / std
    outliers_df["severity"] = np.abs(outliers_df["deviation_std"])
    
    return outliers_df


def plot_outlier_detection(df, column, outliers_df, output_path=None):
    """
    Create time series plot with outliers highlighted.
    
    Args:
        df: Full dataset
        column: Column analyzed
        outliers_df: DataFrame with outliers
        output_path: Path to save figure
    """
    logger.info(f"Creating outlier detection plot for {column}")
    
    if output_path is None:
        safe_col = column.replace("_", " ").title()
        output_path = REPORTS_FIGURES_PATH / f"outliers_{column}_detection.png"
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Group by department and plot
    n_depts = df["department"].nunique()
    n_cols = 4
    n_rows = (n_depts + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle(f"Outlier Detection: {column.replace('_', ' ').title()}", fontsize=14, fontweight="bold")
    
    for idx, dept in enumerate(sorted(df["department"].unique())):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        # Get department data
        dept_df = df[df["department"] == dept].sort_values("year")
        dept_outliers = outliers_df[
            (outliers_df["department"] == dept) & 
            (outliers_df[column].notna())
        ]
        
        # Plot main line
        ax.plot(
            dept_df["year"], dept_df[column],
            linewidth=2, color="#2c3e50", marker="o", markersize=4,
            label="Normal"
        )
        
        # Plot outliers by type
        for outlier_type in ["confirmed", "iqr_only", "arima_only"]:
            type_outliers = dept_outliers[dept_outliers.get("outlier_type") == outlier_type]
            if len(type_outliers) > 0:
                ax.scatter(
                    type_outliers["year"], type_outliers[column],
                    s=100, color=OUTLIER_COLORS.get(outlier_type, "#95a5a6"),
                    marker="X", label=outlier_type.replace("_", " ").title(),
                    zorder=5, edgecolors="black", linewidth=1
                )
        
        ax.set_title(f"{dept}\n({len(dept_outliers)} outliers)", fontweight="bold")
        ax.set_xlabel("Year")
        ax.set_ylabel("Rate per 100k")
        ax.grid(alpha=0.3)
        
        if idx == 0:
            ax.legend(loc="best", fontsize=8)
    
    # Hide extra subplots
    for idx in range(len(df["department"].unique()), n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis("off")
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Outlier detection plot saved to {output_path}")
    plt.close()


def plot_outlier_comparison(df, column, iqr_outliers, arima_outliers, output_path=None):
    """
    Create scatter plot comparing IQR vs ARIMA detection.
    
    Args:
        df: Full dataset
        column: Column analyzed
        iqr_outliers: DataFrame with IQR outliers
        arima_outliers: DataFrame with ARIMA outliers
        output_path: Path to save figure
    """
    logger.info(f"Creating outlier comparison plot for {column}")
    
    if output_path is None:
        output_path = REPORTS_FIGURES_PATH / f"outliers_{column}_comparison.png"
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot all data
    ax.scatter(
        df["year"], df[column],
        s=50, alpha=0.5, color="#95a5a6",
        label="Normal data"
    )
    
    # Plot IQR outliers
    if len(iqr_outliers) > 0:
        ax.scatter(
            iqr_outliers["year"], iqr_outliers[column],
            s=150, color=OUTLIER_COLORS["iqr_only"],
            marker="o", label="IQR only",
            edgecolors="black", linewidth=1.5, zorder=4
        )
    
    # Plot ARIMA outliers
    if len(arima_outliers) > 0:
        ax.scatter(
            arima_outliers["year"], arima_outliers[column],
            s=150, color=OUTLIER_COLORS["arima_only"],
            marker="^", label="ARIMA only",
            edgecolors="black", linewidth=1.5, zorder=4
        )
    
    # Plot confirmed outliers
    confirmed = set(zip(iqr_outliers["department"], iqr_outliers["year"])) & \
                set(zip(arima_outliers["department"], arima_outliers["year"]))
    if confirmed:
        confirmed_df = df[
            df.apply(lambda row: (row["department"], row["year"]) in confirmed, axis=1)
        ]
        ax.scatter(
            confirmed_df["year"], confirmed_df[column],
            s=200, color=OUTLIER_COLORS["confirmed"],
            marker="*", label="Confirmed (both methods)",
            edgecolors="black", linewidth=2, zorder=5
        )
    
    ax.set_title(f"Outlier Detection Comparison: {column.replace('_', ' ').title()}", 
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Rate per 100,000", fontsize=12)
    ax.legend(loc="best", fontsize=11)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Outlier comparison plot saved to {output_path}")
    plt.close()


def generate_outlier_report(df, columns_to_analyze=None):
    """
    Generate comprehensive outlier detection report.
    
    Args:
        df: Dataset with all measures
        columns_to_analyze: List of columns to analyze (default: all rate columns)
        
    Returns:
        Dictionary with summary statistics
    """
    if columns_to_analyze is None:
        columns_to_analyze = RATE_COLUMNS
    
    logger.info("Generating outlier detection report")
    
    report = {}
    
    for column in columns_to_analyze:
        logger.info(f"  Analyzing {column}")
        
        # Detect outliers
        hybrid_outliers = detect_hybrid_outliers(df, column)
        
        if len(hybrid_outliers) == 0:
            logger.info(f"    No outliers detected")
            continue
        
        # Compute severity
        hybrid_outliers = compute_outlier_severity(df, column, hybrid_outliers)
        
        # Classify by type
        confirmed_count = len(hybrid_outliers[hybrid_outliers["outlier_type"] == "confirmed"])
        iqr_only_count = len(hybrid_outliers[hybrid_outliers["outlier_type"] == "iqr_only"])
        arima_only_count = len(hybrid_outliers[hybrid_outliers["outlier_type"] == "arima_only"])
        
        report[column] = {
            "total_outliers": len(hybrid_outliers),
            "confirmed": confirmed_count,
            "iqr_only": iqr_only_count,
            "arima_only": arima_only_count,
            "outliers_df": hybrid_outliers.sort_values("severity", ascending=False),
        }
        
        logger.info(f"    Total: {len(hybrid_outliers)} | Confirmed: {confirmed_count} | IQR: {iqr_only_count} | ARIMA: {arima_only_count}")
    
    return report


def print_outlier_summary(report):
    """
    Print summary of outlier detection results.
    
    Args:
        report: Dictionary from generate_outlier_report
    """
    print("\n" + "=" * 80)
    print("OUTLIER DETECTION REPORT")
    print("=" * 80)
    
    for column, stats in report.items():
        print(f"\n📊 {column.upper()}")
        print(f"   Total outliers: {stats['total_outliers']}")
        print(f"   ✓ Confirmed (both methods): {stats['confirmed']}")
        print(f"   ⚠ IQR only: {stats['iqr_only']}")
        print(f"   ⚠ ARIMA only: {stats['arima_only']}")
        
        if len(stats["outliers_df"]) > 0:
            print(f"\n   Top 3 most severe outliers:")
            for idx, (_, row) in enumerate(stats["outliers_df"].head(3).iterrows(), 1):
                print(f"      {idx}. {row['department']} ({int(row['year'])}) = {row[column]:.2f} "
                      f"({row['outlier_type'].replace('_', ' ').title()}, "
                      f"severity: {row['severity']:.2f}σ)")


def run_outlier_detection():
    """
    Execute complete outlier detection workflow.
    
    Loads data, detects outliers using hybrid approach,
    generates visualizations, and produces report.
    """
    logger.info("=" * 80)
    logger.info("Starting Outlier Detection Analysis")
    logger.info("=" * 80)
    
    try:
        # Load data
        df = load_data()
        
        logger.info("\n" + "=" * 80)
        logger.info("HYBRID OUTLIER DETECTION (IQR + ARIMA)")
        logger.info("=" * 80)
        
        # Generate report
        report = generate_outlier_report(df, RATE_COLUMNS)
        
        # Print summary
        print_outlier_summary(report)
        
        # Create visualizations
        logger.info("\n" + "=" * 80)
        logger.info("GENERATING VISUALIZATIONS")
        logger.info("=" * 80)
        
        for column in RATE_COLUMNS:
            if column not in report:
                continue
            
            outliers_df = report[column]["outliers_df"]
            
            # Detection plot
            plot_outlier_detection(df, column, outliers_df)
            
            # Comparison plot
            iqr_outliers = detect_iqr_outliers(df, column)
            arima_outliers = detect_arima_outliers(df, column)
            plot_outlier_comparison(df, column, iqr_outliers, arima_outliers)
        
        logger.info("=" * 80)
        logger.info("Outlier Detection Analysis completed successfully")
        logger.info("=" * 80)
        
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"✓ Analyzed {len(RATE_COLUMNS)} measures")
        print(f"✓ Processed {df['department'].nunique()} departments")
        print(f"✓ Time period: {df['year'].min()}-{df['year'].max()}")
        total_outliers = sum(stats["total_outliers"] for stats in report.values())
        print(f"✓ Total outliers detected: {total_outliers}")
        print(f"✓ Output saved to: {REPORTS_FIGURES_PATH}")
        print("=" * 80 + "\n")
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    run_outlier_detection()
