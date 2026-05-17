"""
Exploratory Data Analysis module for pneumonia disease measures
"""

from pneumonia.eda.annual_variation import (
    load_disease_measures,
    compute_national_rates,
    compute_departmental_rates,
    plot_national_variation_grid,
    plot_departmental_variation,
    save_annual_tables,
    run_analysis,
)

from pneumonia.eda.regional_analysis import (
    load_weekly_data,
    load_annual_population,
    load_top_regions,
    prepare_regional_data,
    compute_weekly_incidence,
    compute_monthly_seasonality,
    compute_annual_trend,
    decompose_time_series,
    plot_regional_analysis_grid,
    run_regional_analysis,
)

from pneumonia.eda.seasonality_analysis import (
    load_and_prepare_weekly_data as seasonality_load_weekly_data,
    load_population_data as seasonality_load_population,
    compute_incidence,
    aggregate_to_monthly,
    aggregate_to_monthly_by_department,
    compute_monthly_summary_stats,
    plot_national_seasonality_boxplots,
    plot_national_seasonality_heatmaps,
    plot_national_stl_decomposition,
    plot_departmental_boxplots_grid,
    plot_departmental_seasonality_comparison,
    run_seasonality_analysis,
)

from pneumonia.eda.outlier_detection import (
    load_data as outlier_load_data,
    detect_iqr_outliers,
    fit_arima_departmental,
    detect_arima_outliers,
    detect_hybrid_outliers,
    compute_outlier_severity,
    plot_outlier_detection,
    plot_outlier_comparison,
    generate_outlier_report,
    run_outlier_detection,
)

__all__ = [
    # Annual variation
    "load_disease_measures",
    "compute_national_rates",
    "compute_departmental_rates",
    "plot_national_variation_grid",
    "plot_departmental_variation",
    "save_annual_tables",
    "run_analysis",
    # Regional analysis
    "load_weekly_data",
    "load_annual_population",
    "load_top_regions",
    "prepare_regional_data",
    "compute_weekly_incidence",
    "compute_monthly_seasonality",
    "compute_annual_trend",
    "decompose_time_series",
    "plot_regional_analysis_grid",
    "run_regional_analysis",
    # Seasonality analysis
    "seasonality_load_weekly_data",
    "seasonality_load_population",
    "compute_incidence",
    "aggregate_to_monthly",
    "aggregate_to_monthly_by_department",
    "compute_monthly_summary_stats",
    "plot_national_seasonality_boxplots",
    "plot_national_seasonality_heatmaps",
    "plot_national_stl_decomposition",
    "plot_departmental_boxplots_grid",
    "plot_departmental_seasonality_comparison",
    "run_seasonality_analysis",
    # Outlier detection
    "outlier_load_data",
    "detect_iqr_outliers",
    "fit_arima_departmental",
    "detect_arima_outliers",
    "detect_hybrid_outliers",
    "compute_outlier_severity",
    "plot_outlier_detection",
    "plot_outlier_comparison",
    "generate_outlier_report",
    "run_outlier_detection",
]
