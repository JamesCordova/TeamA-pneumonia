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
]
