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

__all__ = [
    "load_disease_measures",
    "compute_national_rates",
    "compute_departmental_rates",
    "plot_national_variation_grid",
    "plot_departmental_variation",
    "save_annual_tables",
    "run_analysis",
]
