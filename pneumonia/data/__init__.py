"""
Data loading, processing, and analysis functions for pneumonia surveillance data
"""

from pneumonia.data.compute_incidence_rates import (
    compute_and_save_pneumonia_incidence_rates,
    compute_disease_rate,
)
from pneumonia.data.interpolate_population import (
    interpolate_and_save_population_data,
)

__version__ = "0.1.0"

__all__ = [
    "compute_disease_rate",
    "interpolate_and_save_population_data",
]
