"""Shared visual theme constants for plotting modules."""

import matplotlib.colors as mcolors


# Color ramps are defined once here so plotting modules can reuse the same
# visual language without re-declaring palette details in each figure helper.
_SEQUENTIAL_BLUE_STOPS = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]

SEQUENTIAL_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "pneumonia_seq_blue", _SEQUENTIAL_BLUE_STOPS
)
DIVERGING_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "pneumonia_div_blue_red", ["#e34948", "#fcfcfb", "#2a78d6"]
)
NO_DATA_COLOR = "#e1e0d9"  # hairline gridline gray — distinct from the palest sequential step