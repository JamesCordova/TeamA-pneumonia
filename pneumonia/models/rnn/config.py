"""
Default hyperparameters shared by all RNN models (LSTM, GRU).

Both architectures use the same sequence length, training schedule, and
feature set — the only difference is the recurrent cell type.
All values can be overridden via model_params in WalkForwardValidator or
via CLI flags in scripts/run_walkforward.py.
"""

RNN_DEFAULT_PARAMS = {
    # Sequence architecture
    "lookback": 52,           # weeks of history fed to the RNN
    "forecast_horizon": 4,    # Dense output size (internal prediction block)

    # Training
    "epochs": 60,
    "batch_size": 8,
    "val_weeks": 26,          # last N sequences reserved for EarlyStopping validation

    # Network
    "units_1": 48,            # units in the first recurrent layer
    "units_2": 24,            # units in the second recurrent layer
    "dropout_rate": 0.3,

    # Preprocessing
    "smooth_window": 3,       # rolling-mean window applied before MinMax scaling
}

# Hyperparameter search ranges (for future tuning)
RNN_SEARCH_RANGES = {
    "lookback": [26, 52, 78],
    "forecast_horizon": [1, 4, 8],
    "epochs": [40, 60, 100],
    "batch_size": [4, 8, 16],
    "units_1": [32, 48, 64, 96],
    "units_2": [16, 24, 32, 48],
    "dropout_rate": [0.1, 0.2, 0.3, 0.4],
    "smooth_window": [1, 3, 5],
}
