# SARIMA Training & Execution Workflow

Complete guide for training, validating, and testing SARIMA models for pneumonia forecasting.

---

## 📋 Quick Start

### 1. Setup

```bash
# Copy environment configuration
cp .env.example .env

# Install dependencies (if not already done)
pip install -r requirements.txt

# Install optional dependency for auto_arima parameter search
pip install pmdarima
```

### 2. Train a Single Department

```bash
# Train SARIMA for AMAZONAS department (under5 age group)
python scripts/train_sarima.py --department AMAZONAS --age_group under5

# Output:
# ✓ models/AMAZONAS/under5/sarima_model.pkl
# ✓ models/AMAZONAS/under5/sarima_model_metadata.json
# ✓ reports/AMAZONAS/under5/results.json
```

### 3. Train All Departments

```bash
# Train SARIMA for all departments with automatic parameter search
python scripts/train_sarima.py --all --age_group under5 --use_auto_arima -v
```

### 4. Review Results

Results are saved to:
- **Model files**: `models/DEPT/AGE_GROUP/sarima_model.pkl`
- **Metadata**: `models/DEPT/AGE_GROUP/sarima_model_metadata.json`
- **Evaluation report**: `reports/DEPT/AGE_GROUP/results.json`

---

## 🔧 Configuration

### Environment Variables (.env file)

```bash
# Reproducibility
RANDOM_SEED=42

# Temporal split strategy
TEMPORAL_SPLIT_STRATEGY=dynamic    # or "years"
TRAIN_RATIO=0.8                    # 80% for training
VAL_RATIO=0.1                      # 10% for validation
TEST_RATIO=0.1                     # 10% for testing

# SARIMA model configuration
USE_AUTO_ARIMA=True                # Auto parameter search
SARIMA_MAX_ITERATIONS=400          # Max iterations for auto_arima
SARIMA_STEPWISE=True               # Use stepwise search (faster)
```

### SARIMA Parameters

Default parameters in `pneumonia/models/sarima/config.py`:

- **Order**: (p, d, q) = (1, 1, 1)
  - p: AR (autoregressive) terms
  - d: Differencing order
  - q: MA (moving average) terms

- **Seasonal Order**: (P, D, Q, s) = (1, 1, 1, 52)
  - P, D, Q: Seasonal AR, differencing, MA
  - s: Seasonal period (52 weeks per year)

When `USE_AUTO_ARIMA=True`, these defaults are overridden by `pmdarima.auto_arima()` search.

---

## 📊 Training Workflow

### Stage-by-Stage Breakdown

The pipeline executes these 6 stages:

#### 1. **Data Loading & Validation**
```python
# Load weekly pneumonia cases
get_departmental_data(department='AMAZONAS', age_group='under5')
# ↓ Aggregates by week, handles missing values, validates
```

#### 2. **Temporal Split**
```python
# Split into train/val/test
temporal_split(data, strategy='dynamic')  # 80% / 10% / 10%
# OR
temporal_split(data, strategy='years')    # 2000-2019 / 2020-2021 / 2022-2023
```

#### 3. **Model Training**
```python
# Create and fit SARIMA
model = SARIMAModel(department='AMAZONAS', age_group='under5')
model.fit(train_data, use_auto_arima=True)  # Auto search for best order
```

If `use_auto_arima=True`:
- Searches parameter space: p ∈ [0,2], d ∈ [0,1], q ∈ [0,2], P ∈ [0,1], D ∈ [0,1], Q ∈ [0,1]
- Uses stepwise search for speed (if `SARIMA_STEPWISE=True`)
- Returns optimal (p,d,q)×(P,D,Q,s)

#### 4. **Validation**
```python
# Predict on validation set (52 weeks)
val_forecast = model.predict(steps=52)
# Compute metrics: MAE, RMSE, MAPE, SMAPE, MDA
val_metrics = compute_all_metrics(val_data, val_forecast)
```

#### 5. **Testing**
```python
# Final evaluation on test set
test_forecast = model.predict(steps=52)
test_metrics = compute_all_metrics(test_data, test_forecast)
```

#### 6. **Reporting**
```python
# Save model and results
model.save()  # → model.pkl + model_metadata.json
# → reports/DEPT/AGE_GROUP/results.json
```

---

## 🚀 CLI Usage Examples

### Single Department Training

```bash
# Basic training (uses config defaults)
python scripts/train_sarima.py --department AMAZONAS

# With specific age group
python scripts/train_sarima.py --department LIMA --age_group 60plus

# With auto_arima parameter search
python scripts/train_sarima.py --department ANCASH --use_auto_arima

# Force manual parameters (no auto_arima)
python scripts/train_sarima.py --department CUSCO --no_auto_arima

# With dynamic temporal split (ratio-based)
python scripts/train_sarima.py --department AMAZONAS --split_strategy dynamic

# With year-based split
python scripts/train_sarima.py --department AMAZONAS --split_strategy years

# Custom forecast steps
python scripts/train_sarima.py --department AMAZONAS --forecast_steps 104  # 2 years

# Verbose output
python scripts/train_sarima.py --department AMAZONAS -v

# Quiet mode (suppress non-critical output)
python scripts/train_sarima.py --department AMAZONAS --quiet
```

### Batch Training

```bash
# Train all departments (under5, with auto_arima)
python scripts/train_sarima.py --all --age_group under5 --use_auto_arima

# Train all for 60+ age group
python scripts/train_sarima.py --all --age_group 60plus

# Train all with verbose logging
python scripts/train_sarima.py --all -v
```

---

## 🧪 Testing

### Run All Tests

```bash
pytest tests/test_sarima_pipeline.py -v
```

### Run Specific Test Classes

```bash
# Test temporal split functionality
pytest tests/test_sarima_pipeline.py::TestTemporalSplit -v

# Test metrics computation
pytest tests/test_sarima_pipeline.py::TestMetrics -v

# Test SARIMA model
pytest tests/test_sarima_pipeline.py::TestSARIMAModel -v

# Test pipeline
pytest tests/test_sarima_pipeline.py::TestSARIMAPipeline -v

# Integration tests
pytest tests/test_sarima_pipeline.py::TestIntegration -v
```

### Run Performance Tests (slow)

```bash
pytest tests/test_sarima_pipeline.py -m slow
```

---

## 📁 Output Structure

After running the pipeline, the following files are created:

```
models/
├── AMAZONAS/
│   ├── under5/
│   │   ├── sarima_model.pkl                    # Trained model
│   │   └── sarima_model_metadata.json          # Training metadata
│   └── 60plus/
│       ├── sarima_model.pkl
│       └── sarima_model_metadata.json
├── ANCASH/
│   └── ...
└── ...

reports/
├── AMAZONAS/
│   ├── under5/
│   │   └── results.json                        # Evaluation results
│   └── 60plus/
│       └── results.json
└── ...

logs/
├── pneumonia.log                               # Training logs
└── ...
```

### Example: Results JSON Structure

```json
{
  "department": "AMAZONAS",
  "age_group": "under5",
  "timestamp": "2026-05-27T...",
  "stages": {
    "data_loading": {
      "n_observations": 1248,
      "date_range": {
        "start": "2000-01-09",
        "end": "2023-12-31"
      },
      "status": "success"
    },
    "training": {
      "order": [2, 1, 1],
      "seasonal_order": [1, 1, 1, 52],
      "aic": 5234.56,
      "bic": 5267.89,
      "fit_method": "auto_arima",
      "status": "success"
    },
    "validation": {
      "metrics": {
        "mae": 1.23,
        "rmse": 1.56,
        "mape": 8.9,
        "smape": 7.2,
        "mda": 62.5
      },
      "status": "success"
    },
    "testing": {
      "metrics": {...},
      "status": "success"
    }
  }
}
```

---

## 🔍 Temporal Split Strategies

### Dynamic Split (Recommended)

Splits data by percentage, automatically calculating boundaries:

```bash
python scripts/train_sarima.py --department AMAZONAS --split_strategy dynamic

# With custom ratios (via .env):
TRAIN_RATIO=0.75
VAL_RATIO=0.15
TEST_RATIO=0.10
```

**Pros:**
- Adapts to data length
- Works with any date range
- No hardcoding required

**Cons:**
- Less reproducible across years
- May not align with calendar years

### Year-Based Split (Legacy)

Splits by year ranges:

```bash
python scripts/train_sarima.py --department AMAZONAS --split_strategy years

# Uses defaults from pneumonia/config.py:
DEFAULT_TRAIN_YEARS = (2000, 2019)
DEFAULT_VAL_YEARS = (2020, 2021)
DEFAULT_TEST_YEARS = (2022, 2023)
```

**Pros:**
- Reproducible
- Clear year boundaries

**Cons:**
- Hardcoded years may not match data
- Less flexible

---

## 📈 Evaluation Metrics

The pipeline computes 5 key metrics:

| Metric | Formula | Notes |
|--------|---------|-------|
| **MAE** | (1/n)Σ\|actual - pred\| | Mean error, interpretable |
| **RMSE** | √((1/n)Σ(actual - pred)²) | Penalizes large errors |
| **MAPE** | (1/n)Σ\|actual - pred\|/\|actual\| × 100 | ⚠️ Undefined for zero actuals |
| **SMAPE** | (1/n)Σ\|actual - pred\|/((actual + pred)/2) × 100 | Symmetric, robust to small values |
| **MDA** | % of time direction changes match | Trend accuracy |

**Note**: MAPE returns NaN when actual values are near zero. Use SMAPE as alternative.

---

## 🐛 Troubleshooting

### Issue: "Insufficient training data"

```
ValueError: Insufficient training data: 50 weeks < 104 required
```

**Solution**: 
- Increase `TRAIN_RATIO` in .env (e.g., 0.9 instead of 0.8)
- Or use more historical data
- Or extend `DEFAULT_TRAIN_YEARS` in config

### Issue: "All actual values are zero or near-zero; MAPE undefined"

```
WARNING: MAPE: All actual values are zero... MAPE undefined. Returning NaN.
```

**Solution**: This is expected when disease counts are very low. Check SMAPE instead.

### Issue: "pmdarima not installed"

```
ImportError: pmdarima is required for auto_arima
```

**Solution**: Install pmdarima:
```bash
pip install pmdarima
```

Or disable auto_arima in .env:
```
USE_AUTO_ARIMA=False
```

### Issue: Model not converging

```
ConvergenceWarning: Maximum number of iterations reached
```

**Solution**: 
- Increase `SARIMA_MAX_ITERATIONS` in .env
- Or disable auto_arima and use manual parameters
- Check data quality (missing values, outliers)

---

## 📚 Key Classes & Functions

### SARIMAPipeline

Main orchestrator class:

```python
from pneumonia.pipelines.sarima_pipeline import SARIMAPipeline

pipeline = SARIMAPipeline(
    department='AMAZONAS',
    age_group='under5',
    use_auto_arima=True,
)

results = pipeline.run()
print(pipeline.summary())
```

### SARIMAModel

Individual model training:

```python
from pneumonia.models.sarima.model import SARIMAModel

model = SARIMAModel(department='AMAZONAS', age_group='under5')
model.fit(train_data, use_auto_arima=True)

forecast = model.predict(steps=52)
diag = model.diagnostics()
model.save()
```

### Utility Functions

```python
from pneumonia.models.utils import (
    get_departmental_data,
    temporal_split,
    handle_missing_values,
)

# Load data
ts = get_departmental_data('AMAZONAS', age_group='under5')

# Split into train/val/test
train, val, test = temporal_split(ts, strategy='dynamic')

# Handle missing values
clean_ts = handle_missing_values(ts, method='interpolate')
```

---

## 🔐 Best Practices

1. **Always set RANDOM_SEED** for reproducibility
2. **Use dynamic split** unless specific years are required
3. **Enable auto_arima** for optimal parameter search
4. **Review SARIMA diagnostics** (ADF test, residuals) after training
5. **Compare SMAPE** instead of MAPE for small counts
6. **Run tests** before batch training: `pytest tests/test_sarima_pipeline.py`
7. **Check logs** for warnings and errors: `logs/pneumonia.log`
8. **Version control models** by storing metadata.json with each model

---

## 📖 Related Documentation

- [SARIMA Configuration](pneumonia/models/sarima/config.py)
- [Metrics Documentation](pneumonia/evaluation/metrics.py)
- [Pipeline Code](pneumonia/pipelines/sarima_pipeline.py)
- [CLI Code](scripts/train_sarima.py)
- [Tests](tests/test_sarima_pipeline.py)
