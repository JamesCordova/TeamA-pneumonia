# ============================================================
# Modelado-Pneumonia — common project commands
# Usage: make <target>   (requires GNU make; on Windows use Git Bash or WSL)
# ============================================================

PYTHON   ?= python
AGE      ?= under5
DEPT     ?= AMAZONAS

.DEFAULT_GOAL := help

# ── Help ────────────────────────────────────────────────────
.PHONY: help
help:
	@echo ""
	@echo "  Modelado-Pneumonia — available targets"
	@echo "  --------------------------------------"
	@echo "  make baselines          Train Naive + SeasonalNaive for all departments"
	@echo "  make baselines-dept     Train baselines for a single dept  (DEPT=AMAZONAS)"
	@echo "  make sarima             Train SARIMA for all departments"
	@echo "  make sarima-dept        Train SARIMA for a single dept      (DEPT=AMAZONAS)"
	@echo "  make train-all          Run baselines + SARIMA for all departments"
	@echo "  make test               Run the test suite"
	@echo "  make lint               Run ruff linter"
	@echo "  make install            Install Python dependencies"
	@echo "  make clean-models       Remove all saved model files"
	@echo "  make clean-reports      Remove all generated reports"
	@echo ""
	@echo "  Options (override as env vars):"
	@echo "    AGE   = under5 | 60plus   (default: under5)"
	@echo "    DEPT  = AMAZONAS | LIMA …  (default: AMAZONAS)"
	@echo ""

# ── Install ─────────────────────────────────────────────────
.PHONY: install
install:
	$(PYTHON) -m pip install -r requirements.txt

# ── Baselines ───────────────────────────────────────────────
.PHONY: baselines
baselines:
	$(PYTHON) scripts/train_baselines.py --all --age_group $(AGE)

.PHONY: baselines-dept
baselines-dept:
	$(PYTHON) scripts/train_baselines.py --department $(DEPT) --age_group $(AGE)

# ── SARIMA ──────────────────────────────────────────────────
.PHONY: sarima
sarima:
	$(PYTHON) scripts/train_sarima.py --all --age_group $(AGE)

.PHONY: sarima-dept
sarima-dept:
	$(PYTHON) scripts/train_sarima.py --department $(DEPT) --age_group $(AGE)

# ── Combined ────────────────────────────────────────────────
.PHONY: train-all
train-all: baselines sarima

# ── Tests ───────────────────────────────────────────────────
.PHONY: test
test:
	$(PYTHON) -m pytest tests/ -v

.PHONY: test-fast
test-fast:
	$(PYTHON) -m pytest tests/ -v -x --tb=short

# ── Lint ────────────────────────────────────────────────────
.PHONY: lint
lint:
	$(PYTHON) -m ruff check pneumonia/ scripts/ tests/

# ── Clean ───────────────────────────────────────────────────
.PHONY: clean-models
clean-models:
	@echo "Removing saved models in models/"
	find models/ -name "*.pkl" -delete
	find models/ -name "*_metadata.json" -delete

.PHONY: clean-reports
clean-reports:
	@echo "Removing generated reports in reports/"
	find reports/ -name "*.json" -delete
	find reports/ -name "*.png" -delete
