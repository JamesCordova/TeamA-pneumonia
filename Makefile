# ============================================================
# Modelado-Pneumonia — common project commands
# Usage: make <target>   (requires GNU make; on Windows use Git Bash or WSL)
# ============================================================

PYTHON     ?= python
AGE        ?= under5
DEPT       ?= AMAZONAS
DOCKER_RUN  = docker compose run --rm pneumonia

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
	@echo "  Docker targets (GPU):"
	@echo "  make docker-build       Build the Docker image"
	@echo "  make docker-shell       Open an interactive bash shell in the container"
	@echo "  make docker-baselines   Train baselines inside Docker (AGE=under5)"
	@echo "  make docker-sarima      Train SARIMA inside Docker    (DEPT=AMAZONAS AGE=under5)"
	@echo "  make docker-test        Run test suite inside Docker"
	@echo "  make docker-run SCRIPT=scripts/foo.py ARGS='--flag val'"
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

# ── Docker ──────────────────────────────────────────────────
.PHONY: docker-build
docker-build:
	docker compose build

.PHONY: docker-shell
docker-shell:
	docker compose run --rm --entrypoint bash pneumonia

# Generic runner: make docker-run SCRIPT=scripts/foo.py ARGS="--dept LIMA"
.PHONY: docker-run
docker-run:
	$(DOCKER_RUN) $(SCRIPT) $(ARGS)

.PHONY: docker-baselines
docker-baselines:
	$(DOCKER_RUN) scripts/train_baselines.py --all --age_group $(AGE)

.PHONY: docker-baselines-dept
docker-baselines-dept:
	$(DOCKER_RUN) scripts/train_baselines.py --department $(DEPT) --age_group $(AGE)

.PHONY: docker-sarima
docker-sarima:
	$(DOCKER_RUN) scripts/train_sarima.py --all --age_group $(AGE)

.PHONY: docker-sarima-dept
docker-sarima-dept:
	$(DOCKER_RUN) scripts/train_sarima.py --department $(DEPT) --age_group $(AGE)

.PHONY: docker-walkforward
docker-walkforward:
	$(DOCKER_RUN) scripts/run_walkforward.py --department $(DEPT) --age_group $(AGE) $(ARGS)

.PHONY: docker-plot
docker-plot:
	$(DOCKER_RUN) scripts/plot_forecasting.py --department $(DEPT) --age_group $(AGE) $(ARGS)

.PHONY: docker-compare
docker-compare:
	$(DOCKER_RUN) scripts/compare_models.py --department $(DEPT) --age_group $(AGE) $(ARGS)

.PHONY: docker-test
docker-test:
	$(DOCKER_RUN) -m pytest tests/ -v
