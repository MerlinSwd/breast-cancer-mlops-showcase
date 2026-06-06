UV ?= uv
CLI := $(UV) run bc-mlops

.PHONY: install lock lint format test train compare validate predict report build clean

install:
	$(UV) sync --extra dev

lock:
	$(UV) lock

lint:
	$(UV) run ruff check .

format:
	$(UV) run ruff format .

test:
	$(UV) run pytest

train:
	$(CLI) train --config configs/train.yaml --output-dir artifacts/runs

compare:
	$(CLI) compare --registry artifacts/registry.json

validate:
	latest_run=$$(find artifacts/runs -mindepth 1 -maxdepth 1 -type d | sort | tail -1); \
	$(CLI) validate --metrics $$latest_run/metrics.json --gates configs/quality_gates.yaml

predict:
	latest_run=$$(find artifacts/runs -mindepth 1 -maxdepth 1 -type d | sort | tail -1); \
	model_artifact=$$(find $$latest_run -maxdepth 1 \( -name 'model.joblib' -o -name 'model.pt' \) | head -1); \
	$(CLI) predict --model $$model_artifact --input sample-inputs/sample.json

report:
	latest_run=$$(find artifacts/runs -mindepth 1 -maxdepth 1 -type d | sort | tail -1); \
	$(CLI) report --run-dir $$latest_run --output $$latest_run/MODEL_CARD.md

build:
	$(UV) run python -m build

clean:
	rm -rf .coverage .pytest_cache .ruff_cache .venv build dist htmlcov *.egg-info artifacts/runs artifacts/registry.json mlruns uv.lock
