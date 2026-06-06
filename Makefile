PYTHON ?= python3
VENV ?= .venv
PIP := $(VENV)/bin/pip
PYTEST := $(VENV)/bin/pytest
RUFF := $(VENV)/bin/ruff
BUILD := $(VENV)/bin/python -m build
CLI := $(VENV)/bin/bc-mlops

.PHONY: install lint format test train compare clean build

install:
	$(PYTHON) -m venv $(VENV)
	. $(VENV)/bin/activate && python -m pip install -U pip
	$(PIP) install -e '.[dev]'

lint:
	$(RUFF) check .

format:
	$(RUFF) format .

test:
	$(PYTEST)

train:
	$(CLI) train --config configs/train.yaml --output-dir artifacts/runs

compare:
	$(CLI) compare --registry artifacts/registry.json

build:
	$(BUILD)

clean:
	rm -rf .coverage .pytest_cache .ruff_cache build dist htmlcov *.egg-info artifacts/runs artifacts/registry.json
