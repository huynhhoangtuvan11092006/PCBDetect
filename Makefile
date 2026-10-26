.PHONY: help install eda test lint format clean

PYTHON := .venv/Scripts/python.exe
PYTEST := .venv/Scripts/pytest.exe
RUFF := .venv/Scripts/ruff.exe

help:
	@echo "PCBDetect Task Automation"
	@echo "  make install       - Install dependencies and editable package"
	@echo "  make prepare-data  - Convert raw DeepPCB to YOLO and COCO formats"
	@echo "  make benchmark     - Run M2 comparative benchmark (Classical vs YOLO)"
	@echo "  make eda           - Run full Exploratory Data Analysis & generate plots"
	@echo "  make test          - Run test suite with pytest"
	@echo "  make lint          - Run ruff linter check"
	@echo "  make format        - Run ruff formatter"
	@echo "  make clean         - Clean up build artifacts and temporary caches"

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

prepare-data:
	$(PYTHON) -m pcbdetect.data.convert --strategy group_disjoint --format both

benchmark:
	$(PYTHON) -m pcbdetect.eval.benchmark_m2

eda:
	$(PYTHON) -m pcbdetect.data.eda

test:
	$(PYTEST) -v tests/

lint:
	$(RUFF) check .

format:
	$(RUFF) format .

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache/ .ruff_cache/
