# One-command local run:  make demo   (creates the venv, builds DuckDB + Chroma index, asks sample questions)
PY      ?= python3
VENV    ?= .venv
BIN     := $(VENV)/bin
CHEMRAG := $(BIN)/chemrag
Q       ?= Which products contain CAS 75-07-0?

.PHONY: setup build build-products ask demo test eval eval-holdout lint doctor graph clean

$(BIN)/python:
	$(PY) -m venv $(VENV)
	$(BIN)/pip install -q --upgrade pip
	# CPU-only torch keeps the download and RAM small (falls back to PyPI if the CPU index is unreachable)
	$(BIN)/pip install -q torch --index-url https://download.pytorch.org/whl/cpu || $(BIN)/pip install -q torch
	$(BIN)/pip install -q -e ".[dev]"

setup: $(BIN)/python

data/processed/cscp.duckdb: data/raw/interviewtestdataset.csv | setup
	$(CHEMRAG) build $(if $(PRODUCTS),--products,)

build: setup
	$(CHEMRAG) build $(if $(PRODUCTS),--products,)

build-products: setup
	$(CHEMRAG) build --products

ask: data/processed/cscp.duckdb
	$(CHEMRAG) ask "$(Q)"

demo: data/processed/cscp.duckdb
	$(CHEMRAG) ask "Which products contain CAS 75-07-0?" --limit 5 --evidence 5
	$(CHEMRAG) ask "What chemicals are reported for Sally Hansen in Nail Polish and Enamel?" --evidence 5
	$(CHEMRAG) ask "Show products discontinued in 2024 that had ChemicalName = talc"
	$(CHEMRAG) ask "Summarize reporting trends over time for Nail Products" --evidence 3
	$(CHEMRAG) ask 'What chemicals are reported for brand "Pure"?' --no-interactive
	$(CHEMRAG) ask "Is titanium dioxide safe for my kids?" --evidence 3

test: setup
	$(BIN)/pytest -q

eval: data/processed/cscp.duckdb
	$(CHEMRAG) eval

eval-holdout: data/processed/cscp.duckdb
	$(CHEMRAG) eval --holdout

lint: setup
	$(BIN)/ruff check chemrag evals tests

doctor: setup
	$(CHEMRAG) doctor

graph: setup
	$(CHEMRAG) graph

clean:
	rm -rf data/processed runs .pytest_cache
