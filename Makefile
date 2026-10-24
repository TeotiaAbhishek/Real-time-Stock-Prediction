.PHONY: install test lint backtest predict download

install:
	pip install torch --index-url https://download.pytorch.org/whl/cpu
	pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check .
	ruff format --check .

backtest:
	stock-forecast backtest

predict:
	stock-forecast predict

download:
	stock-forecast download
