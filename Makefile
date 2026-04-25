.PHONY: lint format

lint:
	python3 -m flake8 src tests run_pipeline.py update_report.py
	python3 -m ruff check src tests run_pipeline.py update_report.py

format:
	python3 -m ruff check --fix src tests run_pipeline.py update_report.py

parser:
	python src/parser.py
