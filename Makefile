.PHONY: help install suite inspect report clean test

help:
	@echo "Benchmark Harness Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  install    - Install dependencies"
	@echo "  suite      - Run full benchmark suite"
	@echo "  inspect    - Run inspection tools"
	@echo "  report     - Generate benchmark report"
	@echo "  test       - Run tests"
	@echo "  clean      - Clean results and cache"
	@echo "  all        - Run suite, inspect, and report"

install:
	pip install -r requirements.txt
	pip install -e .

suite:
	python bench/run_suite.py --config bench/configs/suite.yaml

inspect:
	python inspection/run_inspection.py --results-dir results --inspect-dir results/inspection

report:
	python reports/generate_report.py --results-dir results --output reports/report_latest.md

test:
	pytest tests/ -v

clean:
	rm -rf results/
	rm -rf reports/report_latest.md
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

all: suite inspect report
	@echo ""
	@echo "==================================="
	@echo "Benchmark pipeline complete!"
	@echo "Check reports/report_latest.md"
	@echo "==================================="
