# Makefile for XFUND Generator

.PHONY: help install test clean setup templates run validate test-all test-unit test-integration test-pydantic test-forms test-quick test-coverage test-debug test-failed

help:
	@echo "Available commands:"
	@echo "  install    - Install dependencies"
	@echo "  setup      - Setup project and create sample templates"
	@echo "  templates  - Create sample DOCX templates"
	@echo "  layouts    - Generate layout JSON files for all templates"
	@echo "  validate   - Validate setup without generating dataset"
	@echo "  run        - Generate dataset with default config"
	@echo "  clean      - Clean generated files"
	@echo ""
	@echo "Test commands:"
	@echo "  test           - Run all tests (default)"
	@echo "  test-all       - Run all tests with verbose output"
	@echo "  test-quick     - Run quick tests (exclude slow tests)"
	@echo "  test-unit      - Run unit tests only"
	@echo "  test-integration - Run integration tests only"
	@echo "  test-pydantic  - Run Pydantic model tests"
	@echo "  test-forms     - Run form class tests"
	@echo "  test-coverage  - Run tests with coverage report"
	@echo "  test-debug     - Run tests in debug mode"
	@echo "  test-failed    - Run only failed tests from last run"

install:
	pip install -r requirements.txt

setup: install templates layouts
	@echo "Project setup complete!"
	@echo "Ready to generate XFUND dataset."

templates:
	python create_templates.py

layouts:
	python generate_layouts.py

validate:
	python src/generate_dataset.py --validate-only

run:
	python src/generate_dataset.py

# Test targets
test: test-all

test-all:
	@echo "🧪 Running all tests..."
	python -m pytest tests/ -v --tb=short --disable-warnings

test-quick:
	@echo "⚡ Running quick tests..."
	python -m pytest tests/ -v -m "not slow" --tb=short --disable-warnings

test-unit:
	@echo "🔧 Running unit tests..."
	python -m pytest tests/ -v -m "unit" --tb=short --disable-warnings

test-integration:
	@echo "🔗 Running integration tests..."
	python -m pytest tests/ -v -m "integration" --tb=short --disable-warnings

test-pydantic:
	@echo "📋 Running Pydantic model tests..."
	python -m pytest tests/test_pydantic_models.py -v --tb=short --disable-warnings

test-forms:
	@echo "📝 Running form class tests..."
	python -m pytest tests/test_form_classes.py -v --tb=short --disable-warnings

test-generator:
	@echo "⚙️ Running generator tests..."
	python -m pytest tests/test_generator.py -v --tb=short --disable-warnings

test-coverage:
	@echo "📊 Running tests with coverage..."
	python -m pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html --tb=short --disable-warnings

test-debug:
	@echo "🐛 Running tests in debug mode..."
	python -m pytest tests/ -v -s --tb=long --showlocals

test-failed:
	@echo "🔄 Running failed tests from last run..."
	python -m pytest tests/ -v --lf --tb=short --disable-warnings

clean:
	rm -rf output/*
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -name "*.pyc" -delete

# Development targets
dev-install:
	pip install -r requirements.txt
	pip install pytest pytest-cov black flake8

lint:
	black src/ tests/
	flake8 src/ tests/

test-coverage:
	pytest tests/ --cov=src/ --cov-report=html