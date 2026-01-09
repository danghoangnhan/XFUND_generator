# Makefile for XFUND Generator
# For CI/CD, see .github/workflows/

.PHONY: help install dev test lint format fix type-check clean build

help:
	@echo "Development Commands:"
	@echo "  make install    - Install production dependencies"
	@echo "  make dev        - Install with dev dependencies"
	@echo "  make test       - Run all tests"
	@echo "  make lint       - Check code style (ruff + flake8)"
	@echo "  make format     - Format code with ruff"
	@echo "  make fix        - Auto-fix linting issues"
	@echo "  make type-check - Run mypy type checking"
	@echo "  make clean      - Remove build artifacts"
	@echo "  make build      - Build package"

# Installation
install:
	uv sync --no-dev

dev:
	uv sync

# Quality
lint:
	uv run ruff check xfund_generator/ tests/
	uv run flake8 xfund_generator/ tests/

format:
	uv run ruff format xfund_generator/ tests/

fix:
	uv run ruff check --fix xfund_generator/ tests/
	uv run ruff format xfund_generator/ tests/

type-check:
	uv run mypy xfund_generator/

# Testing
test:
	uv run pytest tests/ -v --tb=short

test-cov:
	uv run pytest tests/ --cov=xfund_generator --cov-report=term-missing --cov-report=html

# Build
build:
	uv build

# Cleanup
clean:
	rm -rf dist/ build/ *.egg-info .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
