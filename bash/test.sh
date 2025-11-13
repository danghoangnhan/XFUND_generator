#!/bin/bash
# Simple bash script for running XFUND Generator tests
# Usage: ./test.sh [options]

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 XFUND Generator Test Suite${NC}"
echo "=================================="

# Default: run all tests with verbose output
if [ $# -eq 0 ]; then
    echo -e "${YELLOW}Running all tests...${NC}"
    python -m pytest tests/ -v --tb=short --disable-warnings
else
    case "$1" in
        "quick"|"fast")
            echo -e "${YELLOW}Running quick tests (excluding slow tests)...${NC}"
            python -m pytest tests/ -v -m "not slow" --tb=short --disable-warnings
            ;;
        "unit")
            echo -e "${YELLOW}Running unit tests...${NC}"
            python -m pytest tests/ -v -m "unit" --tb=short --disable-warnings
            ;;
        "integration")
            echo -e "${YELLOW}Running integration tests...${NC}"
            python -m pytest tests/ -v -m "integration" --tb=short --disable-warnings
            ;;
        "pydantic")
            echo -e "${YELLOW}Running Pydantic model tests...${NC}"
            python -m pytest tests/test_pydantic_models.py -v --tb=short --disable-warnings
            ;;
        "forms")
            echo -e "${YELLOW}Running form class tests...${NC}"
            python -m pytest tests/test_form_classes.py -v --tb=short --disable-warnings
            ;;
        "generator")
            echo -e "${YELLOW}Running generator tests...${NC}"
            python -m pytest tests/test_generator.py -v --tb=short --disable-warnings
            ;;
        "coverage")
            echo -e "${YELLOW}Running tests with coverage...${NC}"
            python -m pytest tests/ -v --cov=src --cov-report=term-missing --tb=short --disable-warnings
            ;;
        "debug")
            echo -e "${YELLOW}Running tests in debug mode...${NC}"
            python -m pytest tests/ -v -s --tb=long --showlocals
            ;;
        "failed")
            echo -e "${YELLOW}Running only failed tests from last run...${NC}"
            python -m pytest tests/ -v --lf --tb=short --disable-warnings
            ;;
        "help"|"-h"|"--help")
            echo "XFUND Generator Test Runner"
            echo ""
            echo "Usage: ./test.sh [option]"
            echo ""
            echo "Options:"
            echo "  (no args)   Run all tests"
            echo "  quick       Run quick tests (exclude slow tests)"
            echo "  unit        Run only unit tests"
            echo "  integration Run only integration tests"
            echo "  pydantic    Run Pydantic model tests"
            echo "  forms       Run form class tests"
            echo "  generator   Run generator tests"
            echo "  coverage    Run tests with coverage report"
            echo "  debug       Run tests in debug mode"
            echo "  failed      Run only failed tests from last run"
            echo "  help        Show this help message"
            echo ""
            echo "Examples:"
            echo "  ./test.sh                 # Run all tests"
            echo "  ./test.sh quick          # Run quick tests"
            echo "  ./test.sh coverage       # Run with coverage"
            echo "  ./test.sh pydantic       # Run Pydantic tests only"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use './test.sh help' for available options"
            exit 1
            ;;
    esac
fi

# Check exit code and provide feedback
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Tests completed successfully!${NC}"
else
    echo -e "${RED}❌ Some tests failed!${NC}"
    echo -e "${YELLOW}💡 Try './test.sh debug' for more detailed output${NC}"
    exit 1
fi