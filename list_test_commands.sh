#!/bin/bash
# 🧪 XFUND Generator - All Test Commands
# Quick reference for running tests

echo "🧪 XFUND Generator Test Commands"
echo "================================"
echo ""

echo "🥇 RECOMMENDED: Self-contained script (uses uv with inline dependencies)"
echo "   uv run run_all_tests.py                  # Run all tests"
echo "   uv run run_all_tests.py --quick          # Quick tests only"
echo "   uv run run_all_tests.py --pydantic       # Pydantic tests only"
echo "   uv run run_all_tests.py --coverage       # With coverage report"
echo "   uv run run_all_tests.py --help           # Show all options"
echo ""

echo "⚡ FAST: Simple bash script"
echo "   ./test.sh                                # Run all tests"  
echo "   ./test.sh quick                          # Quick tests only"
echo "   ./test.sh pydantic                       # Pydantic tests only"
echo "   ./test.sh coverage                       # With coverage"
echo "   ./test.sh help                           # Show options"
echo ""

echo "🏭 STANDARD: Make commands"
echo "   make test                                # Run all tests"
echo "   make test-quick                          # Quick tests only"
echo "   make test-pydantic                       # Pydantic tests only"
echo "   make test-coverage                       # With coverage"
echo "   make help                                # Show all commands"
echo ""

echo "🔧 ADVANCED: Direct pytest"
echo "   python -m pytest tests/ -v              # All tests"
echo "   python -m pytest -m 'not slow' -v       # Quick tests"
echo "   python -m pytest -m 'pydantic' -v       # Pydantic tests"
echo "   python -m pytest --cov=src tests/       # With coverage"
echo ""

echo "📚 Documentation:"
echo "   COMPLETE_TEST_GUIDE.md                   # Comprehensive guide"
echo "   TEST_GUIDE.md                           # Detailed documentation"
echo "   tests/README.md                         # Test structure info"
echo ""

echo "💡 Tips:"
echo "   • Use 'uv run run_all_tests.py' for most reliable results"
echo "   • Use './test.sh quick' for fast development cycles"  
echo "   • Use '--coverage' option to see test coverage"
echo "   • All methods run the same comprehensive test suite"