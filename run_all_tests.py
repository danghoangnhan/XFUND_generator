#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "colorama>=0.4.6",
#     "pydantic>=2.0.0",
#     "pytest>=7.4.0",
#     "pytest-cov>=4.1.0",
# ]
# ///

"""
XFUND Generator - Self-Contained Test Runner

A modern Python script using inline dependencies (PEP 723) for running tests.
This script automatically manages its own environment and dependencies.

Usage:
    uv run run_all_tests.py                    # Run all tests  
    uv run run_all_tests.py --quick            # Quick tests only
    uv run run_all_tests.py --pydantic         # Pydantic tests
    uv run run_all_tests.py --coverage         # With coverage
"""

import argparse
import subprocess
import sys
from pathlib import Path

try:
    from colorama import Fore, Style, init
    init(autoreset=True)
except ImportError:
    class Fore:
        RED = GREEN = YELLOW = BLUE = CYAN = ""
    class Style:
        BRIGHT = RESET_ALL = ""

def print_colored(message: str, color: str = ""):
    """Print colored message"""
    print(f"{color}{message}{Style.RESET_ALL}")

def run_tests(test_type: str = "all") -> int:
    """Run tests based on type"""
    
    print_colored("🧪 XFUND Generator Test Runner", Fore.CYAN + Style.BRIGHT)
    print_colored("=" * 60, Fore.BLUE)
    
    # Check environment
    if not Path("tests").exists():
        print_colored("❌ Tests directory not found!", Fore.RED)
        return 1
    
    print_colored("✅ Test environment ready", Fore.GREEN)
    
    # Build pytest command
    cmd = ["python", "-m", "pytest", "tests/", "-v", "--tb=short", "--disable-warnings"]
    
    # Add test selection
    if test_type == "quick":
        cmd.extend(["-m", "not slow"])
        description = "⚡ Running quick tests"
    elif test_type == "pydantic":
        cmd = ["python", "-m", "pytest", "tests/test_pydantic_models.py", "-v", "--tb=short", "--disable-warnings"]
        description = "📋 Running Pydantic tests"
    elif test_type == "forms":
        cmd = ["python", "-m", "pytest", "tests/test_form_classes.py", "-v", "--tb=short", "--disable-warnings"]
        description = "📝 Running form tests"
    elif test_type == "coverage":
        cmd.extend(["--cov=src", "--cov-report=term-missing"])
        description = "📊 Running tests with coverage"
    else:
        description = "🧪 Running all tests"
    
    print_colored(f"\n{description}", Fore.YELLOW)
    print_colored("=" * 60, Fore.BLUE)
    
    # Run the command
    try:
        result = subprocess.run(cmd, check=True)
        print_colored("\n🎉 Tests completed successfully!", Fore.GREEN + Style.BRIGHT)
        return 0
    except subprocess.CalledProcessError as e:
        print_colored(f"\n💥 Tests failed! Exit code: {e.returncode}", Fore.RED + Style.BRIGHT)
        print_colored("\n💡 Try: uv run run_all_tests.py --help", Fore.YELLOW)
        return 1

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="🧪 XFUND Generator Self-Contained Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--quick", action="store_true", help="Run quick tests only")
    parser.add_argument("--pydantic", action="store_true", help="Run Pydantic model tests")
    parser.add_argument("--forms", action="store_true", help="Run form class tests")
    parser.add_argument("--coverage", action="store_true", help="Run with coverage report")
    
    args = parser.parse_args()
    
    # Determine test type
    if args.quick:
        test_type = "quick"
    elif args.pydantic:
        test_type = "pydantic"
    elif args.forms:
        test_type = "forms"
    elif args.coverage:
        test_type = "coverage"
    else:
        test_type = "all"
    
    return run_tests(test_type)

if __name__ == "__main__":
    sys.exit(main())
