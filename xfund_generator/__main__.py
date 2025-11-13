#!/usr/bin/env python3
"""
Entry point for running xfund_generator as a module.
Usage: python -m xfund_generator [args...]
"""

import sys
from .generate_dataset import main

if __name__ == "__main__":
    sys.exit(main())