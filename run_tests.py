#!/usr/bin/env python3
"""
Test runner for Decant.

Adds src/ to path and runs pytest.
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Run pytest
import pytest

if __name__ == "__main__":
    # Run with verbose output and coverage
    exit_code = pytest.main([
        "tests/",
        "-v",
        "--tb=short",
        "--cov=decant",
        "--cov-report=term-missing",
        "--cov-report=html"
    ])
    sys.exit(exit_code)
