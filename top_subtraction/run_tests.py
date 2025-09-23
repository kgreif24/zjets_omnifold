#!/usr/bin/env python3
"""
Test runner script for top_subtraction module.

This script provides a convenient way to run the test suite with different
configurations and options.

Usage:
    python run_tests.py                    # Run all tests
    python run_tests.py --unit             # Run only unit tests
    python run_tests.py --integration      # Run only integration tests
    python run_tests.py --coverage         # Run with coverage report
    python run_tests.py --verbose          # Run with verbose output
"""

import argparse
import subprocess
import sys
import os


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"✗ Command not found: {cmd[0]}")
        print("Make sure pytest is installed: pip install pytest")
        return False


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="Run top_subtraction tests")
    parser.add_argument("--unit", action="store_true", help="Run only unit tests")
    parser.add_argument(
        "--integration", action="store_true", help="Run only integration tests"
    )
    parser.add_argument(
        "--coverage", action="store_true", help="Run with coverage report"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Run with verbose output"
    )
    parser.add_argument("--file", "-f", type=str, help="Run specific test file")
    parser.add_argument("--pattern", "-k", type=str, help="Run tests matching pattern")
    parser.add_argument(
        "--no-cov", action="store_true", help="Disable coverage even if available"
    )

    args = parser.parse_args()

    # Change to the top_subtraction directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # Build pytest command
    cmd = ["python", "-m", "pytest"]

    # Add test path
    if args.file:
        cmd.append(f"tests/{args.file}")
    else:
        cmd.append("tests/")

    # Add markers
    if args.unit:
        cmd.extend(["-m", "unit"])
    elif args.integration:
        cmd.extend(["-m", "integration"])

    # Add pattern matching
    if args.pattern:
        cmd.extend(["-k", args.pattern])

    # Add verbosity
    if args.verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")

    # Add coverage
    if args.coverage and not args.no_cov:
        try:
            import pytest_cov

            cmd.extend(["--cov=.", "--cov-report=term-missing", "--cov-report=html"])
        except ImportError:
            print("Warning: pytest-cov not installed, skipping coverage")

    # Add other options
    cmd.extend(["--tb=short", "--strict-markers"])

    # Run the tests
    success = run_command(cmd, "Test Suite")

    if success:
        print(f"\n{'='*60}")
        print("✓ All tests passed!")
        if args.coverage and not args.no_cov:
            print("Coverage report generated in htmlcov/index.html")
        print(f"{'='*60}")
        sys.exit(0)
    else:
        print(f"\n{'='*60}")
        print("✗ Some tests failed!")
        print(f"{'='*60}")
        sys.exit(1)


if __name__ == "__main__":
    main()
