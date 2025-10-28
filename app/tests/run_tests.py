"""
Quick test runner for the hybrid multilingual NLP system

Run all tests with:
    python run_tests.py

Run specific test file:
    python run_tests.py --file test_hybrid_pipeline.py

Run with coverage:
    python run_tests.py --coverage
"""

import sys
import subprocess
from pathlib import Path

def run_tests(test_file=None, coverage=False):
    """Run pytest tests"""
    
    # Base command
    cmd = ["pytest"]
    
    # Add specific file or all tests
    if test_file:
        cmd.append(f"app/tests/{test_file}")
    else:
        cmd.append("app/tests/")
    
    # Add verbose output
    cmd.extend(["-v", "--tb=short"])
    
    # Add coverage if requested
    if coverage:
        cmd.extend(["--cov=app", "--cov-report=html", "--cov-report=term"])
    
    print("=" * 70)
    print("🧪 Running Tests for Multilingual Hinglish NLP v2.0")
    print("=" * 70)
    print(f"\nCommand: {' '.join(cmd)}\n")
    
    # Run tests
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except FileNotFoundError:
        print("\n❌ Error: pytest not found!")
        print("Install with: pip install pytest pytest-cov")
        return 1

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run tests for Hinglish NLP")
    parser.add_argument("--file", "-f", help="Specific test file to run")
    parser.add_argument("--coverage", "-c", action="store_true", help="Run with coverage")
    
    args = parser.parse_args()
    
    returncode = run_tests(test_file=args.file, coverage=args.coverage)
    
    if returncode == 0:
        print("\n" + "=" * 70)
        print("✅ All tests passed!")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("❌ Some tests failed. Check output above.")
        print("=" * 70)
    
    return returncode

if __name__ == "__main__":
    sys.exit(main())
