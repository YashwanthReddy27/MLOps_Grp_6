"""
Simple test runner for the data pipeline test suite
Focuses on running tests with pytest or unittest without coverage complexity
"""

import sys
import os
import unittest
from pathlib import Path

# Add project directories to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'dags'))
sys.path.insert(0, str(project_root / 'dags' / 'common'))

# Simple pytest.ini configuration
PYTEST_INI = """
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    -v
    --tb=short
    --strict-markers
    --disable-warnings
markers =
    unit: Unit tests
    integration: Integration tests
    slow: Slow running tests
    performance: Performance tests
"""

# Minimal test requirements
REQUIREMENTS_TEST = """pytest>=7.0.0
pytest-mock>=3.10.0"""


def run_all_tests_unittest():
    """Run all tests with unittest (no external dependencies needed)"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test modules
    test_modules = [
        'test_data_cleaning',
        'test_data_enrichment',
        'test_deduplication',
        'test_database_utils',
        'test_data_validator',
        'test_file_management'
    ]
    
    tests_found = 0
    for module_name in test_modules:
        try:
            module = __import__(module_name)
            module_suite = loader.loadTestsFromModule(module)
            suite.addTests(module_suite)
            tests_found += module_suite.countTestCases()
            print(f"✓ Loaded {module_name}")
        except ImportError as e:
            print(f"✗ Could not import {module_name}: {e}")
    
    if tests_found == 0:
        print("\nNo tests found! Make sure test files are in the tests/ directory")
        return 1
    
    print(f"\nRunning {tests_found} tests...\n")
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.wasSuccessful():
        print("\n✅ All tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed.")
        if result.failures:
            print("\nFailed tests:")
            for test, traceback in result.failures:
                print(f"  - {test}")
        if result.errors:
            print("\nTests with errors:")
            for test, traceback in result.errors:
                print(f"  - {test}")
        return 1


def run_with_pytest():
    """Run tests using pytest (simple, no coverage)"""
    import subprocess
    
    # Check if pytest is installed
    try:
        import pytest
    except ImportError:
        print("pytest is not installed. Install it with: pip install pytest")
        print("Or run tests with unittest: python test_runner.py --unittest")
        return 1
    
    cmd = ["pytest", "-v", "--tb=short", "tests/"]
    
    print("Running tests with pytest...\n")
    result = subprocess.run(cmd)
    
    return result.returncode


def run_specific_module(module_name):
    """Run tests for a specific module"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    try:
        module = __import__(f'test_{module_name}')
        suite.addTests(loader.loadTestsFromModule(module))
        
        print(f"Running tests for {module_name}...\n")
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        return 0 if result.wasSuccessful() else 1
    except ImportError as e:
        print(f"Error: Could not import test_{module_name}")
        print(f"Details: {e}")
        return 1


def setup_minimal():
    """Minimal setup - create config files and test fixtures"""
    print("Creating minimal test configuration...")
    
    # Ensure directories exist
    os.makedirs("tests/fixtures", exist_ok=True)
    
    # Create pytest.ini
    with open("pytest.ini", "w") as f:
        f.write(PYTEST_INI)
    print("✓ Created pytest.ini")
    
    # Create requirements file for reference
    with open("requirements-test.txt", "w") as f:
        f.write(REQUIREMENTS_TEST)
    print("✓ Created requirements-test.txt")
    
    # Create sample fixture files
    print("\nCreating fixture files...")
    
    # Create sample_news.json
    sample_news = {
        "articles": [
            {
                "title": "AI Breakthrough: GPT-5 Announced",
                "description": "OpenAI announces next generation language model",
                "url": "https://example.com/gpt5",
                "publishedAt": "2024-01-15T10:00:00Z",
                "source": {"name": "TechCrunch"},
                "author": "Sarah Johnson"
            },
            {
                "title": "Computer Vision Advances in Medical Imaging",
                "description": "New YOLO variant improves tumor detection",
                "url": "https://example.com/medical-cv",
                "publishedAt": "2024-01-14T15:30:00Z",
                "source": {"name": "Medical AI News"},
                "author": "Dr. Smith"
            }
        ]
    }
    
    with open("tests/fixtures/sample_news.json", "w") as f:
        import json
        json.dump(sample_news, f, indent=2)
    print("✓ Created tests/fixtures/sample_news.json")
    
    # Create sample_arxiv.json
    sample_arxiv = {
        "papers": [
            {
                "arxiv_id": "2401.00123",
                "title": "Attention Is All You Need: Revisited",
                "abstract": "We present improvements to the transformer architecture...",
                "authors": ["Vaswani, A.", "Shazeer, N."],
                "published_date": "2024-01-10T00:00:00Z",
                "categories": ["cs.LG", "cs.CL"],
                "pdf_url": "https://arxiv.org/pdf/2401.00123.pdf"
            }
        ]
    }
    
    with open("tests/fixtures/sample_arxiv.json", "w") as f:
        json.dump(sample_arxiv, f, indent=2)
    print("✓ Created tests/fixtures/sample_arxiv.json")
    
    # Create __init__.py files
    open("tests/__init__.py", "a").close()
    open("tests/fixtures/__init__.py", "a").close()
    print("✓ Created __init__.py files")
    
    print("\nSetup complete!")
    print("\nFiles created:")
    print("  - pytest.ini")
    print("  - requirements-test.txt")
    print("  - tests/fixtures/sample_news.json")
    print("  - tests/fixtures/sample_arxiv.json")
    print("  - tests/__init__.py")
    print("  - tests/fixtures/__init__.py")
    print("\nTo run tests:")
    print("  With unittest: python test_runner.py")
    print("  With pytest:   python test_runner.py --pytest")
    print("\nTo install pytest (optional):")
    print("  pip install pytest")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Simple Test Runner for Data Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_runner.py                    # Run all tests with unittest
  python test_runner.py --pytest          # Run all tests with pytest
  python test_runner.py --module deduplication  # Run specific module tests
  python test_runner.py --setup           # Create config files
        """
    )
    
    parser.add_argument(
        "--module",
        help="Run tests for specific module",
        choices=['data_cleaning', 'data_enrichment', 'deduplication', 
                 'database_utils', 'data_validator', 'file_management']
    )
    parser.add_argument(
        "--pytest",
        action="store_true",
        help="Use pytest instead of unittest"
    )
    parser.add_argument(
        "--unittest",
        action="store_true",
        help="Explicitly use unittest (default)"
    )
    parser.add_argument(
        "--setup",
        action="store_true",
        help="Create minimal config files"
    )
    
    args = parser.parse_args()
    
    if args.setup:
        setup_minimal()
        return
    
    if args.module:
        exit_code = run_specific_module(args.module)
    elif args.pytest:
        exit_code = run_with_pytest()
    else:
        exit_code = run_all_tests_unittest()
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()