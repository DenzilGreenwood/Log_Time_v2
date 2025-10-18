"""
Test Runner for LTQG Test Suite

Comprehensive test runner with different test categories and reporting.
"""

import unittest
import sys
import os
import time
from typing import List, Dict, Any

# Add LTQG source directory to path
LTQG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ltqg_core_implementation_python_10_17_25')
sys.path.insert(0, LTQG_DIR)

class LTQGTestResult(unittest.TextTestResult):
    """Enhanced test result class with timing and categorization."""
    
    def __init__(self, stream, descriptions, verbosity):
        super().__init__(stream, descriptions, verbosity)
        self.start_time = None
        self.test_times = {}
        self.category_results = {}
    
    def startTest(self, test):
        super().startTest(test)
        self.start_time = time.time()
    
    def stopTest(self, test):
        if self.start_time:
            elapsed = time.time() - self.start_time
            self.test_times[str(test)] = elapsed
        super().stopTest(test)
    
    def addSuccess(self, test):
        super().addSuccess(test)
        self._categorize_result(test, 'success')
    
    def addError(self, test, err):
        super().addError(test, err)
        self._categorize_result(test, 'error')
    
    def addFailure(self, test, err):
        super().addFailure(test, err)
        self._categorize_result(test, 'failure')
    
    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self._categorize_result(test, 'skip')
    
    def _categorize_result(self, test, result_type):
        """Categorize test results by module."""
        module_name = test.__class__.__module__
        if module_name not in self.category_results:
            self.category_results[module_name] = {
                'success': 0, 'error': 0, 'failure': 0, 'skip': 0, 'total': 0
            }
        
        self.category_results[module_name][result_type] += 1
        self.category_results[module_name]['total'] += 1

class LTQGTestRunner(unittest.TextTestRunner):
    """Enhanced test runner for LTQG tests."""
    
    def __init__(self, **kwargs):
        kwargs['resultclass'] = LTQGTestResult
        super().__init__(**kwargs)
    
    def run(self, test):
        result = super().run(test)
        
        # Print enhanced results
        self._print_timing_summary(result)
        self._print_category_summary(result)
        
        return result
    
    def _print_timing_summary(self, result):
        """Print timing summary for tests."""
        if not result.test_times:
            return
        
        print("\n" + "="*70)
        print("TIMING SUMMARY")
        print("="*70)
        
        # Sort tests by execution time
        sorted_times = sorted(result.test_times.items(), key=lambda x: x[1], reverse=True)
        
        print("Slowest tests:")
        for test_name, elapsed in sorted_times[:10]:  # Top 10 slowest
            print(f"  {elapsed:.3f}s - {test_name}")
        
        total_time = sum(result.test_times.values())
        avg_time = total_time / len(result.test_times)
        
        print(f"\nTotal test time: {total_time:.3f}s")
        print(f"Average test time: {avg_time:.3f}s")
        print(f"Number of tests: {len(result.test_times)}")
    
    def _print_category_summary(self, result):
        """Print summary by test category."""
        if not result.category_results:
            return
        
        print("\n" + "="*70)
        print("CATEGORY SUMMARY")
        print("="*70)
        
        for module, stats in result.category_results.items():
            success_rate = (stats['success'] / stats['total']) * 100 if stats['total'] > 0 else 0
            
            print(f"\n{module}:")
            print(f"  Total: {stats['total']}")
            print(f"  Success: {stats['success']} ({success_rate:.1f}%)")
            print(f"  Failures: {stats['failure']}")
            print(f"  Errors: {stats['error']}")
            print(f"  Skipped: {stats['skip']}")

def discover_tests(test_dir: str = None, pattern: str = "test_*.py") -> unittest.TestSuite:
    """Discover all tests in the test directory."""
    if test_dir is None:
        test_dir = os.path.dirname(__file__)
    
    loader = unittest.TestLoader()
    suite = loader.discover(test_dir, pattern=pattern)
    
    return suite

def run_test_category(category: str, verbosity: int = 2) -> unittest.TestResult:
    """Run tests for a specific category."""
    test_files = {
        'core': 'test_ltqg_core.py',
        'quantum': 'test_ltqg_quantum.py', 
        'cosmology': 'test_ltqg_cosmology.py',
        'qft': 'test_ltqg_qft.py',
        'integration': 'test_integration.py'
    }
    
    if category not in test_files:
        raise ValueError(f"Unknown test category: {category}")
    
    test_file = test_files[category]
    test_dir = os.path.dirname(__file__)
    
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName(test_file.replace('.py', ''))
    
    runner = LTQGTestRunner(verbosity=verbosity)
    return runner.run(suite)

def run_all_tests(verbosity: int = 2, failfast: bool = False) -> unittest.TestResult:
    """Run all LTQG tests."""
    print("="*80)
    print("LTQG COMPREHENSIVE TEST SUITE")
    print("="*80)
    print("Testing Log-Time Quantum Gravity framework...")
    print()
    
    # Check dependencies
    missing_deps = check_dependencies()
    if missing_deps:
        print("WARNING: Missing dependencies:", missing_deps)
        print("Some tests may be skipped.")
        print()
    
    # Discover and run tests
    suite = discover_tests()
    runner = LTQGTestRunner(verbosity=verbosity, failfast=failfast)
    
    start_time = time.time()
    result = runner.run(suite)
    total_time = time.time() - start_time
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL TEST SUMMARY")
    print("="*80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print(f"Total time: {total_time:.3f}s")
    
    if result.failures:
        print(f"\nFAILED TESTS:")
        for test, traceback in result.failures:
            print(f"  - {test}")
    
    if result.errors:
        print(f"\nERROR TESTS:")
        for test, traceback in result.errors:
            print(f"  - {test}")
    
    return result

def check_dependencies() -> List[str]:
    """Check for required dependencies."""
    required_modules = ['numpy', 'sympy', 'scipy']
    missing = []
    
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    
    return missing

def run_quick_tests() -> unittest.TestResult:
    """Run a quick subset of tests for fast validation."""
    print("Running quick test suite...")
    
    # Run only core tests for quick validation
    return run_test_category('core', verbosity=1)

def run_performance_tests() -> unittest.TestResult:
    """Run performance-focused tests."""
    print("Running performance test suite...")
    
    # Focus on integration tests which include performance testing
    return run_test_category('integration', verbosity=2)

def main():
    """Main test runner entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='LTQG Test Suite Runner')
    parser.add_argument('--category', choices=['core', 'quantum', 'cosmology', 'qft', 'integration'],
                       help='Run tests for specific category')
    parser.add_argument('--quick', action='store_true', help='Run quick test suite')
    parser.add_argument('--performance', action='store_true', help='Run performance tests')
    parser.add_argument('--verbose', '-v', action='count', default=2, 
                       help='Increase verbosity')
    parser.add_argument('--failfast', action='store_true', 
                       help='Stop on first failure')
    
    args = parser.parse_args()
    
    try:
        if args.quick:
            result = run_quick_tests()
        elif args.performance:
            result = run_performance_tests()
        elif args.category:
            result = run_test_category(args.category, args.verbose)
        else:
            result = run_all_tests(args.verbose, args.failfast)
        
        # Exit with appropriate code
        if result.failures or result.errors:
            sys.exit(1)
        else:
            sys.exit(0)
            
    except Exception as e:
        print(f"Test runner error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()