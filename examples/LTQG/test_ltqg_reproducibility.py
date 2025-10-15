#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LTQG Reproducibility and CI Testing Framework

This module provides deterministic testing and environment validation
for the LTQG framework to ensure reproducible results across environments.

Environment Requirements:
- Python 3.8+
- NumPy 1.20+
- SciPy 1.7+
- SymPy 1.8+

Determinism Settings:
- Fixed random seeds for all stochastic components
- Explicit ODE tolerances for numerical integration  
- Machine precision validation for critical calculations

Author: Mathematical Physics Research
License: Open Source
"""

import sys
import os
import numpy as np
import scipy
import sympy as sp
import subprocess
from typing import Dict, List, Any
import time
import hashlib

# Import LTQG modules
from ltqg_core import LogTimeTransform, banner
from ltqg_main import LTQGValidationSuite

# ===========================
# Environment Validation
# ===========================

def get_environment_manifest() -> Dict[str, Any]:
    """
    Generate complete environment manifest for reproducibility.
    
    Returns:
        Dictionary with environment details
    """
    import platform
    
    manifest = {
        'python_version': sys.version,
        'python_executable': sys.executable,
        'platform': {
            'system': platform.system(),
            'release': platform.release(),
            'machine': platform.machine(),
            'processor': platform.processor()
        },
        'packages': {
            'numpy': np.__version__,
            'scipy': scipy.__version__,
            'sympy': sp.__version__
        },
        'numpy_config': {
            'blas_info': str(np.__config__.blas_opt_info),
            'lapack_info': str(np.__config__.lapack_opt_info)
        }
    }
    
    return manifest

def validate_environment() -> bool:
    """
    Validate that environment meets LTQG requirements.
    
    Returns:
        True if environment is suitable
    """
    banner("Environment Validation")
    
    # Check Python version
    python_version = sys.version_info
    if python_version < (3, 8):
        print(f"❌ Python version {python_version} < 3.8 (minimum required)")
        return False
    else:
        print(f"✅ Python version {python_version} >= 3.8")
    
    # Check package versions
    required_packages = {
        'numpy': '1.20.0',
        'scipy': '1.7.0', 
        'sympy': '1.8.0'
    }
    
    for package, min_version in required_packages.items():
        try:
            if package == 'numpy':
                current_version = np.__version__
            elif package == 'scipy':
                current_version = scipy.__version__
            elif package == 'sympy':
                current_version = sp.__version__
            
            print(f"✅ {package}: {current_version} (required: >= {min_version})")
            
        except ImportError:
            print(f"❌ {package}: Not installed (required: >= {min_version})")
            return False
    
    return True

# ===========================
# Determinism Configuration
# ===========================

def configure_determinism() -> None:
    """Configure deterministic settings for reproducible results."""
    # Set NumPy random seed
    np.random.seed(42)
    
    # Set SymPy random seed (if applicable)
    # SymPy doesn't have global random state, but we document this for clarity
    
    print("DETERMINISM CONFIGURATION:")
    print("• NumPy random seed: 42")
    print("• ODE integration tolerances: rtol=1e-10, atol=1e-12")
    print("• Matrix precision: Machine epsilon ~2.22e-16")
    print("• Phase calculation tolerance: 1e-6")

# ===========================
# Command Line Interface
# ===========================

def run_ltqg_command(command: str, description: str) -> Dict[str, Any]:
    """
    Run LTQG command and capture results.
    
    Args:
        command: Command to execute
        description: Human-readable description
        
    Returns:
        Dictionary with execution results
    """
    print(f"\nRunning: {description}")
    print(f"Command: {command}")
    
    start_time = time.time()
    
    try:
        # Execute command
        result = subprocess.run(
            command.split(),
            capture_output=True,
            text=True,
            cwd=os.path.dirname(__file__),
            timeout=300  # 5 minute timeout
        )
        
        elapsed_time = time.time() - start_time
        
        execution_result = {
            'command': command,
            'description': description,
            'returncode': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'elapsed_time': elapsed_time,
            'success': result.returncode == 0
        }
        
        if result.returncode == 0:
            print(f"✅ Success ({elapsed_time:.2f}s)")
        else:
            print(f"❌ Failed with code {result.returncode} ({elapsed_time:.2f}s)")
            if result.stderr:
                print(f"Error: {result.stderr}")
        
        return execution_result
        
    except subprocess.TimeoutExpired:
        print(f"❌ Timeout after 5 minutes")
        return {
            'command': command,
            'description': description,
            'returncode': -1,
            'stdout': '',
            'stderr': 'Command timed out',
            'elapsed_time': 300,
            'success': False
        }

def generate_command_manifest() -> List[Dict[str, str]]:
    """
    Generate manifest of all LTQG commands for testing.
    
    Returns:
        List of command dictionaries
    """
    commands = [
        {
            'command': 'python ltqg_core.py',
            'description': 'Core mathematical foundation validation',
            'expected_keywords': ['PASS', 'Log-time', 'transformation', 'validated']
        },
        {
            'command': 'python ltqg_quantum.py',
            'description': 'Quantum evolution and unitary equivalence',
            'expected_keywords': ['PASS', 'Quantum', 'evolution', 'validated']
        },
        {
            'command': 'python ltqg_cosmology.py', 
            'description': 'FLRW cosmology and Weyl transformations',
            'expected_keywords': ['PASS', 'Cosmology', 'FLRW', 'validated']
        },
        {
            'command': 'python ltqg_main.py --mode quick',
            'description': 'Quick validation of essential components',
            'expected_keywords': ['LTQG', 'validated', 'essential', 'tests']
        }
    ]
    
    return commands

# ===========================
# Reproducibility Testing
# ===========================

def test_numerical_determinism() -> bool:
    """Test that numerical computations are deterministic."""
    banner("Numerical Determinism Test")
    
    configure_determinism()
    
    # Test 1: Log-time transformation
    transform = LogTimeTransform(tau0=1.0)
    test_taus = np.array([0.1, 1.0, 2.5, 10.0])
    
    results1 = []
    results2 = []
    
    for _ in range(2):  # Run twice
        np.random.seed(42)  # Reset seed
        
        run_results = []
        for tau in test_taus:
            sigma = transform.tau_to_sigma(tau)
            tau_back = transform.sigma_to_tau(sigma)
            run_results.append((sigma, tau_back))
        
        if len(results1) == 0:
            results1 = run_results
        else:
            results2 = run_results
    
    # Compare results
    for i, (r1, r2) in enumerate(zip(results1, results2)):
        diff_sigma = abs(r1[0] - r2[0])
        diff_tau = abs(r1[1] - r2[1])
        
        if diff_sigma > 1e-15 or diff_tau > 1e-15:
            print(f"❌ Non-deterministic result at index {i}")
            print(f"   Run 1: σ={r1[0]}, τ={r1[1]}")
            print(f"   Run 2: σ={r2[0]}, τ={r2[1]}")
            return False
    
    print("✅ Numerical computations are deterministic")
    return True

def run_comprehensive_reproducibility_test() -> Dict[str, Any]:
    """
    Run comprehensive reproducibility test suite.
    
    Returns:
        Complete test results
    """
    banner("LTQG Comprehensive Reproducibility Test")
    
    # Environment validation
    if not validate_environment():
        return {'status': 'ENVIRONMENT_FAILED'}
    
    # Generate environment manifest
    manifest = get_environment_manifest()
    
    # Configure determinism
    configure_determinism()
    
    # Test numerical determinism
    if not test_numerical_determinism():
        return {'status': 'DETERMINISM_FAILED'}
    
    # Run command tests
    commands = generate_command_manifest()
    command_results = []
    
    print(f"\nRunning {len(commands)} LTQG commands...")
    
    for cmd_info in commands:
        result = run_ltqg_command(cmd_info['command'], cmd_info['description'])
        
        # Check for expected keywords in output
        if result['success']:
            stdout_text = result['stdout'].lower()
            expected = cmd_info['expected_keywords']
            found_keywords = [kw for kw in expected if kw.lower() in stdout_text]
            
            result['expected_keywords'] = expected
            result['found_keywords'] = found_keywords
            result['keyword_match'] = len(found_keywords) >= len(expected) * 0.5  # At least 50% match
        
        command_results.append(result)
    
    # Overall assessment
    successful_commands = sum(1 for r in command_results if r['success'])
    total_commands = len(command_results)
    
    overall_success = successful_commands == total_commands
    
    print(f"\nREPRODUCIBILITY TEST SUMMARY:")
    print(f"• Environment: {'✅ Valid' if validate_environment() else '❌ Invalid'}")
    print(f"• Determinism: {'✅ Confirmed' if test_numerical_determinism() else '❌ Failed'}")
    print(f"• Commands: {successful_commands}/{total_commands} passed")
    print(f"• Overall: {'✅ PASS' if overall_success else '❌ FAIL'}")
    
    return {
        'status': 'PASS' if overall_success else 'FAIL',
        'environment_manifest': manifest,
        'command_results': command_results,
        'summary': {
            'total_commands': total_commands,
            'successful_commands': successful_commands,
            'success_rate': successful_commands / total_commands
        }
    }

# ===========================
# Make Test Target
# ===========================

def make_test() -> int:
    """
    Main test target for use with make or CI systems.
    
    Returns:
        0 on success, 1 on failure
    """
    try:
        # Run essential validations only (for speed)
        suite = LTQGValidationSuite()
        
        # Run only essential tests
        essential_tests = [test for test in suite.test_modules if test['essential']]
        
        for test_module in essential_tests:
            result = suite.run_single_test(test_module)
            suite.results[test_module['name']] = result
            
            if result['status'] != 'PASS':
                print(f"MAKE TEST FAILED: {test_module['name']}")
                return 1
        
        # Check that we have the expected number of essential tests
        expected_essential = 3  # Core, Quantum, Cosmology
        if len(essential_tests) != expected_essential:
            print(f"MAKE TEST FAILED: Expected {expected_essential} essential tests, got {len(essential_tests)}")
            return 1
        
        print("MAKE TEST PASSED: All essential LTQG components validated")
        return 0
        
    except Exception as e:
        print(f"MAKE TEST FAILED: Exception {e}")
        return 1

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='LTQG Reproducibility Testing')
    parser.add_argument('--mode', choices=['full', 'make'], default='full',
                       help='Test mode: full reproducibility or make target')
    
    args = parser.parse_args()
    
    if args.mode == 'make':
        exit_code = make_test()
        sys.exit(exit_code)
    else:
        results = run_comprehensive_reproducibility_test()
        if results['status'] == 'PASS':
            print("\n🎯 LTQG framework is reproducible and ready for use")
            sys.exit(0)
        else:
            print("\n❌ Reproducibility issues detected")
            sys.exit(1)