"""
Simple Test Runner for LTQG Framework

This runner sets up the proper Python path and runs basic validation tests.
"""

import sys
import os

# Add the LTQG source directory to Python path
current_dir = os.path.dirname(__file__)
ltqg_dir = os.path.join(os.path.dirname(current_dir), 'ltqg_core_implementation_python_10_17_25')

# Check if the LTQG directory exists
if not os.path.exists(ltqg_dir):
    print(f"Error: LTQG source directory not found at: {ltqg_dir}")
    print("Please ensure the 'ltqg_core_implementation_python_10_17_25' directory exists.")
    sys.exit(1)

if ltqg_dir not in sys.path:
    sys.path.insert(0, ltqg_dir)

def test_imports():
    """Test that all LTQG modules can be imported."""
    print("Testing LTQG module imports...")
    
    try:
        import ltqg_core
        print("✓ ltqg_core imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import ltqg_core: {e}")
        return False
    
    try:
        import ltqg_quantum
        print("✓ ltqg_quantum imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import ltqg_quantum: {e}")
        return False
    
    try:
        import ltqg_cosmology
        print("✓ ltqg_cosmology imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import ltqg_cosmology: {e}")
        return False
    
    try:
        import ltqg_curvature
        print("✓ ltqg_curvature imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import ltqg_curvature: {e}")
        return False
    
    try:
        import ltqg_qft
        print("✓ ltqg_qft imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import ltqg_qft: {e}")
        return False
    
    try:
        import ltqg_variational
        print("✓ ltqg_variational imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import ltqg_variational: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic functionality of core modules."""
    print("\nTesting basic functionality...")
    
    try:
        from ltqg_core import LogTimeTransform, validate_log_time_core
        
        # Test log-time transformation
        transform = LogTimeTransform(tau0=1.0)
        tau_test = 2.0
        sigma = transform.tau_to_sigma(tau_test)
        tau_back = transform.sigma_to_tau(sigma)
        
        if abs(tau_back - tau_test) < 1e-12:
            print("✓ Log-time transformation round-trip test passed")
        else:
            print("✗ Log-time transformation round-trip test failed")
            return False
        
        print("✓ Core functionality test passed")
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False

def test_quantum_evolution():
    """Test quantum evolution basic functionality."""
    print("\nTesting quantum evolution...")
    
    try:
        from ltqg_quantum import QuantumEvolutionLTQG
        import numpy as np
        
        evolution = QuantumEvolutionLTQG()
        
        # Simple constant Hamiltonian
        H = np.array([[1., 0.], [0., -1.]], dtype=complex)
        
        def H_func(tau):
            return H
        
        # Short evolution
        tau_grid = np.linspace(1.0, 1.1, 11)
        U = evolution.U_time_ordered_tau(H_func, tau_grid)
        
        # Check unitarity
        identity = np.eye(2)
        product = U.conj().T @ U
        error = np.max(np.abs(product - identity))
        
        if error < 1e-10:
            print("✓ Quantum evolution unitarity test passed")
        else:
            print("✗ Quantum evolution unitarity test failed")
            return False
        
        print("✓ Quantum evolution test passed")
        return True
        
    except Exception as e:
        print(f"✗ Quantum evolution test failed: {e}")
        return False

def test_cosmology():
    """Test cosmology basic functionality."""
    print("\nTesting cosmology...")
    
    try:
        from ltqg_cosmology import FLRWCosmology
        
        # Radiation era
        cosmology = FLRWCosmology(p=0.5)
        
        # Test basic functions
        t_test = 2.0
        a = cosmology.scale_factor(t_test)
        H = cosmology.hubble_parameter(t_test)
        R = cosmology.ricci_scalar_original(t_test)
        R_tilde = cosmology.ricci_scalar_transformed()
        
        # Basic sanity checks
        if a > 0 and H > 0 and abs(a - t_test**0.5) < 1e-12:
            print("✓ FLRW cosmology basic functions test passed")
        else:
            print("✗ FLRW cosmology basic functions test failed")
            return False
        
        print("✓ Cosmology test passed")
        return True
        
    except Exception as e:
        print(f"✗ Cosmology test failed: {e}")
        return False

def run_validation_functions():
    """Run the built-in validation functions."""
    print("\nRunning built-in validation functions...")
    
    try:
        from ltqg_core import validate_log_time_core
        print("Running core validation...")
        validate_log_time_core()
        print("✓ Core validation completed")
    except Exception as e:
        print(f"✗ Core validation failed: {e}")
        return False
    
    try:
        from ltqg_quantum import validate_unitary_equivalence_constant_H
        print("Running quantum validation (constant H)...")
        validate_unitary_equivalence_constant_H()
        print("✓ Quantum validation completed")
    except Exception as e:
        print(f"✗ Quantum validation failed: {e}")
        return False
    
    print("✓ All validation functions completed successfully")
    return True

def main():
    """Main test runner."""
    print("="*80)
    print("LTQG FRAMEWORK TEST RUNNER")
    print("="*80)
    
    all_passed = True
    
    # Test imports
    if not test_imports():
        all_passed = False
    
    # Test basic functionality
    if not test_basic_functionality():
        all_passed = False
    
    # Test quantum evolution
    if not test_quantum_evolution():
        all_passed = False
    
    # Test cosmology
    if not test_cosmology():
        all_passed = False
    
    # Run validation functions
    if not run_validation_functions():
        all_passed = False
    
    print("\n" + "="*80)
    if all_passed:
        print("🎉 ALL TESTS PASSED! 🎉")
        print("The LTQG framework is working correctly.")
    else:
        print("❌ SOME TESTS FAILED ❌")
        print("Please check the error messages above.")
    print("="*80)
    
    return 0 if all_passed else 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)