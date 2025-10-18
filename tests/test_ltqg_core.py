"""
Unit Tests for LTQG Core Module

Tests for the fundamental log-time transformation, asymptotic silence,
and core mathematical utilities.
"""

import unittest
import numpy as np
import sympy as sp
from test_utils import LTQGTestCase, parametrized_test, TestDataGenerator, skip_if_no_sympy

# Import test configuration
try:
    import sys
    import os
    
    # Add the LTQG source directory to Python path
    current_dir = os.path.dirname(__file__)
    ltqg_dir = os.path.join(os.path.dirname(current_dir), 'ltqg_core_implementation_python_10_17_25')
    
    if ltqg_dir not in sys.path:
        sys.path.insert(0, ltqg_dir)
    
    # Import LTQG modules directly
    from ltqg_core import (
        LogTimeTransform, 
        validate_log_time_core,
        validate_asymptotic_silence,
        effective_generator_sigma,
        LTQGConstants
    )
except ImportError as e:
    raise unittest.SkipTest(f"Cannot import LTQG modules: {e}")

class TestLogTimeTransform(LTQGTestCase):
    """Test the LogTimeTransform class."""
    
    def setUp(self):
        super().setUp()
        self.tau0_values = [0.1, 1.0, 2.5, 10.0]
        self.test_times = self.generate_test_times()
    
    def test_initialization(self):
        """Test LogTimeTransform initialization."""
        # Valid initialization
        transform = LogTimeTransform(tau0=1.0)
        self.assertEqual(transform.tau0, 1.0)
        
        # Invalid initialization (negative tau0)
        with self.assertRaises(ValueError):
            LogTimeTransform(tau0=-1.0)
        
        # Invalid initialization (zero tau0)
        with self.assertRaises(ValueError):
            LogTimeTransform(tau0=0.0)
    
    @parametrized_test([(f"tau0_{tau0}", tau0) for tau0 in [0.1, 1.0, 2.5, 10.0]])
    def test_round_trip_transformation(self, case_name, tau0):
        """Test round-trip tau -> sigma -> tau transformation."""
        transform = LogTimeTransform(tau0=tau0)
        
        for tau in self.test_times:
            # Forward transformation
            sigma = transform.tau_to_sigma(tau)
            
            # Backward transformation
            tau_recovered = transform.sigma_to_tau(sigma)
            
            # Assert round-trip accuracy
            self.assertClose(tau, tau_recovered, tol=self.symbolic_tolerance,
                           msg=f"Round-trip failed for tau={tau}, tau0={tau0}")
    
    def test_transformation_arrays(self):
        """Test transformation with numpy arrays."""
        transform = LogTimeTransform(tau0=1.0)
        tau_array = np.array([0.1, 1.0, 5.0, 10.0])
        
        # Forward transformation
        sigma_array = transform.tau_to_sigma(tau_array)
        self.assertEqual(sigma_array.shape, tau_array.shape)
        
        # Backward transformation
        tau_recovered = transform.sigma_to_tau(sigma_array)
        
        # Assert arrays are close
        self.assertClose(tau_array, tau_recovered, tol=self.symbolic_tolerance)
    
    def test_chain_rule_factor(self):
        """Test chain rule factor dσ/dτ = 1/τ."""
        transform = LogTimeTransform(tau0=1.0)
        
        for tau in self.test_times:
            # Test with tau
            factor_tau = transform.chain_rule_factor(tau=tau)
            expected = 1.0 / tau
            self.assertClose(factor_tau, expected, msg=f"Chain rule failed for tau={tau}")
            
            # Test with sigma
            sigma = transform.tau_to_sigma(tau)
            factor_sigma = transform.chain_rule_factor(sigma=sigma)
            self.assertClose(factor_sigma, expected, msg=f"Chain rule failed for sigma={sigma}")
    
    def test_chain_rule_validation_errors(self):
        """Test chain rule factor error handling."""
        transform = LogTimeTransform(tau0=1.0)
        
        # Neither tau nor sigma provided
        with self.assertRaises(ValueError):
            transform.chain_rule_factor()
    
    @skip_if_no_sympy
    def test_symbolic_invertibility(self):
        """Test symbolic validation of invertibility."""
        transform = LogTimeTransform(tau0=1.0)
        self.assertTrue(transform.validate_invertibility())
    
    def test_mathematical_properties(self):
        """Test mathematical properties of the transformation."""
        transform = LogTimeTransform(tau0=1.0)
        
        # Test monotonicity: tau1 < tau2 => sigma1 < sigma2
        tau1, tau2 = 1.0, 2.0
        sigma1 = transform.tau_to_sigma(tau1)
        sigma2 = transform.tau_to_sigma(tau2)
        self.assertLess(sigma1, sigma2, "Transformation should be monotonic")
        
        # Test reference point: sigma(tau0) = 0
        sigma_ref = transform.tau_to_sigma(transform.tau0)
        self.assertClose(sigma_ref, 0.0, tol=self.symbolic_tolerance)
        
        # Test asymptotic behavior: sigma -> -∞ as tau -> 0+
        small_tau = 1e-10
        sigma_small = transform.tau_to_sigma(small_tau)
        self.assertLess(sigma_small, -20, "sigma should be very negative for small tau")

class TestAsymptoticSilence(LTQGTestCase):
    """Test asymptotic silence properties."""
    
    def test_effective_generator_sigma(self):
        """Test effective generator computation in sigma-time."""
        tau0 = 1.0
        
        # Test with constant Hamiltonian
        def H_constant(tau):
            return 2.0
        
        sigma_test = -1.0
        H_eff = effective_generator_sigma(sigma_test, H_constant, tau0)
        
        # Expected: H_eff = tau0 * exp(sigma) * H(tau0 * exp(sigma))
        tau_expected = tau0 * np.exp(sigma_test)
        expected = tau_expected * H_constant(tau_expected)
        
        self.assertClose(H_eff, expected)
    
    def test_asymptotic_silence_behavior(self):
        """Test asymptotic silence for different Hamiltonian types."""
        tau0 = 1.0
        sigma_values = np.linspace(-10, -1, 10)  # Going from distant past to recent past
        
        # Bounded Hamiltonian (should satisfy silence)
        def H_bounded(tau):
            return 1.0  # Constant bounded function
        
        H_eff_values = [effective_generator_sigma(s, H_bounded, tau0) for s in sigma_values]
        
        # Check that H_eff increases as sigma increases (becomes less negative)
        # For H_eff(σ) = τ₀e^σ H(τ₀e^σ), we expect exponential growth as σ increases
        first_value = H_eff_values[0]   # σ = -10 (distant past)
        last_value = H_eff_values[-1]   # σ = -1 (recent past)
        self.assertLess(first_value, last_value, "Effective generator should increase as sigma increases")
        
        # Check approach to zero in the distant past (first value should be small)
        distant_past_value = H_eff_values[0]  # σ = -10
        self.assertLess(distant_past_value, 1e-4, "Effective generator should be small for sigma -> -∞")

class TestLTQGConstants(LTQGTestCase):
    """Test LTQG constants and parameters."""
    
    def test_constant_values(self):
        """Test that constants have reasonable values."""
        # Check default values
        self.assertEqual(LTQGConstants.TAU0_DEFAULT, 1.0)
        self.assertEqual(LTQGConstants.HBAR_DEFAULT, 1.0)
        
        # Check tolerances are positive
        self.assertGreater(LTQGConstants.NUMERICAL_TOL, 0)
        self.assertGreater(LTQGConstants.PHASE_TOL, 0)
        self.assertGreater(LTQGConstants.INTEGRATION_TOL, 0)
        
        # Check physical regime values
        self.assertClose(LTQGConstants.RADIATION_P, 0.5)
        self.assertClose(LTQGConstants.MATTER_P, 2.0/3.0)
        self.assertClose(LTQGConstants.STIFF_P, 1.0/3.0)
    
    def test_era_classifications(self):
        """Test that era parameters correspond to correct physics."""
        # Radiation era: w = 1/3 => p = 1/2
        w_radiation = 1.0/3.0
        p_expected = 2.0 / (3.0 * (1.0 + w_radiation))
        self.assertClose(LTQGConstants.RADIATION_P, p_expected, tol=1e-10)
        
        # Matter era: w = 0 => p = 2/3
        w_matter = 0.0
        p_expected = 2.0 / (3.0 * (1.0 + w_matter))
        self.assertClose(LTQGConstants.MATTER_P, p_expected, tol=1e-10)
        
        # Stiff matter: w = 1 => p = 1/3
        w_stiff = 1.0
        p_expected = 2.0 / (3.0 * (1.0 + w_stiff))
        self.assertClose(LTQGConstants.STIFF_P, p_expected, tol=1e-10)

class TestCoreValidationFunctions(LTQGTestCase):
    """Test the validation functions."""
    
    def test_validate_log_time_core(self):
        """Test that core validation runs without errors."""
        # This should not raise any exceptions
        try:
            validate_log_time_core()
        except Exception as e:
            self.fail(f"Core validation failed with exception: {e}")
    
    def test_validate_asymptotic_silence(self):
        """Test that asymptotic silence validation runs without errors."""
        # This should not raise any exceptions
        try:
            validate_asymptotic_silence()
        except Exception as e:
            self.fail(f"Asymptotic silence validation failed with exception: {e}")

class TestEdgeCases(LTQGTestCase):
    """Test edge cases and boundary conditions."""
    
    def test_very_small_tau(self):
        """Test behavior with very small tau values."""
        transform = LogTimeTransform(tau0=1.0)
        
        # Test with tau close to zero
        small_tau = 1e-15
        sigma = transform.tau_to_sigma(small_tau)
        tau_recovered = transform.sigma_to_tau(sigma)
        
        # Should maintain precision even for very small values
        relative_error = abs(tau_recovered - small_tau) / small_tau
        self.assertLess(relative_error, 1e-10, "Precision lost for very small tau")
    
    def test_very_large_tau(self):
        """Test behavior with very large tau values."""
        transform = LogTimeTransform(tau0=1.0)
        
        # Test with large tau
        large_tau = 1e15
        sigma = transform.tau_to_sigma(large_tau)
        tau_recovered = transform.sigma_to_tau(sigma)
        
        # Should maintain precision for large values
        relative_error = abs(tau_recovered - large_tau) / large_tau
        self.assertLess(relative_error, 1e-10, "Precision lost for very large tau")
    
    def test_different_tau0_scales(self):
        """Test consistency across different tau0 scales."""
        tau_test = 5.0
        tau0_values = [1e-5, 1e-2, 1.0, 1e2, 1e5]
        
        for tau0 in tau0_values:
            transform = LogTimeTransform(tau0=tau0)
            
            # Test round-trip
            sigma = transform.tau_to_sigma(tau_test)
            tau_recovered = transform.sigma_to_tau(sigma)
            
            relative_error = abs(tau_recovered - tau_test) / tau_test
            self.assertLess(relative_error, 1e-12, 
                          f"Round-trip failed for tau0={tau0}")
            
            # Test chain rule
            factor = transform.chain_rule_factor(tau=tau_test)
            expected = 1.0 / tau_test
            self.assertClose(factor, expected, tol=1e-12,
                           msg=f"Chain rule failed for tau0={tau0}")

if __name__ == '__main__':
    unittest.main()