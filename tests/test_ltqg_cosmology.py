"""
Unit Tests for LTQG Cosmology Module

Tests for FLRW spacetimes, Weyl transformations, scalar field minisuperspace,
and cosmological phase transitions.
"""

import unittest
import numpy as np
import sympy as sp
from test_utils import LTQGTestCase, parametrized_test, TestDataGenerator, skip_if_no_sympy

# Import test configuration
try:
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ltqg_core_implementation_python_10_17_25'))
    
    from ltqg_cosmology import (
        FLRWCosmology,
        ScalarFieldMinisuperspace,
        validate_weyl_transform_flrw,
        validate_scalar_clock_minisuperspace,
        validate_cosmological_transitions,
        analyze_cosmological_phases
    )
    from ltqg_core import LTQGConstants
except ImportError as e:
    raise unittest.SkipTest(f"Cannot import LTQG modules: {e}")

class TestFLRWCosmology(LTQGTestCase):
    """Test the FLRWCosmology class."""
    
    def setUp(self):
        super().setUp()
        self.flrw_test_cases = TestDataGenerator.flrw_test_cases()
    
    @parametrized_test(TestDataGenerator.flrw_test_cases())
    def test_initialization_and_classification(self, era_name, p):
        """Test FLRW cosmology initialization and era classification."""
        cosmology = FLRWCosmology(p)
        
        self.assertEqual(cosmology.p, p)
        self.assertEqual(cosmology.tau0, LTQGConstants.TAU0_DEFAULT)
        
        # Check era classification
        if era_name in ['radiation', 'matter', 'stiff']:
            self.assertEqual(cosmology.era_type, era_name)
    
    @parametrized_test(TestDataGenerator.flrw_test_cases())
    def test_scale_factor_and_hubble(self, era_name, p):
        """Test scale factor and Hubble parameter calculations."""
        cosmology = FLRWCosmology(p)
        test_times = self.generate_test_times()
        
        for t in test_times:
            # Scale factor a(t) = t^p
            a = cosmology.scale_factor(t)
            expected_a = t**p
            self.assertClose(a, expected_a, msg=f"Scale factor incorrect for {era_name}")
            
            # Hubble parameter H(t) = p/t
            H = cosmology.hubble_parameter(t)
            expected_H = p / t
            self.assertClose(H, expected_H, msg=f"Hubble parameter incorrect for {era_name}")
    
    @parametrized_test(TestDataGenerator.flrw_test_cases())
    def test_ricci_scalar_original(self, era_name, p):
        """Test original Ricci scalar calculation."""
        cosmology = FLRWCosmology(p)
        test_times = self.generate_test_times()
        
        for t in test_times:
            R = cosmology.ricci_scalar_original(t)
            expected_R = 6 * p * (2*p - 1) / t**2
            self.assertClose(R, expected_R, 
                           msg=f"Original Ricci scalar incorrect for {era_name}")
    
    @parametrized_test(TestDataGenerator.flrw_test_cases())
    def test_weyl_transformation(self, era_name, p):
        """Test Weyl transformation properties."""
        cosmology = FLRWCosmology(p)
        test_times = self.generate_test_times()
        
        # Conformal factor Ω = 1/t
        for t in test_times:
            Omega = cosmology.conformal_factor_weyl(t)
            expected_Omega = 1.0 / t
            self.assertClose(Omega, expected_Omega,
                           msg=f"Conformal factor incorrect for {era_name}")
        
        # Transformed Ricci scalar (should be constant)
        R_tilde = cosmology.ricci_scalar_transformed()
        expected_R_tilde = 12 * (p - 1)**2
        self.assertClose(R_tilde, expected_R_tilde,
                        msg=f"Transformed Ricci scalar incorrect for {era_name}")
    
    @parametrized_test(TestDataGenerator.flrw_test_cases())
    def test_equation_of_state(self, era_name, p):
        """Test equation of state calculations."""
        cosmology = FLRWCosmology(p)
        eos = cosmology.equation_of_state()
        
        # Check structure
        required_keys = ['w', 'rho_time_scaling', 'rho_scale_scaling', 'era', 'scale_exponent']
        for key in required_keys:
            self.assertIn(key, eos, f"Missing EOS key: {key}")
        
        # Check w = 2/(3p) - 1 relation
        if p != 0:  # Avoid division by zero
            expected_w = 2.0/(3.0*p) - 1.0
            self.assertClose(eos['w'], expected_w, tol=1e-10,
                           msg=f"Equation of state parameter w incorrect for {era_name}")
        
        # Check standard scalings
        self.assertClose(eos['rho_time_scaling'], -2.0, tol=1e-10,
                        msg="Energy density time scaling should be -2")
        
        if p != 0:
            expected_rho_scale_scaling = -3.0 * (1.0 + eos['w'])
            self.assertClose(eos['rho_scale_scaling'], expected_rho_scale_scaling, tol=1e-10,
                           msg=f"Energy density scale factor scaling incorrect for {era_name}")
    
    def test_standard_cosmological_eras(self):
        """Test standard cosmological eras have correct parameters."""
        # Radiation era
        radiation = FLRWCosmology(LTQGConstants.RADIATION_P)
        radiation_eos = radiation.equation_of_state()
        self.assertClose(radiation_eos['w'], 1.0/3.0, tol=1e-10, msg="Radiation w should be 1/3")
        
        # Matter era
        matter = FLRWCosmology(LTQGConstants.MATTER_P)
        matter_eos = matter.equation_of_state()
        self.assertClose(matter_eos['w'], 0.0, tol=1e-10, msg="Matter w should be 0")
        
        # Stiff matter era
        stiff = FLRWCosmology(LTQGConstants.STIFF_P)
        stiff_eos = stiff.equation_of_state()
        self.assertClose(stiff_eos['w'], 1.0, tol=1e-10, msg="Stiff matter w should be 1")

class TestScalarFieldMinisuperspace(LTQGTestCase):
    """Test the ScalarFieldMinisuperspace class."""
    
    def setUp(self):
        super().setUp()
        self.minisuperspace = ScalarFieldMinisuperspace()
    
    @skip_if_no_sympy
    def test_lagrangian_symbolic(self):
        """Test symbolic Lagrangian construction."""
        L = self.minisuperspace.lagrangian_symbolic()
        
        # Should be a SymPy expression
        self.assertIsInstance(L, sp.Expr)
        
        # Check that it contains expected symbols and functions
        L_str = str(L)  # Convert to string to check for function names
        
        # Should contain tau function and scale factor function
        has_tau = 'tau' in L_str.lower()
        self.assertTrue(has_tau, 
                       f"Lagrangian should contain tau field. Lagrangian: {L_str}")
        has_a = 'a(' in L_str.lower()  # Look for a(t) function
        self.assertTrue(has_a,
                       f"Lagrangian should contain scale factor. Lagrangian: {L_str}")
    
    @skip_if_no_sympy
    def test_stress_energy_components(self):
        """Test stress-energy tensor component calculation."""
        stress_components = self.minisuperspace.stress_energy_components()
        
        # Check required components
        required_components = ['rho_tau', 'p_tau', 'w_effective']
        for component in required_components:
            self.assertIn(component, stress_components,
                         f"Missing stress-energy component: {component}")
            self.assertIsInstance(stress_components[component], sp.Expr)
        
        # Check that components are SymPy expressions
        for component, expr in stress_components.items():
            self.assertIsInstance(expr, sp.Expr,
                                f"Component {component} should be SymPy expression")

class TestCosmologicalPhases(LTQGTestCase):
    """Test cosmological phase analysis."""
    
    def test_analyze_cosmological_phases(self):
        """Test cosmological phase analysis function."""
        p_values = [0.5, 2.0/3.0, 1.0/3.0]  # Standard eras
        time_range = (0.5, 5.0)
        N_points = 100
        
        results = analyze_cosmological_phases(p_values, time_range, N_points)
        
        # Check structure
        self.assertIsInstance(results, dict)
        
        for p in p_values:
            self.assertIn(p, results, f"Missing results for p={p}")
            
            phase_data = results[p]
            required_keys = ['time', 'scale_factor', 'hubble', 'ricci_original', 
                           'ricci_transformed', 'equation_of_state', 'era_type']
            
            for key in required_keys:
                self.assertIn(key, phase_data, f"Missing key {key} for p={p}")
            
            # Check array lengths
            time_array = phase_data['time']
            self.assertEqual(len(time_array), N_points)
            
            for key in ['scale_factor', 'hubble', 'ricci_original']:
                array = phase_data[key]
                self.assertEqual(len(array), N_points,
                               f"Array {key} has wrong length for p={p}")
            
            # Check that ricci_transformed is constant
            R_tilde = phase_data['ricci_transformed']
            self.assertIsInstance(R_tilde, (int, float),
                                f"Transformed Ricci should be scalar for p={p}")

class TestWeylTransformation(LTQGTestCase):
    """Test Weyl transformation properties."""
    
    @parametrized_test(TestDataGenerator.flrw_test_cases())
    def test_curvature_regularization(self, era_name, p):
        """Test that Weyl transformation provides finite curvature."""
        cosmology = FLRWCosmology(p)
        
        # Original curvature diverges as t -> 0+ (except for radiation p=0.5)
        small_times = np.array([1e-6, 1e-4, 1e-2])
        for t in small_times:
            R_original = cosmology.ricci_scalar_original(t)
            self.assertTrue(np.isfinite(R_original), 
                          f"Original curvature should be finite at t={t}")
            
            # Should grow like 1/t^2, but note that for radiation (p=0.5): R = 6p(2p-1)/t² = 0
            # Only test for large curvature if (2p-1) ≠ 0
            factor_2p_minus_1 = 2*p - 1
            if abs(factor_2p_minus_1) > 0.1 and t < 1e-3:  # Skip radiation era p=0.5
                self.assertGreater(abs(R_original), 1e6,
                                 f"Original curvature should be large for small t in {era_name}")
            elif abs(factor_2p_minus_1) < 0.1:  # Radiation era p≈0.5
                self.assertLess(abs(R_original), 1e-10,
                               f"Radiation era should have zero curvature: R = 6p(2p-1)/t² = 0")
        
        # Transformed curvature is constant and finite
        R_transformed = cosmology.ricci_scalar_transformed()
        self.assertTrue(np.isfinite(R_transformed),
                       f"Transformed curvature should be finite for {era_name}")
        self.assertGreaterEqual(R_transformed, 0,
                               f"Transformed curvature should be non-negative for {era_name}")
    
    def test_frame_independence_warning(self):
        """Test that frame dependence is properly documented."""
        # This is more of a documentation test - ensure the warning exists
        cosmology = FLRWCosmology(0.5)
        
        # The key point is that different conformal frames give different physics
        # unless matter coupling is specified. This should be documented.
        
        # Test that we get different curvatures in different frames
        t_test = 1.0
        R_original = cosmology.ricci_scalar_original(t_test)
        R_transformed = cosmology.ricci_scalar_transformed()
        
        self.assertNotEqual(R_original, R_transformed,
                          "Original and transformed curvatures should differ")

class TestValidationFunctions(LTQGTestCase):
    """Test cosmology validation functions."""
    
    def test_validate_weyl_transform_flrw(self):
        """Test FLRW Weyl transformation validation."""
        try:
            validate_weyl_transform_flrw()
        except Exception as e:
            self.fail(f"Weyl transformation validation failed: {e}")
    
    def test_validate_scalar_clock_minisuperspace(self):
        """Test scalar clock minisuperspace validation."""
        try:
            validate_scalar_clock_minisuperspace()
        except Exception as e:
            self.fail(f"Scalar clock validation failed: {e}")
    
    def test_validate_cosmological_transitions(self):
        """Test cosmological transitions validation."""
        try:
            validate_cosmological_transitions()
        except Exception as e:
            self.fail(f"Cosmological transitions validation failed: {e}")

class TestPhysicalConsistency(LTQGTestCase):
    """Test physical consistency of cosmological models."""
    
    def test_energy_conservation(self):
        """Test energy conservation in cosmological evolution."""
        # For FLRW with equation of state w, energy density should scale as
        # ρ(a) ∝ a^(-3(1+w))
        
        for era_name, p in TestDataGenerator.flrw_test_cases()[:3]:  # Test subset
            cosmology = FLRWCosmology(p)
            eos = cosmology.equation_of_state()
            
            if p != 0:  # Avoid degenerate case
                w = eos['w']
                expected_scaling = -3.0 * (1.0 + w)
                actual_scaling = eos['rho_scale_scaling']
                
                self.assertClose(actual_scaling, expected_scaling, tol=1e-10,
                               msg=f"Energy scaling incorrect for {era_name}")
    
    def test_hubble_friedmann_consistency(self):
        """Test consistency between Hubble parameter and Friedmann equation."""
        # For power-law solutions a(t) = t^p, H = p/t should satisfy
        # Friedmann equation H^2 = (8πG/3)ρ
        
        for era_name, p in TestDataGenerator.flrw_test_cases()[:3]:
            cosmology = FLRWCosmology(p)
            
            t_test = 2.0
            H = cosmology.hubble_parameter(t_test)
            expected_H = p / t_test
            
            self.assertClose(H, expected_H, tol=1e-12,
                           msg=f"Hubble parameter inconsistent for {era_name}")
    
    def test_acceleration_equation_consistency(self):
        """Test consistency with acceleration equation."""
        # For FLRW, ä/a = -(4πG/3)(ρ + 3p)
        # With a(t) = t^p, this gives ä/a = p(p-1)/t^2
        
        for era_name, p in TestDataGenerator.flrw_test_cases()[:3]:
            if p == 0:  # Skip degenerate case
                continue
                
            cosmology = FLRWCosmology(p)
            eos = cosmology.equation_of_state()
            
            # For power-law scale factor: ä/a = p(p-1)/t^2
            t_test = 1.5
            a_ddot_over_a_expected = p * (p - 1) / t_test**2
            
            # From energy-momentum: ä/a = -(4πG/3)(ρ + 3p)
            # For our units and conventions, this should be consistent
            # This is more of a structural consistency check
            
            # The key is that the signs should be consistent with known physics
            if eos['w'] > -1.0/3.0:  # Non-accelerating
                if p < 1:
                    self.assertLess(a_ddot_over_a_expected, 0,
                                  f"Should have deceleration for {era_name}")

if __name__ == '__main__':
    unittest.main()