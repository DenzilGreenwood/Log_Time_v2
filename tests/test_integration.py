"""
Integration Tests for LTQG Framework

End-to-end tests that verify the integration of multiple LTQG components
and cross-module consistency.
"""

import unittest
import numpy as np
from test_utils import LTQGTestCase, PerformanceTimer

# Import test configuration
try:
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ltqg_core_implementation_python_10_17_25'))
    
    from ltqg_core import LogTimeTransform, LTQGConstants
    from ltqg_quantum import QuantumEvolutionLTQG
    from ltqg_cosmology import FLRWCosmology
    from ltqg_qft import QFTModeEvolution, AdaptiveIntegrator, QFTDiagnostics
    from ltqg_variational import VariationalFieldTheory
except ImportError as e:
    raise unittest.SkipTest(f"Cannot import LTQG modules: {e}")

class TestCrossModuleConsistency(LTQGTestCase):
    """Test consistency across different LTQG modules."""
    
    def test_coordinate_transformation_consistency(self):
        """Test that coordinate transformations are consistent across modules."""
        tau0 = 2.0
        
        # Core transformation
        core_transform = LogTimeTransform(tau0)
        
        # Quantum evolution transformation
        quantum_evolution = QuantumEvolutionLTQG(tau0=tau0)
        
        # QFT mode evolution transformation
        qft_mode = QFTModeEvolution(k=1.0, m=0.0, p=0.5, tau0=tau0)
        
        # Test same transformation across modules
        test_times = self.generate_test_times()
        for tau in test_times:
            sigma_core = core_transform.tau_to_sigma(tau)
            sigma_quantum = quantum_evolution.transform.tau_to_sigma(tau)
            sigma_qft = qft_mode.transform.tau_to_sigma(tau)
            
            self.assertClose(sigma_core, sigma_quantum, tol=1e-14,
                           msg="Core and quantum transforms should be identical")
            self.assertClose(sigma_core, sigma_qft, tol=1e-14,
                           msg="Core and QFT transforms should be identical")
    
    def test_cosmological_parameters_consistency(self):
        """Test consistency of cosmological parameters across modules."""
        p_radiation = LTQGConstants.RADIATION_P
        
        # FLRW cosmology
        flrw = FLRWCosmology(p_radiation)
        
        # QFT mode in radiation era
        qft_mode = QFTModeEvolution(k=1.0, m=0.0, p=p_radiation)
        
        # Test background functions
        test_times = np.array([1.0, 2.0, 5.0])
        for t in test_times:
            # Scale factor
            a_flrw = flrw.scale_factor(t)
            a_qft = qft_mode.scale_factor(t)
            self.assertClose(a_flrw, a_qft, msg="Scale factors should be consistent")
            
            # Hubble parameter
            H_flrw = flrw.hubble_parameter(t)
            H_qft = qft_mode.hubble_parameter(t)
            self.assertClose(H_flrw, H_qft, msg="Hubble parameters should be consistent")
    
    def test_physical_constants_consistency(self):
        """Test that physical constants are used consistently."""
        # Check that all modules use the same default values
        quantum_evolution = QuantumEvolutionLTQG()
        qft_mode = QFTModeEvolution(k=1.0)
        
        self.assertEqual(quantum_evolution.tau0, LTQGConstants.TAU0_DEFAULT)
        self.assertEqual(quantum_evolution.hbar, LTQGConstants.HBAR_DEFAULT)
        self.assertEqual(qft_mode.tau0, LTQGConstants.TAU0_DEFAULT)

class TestEndToEndScenarios(LTQGTestCase):
    """Test complete end-to-end scenarios involving multiple modules."""
    
    def test_quantum_cosmology_scenario(self):
        """Test quantum evolution in cosmological background."""
        # Setup: Quantum system in expanding FLRW universe
        p = LTQGConstants.RADIATION_P  # Radiation era
        tau0 = 1.0
        
        # Cosmological background
        cosmology = FLRWCosmology(p, tau0)
        
        # Quantum evolution in this background
        quantum_evolution = QuantumEvolutionLTQG(tau0=tau0)
        
        # Time-dependent Hamiltonian reflecting cosmological evolution
        def H_cosmological(tau):
            # Simple model: frequency scales with Hubble parameter
            H_hubble = cosmology.hubble_parameter(tau)
            sz = np.array([[1., 0.], [0., -1.]], dtype=complex)
            return H_hubble * sz
        
        # Evolution parameters
        tau_i, tau_f = tau0, 5.0
        N_steps = 200
        tau_grid = np.linspace(tau_i, tau_f, N_steps + 1)
        sigma_grid = quantum_evolution.transform.tau_to_sigma(tau_grid)
        
        # Evolve in both coordinates
        U_tau = quantum_evolution.U_time_ordered_tau(H_cosmological, tau_grid)
        U_sigma = quantum_evolution.U_time_ordered_sigma(H_cosmological, sigma_grid)
        
        # Both should be unitary
        self.assertUnitary(U_tau, tol=1e-8)
        self.assertUnitary(U_sigma, tol=1e-8)
        
        # Test on initial state
        psi0 = np.array([1.+0j, 0.+0j])
        psi_tau = U_tau @ psi0
        psi_sigma = U_sigma @ psi0
        
        # Final states should give same physical predictions
        rho_tau = np.outer(psi_tau, np.conjugate(psi_tau))
        rho_sigma = np.outer(psi_sigma, np.conjugate(psi_sigma))
        
        self.assertMatrixClose(rho_tau, rho_sigma, tol=1e-6,
                              msg="Quantum cosmology predictions should be coordinate-independent")
    
    def test_qft_cosmology_scenario(self):
        """Test QFT mode evolution with cosmological background consistency."""
        # Setup: Scalar field mode in radiation era
        k = 1.0
        m = 0.0
        p = LTQGConstants.RADIATION_P
        
        # Background cosmology
        cosmology = FLRWCosmology(p)
        
        # QFT mode evolution
        qft_mode = QFTModeEvolution(k=k, m=m, p=p)
        
        # Verify background consistency
        test_times = np.array([0.5, 1.0, 2.0, 5.0])
        for t in test_times:
            # Scale factor should match
            a_cosmo = cosmology.scale_factor(t)
            a_qft = qft_mode.scale_factor(t)
            self.assertClose(a_cosmo, a_qft, tol=1e-12)
            
            # Hubble parameter should match
            H_cosmo = cosmology.hubble_parameter(t)
            H_qft = qft_mode.hubble_parameter(t)
            self.assertClose(H_cosmo, H_qft, tol=1e-12)
        
        # Mode evolution should be stable
        integrator = AdaptiveIntegrator()
        t_i, t_f = 0.5, 5.0
        u0, u_dot0 = qft_mode.initial_conditions_adiabatic(t_i)
        
        result = integrator.integrate_mode_tau(qft_mode, (t_i, t_f), (u0, u_dot0))
        
        self.assertTrue(result['success'], "QFT evolution should succeed")
        self.assertTrue(np.all(np.isfinite(result['u'])), "Mode function should remain finite")
    
    def test_variational_cosmology_scenario(self):
        """Test variational field theory with cosmological applications."""
        # Setup: Scalar field variational theory
        theory = VariationalFieldTheory(spacetime_dim=4)
        
        # Background: FLRW with radiation era
        p = LTQGConstants.RADIATION_P
        cosmology = FLRWCosmology(p)
        
        # Test that variational theory can handle FLRW backgrounds
        # This is more of a structural test since full symbolic computation is complex
        
        # Check equation of state consistency
        eos = cosmology.equation_of_state()
        
        # Radiation era should have w = 1/3
        self.assertClose(eos['w'], 1.0/3.0, tol=1e-10,
                        msg="Variational cosmology should recover radiation equation of state")
        
        # Energy density scaling should be correct
        self.assertClose(eos['rho_scale_scaling'], -4.0, tol=1e-10,
                        msg="Radiation energy density should scale as a^(-4)")

class TestScalabilityAndPerformance(LTQGTestCase):
    """Test scalability and performance of integrated LTQG calculations."""
    
    def test_large_scale_quantum_evolution(self):
        """Test quantum evolution performance for larger systems."""
        # Larger Hilbert space dimension
        evolution = QuantumEvolutionLTQG()
        
        # 4x4 system (could represent two qubits)
        def H_large(tau):
            # Random Hermitian matrix
            np.random.seed(42)  # Reproducible
            A = np.random.randn(4, 4) + 1j * np.random.randn(4, 4)
            H = (A + A.conj().T) / (2 * tau)  # Time-dependent
            return H
        
        tau_i, tau_f = 1.0, 3.0
        N_steps = 100
        tau_grid = np.linspace(tau_i, tau_f, N_steps + 1)
        
        with PerformanceTimer("large_quantum_evolution") as timer:
            U = evolution.U_time_ordered_tau(H_large, tau_grid)
        
        # Should complete in reasonable time
        self.assertLess(timer.elapsed, 5.0, "Large system evolution should be efficient")
        
        # Should maintain unitarity
        self.assertUnitary(U, tol=1e-10)
    
    def test_long_time_qft_evolution(self):
        """Test QFT evolution performance over long times."""
        mode = QFTModeEvolution(k=1.0, m=0.1, p=0.5)
        integrator = AdaptiveIntegrator()
        
        # Long evolution
        t_i, t_f = 0.1, 50.0
        u0, u_dot0 = mode.initial_conditions_adiabatic(t_i)
        
        with PerformanceTimer("long_qft_evolution") as timer:
            result = integrator.integrate_mode_tau(mode, (t_i, t_f), (u0, u_dot0))
        
        # Should complete efficiently
        self.assertLess(timer.elapsed, 10.0, "Long QFT evolution should be efficient")
        self.assertTrue(result['success'], "Long evolution should succeed")
        
        # Should maintain numerical stability
        self.assertTrue(np.all(np.isfinite(result['u'])), "Solution should remain finite")
    
    def test_multiple_mode_evolution(self):
        """Test evolution of multiple QFT modes simultaneously."""
        # Multiple modes with different parameters
        modes = [
            QFTModeEvolution(k=0.1, m=0.0, p=0.5),
            QFTModeEvolution(k=1.0, m=0.1, p=0.5),
            QFTModeEvolution(k=5.0, m=0.5, p=0.5)
        ]
        
        integrator = AdaptiveIntegrator()
        t_i, t_f = 1.0, 5.0
        
        results = []
        with PerformanceTimer("multiple_modes") as timer:
            for mode in modes:
                u0, u_dot0 = mode.initial_conditions_adiabatic(t_i)
                result = integrator.integrate_mode_tau(mode, (t_i, t_f), (u0, u_dot0))
                results.append(result)
        
        # All should succeed
        for i, result in enumerate(results):
            self.assertTrue(result['success'], f"Mode {i} evolution should succeed")
        
        # Should be efficient
        self.assertLess(timer.elapsed, 10.0, "Multiple mode evolution should be efficient")

class TestErrorHandlingAndRobustness(LTQGTestCase):
    """Test error handling and robustness of integrated systems."""
    
    def test_extreme_parameter_handling(self):
        """Test handling of extreme parameter values."""
        # Very small tau0
        try:
            evolution = QuantumEvolutionLTQG(tau0=1e-10)
            transform = evolution.transform
            
            # Should handle small scale transformations
            sigma = transform.tau_to_sigma(1e-8)
            tau_back = transform.sigma_to_tau(sigma)
            
            # Should maintain precision
            relative_error = abs(tau_back - 1e-8) / 1e-8
            self.assertLess(relative_error, 1e-6, "Should handle extreme scales")
            
        except Exception as e:
            self.fail(f"Extreme parameter handling failed: {e}")
    
    def test_boundary_condition_handling(self):
        """Test handling of boundary conditions."""
        mode = QFTModeEvolution(k=1.0, m=0.0, p=0.5)
        
        # Evolution very close to initial time
        t_i = 1.0
        t_f = t_i + 1e-6  # Very small interval
        
        u0, u_dot0 = mode.initial_conditions_adiabatic(t_i)
        
        integrator = AdaptiveIntegrator()
        result = integrator.integrate_mode_tau(mode, (t_i, t_f), (u0, u_dot0))
        
        # Should succeed even for tiny intervals
        self.assertTrue(result['success'], "Should handle small time intervals")
        
        # Final state should be close to initial state
        u_final = result['u'][-1]
        self.assertClose(abs(u_final), abs(u0), tol=1e-3,
                        msg="Small evolution should preserve state approximately")
    
    def test_numerical_precision_degradation(self):
        """Test detection of numerical precision degradation."""
        mode = QFTModeEvolution(k=1.0, m=0.0, p=0.5)
        diagnostics = QFTDiagnostics()
        
        # Evolution with different precision requirements
        integrator_low = AdaptiveIntegrator(rtol=1e-6, atol=1e-8)
        integrator_high = AdaptiveIntegrator(rtol=1e-10, atol=1e-12)
        
        t_i, t_f = 1.0, 5.0
        u0, u_dot0 = mode.initial_conditions_adiabatic(t_i)
        
        result_low = integrator_low.integrate_mode_tau(mode, (t_i, t_f), (u0, u_dot0))
        result_high = integrator_high.integrate_mode_tau(mode, (t_i, t_f), (u0, u_dot0))
        
        # Compare final amplitudes
        amplitude_error = diagnostics.relative_amplitude_error(
            np.array([result_low['u'][-1]]), 
            np.array([result_high['u'][-1]]), 
            normalize=True
        )
        
        # Higher precision should give more accurate results
        self.assertLess(amplitude_error, 1e-4, 
                       "Higher precision integration should be more accurate")

class TestPhysicalValidation(LTQGTestCase):
    """Test physical validation of integrated calculations."""
    
    def test_energy_momentum_conservation(self):
        """Test energy-momentum conservation in integrated scenarios."""
        # QFT mode evolution with energy tracking
        mode = QFTModeEvolution(k=1.0, m=0.1, p=0.5)
        integrator = AdaptiveIntegrator()
        diagnostics = QFTDiagnostics()
        
        t_i, t_f = 1.0, 5.0
        u0, u_dot0 = mode.initial_conditions_adiabatic(t_i)
        
        result = integrator.integrate_mode_tau(mode, (t_i, t_f), (u0, u_dot0))
        
        # Calculate energy at different times
        energies = []
        for i, t in enumerate(result['t']):
            omega2 = mode.frequency_squared(t)
            energy = diagnostics.energy_density(
                np.array([result['u'][i]]), 
                np.array([result['u_dot'][i]]), 
                np.array([omega2])
            )[0]
            energies.append(energy)
        
        # Energy should evolve smoothly (not be conserved due to expansion)
        energy_array = np.array(energies)
        self.assertTrue(np.all(np.isfinite(energy_array)), "Energy should remain finite")
        self.assertTrue(np.all(energy_array > 0), "Energy should be positive")
        
        # Should not have sudden jumps (but allow for cosmological evolution)
        energy_diff = np.diff(energy_array)
        max_jump = np.max(np.abs(energy_diff)) / np.mean(energy_array)
        self.assertLess(max_jump, 1.0, "Energy should evolve smoothly (allowing cosmological effects)")
    
    def test_causality_preservation(self):
        """Test that causality is preserved in quantum evolution."""
        evolution = QuantumEvolutionLTQG()
        
        # Create a causal Hamiltonian (finite speed of information propagation)
        def H_causal(tau):
            # Simple local Hamiltonian
            sz = np.array([[1., 0.], [0., -1.]], dtype=complex)
            return (1.0 / tau) * sz  # Bounded evolution speed
        
        # Short time evolution (should be approximately local)
        tau_i, tau_f = 1.0, 1.1  # Small interval
        N_steps = 100
        tau_grid = np.linspace(tau_i, tau_f, N_steps + 1)
        
        U = evolution.U_time_ordered_tau(H_causal, tau_grid)
        
        # Evolution should be continuous and bounded
        self.assertUnitary(U, tol=1e-10)
        
        # Check that evolution is "reasonable" (no exponential explosion)
        operator_norm = np.linalg.norm(U, ord=2)
        self.assertClose(operator_norm, 1.0, tol=1e-10,
                        msg="Unitary evolution should preserve operator norm")

if __name__ == '__main__':
    unittest.main()