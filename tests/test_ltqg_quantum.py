"""
Unit Tests for LTQG Quantum Evolution Module

Tests for quantum evolution in both tau and sigma coordinates,
unitary equivalence, and Heisenberg picture observables.
"""

import unittest
import numpy as np
from test_utils import LTQGTestCase, parametrized_test, TestDataGenerator, MockHamiltonian

# Import modules to test
try:
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ltqg_core_implementation_python_10_17_25'))
    
    from ltqg_quantum import (
        QuantumEvolutionLTQG,
        validate_unitary_equivalence_constant_H,
        validate_unitary_equivalence_time_dependent,
        validate_heisenberg_observables,
        analyze_quantum_state_evolution
    )
    from ltqg_core import LTQGConstants
except ImportError as e:
    raise unittest.SkipTest(f"Cannot import LTQG modules: {e}")

class TestQuantumEvolutionLTQG(LTQGTestCase):
    """Test the QuantumEvolutionLTQG class."""
    
    def setUp(self):
        super().setUp()
        self.evolution = QuantumEvolutionLTQG()
        self.test_hamiltonians = self.generate_test_hamiltonians()  # This is now available from LTQGTestCase
    
    def test_initialization(self):
        """Test QuantumEvolutionLTQG initialization."""
        # Default initialization
        evolution = QuantumEvolutionLTQG()
        self.assertEqual(evolution.tau0, LTQGConstants.TAU0_DEFAULT)
        self.assertEqual(evolution.hbar, LTQGConstants.HBAR_DEFAULT)
        
        # Custom initialization
        evolution_custom = QuantumEvolutionLTQG(tau0=2.0, hbar=1.5)
        self.assertEqual(evolution_custom.tau0, 2.0)
        self.assertEqual(evolution_custom.hbar, 1.5)
    
    def test_time_ordered_evolution_constant_hamiltonian(self):
        """Test time-ordered evolution for constant Hamiltonian."""
        # Pauli-X matrix
        H = np.array([[0., 1.], [1., 0.]], dtype=np.complex128)
        
        # Time grid
        tau_i, tau_f = 1.0, 3.0
        N_steps = 100
        tau_grid = np.linspace(tau_i, tau_f, N_steps + 1)
        sigma_grid = self.evolution.transform.tau_to_sigma(tau_grid)
        
        def H_constant(tau):
            return H
        
        # Compute evolution operators
        U_tau = self.evolution.U_time_ordered_tau(H_constant, tau_grid)
        U_sigma = self.evolution.U_time_ordered_sigma(H_constant, sigma_grid)
        
        # Both should be unitary
        self.assertUnitary(U_tau)
        self.assertUnitary(U_sigma)
        
        # For constant H, analytical solution is exp(-iH*Delta_tau/hbar)
        Delta_tau = tau_f - tau_i
        w, V = np.linalg.eig(H)
        U_analytical = V @ np.diag(np.exp(-1j*w*Delta_tau/self.evolution.hbar)) @ np.linalg.inv(V)
        
        # Compare with tau evolution
        self.assertMatrixClose(U_tau, U_analytical, tol=1e-10)
    
    @parametrized_test([
        ("constant_sz", lambda t: 0.5 * np.array([[1., 0.], [0., -1.]], dtype=complex)),
        ("constant_sx", lambda t: 0.8 * np.array([[0., 1.], [1., 0.]], dtype=complex)),
        ("oscillating", lambda t: 0.5 * np.array([[1., 0.], [0., -1.]], dtype=complex) + 
                                  0.3 * np.cos(t) * np.array([[0., 1.], [1., 0.]], dtype=complex))
    ])  # Simplified test cases for speed
    def test_unitary_equivalence(self, name, H_func):
        """Test unitary equivalence between tau and sigma evolution."""
        # Evolution parameters
        tau_i, tau_f = self.evolution.tau0, 3.0
        N_steps = 200
        tau_grid = np.linspace(tau_i, tau_f, N_steps + 1)
        sigma_grid = self.evolution.transform.tau_to_sigma(tau_grid)
        
        # Compute evolution operators
        U_tau = self.evolution.U_time_ordered_tau(H_func, tau_grid)
        U_sigma = self.evolution.U_time_ordered_sigma(H_func, sigma_grid)
        
        # Test on basis states
        basis_states = [np.array([1.+0j, 0.+0j]), np.array([0.+0j, 1.+0j])]
        
        for psi0 in basis_states:
            # Evolve initial state
            psi_tau = U_tau @ psi0
            psi_sigma = U_sigma @ psi0
            
            # Compute density matrices
            rho_tau = np.outer(psi_tau, np.conjugate(psi_tau))
            rho_sigma = np.outer(psi_sigma, np.conjugate(psi_sigma))
            
            # Density matrices should be equal (observable predictions preserved)
            # Use higher tolerance for quantum evolution tests due to numerical precision
            self.assertMatrixClose(rho_tau, rho_sigma, tol=5e-6,
                                 msg=f"Density matrices differ for {name}")
    
    def test_heisenberg_observable_evolution(self):
        """Test Heisenberg picture observable evolution."""
        # Observable to evolve
        sz = np.array([[1., 0.], [0., -1.]], dtype=np.complex128)
        
        # Simple time-dependent Hamiltonian
        sx = np.array([[0., 1.], [1., 0.]], dtype=np.complex128)
        def H_func(tau):
            return 0.5 * sz + 0.3 * np.cos(tau) * sx
        
        # Evolution parameters
        tau_i, tau_f = 1.0, 2.0
        N_steps = 100
        tau_grid = np.linspace(tau_i, tau_f, N_steps + 1)
        
        # Compute evolution operator
        U = self.evolution.U_time_ordered_tau(H_func, tau_grid)
        
        # Evolve observable
        sz_heisenberg = self.evolution.heisenberg_observable(sz, U)
        
        # Should still be Hermitian
        self.assertHermitian(sz_heisenberg)
        
        # Should have same eigenvalues (up to ordering)
        eigenvals_original = np.sort(np.real(np.linalg.eigvals(sz)))
        eigenvals_evolved = np.sort(np.real(np.linalg.eigvals(sz_heisenberg)))
        self.assertClose(eigenvals_original, eigenvals_evolved, tol=1e-10)

class TestQuantumStateEvolution(LTQGTestCase):
    """Test quantum state evolution analysis functions."""
    
    def test_analyze_quantum_state_evolution(self):
        """Test comprehensive quantum state evolution analysis."""
        # Simple oscillating Hamiltonian
        def H_func(tau):
            sz = np.array([[1., 0.], [0., -1.]], dtype=np.complex128)
            sx = np.array([[0., 1.], [1., 0.]], dtype=np.complex128)
            return 0.5 * sz + 0.2 * np.sin(tau) * sx
        
        # Initial state (|+⟩ eigenstate of sx)
        psi0 = np.array([1.+0j, 1.+0j]) / np.sqrt(2)
        
        # Observables to track
        sz = np.array([[1., 0.], [0., -1.]], dtype=np.complex128)
        sx = np.array([[0., 1.], [1., 0.]], dtype=np.complex128)
        observables = [sz, sx]
        
        # Run analysis
        results = analyze_quantum_state_evolution(
            H_func, psi0, tau_final=3.0, observables=observables, N_steps=200
        )
        
        # Check results structure
        required_keys = ['tau_grid', 'sigma_grid', 'U_tau', 'U_sigma', 
                        'psi_tau_final', 'psi_sigma_final', 'fidelity',
                        'unitarity_tau', 'unitarity_sigma', 'observables_tau', 'observables_sigma']
        
        for key in required_keys:
            self.assertIn(key, results, f"Missing key: {key}")
        
        # Check fidelity (should be close to 1)
        self.assertGreater(results['fidelity'], 0.99, "Low fidelity between tau and sigma evolution")
        
        # Check unitarity
        self.assertLess(results['unitarity_tau'], 1e-10, "Poor unitarity for tau evolution")
        self.assertLess(results['unitarity_sigma'], 1e-10, "Poor unitarity for sigma evolution")
        
        # Check observable agreement (use higher tolerance for quantum evolution)
        for obs_tau, obs_sigma in zip(results['observables_tau'], results['observables_sigma']):
            self.assertClose(obs_tau, obs_sigma, tol=5e-6,
                           msg="Observable expectations differ between tau and sigma")

class TestValidationFunctions(LTQGTestCase):
    """Test the validation functions."""
    
    def test_validate_unitary_equivalence_constant_H(self):
        """Test constant Hamiltonian validation function."""
        try:
            validate_unitary_equivalence_constant_H()
        except Exception as e:
            self.fail(f"Constant H validation failed: {e}")
    
    def test_validate_unitary_equivalence_time_dependent(self):
        """Test time-dependent Hamiltonian validation function."""
        try:
            validate_unitary_equivalence_time_dependent()
        except Exception as e:
            self.fail(f"Time-dependent H validation failed: {e}")
    
    def test_validate_heisenberg_observables(self):
        """Test Heisenberg observable validation function."""
        try:
            validate_heisenberg_observables()
        except Exception as e:
            self.fail(f"Heisenberg observable validation failed: {e}")

class TestQuantumPhases(LTQGTestCase):
    """Test quantum phases and Berry phases."""
    
    def test_geometric_phase_preservation(self):
        """Test that geometric phases are preserved under coordinate transformation."""
        evolution = QuantumEvolutionLTQG()
        
        # Adiabatic Hamiltonian with rotating magnetic field
        def H_adiabatic(tau):
            omega = 0.1  # Adiabatic parameter
            theta = omega * tau
            
            # Rotating field in x-y plane
            sz = np.array([[1., 0.], [0., -1.]], dtype=np.complex128)
            sx = np.array([[0., 1.], [1., 0.]], dtype=np.complex128)
            sy = np.array([[0., -1j], [1j, 0.]], dtype=np.complex128)
            
            return 0.5 * sz + 0.1 * (np.cos(theta) * sx + np.sin(theta) * sy)
        
        # Initial eigenstate
        psi0 = np.array([1.+0j, 0.+0j])  # |↑⟩ state
        
        # Evolution over full cycle
        tau_i, tau_f = 1.0, 1.0 + 2*np.pi/0.1  # One full rotation
        N_steps = 1000
        tau_grid = np.linspace(tau_i, tau_f, N_steps + 1)
        sigma_grid = evolution.transform.tau_to_sigma(tau_grid)
        
        # Evolve in both coordinates
        U_tau = evolution.U_time_ordered_tau(H_adiabatic, tau_grid)
        U_sigma = evolution.U_time_ordered_sigma(H_adiabatic, sigma_grid)
        
        psi_tau_final = U_tau @ psi0
        psi_sigma_final = U_sigma @ psi0
        
        # Extract phases
        phase_tau = np.angle(np.vdot(psi0, psi_tau_final))
        phase_sigma = np.angle(np.vdot(psi0, psi_sigma_final))
        
        # Phases should be equal (modulo 2π)
        phase_diff = np.abs(phase_tau - phase_sigma)
        phase_diff = min(phase_diff, 2*np.pi - phase_diff)  # Wrap to [0, π]
        
        self.assertLess(phase_diff, 0.1, "Geometric phases differ significantly")

class TestNumericalStability(LTQGTestCase):
    """Test numerical stability of quantum evolution."""
    
    def test_large_time_evolution(self):
        """Test stability for large time evolution."""
        evolution = QuantumEvolutionLTQG()
        
        # Stable Hamiltonian
        def H_stable(tau):
            return 0.1 * np.array([[1., 0.], [0., -1.]], dtype=np.complex128)
        
        # Large time evolution
        tau_i, tau_f = 1.0, 100.0
        N_steps = 2000
        tau_grid = np.linspace(tau_i, tau_f, N_steps + 1)
        
        U = evolution.U_time_ordered_tau(H_stable, tau_grid)
        
        # Should remain unitary
        self.assertUnitary(U, tol=1e-8)
        
        # Determinant should have unit magnitude
        det_U = np.linalg.det(U)
        self.assertClose(np.abs(det_U), 1.0, tol=1e-8)
    
    def test_high_frequency_hamiltonian(self):
        """Test evolution with high-frequency time dependence."""
        evolution = QuantumEvolutionLTQG()
        
        # High-frequency oscillating Hamiltonian
        def H_high_freq(tau):
            sz = np.array([[1., 0.], [0., -1.]], dtype=np.complex128)
            sx = np.array([[0., 1.], [1., 0.]], dtype=np.complex128)
            return 0.5 * sz + 0.3 * np.cos(10*tau) * sx
        
        # Short time with fine resolution
        tau_i, tau_f = 1.0, 2.0
        N_steps = 1000
        tau_grid = np.linspace(tau_i, tau_f, N_steps + 1)
        
        U = evolution.U_time_ordered_tau(H_high_freq, tau_grid)
        
        # Should remain unitary despite high frequency
        self.assertUnitary(U, tol=1e-6)

if __name__ == '__main__':
    unittest.main()