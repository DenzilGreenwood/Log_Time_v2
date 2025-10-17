#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LTQG Quantum Evolution Module

This module extends the core LTQG framework to quantum mechanical systems,
implementing the σ-Schrödinger equation and demonstrating unitary equivalence
between τ-time and σ-time evolution.

Key Features:
- σ-Schrödinger equation: iℏ ∂_σ ψ = τ₀e^σ H(τ₀e^σ) ψ
- Time-ordered evolution operators in both coordinates
- Heisenberg picture observables
- Unitary equivalence validation for constant and time-dependent Hamiltonians

Physical Interpretation:
The log-time reparameterization preserves all quantum mechanical predictions
while providing mathematical advantages in cosmological contexts.

Author: Mathematical Physics Research  
License: Open Source
"""

import numpy as np
import sympy as sp
from typing import Callable, Tuple, Union, Optional, List
from ltqg_core import LogTimeTransform, banner, assert_close, mat_expm, LTQGConstants

# ===========================
# Unitary Equivalence Theorem (τ ↔ σ)
# ===========================

def unitary_equivalence_theorem():
    """
    ╔═══════════════════════════════════════════════════════════════════════════════╗
    ║                        UNITARY EQUIVALENCE THEOREM                           ║
    ╠═══════════════════════════════════════════════════════════════════════════════╣
    ║ Proposition (Unitary equivalence under log-time).                            ║
    ║                                                                               ║
    ║ Let H: (0,τf] → B(H) be strongly measurable with ||H(τ)|| locally bounded    ║
    ║ on (0,τf] and suppose H(τ) generates a unique unitary propagator             ║
    ║ U_τ(τf,τi) (Kato conditions).                                                ║
    ║                                                                               ║
    ║ Define σ = log(τ/τ₀) and H_eff(σ) = τ₀e^σ H(τ₀e^σ).                        ║
    ║                                                                               ║
    ║ Then the σ-ordered propagator                                                 ║
    ║     U_σ(σf,σi) = T exp(-iℏ ∫[σi to σf] H_eff(s) ds)                        ║
    ║                                                                               ║
    ║ exists and equals U_τ(τf,τi) with τi,f = τ₀e^(σi,f).                       ║
    ║                                                                               ║
    ║ Proof outline:                                                                ║
    ║ 1. Variable change τ = τ₀e^σ in Dyson series                                ║
    ║ 2. dτ = τ₀e^σ dσ transforms integration measure                              ║
    ║ 3. H(τ₀e^σ) dτ = τ₀e^σ H(τ₀e^σ) dσ = H_eff(σ) dσ                         ║
    ║ 4. Time-ordering preserved: T[τ₁ < τ₂] ↔ T[σ₁ < σ₂]                        ║
    ║ 5. Dominated convergence justifies equality by local boundedness             ║
    ╚═══════════════════════════════════════════════════════════════════════════════╝
    """
    pass

# ===========================
# Quantum Evolution Framework
# ===========================

class QuantumEvolutionLTQG:
    """
    Quantum evolution in LTQG framework supporting both τ-time and σ-time descriptions.
    
    Provides time-ordered evolution operators and validates unitary equivalence
    between coordinate systems.
    """
    
    def __init__(self, tau0: float = LTQGConstants.TAU0_DEFAULT, 
                 hbar: float = LTQGConstants.HBAR_DEFAULT):
        """
        Initialize quantum evolution framework.
        
        Args:
            tau0: Reference time scale
            hbar: Reduced Planck constant
        """
        self.transform = LogTimeTransform(tau0)
        self.tau0 = tau0
        self.hbar = hbar
    
    def U_time_ordered_tau(self, H_of_tau: Callable[[float], np.ndarray], 
                          tau_grid: np.ndarray) -> np.ndarray:
        """
        Time-ordered evolution operator in τ-time.
        
        Computes U(τf,τi) for Hamiltonian H(τ) via time-slicing.
        
        Args:
            H_of_tau: Hamiltonian as function of proper time τ
            tau_grid: Array of time points from τi to τf
            
        Returns:
            Time-ordered evolution operator
        """
        dim = H_of_tau(tau_grid[0]).shape[0]
        U = np.eye(dim, dtype=np.complex128)
        
        for i in range(len(tau_grid)-1):
            t_mid = 0.5 * (tau_grid[i] + tau_grid[i+1])
            dt = tau_grid[i+1] - tau_grid[i]
            H = H_of_tau(t_mid)
            
            # Matrix exponential step
            w, V = np.linalg.eig(H)
            step = V @ np.diag(np.exp(-1j*w*dt/self.hbar)) @ np.linalg.inv(V)
            U = step @ U
            
        return U
    
    def U_time_ordered_sigma(self, H_of_tau: Callable[[float], np.ndarray], 
                           sigma_grid: np.ndarray) -> np.ndarray:
        """
        Time-ordered evolution operator in σ-time.
        
        Computes U(σf,σi) using effective Hamiltonian H_eff(σ) = τ₀e^σ H(τ₀e^σ).
        
        Args:
            H_of_tau: Original Hamiltonian as function of proper time τ
            sigma_grid: Array of log-time points from σi to σf
            
        Returns:
            Time-ordered evolution operator
        """
        dim = H_of_tau(self.tau0).shape[0]
        U = np.eye(dim, dtype=np.complex128)
        
        for i in range(len(sigma_grid)-1):
            s_mid = 0.5 * (sigma_grid[i] + sigma_grid[i+1])
            ds = sigma_grid[i+1] - sigma_grid[i]
            
            # Convert to τ for Hamiltonian evaluation
            tau_mid = self.transform.sigma_to_tau(s_mid)
            H = H_of_tau(tau_mid)
            
            # Effective evolution with τ₀e^σ factor
            factor = tau_mid / self.hbar
            w, V = np.linalg.eig(H)
            step = V @ np.diag(np.exp(-1j*w*factor*ds)) @ np.linalg.inv(V)
            U = step @ U
            
        return U
    
    def heisenberg_observable(self, A: np.ndarray, U: np.ndarray) -> np.ndarray:
        """
        Evolve observable to Heisenberg picture: A_H = U† A U.
        
        Args:
            A: Observable operator in Schrödinger picture
            U: Time evolution operator
            
        Returns:
            Heisenberg picture observable
        """
        return U.conj().T @ A @ U

# ===========================
# Validation Functions
# ===========================

def validate_unitary_equivalence_constant_H() -> None:
    """Validate unitary equivalence for constant Hamiltonian."""
    banner("Quantum Evolution: Constant Hamiltonian Equivalence")
    
    evolution = QuantumEvolutionLTQG()
    
    # Test parameters
    H = np.array([[0., 1.], [1., 0.]], dtype=np.complex128)  # Pauli-X
    psi0 = np.array([1.+0j, 0.+0j])
    
    print("QUANTUM EVOLUTION EQUIVALENCE TEST:")
    print("τ-Schrödinger: iℏ ∂_τ ψ = H ψ")
    print("σ-Schrödinger: iℏ ∂_σ ψ = τ₀e^σ H ψ")
    print("For constant H, both should yield identical density matrices")
    
    # Direct evolution operators for constant H
    def U_tau_direct(tau_i: float, tau_f: float) -> np.ndarray:
        """Direct τ-evolution for constant H."""
        delta_tau = tau_f - tau_i
        w, V = np.linalg.eig(H)
        return V @ np.diag(np.exp(-1j*w*delta_tau/evolution.hbar)) @ np.linalg.inv(V)
    
    def U_sigma_direct(sig_i: float, sig_f: float) -> np.ndarray:
        """Direct σ-evolution for constant H."""
        factor = (evolution.tau0/evolution.hbar) * (np.exp(sig_f) - np.exp(sig_i))
        w, V = np.linalg.eig(H)
        return V @ np.diag(np.exp(-1j*w*factor)) @ np.linalg.inv(V)
    
    print("\nTesting evolution operators:")
    print("U_τ(τᵢ,τf) = exp(-iH(τf-τᵢ)/ℏ)")
    print("U_σ(σᵢ,σf) = exp(-iH(τ₀/ℏ)(e^σf - e^σᵢ))")
    print("With τ = τ₀e^σ: τf - τᵢ = τ₀(e^σf - e^σᵢ) → operators identical")
    
    # Test over range of times
    taus = np.linspace(evolution.tau0, 3.0, 9)
    for tau in taus:
        sig_f = evolution.transform.tau_to_sigma(tau)
        sig_i = 0.0
        
        # Compute density matrices
        rho_tau = np.outer(U_tau_direct(evolution.tau0, tau) @ psi0, 
                          np.conjugate(U_tau_direct(evolution.tau0, tau) @ psi0))
        rho_sig = np.outer(U_sigma_direct(sig_i, sig_f) @ psi0, 
                          np.conjugate(U_sigma_direct(sig_i, sig_f) @ psi0))
        
        assert_close(rho_tau, rho_sig, tol=1e-10)
    
    print("✓ MATHEMATICAL RESULT: Density matrices ρ_τ = ρ_σ to machine precision")
    print("  Quantum evolution is unitarily equivalent under log-time reparameterization")
    print("PASS: Constant-H σ reproduces τ (up to unobservable phase).")

def validate_unitary_equivalence_time_dependent() -> None:
    """Validate unitary equivalence for time-dependent, non-commuting Hamiltonian."""
    banner("Quantum Evolution: Time-Dependent Non-Commuting Hamiltonian")
    
    evolution = QuantumEvolutionLTQG()
    
    # Test parameters - oscillating two-level system
    Delta = 0.7    # Energy splitting
    Omega0 = 1.2   # Coupling strength
    nu = 0.9       # Oscillation frequency
    
    # Pauli matrices
    sx = np.array([[0., 1.], [1., 0.]], dtype=np.complex128)
    sz = np.array([[1., 0.], [0., -1.]], dtype=np.complex128)
    
    def H_of_tau(tau: float) -> np.ndarray:
        """Time-dependent Hamiltonian with non-commuting terms."""
        return Delta * sz + (Omega0 * np.cos(nu * tau)) * sx
    
    # Evolution parameters
    tau_i, tau_f = evolution.tau0, 3.0
    N = 600  # Fine time discretization for accuracy
    
    # Construct time grids
    tau_grid = np.linspace(tau_i, tau_f, N+1)
    sigma_grid = evolution.transform.tau_to_sigma(tau_grid)
    
    # Compute evolution operators
    U_tau = evolution.U_time_ordered_tau(H_of_tau, tau_grid)
    U_sig = evolution.U_time_ordered_sigma(H_of_tau, sigma_grid)
    
    # Test on computational basis states
    basis = [np.array([1.+0j, 0.+0j]), np.array([0.+0j, 1.+0j])]
    
    for psi0 in basis:
        rho_tau = np.outer(U_tau @ psi0, np.conjugate(U_tau @ psi0))
        rho_sig = np.outer(U_sig @ psi0, np.conjugate(U_sig @ psi0))
        assert_close(rho_tau, rho_sig, tol=5e-6)  # Numerical tolerance for time-ordering
    
    print("✓ Time-ordered evolution preserves unitary equivalence")
    print("✓ Non-commuting Hamiltonians handled correctly")
    print("PASS: Non-commuting H(τ) equivalence verified to numerical tolerance.")

def validate_heisenberg_observables() -> None:
    """Validate Heisenberg picture observables under both evolution schemes."""
    banner("Quantum Evolution: Heisenberg Picture Observable Evolution")
    
    evolution = QuantumEvolutionLTQG()
    
    # Test parameters
    Delta = 0.5
    Omega0 = 0.8
    nu = 1.1
    
    sx = np.array([[0., 1.], [1., 0.]], dtype=np.complex128)
    sz = np.array([[1., 0.], [0., -1.]], dtype=np.complex128)
    
    def H_of_tau(tau: float) -> np.ndarray:
        return Delta * sz + (Omega0 * np.sin(nu * tau)) * sx
    
    # Observable to evolve
    A = sz  # σ_z observable
    
    # Evolution setup
    tau_i, tau_f = evolution.tau0, 2.5
    N = 500
    tau_grid = np.linspace(tau_i, tau_f, N+1)
    sigma_grid = evolution.transform.tau_to_sigma(tau_grid)
    
    # Compute evolution operators
    U_tau = evolution.U_time_ordered_tau(H_of_tau, tau_grid)
    U_sig = evolution.U_time_ordered_sigma(H_of_tau, sigma_grid)
    
    # Evolve observable to Heisenberg picture
    A_tau = evolution.heisenberg_observable(A, U_tau)
    A_sig = evolution.heisenberg_observable(A, U_sig)
    
    assert_close(A_tau, A_sig, tol=5e-6)
    
    print("✓ Heisenberg observables agree under τ vs σ evolution")
    print("✓ Physical predictions preserved in both coordinate systems")
    print("PASS: Heisenberg observable equivalence confirmed.")

# ===========================
# Quantum State Analysis
# ===========================

def analyze_quantum_state_evolution(H_of_tau: Callable[[float], np.ndarray],
                                   psi0: np.ndarray, tau_final: float,
                                   observables: List[np.ndarray] = None,
                                   N_steps: int = 1000) -> dict:
    """
    Comprehensive analysis of quantum state evolution in both coordinate systems.
    
    Args:
        H_of_tau: Time-dependent Hamiltonian
        psi0: Initial state
        tau_final: Final evolution time
        observables: List of observables to track
        N_steps: Number of time steps
        
    Returns:
        Dictionary with evolution data and analysis
    """
    evolution = QuantumEvolutionLTQG()
    
    # Set up evolution
    tau_grid = np.linspace(evolution.tau0, tau_final, N_steps+1)
    sigma_grid = evolution.transform.tau_to_sigma(tau_grid)
    
    # Compute evolution operators
    U_tau = evolution.U_time_ordered_tau(H_of_tau, tau_grid)
    U_sig = evolution.U_time_ordered_sigma(H_of_tau, sigma_grid)
    
    # Evolve initial state
    psi_tau_final = U_tau @ psi0
    psi_sig_final = U_sig @ psi0
    
    # Analyze results
    results = {
        'tau_grid': tau_grid,
        'sigma_grid': sigma_grid,
        'U_tau': U_tau,
        'U_sigma': U_sig,
        'psi_tau_final': psi_tau_final,
        'psi_sigma_final': psi_sig_final,
        'fidelity': np.abs(np.vdot(psi_tau_final, psi_sig_final))**2,
        'unitarity_tau': np.linalg.norm(U_tau @ U_tau.conj().T - np.eye(len(psi0))),
        'unitarity_sigma': np.linalg.norm(U_sig @ U_sig.conj().T - np.eye(len(psi0)))
    }
    
    # Track observables if provided
    if observables:
        results['observables_tau'] = []
        results['observables_sigma'] = []
        
        for A in observables:
            exp_tau = np.real(np.vdot(psi_tau_final, A @ psi_tau_final))
            exp_sig = np.real(np.vdot(psi_sig_final, A @ psi_sig_final))
            
            results['observables_tau'].append(exp_tau)
            results['observables_sigma'].append(exp_sig)
    
    return results

# ===========================
# Validation Suite
# ===========================

def run_quantum_evolution_validation() -> None:
    """Run complete validation suite for quantum evolution in LTQG."""
    print("="*80)
    print("LTQG QUANTUM EVOLUTION VALIDATION SUITE")
    print("="*80)
    
    validate_unitary_equivalence_constant_H()
    validate_unitary_equivalence_time_dependent()
    validate_heisenberg_observables()
    
    print("\n" + "="*80)
    print("QUANTUM EVOLUTION VALIDATION SUMMARY:")
    print("="*80)
    print("✅ Constant Hamiltonian: Perfect unitary equivalence τ ⟺ σ")
    print("✅ Time-dependent H(τ): Numerical equivalence with time-ordering")
    print("✅ Heisenberg observables: Physical predictions preserved")
    print("✅ Quantum mechanics fully compatible with log-time reparameterization")
    print("="*80)

if __name__ == "__main__":
    run_quantum_evolution_validation()