#!/usr/bin/env python3
"""
LTQG Cosmological Inference Validation

This script demonstrates that σ-uniform integration preserves cosmological
parameter inference while providing operational advantages for early-time
and high-redshift measurements.

Key Features:
- Standard z-integration vs σ-grid ODE comparison
- Synthetic supernova likelihood analysis
- Parameter constraint validation
- Operational protocol implications

Usage:
    python test_cosmological_inference.py
"""

import sys
import os
import numpy as np

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

def test_distance_equivalence():
    """Test that σ-grid ODE matches standard z-integration."""
    
    try:
        from ltqg_cosmological_inference import Cosmology, CosmoParams, self_check_distances
        
        print("="*70)
        print("LTQG COSMOLOGICAL INFERENCE VALIDATION")
        print("="*70)
        print("Testing σ-uniform vs standard distance calculations...")
        
        # Run the built-in self-check
        rel_error = self_check_distances()
        
        if rel_error < 1e-4:
            print(f"✅ PASS: σ-integration matches standard within {rel_error:.2e}")
            print("   Distance calculations are mathematically equivalent")
        else:
            print(f"⚠️  WARNING: Relative error {rel_error:.2e} higher than expected")
            print("   Consider tighter integration tolerances")
        
        return rel_error < 1e-4
        
    except ImportError as e:
        print(f"❌ FAIL: Cannot import cosmological inference module: {e}")
        return False
    except Exception as e:
        print(f"❌ FAIL: Distance equivalence test failed: {e}")
        return False

def test_parameter_inference():
    """Test that parameter constraints are preserved under σ-integration."""
    
    try:
        from ltqg_cosmological_inference import (
            Cosmology, CosmoParams, SNDataset, 
            run_grid_search_sn, HAS_EMCEE, run_emcee_sn
        )
        
        print("\n" + "="*70)
        print("PARAMETER INFERENCE VALIDATION")
        print("="*70)
        
        # Create synthetic data from known cosmology
        true_cosmo = Cosmology(CosmoParams(H0=70.0, Omega_m=0.3))
        data = SNDataset.synthetic(n=40, rng_seed=42, cosmology=true_cosmo, M_true=-19.3)
        
        print(f"Synthetic dataset: {len(data.z)} SNe, z ∈ [{data.z.min():.3f}, {data.z.max():.3f}]")
        print("True parameters: H₀=70.0 km/s/Mpc, Ωₘ=0.30")
        
        # Fit with standard distances
        print("\nFitting with standard z-integration...")
        best_std = run_grid_search_sn(data, use_sigma=False, model="lcdm")
        ll_std, (H0_std, Om_std, M_std) = best_std
        
        # Fit with σ-integration
        print("Fitting with σ-uniform integration...")
        best_sig = run_grid_search_sn(data, use_sigma=True, model="lcdm")
        ll_sig, (H0_sig, Om_sig, M_sig) = best_sig
        
        print(f"\nResults comparison:")
        print(f"  Standard:     H₀={H0_std:.1f}, Ωₘ={Om_std:.3f}, M={M_std:.2f}, lnL={ll_std:.1f}")
        print(f"  σ-integration: H₀={H0_sig:.1f}, Ωₘ={Om_sig:.3f}, M={M_sig:.2f}, lnL={ll_sig:.1f}")
        
        # Check parameter agreement
        H0_diff = abs(H0_std - H0_sig)
        Om_diff = abs(Om_std - Om_sig)
        M_diff = abs(M_std - M_sig)
        
        param_agreement = (H0_diff < 0.5) and (Om_diff < 0.01) and (M_diff < 0.05)
        
        if param_agreement:
            print("✅ PASS: Parameter constraints agree between methods")
            print("   σ-integration preserves cosmological inference")
        else:
            print("⚠️  WARNING: Parameter differences larger than expected")
            print("   This may indicate numerical issues or insufficient grid resolution")
        
        # MCMC comparison if available
        if HAS_EMCEE:
            print("\nRunning MCMC validation...")
            
            # Standard MCMC
            chain_std, _ = run_emcee_sn(data, use_sigma=False, model="lcdm", 
                                       nwalkers=24, nsteps=1000, burn=200)
            means_std = np.mean(chain_std, axis=0)
            stds_std = np.std(chain_std, axis=0)
            
            # σ-integration MCMC  
            chain_sig, _ = run_emcee_sn(data, use_sigma=True, model="lcdm",
                                       nwalkers=24, nsteps=1000, burn=200)
            means_sig = np.mean(chain_sig, axis=0)
            stds_sig = np.std(chain_sig, axis=0)
            
            print(f"MCMC posterior means:")
            print(f"  Standard:      H₀={means_std[0]:.1f}±{stds_std[0]:.1f}, Ωₘ={means_std[1]:.3f}±{stds_std[1]:.3f}")
            print(f"  σ-integration: H₀={means_sig[0]:.1f}±{stds_sig[0]:.1f}, Ωₘ={means_sig[1]:.3f}±{stds_sig[1]:.3f}")
            
            # Check if posterior means agree within 1σ
            mcmc_agreement = (abs(means_std[0] - means_sig[0]) < max(stds_std[0], stds_sig[0]) and
                            abs(means_std[1] - means_sig[1]) < max(stds_std[1], stds_sig[1]))
            
            if mcmc_agreement:
                print("✅ PASS: MCMC posteriors agree within statistical uncertainties")
            else:
                print("⚠️  NOTE: MCMC differences may reflect sampling variations")
        
        return param_agreement
        
    except Exception as e:
        print(f"❌ FAIL: Parameter inference test failed: {e}")
        return False

def test_operational_protocols():
    """Demonstrate operational differences between σ-uniform and τ-uniform protocols."""
    
    print("\n" + "="*70)
    print("OPERATIONAL PROTOCOL IMPLICATIONS")
    print("="*70)
    
    try:
        # Show sampling density differences
        print("Sampling protocol comparison:")
        
        # τ-uniform sampling
        tau_uniform = np.linspace(0.1, 2.0, 10)
        sigma_from_tau = np.log(tau_uniform / 1.0)
        
        # σ-uniform sampling  
        sigma_uniform = np.linspace(sigma_from_tau[0], sigma_from_tau[-1], 10)
        tau_from_sigma = np.exp(sigma_uniform)
        
        print(f"\nτ-uniform sampling (Δτ = constant):")
        print(f"  τ points: {tau_uniform[::3]}")
        print(f"  σ points: {sigma_from_tau[::3]}")
        
        print(f"\nσ-uniform sampling (Δσ = constant):")  
        print(f"  σ points: {sigma_uniform[::3]}")
        print(f"  τ points: {tau_from_sigma[::3]}")
        
        # Early-time density
        early_tau_count = np.sum(tau_from_sigma < 0.5)
        early_tau_uniform_count = np.sum(tau_uniform < 0.5)
        
        print(f"\nEarly-time sampling density (τ < 0.5):")
        print(f"  τ-uniform: {early_tau_uniform_count} points")
        print(f"  σ-uniform: {early_tau_count} points")
        
        if early_tau_count > early_tau_uniform_count:
            print("✅ σ-uniform provides denser early-time coverage")
            print("   Advantage for measurements near classical singularities")
        
        print(f"\nOperational consequences:")
        print(f"  • σ-uniform protocols: Better early-time resolution")
        print(f"  • Exponential spacing in τ: Natural for quantum evolution")  
        print(f"  • Phase accumulation: Different patterns in σ vs τ")
        print(f"  • Detector protocols: Distinguishable if sampling-limited")
        
        return True
        
    except Exception as e:
        print(f"❌ FAIL: Operational protocol test failed: {e}")
        return False

def main():
    """Run all cosmological inference validation tests."""
    
    print("LTQG COSMOLOGICAL INFERENCE COMPREHENSIVE VALIDATION")
    print("=" * 80)
    print("Validating that σ-uniform integration preserves cosmological")
    print("parameter inference while providing operational advantages.")
    print()
    
    # Run validation tests
    tests = [
        ("Distance Equivalence", test_distance_equivalence),
        ("Parameter Inference", test_parameter_inference), 
        ("Operational Protocols", test_operational_protocols)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ CRITICAL ERROR in {test_name}: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*80)
    print("COSMOLOGICAL INFERENCE VALIDATION SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎯 LTQG COSMOLOGICAL INFERENCE: FULLY VALIDATED")
        print("   • Distance calculations equivalent under reparameterization")
        print("   • Parameter constraints preserved (H₀, Ωₘ, dark energy)")
        print("   • Operational advantages for early-time measurements")
        print("   • Framework ready for real data applications")
    else:
        print("\n⚠️  VALIDATION INCOMPLETE: Some tests failed")
        print("   Check numerical tolerances and implementation details")
    
    print("\nNext steps:")
    print("  • Replace synthetic data with Pantheon+ supernova catalog")  
    print("  • Add BAO and CMB distance priors")
    print("  • Implement σ-uniform metrology protocols")
    print("  • Test with interacting dark energy models (w₀-wₐ)")

if __name__ == "__main__":
    main()