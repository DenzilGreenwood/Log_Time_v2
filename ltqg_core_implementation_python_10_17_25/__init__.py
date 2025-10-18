"""
LTQG (Log-Time Quantum Gravity) Framework

A comprehensive implementation of log-time quantum gravity theory including:
- Core log-time transformations and coordinate systems
- Quantum evolution in both τ and σ coordinates  
- FLRW cosmological models with Weyl transformations
- Quantum field theory mode evolution
- Variational field theory and constraint analysis
- Differential geometry and curvature calculations

This package provides research-grade mathematical tools for theoretical physics
applications in quantum gravity, cosmology, and quantum field theory.

Author: LTQG Development Team
Version: 1.0.0
Date: October 2025
"""

# Core transformation module
from .ltqg_core import (
    LogTimeTransform,
    validate_log_time_core,
    AsymptoticSilenceValidator
)

# Quantum evolution module
try:
    from .ltqg_quantum import (
        QuantumEvolutionLTQG,
        validate_quantum_equivalence
    )
except ImportError:
    # Handle gracefully if dependencies are missing
    QuantumEvolutionLTQG = None
    validate_quantum_equivalence = None

# Cosmology module
try:
    from .ltqg_cosmology import (
        FLRWCosmology,
        validate_cosmology_consistency
    )
except ImportError:
    FLRWCosmology = None
    validate_cosmology_consistency = None

# Quantum field theory module
try:
    from .ltqg_qft import (
        QFTModeEvolution,
        AdaptiveIntegrator,
        validate_qft_evolution
    )
except ImportError:
    QFTModeEvolution = None
    AdaptiveIntegrator = None
    validate_qft_evolution = None

# Variational mechanics module
try:
    from .ltqg_variational import (
        VariationalFieldTheory,
        validate_variational_consistency
    )
except ImportError:
    VariationalFieldTheory = None
    validate_variational_consistency = None

# Differential geometry module
try:
    from .ltqg_curvature import (
        SymbolicCurvature,
        validate_curvature_calculations
    )
except ImportError:
    SymbolicCurvature = None
    validate_curvature_calculations = None

# Extended validation tools
try:
    from .ltqg_extended_validation import (
        ExtendedValidator,
        ComprehensiveTestSuite
    )
except ImportError:
    ExtendedValidator = None
    ComprehensiveTestSuite = None

# Frame analysis tools
try:
    from .ltqg_frame_analysis import (
        FrameAnalyzer,
        validate_frame_consistency
    )
except ImportError:
    FrameAnalyzer = None
    validate_frame_consistency = None

# Geodesic calculations
try:
    from .ltqg_geodesics import (
        GeodesicCalculator,
        validate_geodesic_equations
    )
except ImportError:
    GeodesicCalculator = None
    validate_geodesic_equations = None

# Deparameterization tools
try:
    from .ltqg_deparameterization import (
        DeparameterizationAnalyzer,
        validate_deparameterization
    )
except ImportError:
    DeparameterizationAnalyzer = None
    validate_deparameterization = None

# Package metadata
__version__ = "1.0.0"
__author__ = "LTQG Development Team"
__email__ = "ltqg@example.com"
__description__ = "Log-Time Quantum Gravity Framework"

# Public API - Main classes and functions
__all__ = [
    # Core functionality (always available)
    'LogTimeTransform',
    'validate_log_time_core',
    'AsymptoticSilenceValidator',
    
    # Quantum mechanics (if available)
    'QuantumEvolutionLTQG',
    'validate_quantum_equivalence',
    
    # Cosmology (if available)
    'FLRWCosmology', 
    'validate_cosmology_consistency',
    
    # Quantum field theory (if available)
    'QFTModeEvolution',
    'AdaptiveIntegrator',
    'validate_qft_evolution',
    
    # Variational mechanics (if available)
    'VariationalFieldTheory',
    'validate_variational_consistency',
    
    # Differential geometry (if available)
    'SymbolicCurvature',
    'validate_curvature_calculations',
    
    # Extended tools (if available)
    'ExtendedValidator',
    'ComprehensiveTestSuite',
    'FrameAnalyzer',
    'validate_frame_consistency',
    'GeodesicCalculator',
    'validate_geodesic_equations',
    'DeparameterizationAnalyzer',
    'validate_deparameterization',
    
    # Package metadata
    '__version__',
    '__author__',
    '__description__'
]

# Remove None values from __all__ for unavailable modules
__all__ = [name for name in __all__ if globals().get(name) is not None]

def get_available_modules():
    """
    Return a dictionary of available LTQG modules and their status.
    
    Returns:
        dict: Module names mapped to availability status
    """
    modules = {
        'core': LogTimeTransform is not None,
        'quantum': QuantumEvolutionLTQG is not None,
        'cosmology': FLRWCosmology is not None,
        'qft': QFTModeEvolution is not None,
        'variational': VariationalFieldTheory is not None,
        'curvature': SymbolicCurvature is not None,
        'extended_validation': ExtendedValidator is not None,
        'frame_analysis': FrameAnalyzer is not None,
        'geodesics': GeodesicCalculator is not None,
        'deparameterization': DeparameterizationAnalyzer is not None
    }
    return modules

def print_module_status():
    """Print the availability status of all LTQG modules."""
    print("LTQG Framework Module Status:")
    print("=" * 40)
    
    modules = get_available_modules()
    for module_name, available in modules.items():
        status = "✓ Available" if available else "✗ Not Available"
        print(f"{module_name:20s}: {status}")
    
    available_count = sum(modules.values())
    total_count = len(modules)
    print(f"\nModules loaded: {available_count}/{total_count}")

def validate_installation():
    """
    Validate the LTQG installation by checking core functionality.
    
    Returns:
        bool: True if installation is valid, False otherwise
    """
    try:
        # Test core functionality
        if LogTimeTransform is None:
            print("ERROR: Core LTQG functionality not available")
            return False
        
        # Quick validation test
        transform = LogTimeTransform(tau0=1.0)
        tau = 2.0
        sigma = transform.tau_to_sigma(tau)
        tau_back = transform.sigma_to_tau(sigma)
        error = abs(tau_back - tau)
        
        if error > 1e-12:
            print(f"ERROR: Core transformation validation failed (error: {error})")
            return False
        
        print("✓ LTQG installation validated successfully")
        return True
        
    except Exception as e:
        print(f"ERROR: Installation validation failed: {e}")
        return False

# Convenience functions for common use cases
def create_log_time_system(tau0=1.0):
    """
    Create a complete log-time coordinate system.
    
    Args:
        tau0 (float): Reference time scale
        
    Returns:
        dict: Dictionary containing available system components
    """
    system = {
        'transform': LogTimeTransform(tau0=tau0)
    }
    
    if QuantumEvolutionLTQG is not None:
        system['quantum'] = QuantumEvolutionLTQG(tau0=tau0)
    
    if FLRWCosmology is not None:
        # Default to radiation-dominated era (p=0.5)
        system['cosmology'] = FLRWCosmology(p=0.5)
    
    return system

def run_comprehensive_validation():
    """
    Run comprehensive validation of all available LTQG modules.
    
    Returns:
        dict: Validation results for each module
    """
    results = {}
    
    # Core validation
    try:
        core_result = validate_log_time_core() if 'validate_log_time_core' in globals() else True
        results['core'] = core_result
    except Exception as e:
        results['core'] = f"Error: {e}"
    
    # Quantum validation
    if QuantumEvolutionLTQG is not None and validate_quantum_equivalence is not None:
        try:
            quantum_result = validate_quantum_equivalence()
            results['quantum'] = quantum_result
        except Exception as e:
            results['quantum'] = f"Error: {e}"
    
    # Cosmology validation
    if FLRWCosmology is not None and validate_cosmology_consistency is not None:
        try:
            cosmology_result = validate_cosmology_consistency()
            results['cosmology'] = cosmology_result
        except Exception as e:
            results['cosmology'] = f"Error: {e}"
    
    # QFT validation
    if QFTModeEvolution is not None and validate_qft_evolution is not None:
        try:
            qft_result = validate_qft_evolution()
            results['qft'] = qft_result
        except Exception as e:
            results['qft'] = f"Error: {e}"
    
    return results

# Package initialization message (only shown on first import)
import sys
if __name__ != '__main__':
    # Only print during normal import, not during direct execution
    pass
else:
    print(f"LTQG Framework v{__version__} loaded")
    print_module_status()