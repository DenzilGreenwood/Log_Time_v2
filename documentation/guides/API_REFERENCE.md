# LTQG API Reference

## Core Modules

### `ltqg_core.py` - Fundamental Log-Time Mathematics

#### Classes

##### `LogTimeTransformation`
Mathematical transformation between proper time (τ) and log-time (σ).

```python
class LogTimeTransformation:
    def __init__(self, tau0=1.0):
        """Initialize with reference time tau0."""
        
    def tau_to_sigma(self, tau):
        """Convert proper time to log-time: σ = log(τ/τ₀)"""
        
    def sigma_to_tau(self, sigma):
        """Convert log-time to proper time: τ = τ₀e^σ"""
        
    def validate_round_trip(self, tau_test_values):
        """Validate τ → σ → τ round-trip accuracy"""
```

##### `LTQGConstants`
Physical and mathematical constants for the framework.

```python
class LTQGConstants:
    TAU0_DEFAULT = 1.0      # Reference time scale
    HBAR_DEFAULT = 1.0      # Reduced Planck constant
    C_DEFAULT = 1.0         # Speed of light (natural units)
    TOLERANCE = 1e-10       # Numerical tolerance
```

#### Functions

```python
def validate_log_time_transformation():
    """Comprehensive validation of log-time coordinate transformation."""
    
def validate_asymptotic_silence():
    """Validate asymptotic silence property for physical Hamiltonians."""
```

---

### `ltqg_quantum.py` - Quantum Evolution

#### Classes

##### `QuantumEvolutionLTQG`
Quantum state evolution in both τ and σ coordinates.

```python
class QuantumEvolutionLTQG:
    def __init__(self, tau0=1.0, hbar=1.0):
        """Initialize quantum evolution framework."""
        
    def U_time_ordered_tau(self, H_func, tau_grid):
        """Compute time-ordered evolution operator in τ coordinates."""
        
    def U_time_ordered_sigma(self, H_func, sigma_grid):
        """Compute time-ordered evolution operator in σ coordinates."""
        
    def heisenberg_observable(self, A, U):
        """Evolve observable in Heisenberg picture: A_H = U†AU"""
```

#### Functions

```python
def validate_unitary_equivalence_constant_H():
    """Test unitary equivalence for constant Hamiltonians."""
    
def validate_unitary_equivalence_time_dependent():
    """Test equivalence for time-dependent Hamiltonians."""
    
def validate_heisenberg_observables():
    """Validate observable evolution in Heisenberg picture."""
    
def analyze_quantum_state_evolution(H_func, psi0, tau_final, observables=None):
    """Comprehensive quantum state analysis."""
```

---

### `ltqg_cosmology.py` - Cosmological Applications

#### Classes

##### `FLRWCosmology`
FLRW spacetime with log-time and Weyl transformations.

```python
class FLRWCosmology:
    def __init__(self, p=0.5):
        """Initialize with expansion parameter p (a ∝ t^p)."""
        
    def scale_factor(self, t):
        """FLRW scale factor a(t) = t^p"""
        
    def ricci_scalar(self, t):
        """Original Ricci scalar R(t) = 6p(2p-1)/t²"""
        
    def weyl_ricci_scalar(self):
        """Weyl-regularized scalar R̃ = 12(p-1)²"""
        
    def equation_of_state(self):
        """Equation of state parameter w = 2/(3p) - 1"""
```

#### Functions

```python
def validate_flrw_curvature_regularization():
    """Validate curvature regularization for all cosmological eras."""
    
def validate_cosmological_parameter_inference():
    """Test preservation of cosmological parameter inference."""
    
def analyze_minisuperspace_dynamics():
    """Complete minisuperspace analysis with scalar field."""
```

---

### `ltqg_qft.py` - Quantum Field Theory

#### Classes

##### `QuantumFieldLTQG`
Scalar quantum fields on FLRW backgrounds.

```python
class QuantumFieldLTQG:
    def __init__(self, k=1.0, m=0.0, p=0.5):
        """Initialize with mode k, mass m, expansion parameter p."""
        
    def mode_equation_tau(self, tau, phi, dphi_dtau):
        """Klein-Gordon equation in τ coordinates."""
        
    def mode_equation_sigma(self, sigma, phi, dphi_dsigma):
        """Klein-Gordon equation in σ coordinates."""
        
    def bogoliubov_coefficients(self, phi_tau, phi_sigma):
        """Compute Bogoliubov coefficients between coordinate frames."""
```

#### Functions

```python
def validate_qft_mode_evolution():
    """Validate quantum field mode evolution equivalence."""
    
def validate_wronskian_conservation():
    """Test Wronskian conservation in both coordinate systems."""
    
def validate_bogoliubov_unitarity():
    """Validate |α|² - |β|² = 1 relation."""
```

---

### `ltqg_curvature.py` - Geometric Analysis

#### Classes

##### `CurvatureAnalysisLTQG`
Riemann tensor and curvature invariants for LTQG spacetimes.

```python
class CurvatureAnalysisLTQG:
    def __init__(self, metric_type='FLRW', **params):
        """Initialize with spacetime metric."""
        
    def riemann_tensor(self, coordinates):
        """Compute full Riemann curvature tensor."""
        
    def ricci_tensor(self, coordinates):
        """Compute Ricci tensor R_μν."""
        
    def weyl_tensor(self, coordinates):
        """Compute Weyl conformal tensor."""
        
    def kretschmann_scalar(self, coordinates):
        """Compute Kretschmann scalar K = R_μνρσ R^μνρσ"""
```

---

### `ltqg_variational.py` - Variational Mechanics

#### Functions

```python
def einstein_equations_minisuperspace():
    """Derive Einstein equations in minisuperspace."""
    
def constraint_analysis():
    """Analyze diffeomorphism and Hamiltonian constraints."""
    
def validate_variational_principle():
    """Validate action principle for FLRW + scalar field."""
```

---

## Validation Functions

### Quick Validation
```python
from src.ltqg_main import run_essential_validation
run_essential_validation()  # ~1 minute comprehensive test
```

### Individual Component Testing
```python
from src.ltqg_core import validate_log_time_transformation
from src.ltqg_quantum import validate_unitary_equivalence_constant_H
from src.ltqg_cosmology import validate_flrw_curvature_regularization
from src.ltqg_qft import validate_qft_mode_evolution

# Run individual validations
validate_log_time_transformation()        # Mathematical foundation
validate_unitary_equivalence_constant_H()  # Quantum equivalence
validate_flrw_curvature_regularization()   # Cosmological applications
validate_qft_mode_evolution()             # Quantum field theory
```

## Error Handling

All functions include comprehensive error checking:

```python
try:
    from src.ltqg_core import validate_log_time_transformation
    validate_log_time_transformation()
    print("✅ Validation successful")
except Exception as e:
    print(f"❌ Validation failed: {e}")
```

## Physical Units

The framework uses natural units (ℏ = c = 1) by default. For dimensional analysis:

```python
# Example: Convert to SI units
hbar_SI = 1.054571817e-34  # J⋅s
c_SI = 299792458           # m/s
tau0_SI = 1e-43           # Planck time scale

# Use with dimensional quantities
evolution = QuantumEvolutionLTQG(tau0=tau0_SI, hbar=hbar_SI)
```

## Numerical Precision

Default tolerances are optimized for each physical domain:

- **Mathematical operations**: 1e-14 (machine precision)
- **Quantum evolution**: 1e-10 (unitarity preservation)
- **Cosmological parameters**: 1e-12 (observational precision)
- **QFT modes**: 1e-6 (integration tolerance)

## Performance Considerations

For optimal performance:

1. **Use vectorized operations**: NumPy arrays for large computations
2. **Adaptive integration**: Default RK45 for quantum evolution
3. **Symbolic validation**: SymPy for exact mathematical verification
4. **Caching**: Results cached for repeated calculations

## Dependencies

- **NumPy** ≥ 1.20.0: Numerical arrays and linear algebra
- **SciPy** ≥ 1.7.0: Integration and optimization
- **SymPy** ≥ 1.8.0: Symbolic mathematics
- **Matplotlib** ≥ 3.3.0: Plotting (optional)

## Thread Safety

All core functions are thread-safe for read operations. For parallel computation:

```python
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# Safe for parallel validation
test_cases = [0.5, 2/3, 1/3]  # Different cosmological eras
with ThreadPoolExecutor() as executor:
    results = list(executor.map(validate_era, test_cases))
```