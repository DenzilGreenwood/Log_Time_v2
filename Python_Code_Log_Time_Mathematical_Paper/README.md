# LTQG - Log-Time Quantum Gravity Framework

A comprehensive, well-structured codebase for Log-Time Quantum Gravity (LTQG) research, providing a modular framework that covers all aspects of this complex mathematical and physical concept.

## Architecture Overview

The codebase is organized into specialized modules, each focusing on a specific aspect of LTQG:

```
LTQG Framework Structure:
├── ltqg_core.py           - Fundamental log-time transformation
├── ltqg_quantum.py        - Quantum mechanical applications  
├── ltqg_cosmology.py      - FLRW cosmology and Weyl transformations
├── ltqg_qft.py           - Quantum field theory mode evolution
├── ltqg_curvature.py     - Riemann tensor and curvature invariants
├── ltqg_variational.py   - Einstein equations and constraints
├── ltqg_main.py          - Comprehensive validation suite
└── README.md             - This documentation
```

## Module Descriptions

### 1. Core Foundation (`ltqg_core.py`)
**The mathematical heart of LTQG**
- Log-time transformation: σ = log(τ/τ₀) ⟺ τ = τ₀e^σ
- Chain rule: d/dτ = τ d/dσ
- Asymptotic silence: generators vanish as σ → -∞
- Mathematical rigor validation and numerical stability

### 2. Quantum Evolution (`ltqg_quantum.py`)
**Quantum mechanics in log-time coordinates**
- σ-Schrödinger equation: iℏ ∂_σ ψ = τ₀e^σ H(τ₀e^σ) ψ
- Time-ordered evolution operators in both τ and σ coordinates
- Unitary equivalence validation
- Heisenberg picture observables

### 3. Cosmology (`ltqg_cosmology.py`)
**Cosmological applications and spacetime geometry**
- FLRW metrics with scale factor a(t) = t^p
- Weyl conformal transformations with Ω = 1/t
- Scalar field minisuperspace models
- Curvature regularization and phase transitions

### 4. Quantum Field Theory (`ltqg_qft.py`)
**QFT mode evolution in expanding spacetimes**
- Scalar field mode equations in τ and σ coordinates
- Robust adaptive numerical integration
- Bogoliubov transformations and particle creation
- Phase-sensitive diagnostics and validation

### 5. Curvature Analysis (`ltqg_curvature.py`)
**Rigorous geometric analysis**
- Complete Riemann tensor computation
- Curvature invariants: Ricci scalar, Kretschmann scalar
- Weyl transformation effects on geometry
- Direct metric analysis without computational shortcuts

### 6. Variational Mechanics (`ltqg_variational.py`)
**Field theory and constraint analysis**
- Einstein tensor and field equations
- Scalar field stress-energy tensors
- Hamiltonian and momentum constraints
- Phase space formulation and conservation laws

### 7. Main Coordination (`ltqg_main.py`)
**Comprehensive validation and integration**
- Complete framework validation suite
- Integration testing across all modules
- Performance monitoring and error reporting
- Research applications demonstration

## Key Features

### Mathematical Rigor
- **Exact transformations**: No approximations in core mathematics
- **Symbolic computation**: SymPy for exact analytical results
- **Numerical validation**: High-precision numerical verification
- **Error handling**: Comprehensive validation and consistency checks

### Physical Applications
- **Early universe cosmology**: Curvature regularization
- **Quantum gravity**: Natural time coordinate systems
- **Black hole physics**: Improved coordinate choices
- **Inflation models**: Dark energy and scalar field dynamics

### Software Quality
- **Modular design**: Clean separation of concerns
- **Comprehensive testing**: Each module validates its core concepts
- **Documentation**: Extensive docstrings and mathematical explanations
- **Type hints**: Modern Python coding standards

## Quick Start

### Run Complete Validation
```bash
python ltqg_main.py
```

### Run Essential Tests Only
```bash
python ltqg_main.py --mode quick
```

### View Applications Demo
```bash
python ltqg_main.py --mode demo
```

### Use Individual Modules
```python
from ltqg_core import LogTimeTransform, validate_log_time_core
from ltqg_quantum import QuantumEvolutionLTQG
from ltqg_cosmology import FLRWCosmology

# Example: Basic log-time transformation
transform = LogTimeTransform(tau0=1.0)
sigma = transform.tau_to_sigma(2.5)  # Convert τ=2.5 to σ
tau_back = transform.sigma_to_tau(sigma)  # Verify invertibility

# Example: FLRW cosmology analysis
cosmology = FLRWCosmology(p=0.5)  # Radiation era
R_original = cosmology.ricci_scalar_original(1.0)
R_transformed = cosmology.ricci_scalar_transformed()
```

## Mathematical Validation

The framework provides comprehensive validation of:

1. **Core Mathematics** (Essential)
   - Log-time transformation invertibility
   - Chain rule exactness
   - Asymptotic silence properties

2. **Quantum Mechanics** (Essential)
   - Unitary equivalence τ ⟺ σ
   - Time-ordering preservation
   - Observable predictions

3. **Cosmological Applications** (Essential)
   - FLRW dynamics
   - Weyl transformation properties
   - Curvature regularization

4. **Advanced Features** (Optional)
   - QFT mode evolution
   - Complete curvature analysis
   - Variational field theory

## Research Applications

### Current Capabilities
- **Cosmological perturbations**: Mode evolution in expanding backgrounds
- **Curvature singularities**: Regularization via Weyl transformations  
- **Quantum cosmology**: Unitary evolution with natural time coordinates
- **Field theory**: Complete Einstein equations with scalar field matter

### Future Extensions
- **Black hole spacetimes**: Schwarzschild and Kerr geometries
- **Quantum gravity**: Loop quantum cosmology applications
- **Phenomenology**: Dark energy and inflation models
- **Numerical relativity**: Advanced integration schemes

## Dependencies

### Required
- `numpy`: Numerical computations
- `sympy`: Symbolic mathematics
- `scipy`: Advanced numerical methods (for QFT module)

### Optional
- `matplotlib`: Plotting and visualization
- `jupyter`: Interactive development

## Installation

```bash
# Clone or download the LTQG codebase
# Install dependencies
pip install numpy sympy scipy

# Run validation
python ltqg_main.py
```

## Contributing

This codebase provides a solid foundation for LTQG research. Contributions welcome in:

- **New physical applications**: Additional spacetime geometries
- **Numerical methods**: Advanced integration techniques  
- **Theoretical extensions**: Higher-dimensional models
- **Computational optimization**: Performance improvements

## Citation

If you use this LTQG framework in research, please cite the underlying mathematical papers and this computational implementation.

## License

Open Source - Available for research and educational use.

---

**Mathematical Physics Research**  
*Log-Time Quantum Gravity Framework*  
*Version 1.0 - Comprehensive Implementation*