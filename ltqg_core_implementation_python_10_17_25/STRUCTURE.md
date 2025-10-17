# LTQG Codebase Structure

This document describes the organized structure of the Log-Time Quantum Gravity (LTQG) framework codebase, as outlined in the formal paper.

## Directory Structure

```
ltqg/
├── ltqg_core.py                    # Core mathematical foundations
├── ltqg_quantum.py                 # Quantum mechanical applications
├── ltqg_cosmology.py               # Cosmological dynamics
├── ltqg_qft.py                     # Quantum field theory
├── ltqg_curvature.py               # Curvature analysis
├── ltqg_variational.py             # Variational mechanics
├── ltqg_main.py                    # Validation orchestration
├── ltqg_validation_extended.py     # Extended validation suite
├── webgl/                          # Interactive visualizations
│   ├── ltqg_black_hole_webgl.html
│   ├── ltqg_bigbang_funnel.html
│   └── serve_webgl.py
└── backup_extra_files/             # Additional utility files
    ├── demo_ltqg.py
    ├── dm_fast.py
    ├── ltqg_cosmological_inference.py
    └── test_cosmological_inference.py
```

## Module Descriptions

### Core Framework Modules

- **ltqg_core.py**: Fundamental log-time transformation mathematics, differential calculus, and asymptotic behavior
- **ltqg_quantum.py**: Quantum mechanical evolution in log-time coordinates, unitary equivalence proofs
- **ltqg_cosmology.py**: FLRW spacetime dynamics, Weyl transformations, and curvature regularization
- **ltqg_qft.py**: Quantum field theory mode evolution, Bogoliubov transformations
- **ltqg_curvature.py**: Complete curvature tensor analysis, geometric invariants
- **ltqg_variational.py**: Variational principles, Einstein equations, constraint analysis

### Validation and Testing

- **ltqg_main.py**: Comprehensive validation suite orchestration and framework integration
- **ltqg_validation_extended.py**: Extended test suite with detailed mathematical verifications

### Interactive Components

- **webgl/**: Web-based interactive visualizations for educational and research purposes
  - Black hole visualization with LTQG coordinate systems
  - Big Bang funnel visualization for early universe dynamics
  - Web server for hosting interactive demonstrations

### Additional Utilities

- **backup_extra_files/**: Specialized applications and inference tools
  - Cosmological parameter inference demonstrations
  - Fast distance modulus calculations
  - Extended testing frameworks

## Usage

### Running the Main Validation Suite

```bash
# Complete framework validation
python ltqg_main.py

# Quick essential tests only
python ltqg_main.py --mode quick

# Applications demonstration
python ltqg_main.py --mode demo
```

### Individual Module Testing

```bash
# Test specific components
python -c "from ltqg_core import run_core_validation_suite; run_core_validation_suite()"
python -c "from ltqg_quantum import run_quantum_evolution_validation; run_quantum_evolution_validation()"
```

### Interactive Visualizations

```bash
# Start WebGL server
cd webgl
python serve_webgl.py
# Navigate to http://localhost:8000
```

## Mathematical Architecture

The codebase follows the mathematical structure outlined in the LTQG framework paper:

1. **Foundation**: Log-time coordinate transformation `σ = log(τ/τ₀)`
2. **Quantum Evolution**: Schrödinger equation in σ-coordinates
3. **Cosmological Applications**: FLRW with Weyl regularization
4. **Field Theory**: Mode evolution and particle creation
5. **Geometric Analysis**: Complete curvature tensor computations
6. **Variational Formulation**: Einstein equations and constraints

## Import Structure

The modules are designed with clear dependency hierarchies:

```python
# Core mathematics (no dependencies)
from ltqg_core import *

# Quantum applications (depends on core)
from ltqg_quantum import *

# Cosmological analysis (depends on core)
from ltqg_cosmology import *

# Field theory (depends on core, quantum)
from ltqg_qft import *

# Geometric analysis (independent)
from ltqg_curvature import *

# Variational methods (depends on curvature)
from ltqg_variational import *
```

## Computational Validation

Each module includes comprehensive validation:
- Analytical result verification
- Numerical accuracy testing
- Cross-module consistency checks
- Performance benchmarking
- Mathematical identity validation

The framework ensures all physical predictions are coordinate-invariant while providing computational advantages through the log-time parameterization.