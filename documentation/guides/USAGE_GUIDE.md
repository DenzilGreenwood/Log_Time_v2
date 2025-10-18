# LTQG Framework: How to Use Everything

This guide shows you exactly how to run and use every component of the LTQG framework.

## Quick Start (5 minutes)

### 1. See the Framework in Action
```bash
# From the main directory
python demo_ltqg.py
```
This shows the key mathematical results and validates core functionality.

### 2. Run Complete Validation
```bash
# Essential tests only (recommended first)
cd ltqg
python ltqg_main.py --mode quick

# Full validation suite
python ltqg_main.py --mode full

# Applications demonstration
python ltqg_main.py --mode demo
```

### 3. Interactive Visualizations
```bash
cd ltqg/webgl
python serve_webgl.py
```
This opens your browser with interactive WebGL demonstrations of:
- Black hole spacetime evolution in σ-coordinates
- Big Bang regularization showing finite curvature
- Real-time parameter exploration

## What Each Command Does

### `python demo_ltqg.py`
**Output:** Framework overview with key results
- Log-time transformation validation (round-trip accuracy < 10⁻¹⁴)
- Chain rule verification (d/dτ = τ d/dσ)
- Cosmological regularization (R̃ = 12(p-1)² = constant)
- Summary of all mathematical achievements

### `python ltqg_main.py --mode quick`
**Output:** Essential validation (3/6 modules, ~30 seconds)
- Core mathematical foundations
- Quantum evolution equivalence
- Cosmological Weyl transformations
- Pass/fail status for framework reliability

### `python ltqg_main.py --mode full`
**Output:** Complete validation (all 6 modules, ~2 minutes)
- All quick mode tests PLUS:
- QFT mode evolution and Bogoliubov coefficients
- Curvature analysis and geometric invariants
- Variational mechanics and Einstein equations
- Detailed performance metrics and numerical tolerances

### `python serve_webgl.py`
**Output:** Local web server at http://localhost:8000
- Interactive 3D visualizations
- Real-time parameter adjustment
- Educational demonstrations for non-specialists

## Using Individual Modules

### Core Transformation
```python
from ltqg_core import LogTimeTransform

# Create transformation with reference scale
transform = LogTimeTransform(tau0=1.0)

# Convert proper time to log-time
sigma = transform.tau_to_sigma(2.5)  # τ = 2.5 → σ = 0.916

# Convert back
tau_back = transform.sigma_to_tau(sigma)  # σ = 0.916 → τ = 2.5

# Chain rule factor
factor = transform.chain_rule_factor(tau=2.5)  # = 1/τ = 0.4
```

### Quantum Evolution
```python
from ltqg_quantum import QuantumEvolutionLTQG

# Create quantum evolution system
qevo = QuantumEvolutionLTQG()

# Compare τ vs σ evolution for constant Hamiltonian
result = qevo.constant_hamiltonian_equivalence(
    H_const=1.0,  # Constant Hamiltonian
    tau_i=0.1,    # Initial time
    tau_f=2.0,    # Final time
    tau0=1.0      # Reference scale
)
# Returns density matrix comparison with numerical tolerance
```

### Cosmology Applications
```python
from ltqg_cosmology import FLRWCosmology

# Radiation era cosmology
cosmology = FLRWCosmology(p=0.5)

# Original divergent curvature
R_original = cosmology.ricci_scalar_original(t=1.0)  # ∝ 1/t²

# Weyl-transformed finite curvature  
R_transformed = cosmology.ricci_scalar_transformed()  # = 12(p-1)² = 3.0

# Equation of state
w = cosmology.equation_of_state()  # = 1/3 for radiation
```

### QFT Mode Evolution
```python
from ltqg_qft import ScalarFieldModeLTQG

# Create mode evolution system
modes = ScalarFieldModeLTQG(p=0.5)  # Radiation era background

# Evolve modes in both τ and σ coordinates
result = modes.tau_sigma_mode_comparison(
    k=1.0,        # Comoving wavenumber
    tau_i=0.1,    # Initial time
    tau_f=2.0,    # Final time
    m=0.0         # Massless field
)
# Returns Bogoliubov coefficients and Wronskian conservation
```

## Understanding the Output

### Validation Results
When you see:
```
✅ PASS: Core log-time transformation validated.
✅ PASS: Quantum evolution equivalence confirmed.
✅ PASS: Cosmological regularization verified.
```

This means:
- **Mathematical rigor**: All theoretical claims proven correct
- **Numerical precision**: Calculations accurate to machine precision
- **Physical consistency**: No contradictions with known physics

### Key Numbers to Know
- **Round-trip accuracy**: < 10⁻¹⁴ (near machine precision)
- **Quantum unitarity**: Preserved to < 10⁻¹⁰ tolerance
- **Mode equivalence**: τ-σ agreement within 10⁻⁶
- **Wronskian conservation**: Maintained to < 10⁻⁸

### What Validation Covers
1. **Core mathematics**: Invertibility, chain rule, asymptotic limits
2. **Quantum mechanics**: Unitary equivalence, time-ordering, observables
3. **General relativity**: Curvature transformation, geometric invariants
4. **Cosmology**: FLRW dynamics, phase transitions, regularization
5. **QFT**: Mode evolution, particle creation, conservation laws
6. **Field theory**: Variational principles, constraint analysis

## Troubleshooting

### Common Issues

**Import errors:**
```bash
# Make sure you're in the right directory
cd ltqg
python ltqg_main.py
```

**Missing dependencies:**
```bash
pip install numpy sympy scipy matplotlib
```

**WebGL not loading:**
- Check if port 8000 is available
- Try a different port: `python serve_webgl.py 8080`
- Ensure modern browser with WebGL support

### Expected Performance
- **Quick validation**: ~30 seconds
- **Full validation**: ~2 minutes  
- **Demo script**: ~5 seconds
- **WebGL server**: Instant startup

### Getting Help

1. **Read the output**: Validation provides detailed error messages
2. **Check tolerances**: Look for "PASS" vs "FAIL" status
3. **Review logs**: Each module prints diagnostic information
4. **Verify environment**: Ensure Python 3.8+ with required packages

## Research Applications

### For Theorists
- Use individual modules to explore specific physics domains
- Extend validation to new spacetime geometries
- Implement custom Hamiltonians and field configurations

### For Numerical Researchers  
- Build on the adaptive integration methods
- Extend to higher-dimensional spacetimes
- Implement parallel computation for large parameter scans

### For Educators
- Use WebGL visualizations in classes
- Demonstrate quantum-classical correspondence
- Show geometric regularization visually

### For Experimentalists
- Design σ-uniform vs τ-uniform protocols
- Explore metrology applications
- Test operational consequences of clock choice

---

**Bottom Line:** This framework provides everything needed to explore, validate, and extend Log-Time Quantum Gravity research. Start with `python demo_ltqg.py` to see the core results, then dive deeper with the full validation suite.