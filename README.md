# Log-Time Quantum Gravity (LTQG) Framework

A comprehensive mathematical framework for quantum gravity based on log-time reparameterization, providing curvature regularization and unified treatment of quantum mechanics and general relativity.

## Overview

**LTQG is a reparameterization approach, not a new physical theory.** The framework introduces the logarithmic time coordinate σ = log(τ/τ₀) where τ is a scalar field serving as internal time. This provides operational and regularity advantages while preserving all physical predictions.

### Key Features

- **Mathematical Rigor**: Exact unitary equivalence between τ-time and σ-time quantum evolution
- **Curvature Regularization**: Weyl transformations provide finite curvature in cosmological contexts
- **Unified Framework**: Consistent treatment across quantum mechanics, cosmology, and quantum field theory
- **Operational Advantages**: Asymptotic silence and improved numerical stability
- **Reproducible Results**: Deterministic testing with comprehensive validation suite

## Mathematical Summary

### Core Transformation
- **Log-time mapping**: σ = log(τ/τ₀) ⟺ τ = τ₀e^σ
- **Chain rule**: d/dτ = τ d/dσ (exact derivative transformation)
- **Asymptotic silence**: Generators vanish as σ → -∞ with finite total phase

### Unitary Equivalence Theorem
For strongly measurable H(τ) with locally bounded norm satisfying Kato conditions:
```
U_σ(σf,σi) = T exp(-iℏ ∫[σi to σf] H_eff(s) ds) = U_τ(τf,τi)
```
where H_eff(σ) = τ₀e^σ H(τ₀e^σ) and τi,f = τ₀e^(σi,f).

### Cosmological Applications
- **FLRW with Weyl transformation**: R̃ = 12(p-1)² (constant curvature)
- **Corrected equation of state**: w = 2/(3p) - 1 for scale factor a(t) = t^p
- **Minisuperspace formulation**: Unified action with Einstein + scalar field equations

## Quick Start

### Installation Requirements
- Python 3.8+
- NumPy 1.20+
- SciPy 1.7+
- SymPy 1.8+

### Basic Usage

#### Essential Validation (< 1 minute)
```bash
cd Python_Code_Log_Time_Mathematical_Paper
python ltqg_main.py --mode quick
```

#### Complete Framework Validation
```bash
python ltqg_main.py --mode full
```

#### Individual Module Testing
```bash
python ltqg_core.py          # Mathematical foundation
python ltqg_quantum.py       # Quantum evolution
python ltqg_cosmology.py     # FLRW cosmology + Weyl transformations  
python ltqg_qft.py          # Quantum field theory modes
python ltqg_curvature.py    # Riemann tensor computation
python ltqg_variational.py  # Einstein equations + constraints
```

#### Reproducibility Testing
```bash
python test_ltqg_reproducibility.py --mode make  # CI target
python test_ltqg_reproducibility.py --mode full  # Complete reproducibility test
```

## Framework Architecture

```
Python_Code_Log_Time_Mathematical_Paper/
├── ltqg_core.py           # Fundamental log-time transformation & asymptotic silence
├── ltqg_quantum.py        # Unitary equivalence & σ-Schrödinger equation  
├── ltqg_cosmology.py      # FLRW dynamics & Weyl regularization
├── ltqg_qft.py           # Scalar field modes & Bogoliubov analysis
├── ltqg_curvature.py     # Riemann tensor & curvature invariants
├── ltqg_variational.py   # Einstein equations & constraint analysis
├── ltqg_main.py          # Comprehensive validation orchestration
└── test_ltqg_reproducibility.py  # Deterministic testing & CI
```

## Key Validation Results

### 1. Mathematical Foundation ✅
- Log-time transformation is rigorously invertible
- Round-trip accuracy to machine precision (< 2×10^-15)
- Asymptotic silence with explicit L¹ conditions

### 2. Quantum Mechanics ✅  
- Unitary equivalence: ρ_τ = ρ_σ for all physical states
- Time-ordering preserved under coordinate transformation
- Heisenberg picture observables identical in both frames

### 3. Cosmology ✅
- **Radiation era** (p=1/2): w=1/3, R̃=3.0
- **Matter era** (p=2/3): w=0, R̃=1.33
- **Stiff matter** (p=1/3): w=1, R̃=5.33
- Frame dependence warning: Weyl rescaling ≠ diffeomorphism

### 4. Quantum Field Theory ✅
- Bogoliubov coefficients |β_k|² invariant across coordinates
- Wronskian conservation < 10^-8
- Relative error < 10^-5 for all tested modes
- σ-coordinate anti-damping is coordinate effect, not physical creation

## Research Applications

### Early Universe Cosmology
- Curvature regularization near big bang singularity
- Scalar field dark energy with internal time clock
- Phase transitions between radiation/matter/dark energy eras

### Quantum Gravity Models
- Natural time coordinate for quantum gravitational dynamics
- Black hole physics with improved coordinate systems
- Inflation and dark energy phenomenology

### Mathematical Physics
- Asymptotic analysis with regularized evolution equations
- Variational principles with scalar field internal time
- Constraint analysis and canonical formulation

## Figures and Results

### Cosmological Eras Summary
| Era       | p     | w        | H      | ρ(a)   | ρ(t)   | R̃        |
|-----------|-------|----------|--------|--------|--------|----------|
| Radiation | 0.500 | 0.333    | 0.5/t  | a^-4.0 | t^-2.0 | 3.0      |
| Matter    | 0.667 | 0.000    | 0.667/t| a^-3.0 | t^-2.0 | 1.3      |
| Stiff     | 0.333 | 1.000    | 0.333/t| a^-6.0 | t^-2.0 | 5.3      |

### Bogoliubov Cross-Check Results
Cross-coordinate validation shows |β_k|²_τ = |β_k|²_σ within numerical precision:
- Maximum relative error: < 10^-5
- Wronskian conservation: < 10^-8
- Physical particle creation invariant under time coordinate choice

## Testing and Validation

### Make Test Target
```bash
python test_ltqg_reproducibility.py --mode make
# Returns 0 on success, 1 on failure
# Tests essential components: Core + Quantum + Cosmology
```

### Comprehensive Test Suite
```bash
pytest tests/          # Unit tests (when available)
python ltqg_main.py    # Integration validation
```

### Expected Output Patterns
- **Core Foundation**: "PASS: Core log-time transformation validated"
- **Quantum Evolution**: "PASS: Quantum evolution validated"  
- **Cosmology**: "PASS: Cosmological applications validated"
- **Overall**: "Essential tests: 3/3 passed ✅"

## Frame Dependence Warning

⚠️ **Important**: Weyl rescaling g̃_μν = Ω²g_μν is NOT a diffeomorphism. Matter coupling choice (Einstein/Jordan-style) required for observable equivalence. The constant curvature R̃ is a geometric property of the conformal frame, not a gauge redundancy.

## Limitations and Future Work

### Current Limitations
- No back-reaction computation between quantum fields and gravity
- Frame-dependent claims require careful matter coupling prescription
- Focus on minisuperspace models (full field theory extensions planned)

### Research Roadmap
1. **Back-reaction computation**: Calculate ⟨T_μν⟩ and iterate background in σ-coordinate
2. **Near-horizon physics**: Compare σ-slices to Eddington-Finkelstein time near r_s
3. **Full field theory**: Extend beyond minisuperspace to complete quantum field theory

## Citation and Acknowledgments

If you use this framework in research, please cite:
```
LTQG Framework: Log-Time Quantum Gravity Mathematical Implementation
Mathematical Physics Research Group
```

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Contributing

Contributions welcome! Please ensure:
1. All tests pass: `python test_ltqg_reproducibility.py --mode make`
2. Code follows existing mathematical rigor standards
3. New features include validation functions
4. Documentation includes physical interpretation

## Support

For questions, issues, or collaboration:
- Create GitHub issues for bugs/features
- Check validation output for debugging hints
- Review mathematical documentation in module docstrings

**FLRW + Weyl Transformation (Ω = 1/t):**
- Scalar curvature: R̃ = 12(p-1)² (constant)
- Ricci invariant: R̃_{μν}R̃^{μν} = 36(p-1)⁴ (constant)
- Kretschmann scalar: K̃ = 24(p-1)⁴ (constant)
- Einstein condition: E_{μν} = 0 (maximally symmetric spacetime)

**Quantum Mechanics:**
- Perfect unitary equivalence between τ and σ evolution
- Asymptotic silence: lim_{σ→-∞} H_eff(σ) = 0
- Finite accumulated phase: ∫_{-∞}^σ H_eff(s) ds = τ₀e^σ < ∞

## Installation

### Prerequisites
- Python 3.8+
- SymPy for symbolic mathematics
- NumPy for numerical computations
- SciPy for adaptive integration

### Setup
```bash
git clone https://github.com/DenzilGreenwood/Log_Time_v2.git
cd Log_Time_v2
pip install sympy numpy scipy
```

## Usage

### Running the Validation Suite
```bash
cd Code_Log_Time_Mathematical_Paper
python ltqg_validation_updated_extended.py
```

This executes all 11 validation tests and produces a comprehensive report verifying:
- Mathematical consistency of the LTQG framework
- Geometric properties of Weyl-transformed spacetimes
- Quantum mechanical equivalence under time reparameterization
- Numerical validation of theoretical predictions

### Expected Output
The validation suite produces detailed mathematical analysis including:
- Symbolic verification of all theoretical claims
- Numerical confirmation with error bounds
- Geometric analysis of curvature invariants
- Complete diagnostic information

## Mathematical Foundation

### Core Transformation
The fundamental insight is the logarithmic time transformation:
```
σ = log(τ/τ₀)
τ = τ₀e^σ
```

This transforms the Schrödinger equation:
```
iℏ ∂_τ ψ = H(τ) ψ  →  iℏ ∂_σ ψ = τ₀e^σ H(τ₀e^σ) ψ
```

### Weyl Conformal Transformation
Spacetime metrics are conformally transformed:
```
g̃_μν = Ω²g_μν  with  Ω = 1/t
```

This regularizes the FLRW metric, converting:
```
R(t) = 6p(2p-1)/t²  →  R̃ = 12(p-1)²  (constant)
```

### Key Theorems
1. **Invertibility**: The log-time map σ(τ(σ)) = σ is exactly invertible
2. **Unitary Equivalence**: Evolution operators U_τ and U_σ are unitarily equivalent
3. **Einstein Condition**: Weyl-transformed FLRW satisfies R̃_{μν} = (R̃/4)g̃_{μν}
4. **Asymptotic Silence**: Generator vanishes as H_eff(σ→-∞) → 0

## Applications

### Cosmological Quantum Mechanics
- Resolves initial condition problems through asymptotic silence
- Provides finite, well-defined quantum evolution from t=0
- Maintains causal structure while regularizing singularities

### Black Hole Physics
- Extends Schwarzschild analysis to include temporal dynamics
- Provides framework for quantum black hole evolution
- Maintains spherical symmetry under conformal transformations

### Quantum Field Theory
- Validates mode evolution in curved spacetimes
- Provides robust numerical methods for QFT calculations
- Maintains Bogoliubov coefficient relationships

## Technical Details

### Numerical Methods
- **Adaptive RK45**: High-precision integration for quantum evolution
- **Symbolic computation**: Exact mathematical verification using SymPy
- **Error analysis**: Comprehensive error bounds and convergence testing
- **Phase-robust diagnostics**: Wronskian and Bogoliubov coefficient tracking

### Geometric Analysis
- **Ricci tensor computation**: Direct calculation from metric components
- **Curvature invariants**: Proper raised-index contractions
- **Einstein condition**: Systematic verification of maximal symmetry
- **Weyl identity application**: Exact conformal transformation analysis

## Citation

If you use this work in your research, please cite:

```bibtex
@article{greenwood2025ltqg,
  title={Log-Time Quantum Gravity: Mathematical Framework and Validation},
  author={Greenwood, Denzil James},
  journal={arXiv preprint},
  year={2025},
  url={https://github.com/DenzilGreenwood/Log_Time_v2}
}
```

## Contributing

We welcome contributions to the LTQG framework:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- All mathematical claims must be verified in the validation suite
- New features should include comprehensive tests
- Code should be well-documented with mathematical explanations
- Maintain symbolic accuracy alongside numerical precision

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

**Denzil James Greenwood**
- GitHub: [@DenzilGreenwood](https://github.com/DenzilGreenwood)
- Repository: [Log_Time_v2](https://github.com/DenzilGreenwood/Log_Time_v2)

## Acknowledgments

- Mathematical validation using SymPy symbolic computation
- Numerical methods implemented with NumPy and SciPy
- LaTeX documentation for mathematical rigor
- Community feedback and collaboration

---

**Note**: This is a research framework for quantum gravity. While mathematically rigorous, experimental validation remains an open area of investigation.