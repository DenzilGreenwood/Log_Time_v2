# Log-Time Quantum Gravity (LTQG) Framework

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-passing-green.svg)](tests/)

A mathematically rigorous framework bridging General Relativity and Quantum Mechanics through temporal reparameterization. LTQG introduces the logarithmic time coordinate σ = log(τ/τ₀) to achieve unitary equivalence between quantum evolution descriptions while providing curvature regularization in cosmological contexts.

## 🎯 Quick Start

### Prerequisites
```bash
pip install numpy scipy sympy matplotlib
```

### Basic Usage
```bash
git clone https://github.com/DenzilGreenwood/Log_Time_v2.git
cd Log_Time_v2
python src/ltqg_main.py --mode quick  # Essential validation (~1 min)
```

## 📊 Framework Overview

LTQG is a **reparameterization approach**, not a new physical theory. It preserves all predictions of General Relativity and Quantum Mechanics while providing:

- **Mathematical Rigor**: Exact unitary equivalence with machine precision validation
- **Curvature Regularization**: Finite scalar curvature through Weyl transformations  
- **Computational Advantages**: Improved numerical stability near classical singularities
- **Unified Framework**: Consistent treatment across quantum mechanics, cosmology, and QFT

### Core Mathematical Results

| Component | Key Result | Validation |
|-----------|------------|------------|
| **Core Transform** | σ = log(τ/τ₀) with round-trip accuracy | < 10⁻¹⁴ |
| **Quantum Evolution** | Unitary equivalence: ρ_τ = ρ_σ | < 10⁻¹⁰ |
| **FLRW Cosmology** | Constant curvature: R̃ = 12(p-1)² | Exact |
| **QFT Modes** | Bogoliubov coefficient invariance | < 10⁻⁶ |

## 🏗️ Architecture

```
Log_Time_v2/
├── src/                          # Core implementation
│   ├── ltqg_core.py             # Log-time transformation & asymptotic silence
│   ├── ltqg_quantum.py          # Quantum evolution & unitary equivalence
│   ├── ltqg_cosmology.py        # FLRW dynamics & Weyl regularization
│   ├── ltqg_qft.py              # Quantum field theory on curved spacetime
│   └── ltqg_main.py             # Main validation orchestrator
├── tests/                        # Comprehensive test suite
├── examples/                     # Jupyter notebooks & demonstrations
├── documentation/                # Papers, analysis, and guides
└── formal_paper/                # LaTeX source for academic paper
```

## 🧮 Mathematical Foundation

### Temporal Reparameterization
The logarithmic coordinate transformation:
```
σ = log(τ/τ₀) ↔ τ = τ₀e^σ
```
converts the Schrödinger equation:
```
iℏ ∂_τ ψ = H(τ) ψ  →  iℏ ∂_σ ψ = K(σ) ψ
```
where `K(σ) = τ₀e^σ H(τ₀e^σ)` exhibits asymptotic silence as σ → -∞.

### Cosmological Applications
For FLRW spacetimes with scale factor a(t) = t^p, the Weyl transformation Ω = 1/t yields:
- **Regularized curvature**: R̃ = 12(p-1)² (finite constant)
- **Era classification**: Radiation (R̃=3), Matter (R̃=4/3), Stiff matter (R̃=16/3)
- **Frame caveat**: Curvature regularization in Weyl frame; geodesic completeness requires analysis

## 🚀 Usage Examples

### Core Validation
```python
from src.ltqg_core import validate_log_time_transformation
validate_log_time_transformation()  # Tests round-trip accuracy
```

### Quantum Evolution
```python
from src.ltqg_quantum import validate_unitary_equivalence_constant_H
validate_unitary_equivalence_constant_H()  # Verifies ρ_τ = ρ_σ
```

### Cosmological Analysis
```python
from src.ltqg_cosmology import validate_flrw_curvature_regularization
validate_flrw_curvature_regularization()  # Confirms R̃ = 12(p-1)²
```

## 📈 Validation Results

### Physical Regimes Tested
| Era | p | w | Curvature R̃ | Validation |
|-----|---|---|-------------|------------|
| Radiation | 0.5 | 1/3 | 3.0 | ✅ Exact |
| Matter | 2/3 | 0 | 4/3 | ✅ Exact |
| Stiff | 1/3 | 1 | 16/3 | ✅ Exact |

### Numerical Precision
- **Round-trip accuracy**: < 2×10⁻¹⁵ across 44 orders of magnitude
- **Quantum unitarity**: |U†U - I| < 10⁻¹⁰
- **QFT mode evolution**: Relative error < 10⁻⁶

## ⚠️ Important Limitations

### Conceptual Boundaries
1. **Singularity Resolution**: Curvature regularization does NOT resolve geodesic incompleteness in the original frame
2. **Problem of Time**: Framework sidesteps rather than solves the canonical constraint H̄ψ = 0
3. **Frame Dependence**: Weyl transformation requires matter coupling prescriptions for physical interpretation

### Scope Classification
- ✅ **Sophisticated reparameterization technique**
- ✅ **Powerful computational tool for cosmology**  
- ✅ **Educational framework for quantum-gravitational concepts**
- ❌ **NOT a fundamental theory of quantum gravity**
- ❌ **NOT complete resolution of spacetime singularities**

## 🧪 Testing

### Quick Validation (CI/Make target)
```bash
python tests/simple_test_runner.py  # Essential tests
```

### Comprehensive Test Suite
```bash
python tests/test_ltqg_core.py      # Mathematical foundation
python tests/test_ltqg_quantum.py   # Quantum evolution
python tests/test_ltqg_cosmology.py # Cosmological applications
```

### Expected Output
```
🎉 ALL TESTS PASSED! 🎉
The LTQG framework is working correctly.
```

## 📚 Documentation

- **[Academic Paper](formal_paper/main.pdf)**: Complete mathematical exposition
- **[Usage Guide](documentation/guides/USAGE_GUIDE.md)**: Detailed tutorials
- **[API Reference](documentation/guides/API_REFERENCE.md)**: Function documentation
- **[Mathematical Analysis](documentation/analysis/)**: Rigorous proofs and validations

## 🤝 Contributing

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Test** thoroughly: All tests must pass
4. **Submit** a Pull Request with detailed description

### Development Guidelines
- Maintain mathematical rigor with symbolic verification
- Include comprehensive tests for new features
- Follow existing code structure and documentation standards
- Ensure reproducible results with deterministic testing

## 📄 Citation

If you use this framework in research, please cite:

```bibtex
@software{greenwood2025ltqg,
  title={Log-Time Quantum Gravity Framework: Mathematical Implementation},
  author={Greenwood, Denzil James},
  year={2025},
  url={https://github.com/DenzilGreenwood/Log_Time_v2},
  note={MIT License}
}
```

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/DenzilGreenwood/Log_Time_v2/issues)
- **Discussions**: [GitHub Discussions](https://github.com/DenzilGreenwood/Log_Time_v2/discussions)
- **Email**: [Contact via GitHub profile](https://github.com/DenzilGreenwood)

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Research Impact**: This framework provides valuable computational tools for early universe physics while maintaining intellectual honesty about its scope as a reparameterization technique rather than a fundamental quantum gravity theory.