# LTQG Test Suite

A comprehensive unit test suite for the Log-Time Quantum Gravity (LTQG) framework.

## Overview

This test suite provides rigorous validation of all LTQG mathematical components with multiple testing approaches:

- **Unit Tests**: Individual component testing
- **Integration Tests**: Cross-module consistency validation  
- **Performance Tests**: Efficiency and scalability benchmarks
- **Physical Validation**: Conservation laws and physics consistency

## Test Structure

```
tests/
├── __init__.py              # Test package initialization
├── test_utils.py           # Common testing utilities and fixtures
├── test_ltqg_core.py       # Core transformation and utilities
├── test_ltqg_quantum.py    # Quantum evolution and unitary equivalence
├── test_ltqg_cosmology.py  # FLRW spacetimes and Weyl transformations
├── test_ltqg_qft.py        # Quantum field theory mode evolution
├── test_integration.py     # End-to-end integration tests
├── run_tests.py           # Enhanced test runner with reporting
├── pytest.ini            # pytest configuration
└── README.md             # This file
```

## Test Categories

### Core Tests (`test_ltqg_core.py`)
- **Log-time transformation**: Round-trip accuracy, chain rule validation
- **Asymptotic silence**: Mathematical properties and sufficient conditions
- **Symbolic validation**: SymPy-based exact verification
- **Edge cases**: Extreme parameter values and boundary conditions

### Quantum Tests (`test_ltqg_quantum.py`)
- **Unitary equivalence**: τ ⟺ σ coordinate transformation validation
- **Time-ordered evolution**: Constant and time-dependent Hamiltonians
- **Heisenberg observables**: Physical prediction preservation
- **Geometric phases**: Berry phase preservation under coordinate changes

### Cosmology Tests (`test_ltqg_cosmology.py`)
- **FLRW spacetimes**: Scale factor, Hubble parameter, equation of state
- **Weyl transformations**: Curvature regularization and conformal properties
- **Scalar field minisuperspace**: Variational formulation validation
- **Phase transitions**: Standard cosmological era consistency

### QFT Tests (`test_ltqg_qft.py`)
- **Mode evolution**: τ and σ coordinate equivalence
- **Bogoliubov analysis**: Particle creation and coordinate invariance
- **Numerical stability**: Long-time evolution and high-frequency modes
- **Anti-damping regime**: σ-coordinate effects vs physical phenomena

### Integration Tests (`test_integration.py`)
- **Cross-module consistency**: Parameter synchronization across modules
- **End-to-end scenarios**: Complete physical calculations
- **Performance benchmarks**: Scalability and efficiency tests
- **Error handling**: Robustness and boundary condition management

## Running Tests

### Quick Start
```bash
# Run all tests
cd tests
python run_tests.py

# Run specific category
python run_tests.py --category core
python run_tests.py --category quantum
python run_tests.py --category cosmology
python run_tests.py --category qft
python run_tests.py --category integration

# Quick validation
python run_tests.py --quick

# Performance benchmarks
python run_tests.py --performance
```

### Using unittest directly
```bash
# Run all tests
python -m unittest discover tests

# Run specific test file
python -m unittest tests.test_ltqg_core

# Run specific test class
python -m unittest tests.test_ltqg_core.TestLogTimeTransform

# Run specific test method
python -m unittest tests.test_ltqg_core.TestLogTimeTransform.test_round_trip_transformation
```

### Using pytest (if installed)
```bash
# Install pytest (optional)
pip install pytest pytest-xdist pytest-cov

# Run all tests
pytest tests/

# Run with coverage
pytest --cov=ltqg_core_implementation_python_10_17_25 tests/

# Run in parallel
pytest -n auto tests/

# Run only fast tests
pytest -m "not slow" tests/
```

## Test Utilities

The `test_utils.py` module provides:

### LTQGTestCase Base Class
- **assertClose**: Numerical tolerance comparisons
- **assertSymbolicEqual**: SymPy expression equality
- **assertMatrixClose**: Matrix element-wise comparison
- **assertUnitary**: Unitary matrix validation
- **assertHermitian**: Hermitian matrix validation

### Test Data Generators
- **FLRW test cases**: Standard cosmological eras
- **QFT mode cases**: Various wave numbers and masses
- **Test Hamiltonians**: Quantum system examples
- **Weyl transformation cases**: Conformal factor examples

### Performance Tools
- **PerformanceTimer**: Execution time measurement
- **benchmark_function**: Multi-run performance analysis

### Decorators
- **parametrized_test**: Run tests with multiple parameter sets
- **skip_if_no_sympy**: Conditional symbolic computation tests
- **skip_if_no_scipy**: Conditional numerical integration tests

## Test Configuration

### Tolerances
- **Symbolic tolerance**: `1e-14` (machine precision)
- **Numerical tolerance**: `1e-6` (physics calculations)
- **General tolerance**: `1e-10` (default comparisons)

### Performance Benchmarks
- **Integration timeout**: 10 seconds maximum
- **Function evaluations**: < 10,000 for standard cases
- **Memory usage**: Monitored for large systems

## Expected Results

### Success Criteria
- **Core tests**: 100% pass rate (mathematical exactness)
- **Quantum tests**: > 99% pass rate (numerical precision)
- **Cosmology tests**: 100% pass rate (analytical validation)
- **QFT tests**: > 95% pass rate (challenging numerics)
- **Integration tests**: > 90% pass rate (complex scenarios)

### Known Issues
- **Anti-damped QFT regimes**: May show numerical sensitivity
- **Long-time evolution**: Precision degradation expected
- **High-frequency modes**: Requires fine time resolution
- **Symbolic computation**: May be slow for complex expressions

## Dependencies

### Required
- **NumPy**: Numerical arrays and linear algebra
- **SymPy**: Symbolic mathematics (core validation)
- **SciPy**: Numerical integration (QFT tests)

### Optional
- **pytest**: Enhanced test runner
- **pytest-xdist**: Parallel test execution
- **pytest-cov**: Coverage analysis

## Contributing

### Adding New Tests
1. Follow the naming convention: `test_*.py`
2. Inherit from `LTQGTestCase` for enhanced assertions
3. Use `parametrized_test` decorator for multiple test cases
4. Include docstrings explaining the physics being tested
5. Add appropriate tolerance values for numerical comparisons

### Test Categories
Mark tests with appropriate categories:
- `@unittest.skipIf`: Conditional execution
- `parametrized_test`: Multiple parameter sets
- Performance tests: Include timing assertions

### Validation Standards
- **Mathematical rigor**: Symbolic validation where possible
- **Numerical accuracy**: Appropriate tolerances for physics
- **Physical consistency**: Conservation laws and invariances
- **Edge case handling**: Boundary conditions and extreme values

## Mathematical Validation Approach

The test suite implements a **hierarchical validation strategy**:

1. **Symbolic Level**: Exact mathematical verification using SymPy
2. **Numerical Level**: High-precision floating-point validation
3. **Physical Level**: Conservation laws and physical consistency
4. **Integration Level**: Cross-module coherence and end-to-end scenarios

This ensures that:
- **Theoretical claims** are mathematically proven
- **Numerical implementations** maintain required precision
- **Physical predictions** remain consistent across coordinate systems
- **Software architecture** supports reliable scientific computation

## Continuous Integration

The test suite is designed for:
- **Automated testing**: Complete validation without manual intervention
- **Regression detection**: Catch mathematical or numerical errors
- **Performance monitoring**: Track computational efficiency
- **Documentation validation**: Ensure code matches mathematical claims

Run the full test suite before any commits to ensure mathematical and computational integrity of the LTQG framework.