"""
LTQG Test Suite

This package contains comprehensive unit tests for the Log-Time Quantum Gravity (LTQG) framework.

Test Categories:
- Core: Fundamental log-time transformations and utilities
- Quantum: Quantum evolution and unitary equivalence
- Cosmology: FLRW spacetimes and Weyl transformations
- Curvature: Differential geometry and tensor calculations
- QFT: Quantum field theory mode evolution
- Variational: Field equations and constraint analysis
- Integration: End-to-end integration tests

Author: Mathematical Physics Research
License: Open Source
"""

__version__ = "1.0.0"
__author__ = "Mathematical Physics Research Team"

# Test configuration
import sys
import os

# Add the ltqg_core_implementation_python_10_17_25 directory to Python path
LTQG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ltqg_core_implementation_python_10_17_25')
if LTQG_DIR not in sys.path:
    sys.path.insert(0, LTQG_DIR)

# Test utilities will be imported by individual test modules as needed