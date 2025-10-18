"""
Test Utilities for LTQG Test Suite

Common utilities, fixtures, and helper functions for LTQG testing.
"""

import numpy as np
import sympy as sp
import unittest
from typing import Callable, Any, Union, Tuple, List
import warnings

class LTQGTestCase(unittest.TestCase):
    """Base test case class for LTQG tests with common assertions and utilities."""
    
    def setUp(self):
        """Set up common test fixtures."""
        self.tolerance = 1e-10
        self.numerical_tolerance = 1e-6
        self.symbolic_tolerance = 1e-14
        
        # Suppress SymPy warnings during tests
        warnings.filterwarnings("ignore", category=UserWarning, module="sympy")
    
    def assertClose(self, a: Union[float, np.ndarray], b: Union[float, np.ndarray], 
                    tol: float = None, msg: str = ""):
        """Assert two values are close within tolerance."""
        if tol is None:
            tol = self.tolerance
            
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            diff = abs(a - b)
        else:
            diff = np.max(np.abs(np.array(a) - np.array(b)))
            
        self.assertLess(diff, tol, msg or f"Values not close: {a} ≠ {b} (diff={diff}, tol={tol})")
    
    def assertSymbolicEqual(self, expr1: sp.Expr, expr2: sp.Expr, msg: str = ""):
        """Assert two symbolic expressions are equal."""
        diff = sp.simplify(expr1 - expr2)
        self.assertEqual(diff, 0, msg or f"Symbolic expressions not equal: {expr1} ≠ {expr2}")
    
    def assertMatrixClose(self, A: np.ndarray, B: np.ndarray, tol: float = None, msg: str = ""):
        """Assert two matrices are close element-wise."""
        if tol is None:
            tol = self.tolerance
            
        self.assertEqual(A.shape, B.shape, "Matrix shapes must match")
        diff = np.max(np.abs(A - B))
        self.assertLess(diff, tol, msg or f"Matrices not close (max diff={diff}, tol={tol})")
    
    def assertUnitary(self, U: np.ndarray, tol: float = None, msg: str = ""):
        """Assert matrix is unitary: U†U = I."""
        if tol is None:
            tol = self.tolerance
            
        identity = np.eye(U.shape[0])
        product = U.conj().T @ U
        self.assertMatrixClose(product, identity, tol, msg or "Matrix is not unitary")
    
    def assertHermitian(self, H: np.ndarray, tol: float = None, msg: str = ""):
        """Assert matrix is Hermitian: H† = H."""
        if tol is None:
            tol = self.tolerance
            
        self.assertMatrixClose(H, H.conj().T, tol, msg or "Matrix is not Hermitian")
    
    def generate_test_times(self, t_min: float = 0.1, t_max: float = 10.0, 
                           n_points: int = 50) -> np.ndarray:
        """Generate logarithmically spaced test time points."""
        return np.logspace(np.log10(t_min), np.log10(t_max), n_points)
    
    def generate_test_hamiltonians(self) -> List[Tuple[str, Callable[[float], np.ndarray]]]:
        """Generate test Hamiltonians for quantum evolution tests."""
        # Pauli matrices
        sx = np.array([[0., 1.], [1., 0.]], dtype=complex)
        sy = np.array([[0., -1j], [1j, 0.]], dtype=complex)
        sz = np.array([[1., 0.], [0., -1.]], dtype=complex)
        
        test_hamiltonians = [
            ("constant_sz", lambda t: 0.5 * sz),
            ("constant_sx", lambda t: 0.8 * sx),
            ("oscillating", lambda t: 0.5 * sz + 0.3 * np.cos(t) * sx),
            ("time_dependent", lambda t: (1.0 / t) * sz + 0.2 * np.sin(2*t) * sy),
            ("non_commuting", lambda t: 0.7 * sz + 0.4 * np.cos(0.5*t) * sx + 0.1 * np.sin(t) * sy)
        ]
        
        return test_hamiltonians

class TestDataGenerator:
    """Generate test data for various LTQG components."""
    
    @staticmethod
    def flrw_test_cases() -> List[Tuple[str, float]]:
        """Generate FLRW test cases with physical interpretation."""
        return [
            ("radiation", 0.5),
            ("matter", 2.0/3.0),
            ("stiff", 1.0/3.0),
            ("inflation", 2.0),
            ("generic_1", 0.4),
            ("generic_2", 0.8)
        ]
    
    @staticmethod
    def qft_mode_test_cases() -> List[Tuple[str, float, float, float]]:
        """Generate QFT mode test cases: (name, k, m, p)."""
        return [
            ("massless_radiation", 1.0, 0.0, 0.5),
            ("massive_radiation", 1.0, 0.5, 0.5),
            ("massless_matter", 2.0, 0.0, 2.0/3.0),
            ("massive_matter", 2.0, 0.3, 2.0/3.0),
            ("high_k_radiation", 10.0, 0.0, 0.5),
            ("low_k_matter", 0.1, 0.0, 2.0/3.0)
        ]
    
    @staticmethod
    def weyl_transformation_test_cases() -> List[Tuple[str, sp.Expr, Tuple]]:
        """Generate Weyl transformation test cases."""
        t = sp.Symbol('t', positive=True, real=True)
        
        return [
            ("inverse_time", 1/t, (t,)),
            ("power_law", t**(-0.5), (t,)),
            ("exponential", sp.exp(-t), (t,)),
            ("logarithmic", sp.log(t), (t,))
        ]

def skip_if_no_sympy(test_func):
    """Decorator to skip tests if SymPy is not available."""
    def wrapper(*args, **kwargs):
        try:
            import sympy
            return test_func(*args, **kwargs)
        except ImportError:
            raise unittest.SkipTest("SymPy not available")
    return wrapper

def skip_if_no_scipy(test_func):
    """Decorator to skip tests if SciPy is not available."""
    def wrapper(*args, **kwargs):
        try:
            import scipy
            return test_func(*args, **kwargs)
        except ImportError:
            raise unittest.SkipTest("SciPy not available")
    return wrapper

def parametrized_test(test_cases):
    """Decorator to run a test with multiple parameter sets."""
    def decorator(test_func):
        def wrapper(self):
            for case_name, *params in test_cases:
                with self.subTest(case=case_name):
                    test_func(self, case_name, *params)  # Fixed: pass case_name as first argument
        return wrapper
    return decorator

class MockHamiltonian:
    """Mock Hamiltonian for testing purposes."""
    
    def __init__(self, size: int = 2, time_dependent: bool = False):
        self.size = size
        self.time_dependent = time_dependent
        
        # Generate random Hermitian matrix
        np.random.seed(42)  # For reproducibility
        A = np.random.randn(size, size) + 1j * np.random.randn(size, size)
        self.H0 = (A + A.conj().T) / 2
        
        if time_dependent:
            B = np.random.randn(size, size) + 1j * np.random.randn(size, size)
            self.H1 = (B + B.conj().T) / 2
    
    def __call__(self, t: float) -> np.ndarray:
        if self.time_dependent:
            return self.H0 + np.sin(t) * self.H1
        else:
            return self.H0

# Performance testing utilities
class PerformanceTimer:
    """Context manager for timing code execution."""
    
    def __init__(self, name: str = ""):
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        import time
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        self.end_time = time.perf_counter()
    
    @property
    def elapsed(self) -> float:
        """Return elapsed time in seconds."""
        if self.start_time is None or self.end_time is None:
            return 0.0
        return self.end_time - self.start_time

def benchmark_function(func: Callable, *args, n_runs: int = 10, **kwargs) -> Tuple[float, float]:
    """Benchmark a function over multiple runs.
    
    Returns:
        Tuple of (mean_time, std_time) in seconds
    """
    import time
    
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        func(*args, **kwargs)
        end = time.perf_counter()
        times.append(end - start)
    
    return np.mean(times), np.std(times)