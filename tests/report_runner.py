"""
Enhanced Test Runner with Report Generation

This runner provides clean, organized test reports instead of overwhelming CLI output.
"""

import sys
import os
import time
import json
from pathlib import Path
from typing import Dict, List, Any

# Add the LTQG source directory to Python path
current_dir = os.path.dirname(__file__)
ltqg_dir = os.path.join(os.path.dirname(current_dir), 'ltqg_core_implementation_python_10_17_25')
if ltqg_dir not in sys.path:
    sys.path.insert(0, ltqg_dir)

class TestReportGenerator:
    """Generate comprehensive test reports."""
    
    def __init__(self, reports_dir: str = "reports"):
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(exist_ok=True)
        self.results = {
            'summary': {},
            'categories': {},
            'failures': [],
            'errors': [],
            'skipped': [],
            'timing': {},
            'execution_info': {}
        }
    
    def run_test_category(self, category: str, verbose: bool = False) -> Dict[str, Any]:
        """Run tests for a specific category and capture results."""
        print(f"Running {category} tests...")
        
        # Import test modules dynamically to avoid initial import errors
        test_modules = {
            'core': self._test_core_functionality,
            'quantum': self._test_quantum_functionality,
            'cosmology': self._test_cosmology_functionality,
            'qft': self._test_qft_functionality,
            'integration': self._test_integration_functionality
        }
        
        if category in test_modules:
            start_time = time.time()
            try:
                result = test_modules[category]()
                elapsed = time.time() - start_time
                
                self.results['categories'][category] = {
                    'status': 'PASSED' if result['success'] else 'FAILED',
                    'tests_run': result.get('tests_run', 1),
                    'failures': result.get('failures', 0),
                    'errors': result.get('errors', 0),
                    'elapsed_time': elapsed,
                    'details': result.get('details', [])
                }
                
                if not result['success']:
                    self.results['failures'].extend(result.get('failure_details', []))
                
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                self.results['categories'][category] = {
                    'status': 'ERROR',
                    'tests_run': 0,
                    'failures': 0,
                    'errors': 1,
                    'elapsed_time': elapsed,
                    'details': [f"Category error: {str(e)}"]
                }
                self.results['errors'].append(f"{category}: {str(e)}")
                return {'success': False, 'error': str(e)}
        else:
            raise ValueError(f"Unknown test category: {category}")
    
    def _test_core_functionality(self) -> Dict[str, Any]:
        """Test core LTQG functionality."""
        tests_passed = 0
        tests_total = 0
        details = []
        
        try:
            from ltqg_core import LogTimeTransform, validate_log_time_core
            tests_total += 1
            
            # Test 1: Basic transformation
            transform = LogTimeTransform(tau0=1.0)
            tau_values = [0.1, 1.0, 5.0, 10.0]
            max_error = 0
            
            for tau in tau_values:
                sigma = transform.tau_to_sigma(tau)
                tau_back = transform.sigma_to_tau(sigma)
                error = abs(tau_back - tau)
                max_error = max(max_error, error)
            
            if max_error < 1e-12:
                tests_passed += 1
                details.append("[PASS] Log-time transformation round-trip test")
            else:
                details.append(f"[FAIL] Round-trip error: {max_error}")
            
            tests_total += 1
            
            # Test 2: Chain rule validation
            chain_errors = []
            for tau in tau_values:
                factor = transform.chain_rule_factor(tau=tau)
                expected = 1.0 / tau
                error = abs(factor - expected)
                chain_errors.append(error)
            
            if max(chain_errors) < 1e-12:
                tests_passed += 1
                details.append("[PASS] Chain rule validation")
            else:
                details.append(f"[FAIL] Chain rule max error: {max(chain_errors)}")
            
            tests_total += 1
            
            # Test 3: Symbolic validation
            if transform.validate_invertibility():
                tests_passed += 1
                details.append("[PASS] Symbolic invertibility validation")
            else:
                details.append("[FAIL] Symbolic validation failed")
            
            return {
                'success': tests_passed == tests_total,
                'tests_run': tests_total,
                'failures': tests_total - tests_passed,
                'details': details
            }
            
        except Exception as e:
            return {
                'success': False,
                'tests_run': tests_total,
                'failures': tests_total - tests_passed,
                'errors': 1,
                'details': details + [f"Error: {str(e)}"]
            }
    
    def _test_quantum_functionality(self) -> Dict[str, Any]:
        """Test quantum evolution functionality."""
        tests_passed = 0
        tests_total = 0
        details = []
        
        try:
            from ltqg_quantum import QuantumEvolutionLTQG
            import numpy as np
            
            evolution = QuantumEvolutionLTQG()
            tests_total += 1
            
            # Test 1: Unitary evolution
            H = np.array([[1., 0.], [0., -1.]], dtype=complex)
            def H_func(tau): return H
            
            tau_grid = np.linspace(1.0, 1.5, 21)
            U = evolution.U_time_ordered_tau(H_func, tau_grid)
            
            # Check unitarity
            identity = np.eye(2)
            product = U.conj().T @ U
            unitarity_error = np.max(np.abs(product - identity))
            
            if unitarity_error < 1e-10:
                tests_passed += 1
                details.append("✓ Unitary evolution validation")
            else:
                details.append(f"✗ Unitarity error: {unitarity_error}")
            
            tests_total += 1
            
            # Test 2: Coordinate equivalence
            sigma_grid = evolution.transform.tau_to_sigma(tau_grid)
            U_sigma = evolution.U_time_ordered_sigma(H_func, sigma_grid)
            
            # Test on basis states
            psi0 = np.array([1.+0j, 0.+0j])
            psi_tau = U @ psi0
            psi_sigma = U_sigma @ psi0
            
            rho_tau = np.outer(psi_tau, np.conjugate(psi_tau))
            rho_sigma = np.outer(psi_sigma, np.conjugate(psi_sigma))
            
            equiv_error = np.max(np.abs(rho_tau - rho_sigma))
            
            if equiv_error < 1e-6:
                tests_passed += 1
                details.append("✓ Coordinate equivalence validation")
            else:
                details.append(f"✗ Equivalence error: {equiv_error}")
            
            return {
                'success': tests_passed == tests_total,
                'tests_run': tests_total,
                'failures': tests_total - tests_passed,
                'details': details
            }
            
        except Exception as e:
            return {
                'success': False,
                'tests_run': tests_total,
                'failures': tests_total - tests_passed,
                'errors': 1,
                'details': details + [f"Error: {str(e)}"]
            }
    
    def _test_cosmology_functionality(self) -> Dict[str, Any]:
        """Test cosmology functionality."""
        tests_passed = 0
        tests_total = 0
        details = []
        
        try:
            from ltqg_cosmology import FLRWCosmology
            
            # Test standard eras
            eras = [
                ("radiation", 0.5, 1.0/3.0),
                ("matter", 2.0/3.0, 0.0),
                ("stiff", 1.0/3.0, 1.0)
            ]
            
            for era_name, p, expected_w in eras:
                tests_total += 1
                
                cosmology = FLRWCosmology(p)
                eos = cosmology.equation_of_state()
                
                w_error = abs(eos['w'] - expected_w)
                
                if w_error < 1e-10:
                    tests_passed += 1
                    details.append(f"✓ {era_name} era equation of state")
                else:
                    details.append(f"✗ {era_name} w error: {w_error}")
            
            tests_total += 1
            
            # Test Weyl transformation
            radiation = FLRWCosmology(0.5)
            R_tilde = radiation.ricci_scalar_transformed()
            expected_R_tilde = 12 * (0.5 - 1)**2
            
            if abs(R_tilde - expected_R_tilde) < 1e-10:
                tests_passed += 1
                details.append("✓ Weyl transformation validation")
            else:
                details.append(f"✗ Weyl transform error: {abs(R_tilde - expected_R_tilde)}")
            
            return {
                'success': tests_passed == tests_total,
                'tests_run': tests_total,
                'failures': tests_total - tests_passed,
                'details': details
            }
            
        except Exception as e:
            return {
                'success': False,
                'tests_run': tests_total,
                'failures': tests_total - tests_passed,
                'errors': 1,
                'details': details + [f"Error: {str(e)}"]
            }
    
    def _test_qft_functionality(self) -> Dict[str, Any]:
        """Test QFT functionality."""
        tests_passed = 0
        tests_total = 0
        details = []
        
        try:
            from ltqg_qft import QFTModeEvolution, AdaptiveIntegrator
            
            tests_total += 1
            
            # Test mode evolution
            mode = QFTModeEvolution(k=1.0, m=0.0, p=0.5)
            integrator = AdaptiveIntegrator()
            
            t_i, t_f = 1.0, 2.0
            u0, u_dot0 = mode.initial_conditions_adiabatic(t_i)
            
            result = integrator.integrate_mode_tau(mode, (t_i, t_f), (u0, u_dot0))
            
            if result['success'] and len(result['u']) > 0:
                details.append("✓ QFT mode evolution integration")
                tests_passed += 1
            else:
                details.append("✗ QFT integration failed")
            
            tests_total += 1
            
            # Test coordinate equivalence (simplified)
            s_i = mode.transform.tau_to_sigma(t_i)
            s_f = mode.transform.tau_to_sigma(t_f)
            w0 = t_i * u_dot0
            
            result_sigma = integrator.integrate_mode_sigma(mode, (s_i, s_f), (u0, w0))
            
            if result_sigma['success']:
                details.append("✓ QFT sigma-coordinate evolution")
                tests_passed += 1
            else:
                details.append("✗ QFT sigma evolution failed")
            
            return {
                'success': tests_passed == tests_total,
                'tests_run': tests_total,
                'failures': tests_total - tests_passed,
                'details': details
            }
            
        except Exception as e:
            return {
                'success': False,
                'tests_run': tests_total,
                'failures': tests_total - tests_passed,
                'errors': 1,
                'details': details + [f"Error: {str(e)}"]
            }
    
    def _test_integration_functionality(self) -> Dict[str, Any]:
        """Test cross-module integration."""
        tests_passed = 0
        tests_total = 0
        details = []
        
        try:
            from ltqg_core import LogTimeTransform
            from ltqg_quantum import QuantumEvolutionLTQG
            from ltqg_cosmology import FLRWCosmology
            
            tests_total += 1
            
            # Test coordinate consistency across modules
            tau0 = 2.0
            
            core_transform = LogTimeTransform(tau0)
            quantum_evolution = QuantumEvolutionLTQG(tau0=tau0)
            
            test_times = [1.0, 2.0, 5.0]
            max_error = 0
            
            for tau in test_times:
                sigma_core = core_transform.tau_to_sigma(tau)
                sigma_quantum = quantum_evolution.transform.tau_to_sigma(tau)
                error = abs(sigma_core - sigma_quantum)
                max_error = max(max_error, error)
            
            if max_error < 1e-14:
                tests_passed += 1
                details.append("✓ Cross-module coordinate consistency")
            else:
                details.append(f"✗ Coordinate inconsistency: {max_error}")
            
            tests_total += 1
            
            # Test parameter consistency
            p = 0.5
            flrw = FLRWCosmology(p)
            
            t_test = 2.0
            a_flrw = flrw.scale_factor(t_test)
            H_flrw = flrw.hubble_parameter(t_test)
            
            # These should match mathematical expectations
            expected_a = t_test**p
            expected_H = p / t_test
            
            a_error = abs(a_flrw - expected_a)
            H_error = abs(H_flrw - expected_H)
            
            if a_error < 1e-12 and H_error < 1e-12:
                tests_passed += 1
                details.append("✓ Cosmological parameter consistency")
            else:
                details.append(f"✗ Parameter errors: a={a_error}, H={H_error}")
            
            return {
                'success': tests_passed == tests_total,
                'tests_run': tests_total,
                'failures': tests_total - tests_passed,
                'details': details
            }
            
        except Exception as e:
            return {
                'success': False,
                'tests_run': tests_total,
                'failures': tests_total - tests_passed,
                'errors': 1,
                'details': details + [f"Error: {str(e)}"]
            }
    
    def generate_html_report(self) -> str:
        """Generate HTML test report."""
        total_tests = sum(cat.get('tests_run', 0) for cat in self.results['categories'].values())
        total_failures = sum(cat.get('failures', 0) for cat in self.results['categories'].values())
        total_errors = sum(cat.get('errors', 0) for cat in self.results['categories'].values())
        total_passed = total_tests - total_failures - total_errors
        
        success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>LTQG Test Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .summary {{ margin: 20px 0; }}
        .category {{ margin: 15px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        .passed {{ background-color: #d4edda; }}
        .failed {{ background-color: #f8d7da; }}
        .error {{ background-color: #fff3cd; }}
        .details {{ margin-top: 10px; font-size: 0.9em; }}
        .metric {{ display: inline-block; margin: 0 20px; }}
        pre {{ background-color: #f8f9fa; padding: 10px; border-radius: 3px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>LTQG Framework Test Report</h1>
        <p>Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="summary">
        <h2>Test Summary</h2>
        <div class="metric"><strong>Total Tests:</strong> {total_tests}</div>
        <div class="metric"><strong>Passed:</strong> {total_passed}</div>
        <div class="metric"><strong>Failed:</strong> {total_failures}</div>
        <div class="metric"><strong>Errors:</strong> {total_errors}</div>
        <div class="metric"><strong>Success Rate:</strong> {success_rate:.1f}%</div>
    </div>
    
    <h2>Test Categories</h2>
"""
        
        for category, result in self.results['categories'].items():
            status_class = result['status'].lower()
            html_content += f"""
    <div class="category {status_class}">
        <h3>{category.title()} Tests - {result['status']}</h3>
        <p><strong>Tests Run:</strong> {result['tests_run']} | 
           <strong>Failures:</strong> {result['failures']} | 
           <strong>Errors:</strong> {result['errors']} | 
           <strong>Time:</strong> {result['elapsed_time']:.2f}s</p>
        <div class="details">
"""
            
            for detail in result.get('details', []):
                # Clean detail text for HTML display
                clean_detail = detail.replace('✓', '[PASS]').replace('✗', '[FAIL]')
                html_content += f"            <div>{clean_detail}</div>\n"
            
            html_content += """        </div>
    </div>
"""
        
        html_content += """
</body>
</html>
"""
        
        report_path = self.reports_dir / "test_report.html"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return str(report_path)
    
    def generate_json_report(self) -> str:
        """Generate JSON test report."""
        self.results['execution_info'] = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_tests': sum(cat.get('tests_run', 0) for cat in self.results['categories'].values()),
            'total_time': sum(cat.get('elapsed_time', 0) for cat in self.results['categories'].values())
        }
        
        report_path = self.reports_dir / "test_results.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2)
        
        return str(report_path)
    
    def print_summary(self):
        """Print a clean summary to console."""
        print("\n" + "="*80)
        print("LTQG TEST SUMMARY")
        print("="*80)
        
        total_tests = sum(cat.get('tests_run', 0) for cat in self.results['categories'].values())
        total_failures = sum(cat.get('failures', 0) for cat in self.results['categories'].values())
        total_errors = sum(cat.get('errors', 0) for cat in self.results['categories'].values())
        total_passed = total_tests - total_failures - total_errors
        
        print(f"Tests Run: {total_tests}")
        print(f"Passed: {total_passed}")
        print(f"Failed: {total_failures}")
        print(f"Errors: {total_errors}")
        print(f"Success Rate: {(total_passed/total_tests*100):.1f}%" if total_tests > 0 else "N/A")
        
        print("\nCategory Results:")
        for category, result in self.results['categories'].items():
            status_icon = "✓" if result['status'] == 'PASSED' else "✗"
            print(f"  {status_icon} {category.title()}: {result['status']} "
                  f"({result['tests_run']} tests, {result['elapsed_time']:.2f}s)")
        
        if self.results['failures'] or self.results['errors']:
            print("\nIssues Found:")
            for failure in self.results['failures']:
                print(f"  ✗ {failure}")
            for error in self.results['errors']:
                print(f"  ⚠ {error}")
        
        print("="*80)

def main():
    """Main test runner with report generation."""
    print("LTQG Test Suite with Report Generation")
    print("="*50)
    
    reporter = TestReportGenerator()
    
    # Test categories to run
    categories = ['core', 'quantum', 'cosmology', 'qft', 'integration']
    
    # Run tests for each category
    for category in categories:
        try:
            result = reporter.run_test_category(category)
            status = "✓" if result.get('success', False) else "✗"
            print(f"{status} {category.title()} tests completed")
        except Exception as e:
            print(f"✗ {category.title()} tests failed: {e}")
    
    # Generate reports
    print("\nGenerating reports...")
    
    html_path = reporter.generate_html_report()
    json_path = reporter.generate_json_report()
    
    print(f"HTML Report: {html_path}")
    print(f"JSON Report: {json_path}")
    
    # Print summary
    reporter.print_summary()
    
    # Determine exit code
    total_failures = sum(cat.get('failures', 0) for cat in reporter.results['categories'].values())
    total_errors = sum(cat.get('errors', 0) for cat in reporter.results['categories'].values())
    
    if total_failures > 0 or total_errors > 0:
        print(f"\n⚠ Tests completed with {total_failures} failures and {total_errors} errors")
        print(f"Check the HTML report for details: {html_path}")
        return 1
    else:
        print("\n🎉 All tests passed! Check the HTML report for details.")
        return 0

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)