#!/usr/bin/env python3
"""
Debug QFT Import Issues
"""

import sys
import os

# Add the LTQG source directory to Python path
current_dir = os.path.dirname(__file__)
ltqg_dir = os.path.join(os.path.dirname(current_dir), 'ltqg_core_implementation_python_10_17_25')

print(f"Adding to Python path: {ltqg_dir}")
print(f"Directory exists: {os.path.exists(ltqg_dir)}")

if ltqg_dir not in sys.path:
    sys.path.insert(0, ltqg_dir)

print(f"Python path: {sys.path[:3]}...")

try:
    print("Importing ltqg_core...")
    from ltqg_core import LogTimeTransform, LTQGConstants
    print("✓ ltqg_core imported successfully")
    
    print("Importing ltqg_qft...")
    from ltqg_qft import QFTModeEvolution, AdaptiveIntegrator
    print("✓ ltqg_qft imported successfully")
    
    print("Testing QFTModeEvolution...")
    mode = QFTModeEvolution(k=1.0, m=0.0, p=0.5)
    print(f"✓ QFTModeEvolution created: k={mode.k}, m={mode.m}, p={mode.p}")
    
    print("Importing validation functions...")
    from ltqg_qft import (
        validate_qft_mode_evolution_basic,
        validate_qft_robust_integration,
        bogoliubov_cross_check_comprehensive,
        QFTDiagnostics
    )
    print("✓ All QFT validation functions imported successfully")
    
    print("\n🎉 All imports successful!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    
    # List available files
    if os.path.exists(ltqg_dir):
        print(f"\nFiles in {ltqg_dir}:")
        for file in sorted(os.listdir(ltqg_dir)):
            if file.endswith('.py'):
                print(f"  - {file}")
    
    import traceback
    traceback.print_exc()

except Exception as e:
    print(f"❌ Other error: {e}")
    import traceback
    traceback.print_exc()