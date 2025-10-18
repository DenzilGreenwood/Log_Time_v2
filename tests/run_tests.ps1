#!/usr/bin/env pwsh
# run_tests.ps1 - PowerShell script for clean LTQG test execution

Write-Host "LTQG Test Suite - Clean Report Generation" -ForegroundColor Cyan
Write-Host "=" * 50 -ForegroundColor Cyan

# Change to the tests directory
Set-Location $PSScriptRoot

# Ensure reports directory exists
if (!(Test-Path "reports")) {
    New-Item -ItemType Directory -Name "reports" | Out-Null
    Write-Host "Created reports directory" -ForegroundColor Green
}

# Option 1: Run with our custom report runner
Write-Host "`nOption 1: Custom Report Runner" -ForegroundColor Yellow
Write-Host "Running: python report_runner.py" -ForegroundColor Gray

try {
    python report_runner.py
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`n✓ All tests passed with custom runner" -ForegroundColor Green
    } else {
        Write-Host "`n⚠ Some tests failed - check HTML report" -ForegroundColor Yellow
    }
} catch {
    Write-Host "`n✗ Error running custom test runner: $_" -ForegroundColor Red
}

Write-Host "`n" + "-" * 50

# Option 2: Run pytest with clean configuration
Write-Host "`nOption 2: Pytest with Clean Reports" -ForegroundColor Yellow
Write-Host "Running: pytest --quiet --tb=short --html=reports/pytest_report.html --self-contained-html" -ForegroundColor Gray

try {
    pytest --quiet --tb=short --html=reports/pytest_report.html --self-contained-html --disable-warnings
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`n✓ All pytest tests passed" -ForegroundColor Green
    } else {
        Write-Host "`n⚠ Some pytest tests failed" -ForegroundColor Yellow
    }
} catch {
    Write-Host "`n⚠ Pytest not available or error occurred" -ForegroundColor Yellow
    Write-Host "Error: $_" -ForegroundColor Red
}

Write-Host "`n" + "=" * 50

# Show available reports
Write-Host "`nGenerated Reports:" -ForegroundColor Cyan

$reportFiles = @(
    "reports/test_report.html",
    "reports/test_results.json", 
    "reports/pytest_report.html",
    "reports/junit_report.xml"
)

foreach ($reportFile in $reportFiles) {
    if (Test-Path $reportFile) {
        $fullPath = Resolve-Path $reportFile
        $size = (Get-Item $reportFile).Length
        $sizeKB = [math]::Round($size/1024, 1)
        Write-Host "  + $reportFile ($sizeKB KB)" -ForegroundColor Green
        $filePath = $fullPath.Path.Replace('\', '/')
        Write-Host "    file:///$filePath" -ForegroundColor Gray
    } else {
        Write-Host "  - $reportFile (not generated)" -ForegroundColor Gray
    }
}

# Option 3: Run a quick validation test
Write-Host "`n" + "-" * 50
Write-Host "`nOption 3: Quick Validation" -ForegroundColor Yellow

$quickTestScript = @'
import sys
import os

# Add LTQG path
ltqg_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ltqg_core_implementation_python_10_17_25')
sys.path.insert(0, ltqg_dir)

try:
    from ltqg_core import LogTimeTransform
    transform = LogTimeTransform(tau0=1.0)
    
    # Quick test
    tau = 2.0
    sigma = transform.tau_to_sigma(tau)
    tau_back = transform.sigma_to_tau(sigma)
    error = abs(tau_back - tau)
    
    if error < 1e-12:
        print(f'+ Quick validation passed (error: {error:.2e})')
        sys.exit(0)
    else:
        print(f'- Quick validation failed (error: {error:.2e})')
        sys.exit(1)
        
except ImportError as e:
    print(f'! Import error: {e}')
    sys.exit(2)
except Exception as e:
    print(f'- Error: {e}')
    sys.exit(3)
'@

$quickTestScript | Out-File -FilePath "quick_test.py" -Encoding UTF8

try {
    python quick_test.py
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Quick validation confirms LTQG core functionality" -ForegroundColor Green
    } else {
        Write-Host "Quick validation detected issues" -ForegroundColor Yellow
    }
} catch {
    Write-Host "Could not run quick validation" -ForegroundColor Red
}

# Clean up
Remove-Item "quick_test.py" -ErrorAction SilentlyContinue

Write-Host "`n" + "=" * 50
Write-Host "Test execution complete!" -ForegroundColor Cyan
Write-Host "Check the HTML reports for detailed results." -ForegroundColor White