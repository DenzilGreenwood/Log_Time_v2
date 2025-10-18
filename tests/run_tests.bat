@echo off
REM Windows Batch file for running LTQG tests with clean output

echo LTQG Test Suite - Clean Output Mode
echo =====================================

cd /d "%~dp0"

REM Ensure reports directory exists
if not exist "reports" mkdir reports

REM Run the simple test runner
echo.
echo Running comprehensive LTQG validation...
python simple_runner.py

echo.
echo =====================================
echo Test execution complete!
echo.
echo Reports generated:
if exist "reports\simple_test_report.txt" (
    echo   - Text Report: reports\simple_test_report.txt
)
if exist "reports\test_report.html" (
    echo   - HTML Report: reports\test_report.html  
)

echo.
echo Press any key to exit...
pause >nul