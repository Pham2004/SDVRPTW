@echo off
setlocal
python "%~dp0python_src\run_benchmark_suite.py" %*
exit /b %errorlevel%
