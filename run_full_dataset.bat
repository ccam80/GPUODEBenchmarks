@echo off
REM Forwards to bench.py, the benchmark entry point; every flag is the same.
python "%~dp0bench.py" %*
exit /b %errorlevel%
