@echo off
REM Batch wrapper for PowerShell script
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_full_dataset.ps1" %*
exit /b %errorlevel%
