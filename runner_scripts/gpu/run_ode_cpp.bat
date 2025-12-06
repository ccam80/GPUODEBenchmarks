@echo off
REM Batch wrapper for PowerShell script
powershell -ExecutionPolicy Bypass -File "%~dp0run_ode_cpp.ps1" %*
