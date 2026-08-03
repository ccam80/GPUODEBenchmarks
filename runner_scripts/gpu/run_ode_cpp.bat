@echo off
setlocal enabledelayedexpansion
REM Batch wrapper for PowerShell script; delayed expansion keeps metacharacters in args inert.
set "RAW=%*"
if defined RAW (
    powershell -ExecutionPolicy Bypass -File "%~dp0run_ode_cpp.ps1" !RAW!
) else (
    powershell -ExecutionPolicy Bypass -File "%~dp0run_ode_cpp.ps1"
)
endlocal & exit /b %errorlevel%
