@echo off
setlocal enabledelayedexpansion
REM Batch wrapper for PowerShell script; delayed expansion keeps metacharacters in args inert.
set "RAW=%*"
if defined RAW (
    powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_full_dataset.ps1" !RAW!
) else (
    powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_full_dataset.ps1"
)
endlocal & exit /b %errorlevel%
