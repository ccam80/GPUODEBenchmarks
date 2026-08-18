@echo off
setlocal enabledelayedexpansion
set "PA_RAW=%*"
call "%~dp0..\parse_args.bat"
if errorlevel 1 exit /b 1

call GPU_ODE_JAX\venv\Scripts\activate.bat

set XLA_PYTHON_CLIENT_PREALLOCATE=false

if /i "%ANALYSIS%"=="work-precision" (
    python GPU_ODE_JAX\bench_diffrax.py wp "%ALGORITHM%" --problem "%PROBLEM%"
    if errorlevel 1 exit /b 1
    call deactivate
    endlocal
    exit /b 0
)

REM The whole ascending N sweep runs in one process on kernels compiled once.
set "NLIST_CSV=!NLIST: =,!"
if "!NLIST_CSV:~0,1!"=="," set "NLIST_CSV=!NLIST_CSV:~1!"
echo N sweep = !NLIST_CSV!
python GPU_ODE_JAX\bench_diffrax.py "!NLIST_CSV!" "%ALGORITHM%" --problem "%PROBLEM%"
if !errorlevel! neq 0 exit /b 1

call deactivate
endlocal
