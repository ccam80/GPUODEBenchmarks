@echo off
setlocal enabledelayedexpansion
set "PA_RAW=%*"
call "%~dp0..\parse_args.bat"
if errorlevel 1 exit /b 1

call GPU_ODE_JAX\venv\Scripts\activate.bat

set XLA_PYTHON_CLIENT_PREALLOCATE=false

if /i "%ANALYSIS%"=="work-precision" (
    python GPU_ODE_JAX\bench_diffrax.py 32768 wp "%ALGORITHM%" --problem "%PROBLEM%"
    if errorlevel 1 exit /b 1
    call deactivate
    endlocal
    exit /b 0
)

for %%a in (!NLIST!) do (
    echo No. of trajectories = %%a
    python GPU_ODE_JAX\bench_diffrax.py %%a "%ALGORITHM%" --problem "%PROBLEM%"
    if !errorlevel! neq 0 exit /b 1
)

call deactivate
endlocal
