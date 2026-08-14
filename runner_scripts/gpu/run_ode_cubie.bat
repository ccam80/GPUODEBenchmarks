@echo off
setlocal enabledelayedexpansion
set "PA_RAW=%*"
call "%~dp0..\parse_args.bat"
if errorlevel 1 exit /b 1

call GPU_ODE_CUBIE\venv\Scripts\activate.bat

REM The venv is shared with the MLIR suite; cubie picks its backend from this at import time.
set CUBIE_CUDA_BACKEND=numba-cuda

if /i "%ANALYSIS%"=="work-precision" (
    python GPU_ODE_CUBIE\bench_cubie.py 131072 wp "%ALGORITHM%"
    if errorlevel 1 exit /b 1
    call deactivate
    endlocal
    exit /b 0
)

for %%a in (!NLIST!) do (
    echo No. of trajectories = %%a
    python GPU_ODE_CUBIE\bench_cubie.py %%a "%ALGORITHM%"
    if !errorlevel! neq 0 exit /b 1
)

call deactivate
endlocal
