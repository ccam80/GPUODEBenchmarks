@echo off
setlocal enabledelayedexpansion
set "PA_RAW=%*"
call "%~dp0..\parse_args.bat"
if errorlevel 1 exit /b 1

call GPU_ODE_CUBIE_MLIR\venv\Scripts\activate.bat

REM Cubie picks its backend from this at import time.
set CUBIE_CUDA_BACKEND=mlir

if /i "%ANALYSIS%"=="work-precision" (
    python GPU_ODE_CUBIE_MLIR\bench_cubie_mlir.py 131072 wp "%ALGORITHM%" --problem "%PROBLEM%"
    if errorlevel 1 exit /b 1
    call deactivate
    endlocal
    exit /b 0
)

for %%a in (!NLIST!) do (
    echo No. of trajectories = %%a
    python GPU_ODE_CUBIE_MLIR\bench_cubie_mlir.py %%a "%ALGORITHM%" --problem "%PROBLEM%"
    if !errorlevel! neq 0 exit /b 1
)

call deactivate
endlocal
