@echo off
setlocal enabledelayedexpansion

REM Algorithm request (issue #29): forwarded to the bench script, which runs
REM every supported algorithm for "all" and skips cleanly when unsupported.
set "alg=%~2"
if "%alg%"=="" set "alg=all"

REM Activate virtual environment
call GPU_ODE_CUBIE_MLIR\venv\Scripts\activate.bat

REM Pin cubie to the MLIR backend (single cubie install, backend chosen at
REM import time via this env var).
set CUBIE_CUDA_BACKEND=mlir

REM Work-precision mode: `run_ode_cubie_mlir.bat wp` sweeps dt/tolerance at N=32768.
if /i "%~1"=="wp" (
    python GPU_ODE_CUBIE_MLIR\bench_cubie_mlir.py 32768 wp %alg%
    if errorlevel 1 exit /b 1
    call deactivate
    endlocal
    exit /b 0
)

set a=8
set max_a=%1

:loop
if %a% gtr %max_a% goto end

echo No. of trajectories = %a%
python GPU_ODE_CUBIE_MLIR\bench_cubie_mlir.py %a% %alg%
if errorlevel 1 exit /b 1

REM Increment the value
set /a a=%a%*4
goto loop

:end
REM Deactivate virtual environment
call deactivate

endlocal
