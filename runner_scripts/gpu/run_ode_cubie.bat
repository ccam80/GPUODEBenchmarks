@echo off
setlocal enabledelayedexpansion

REM Algorithm filter; "all" runs every algorithm this framework supports.
set "alg=%~2"
if "%alg%"=="" set "alg=all"

REM Activate virtual environment
call GPU_ODE_CUBIE\venv\Scripts\activate.bat

REM Pin cubie to the stock numba-cuda backend. The venv is shared with the
REM CUBIE_MLIR suite and holds both backends, so state it rather than relying
REM on the default (backend is chosen at import time from this env var).
set CUBIE_CUDA_BACKEND=numba-cuda

REM Work-precision mode: `run_ode_cubie.bat wp` sweeps dt/tolerance at N=32768.
if /i "%~1"=="wp" (
    python GPU_ODE_CUBIE\bench_cubie.py 32768 wp %alg%
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
python GPU_ODE_CUBIE\bench_cubie.py %a% %alg%
if errorlevel 1 exit /b 1

REM Increment the value
set /a a=%a%*4
goto loop

:end
REM Deactivate virtual environment
call deactivate

endlocal
