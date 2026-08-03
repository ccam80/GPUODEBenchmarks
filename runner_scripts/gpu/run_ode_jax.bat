@echo off
setlocal enabledelayedexpansion

REM Algorithm request (issue #29): forwarded to the bench script, which runs
REM every supported algorithm for "all" and skips cleanly when unsupported.
set "alg=%~2"
if "%alg%"=="" set "alg=all"

set a=8
set max_a=%1
set XLA_PYTHON_CLIENT_PREALLOCATE=false

REM Activate virtual environment
call GPU_ODE_JAX\venv\Scripts\activate.bat

REM Work-precision mode: `run_ode_jax.bat wp` sweeps dt/tolerance at N=32768.
if /i "%~1"=="wp" (
    python GPU_ODE_JAX\bench_diffrax.py 32768 wp %alg%
    if errorlevel 1 exit /b 1
    call deactivate
    endlocal
    exit /b 0
)

:loop
if %a% gtr %max_a% goto end

REM Print the values
echo No. of trajectories = %a%
python GPU_ODE_JAX\bench_diffrax.py %a% %alg%
if errorlevel 1 exit /b 1

REM Increment the value
set /a a=%a%*4
goto loop

:end
REM Deactivate virtual environment
call deactivate

endlocal
