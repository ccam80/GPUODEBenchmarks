@echo off
setlocal enabledelayedexpansion

set a=8
set max_a=%1
set XLA_PYTHON_CLIENT_PREALLOCATE=false

REM Activate virtual environment
call GPU_ODE_JAX\venv\Scripts\activate.bat

REM Work-precision mode: `run_ode_jax.bat wp` sweeps dt/tolerance at N=32768.
if /i "%~1"=="wp" (
    python GPU_ODE_JAX\bench_diffrax.py 32768 wp
    if errorlevel 1 exit /b 1
    call deactivate
    endlocal
    exit /b 0
)

:loop
if %a% gtr %max_a% goto end

REM Print the values
echo No. of trajectories = %a%
python GPU_ODE_JAX\bench_diffrax.py %a%
if errorlevel 1 exit /b 1

REM Increment the value
set /a a=%a%*4
goto loop

:end
REM Deactivate virtual environment
call deactivate

endlocal
