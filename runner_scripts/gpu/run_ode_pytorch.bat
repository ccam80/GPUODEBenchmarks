@echo off
setlocal enabledelayedexpansion

set a=8
set max_a=%1

REM Activate virtual environment
call GPU_ODE_PyTorch\venv\Scripts\activate.bat

REM Work-precision mode: `run_ode_pytorch.bat wp` sweeps dt at N=32768.
if /i "%~1"=="wp" (
    python GPU_ODE_PyTorch\bench_torchdiffeq.py 32768 wp
    if errorlevel 1 exit /b 1
    call deactivate
    endlocal
    exit /b 0
)

:loop
if %a% gtr %max_a% goto end

REM Print the values
echo No. of trajectories = %a%
python GPU_ODE_PyTorch\bench_torchdiffeq.py %a%
if errorlevel 1 exit /b 1

REM Increment the value
set /a a=%a%*4
goto loop

:end
REM Deactivate virtual environment
call deactivate

endlocal
