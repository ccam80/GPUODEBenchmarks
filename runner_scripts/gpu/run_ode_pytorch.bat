@echo off
setlocal enabledelayedexpansion

set a=8
set max_a=%1

REM Activate virtual environment
call GPU_ODE_PyTorch\venv_torch\Scripts\activate.bat

:loop
if %a% gtr %max_a% goto end

REM Print the values
echo No. of trajectories = %a%
python GPU_ODE_PyTorch\bench_torchdiffeq.py %a%

REM Increment the value
set /a a=%a%*4
goto loop

:end
REM Deactivate virtual environment
call deactivate

endlocal
