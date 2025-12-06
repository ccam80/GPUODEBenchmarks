@echo off
setlocal enabledelayedexpansion

REM Activate virtual environment
call GPU_ODE_CUBIE\venv_cubie\Scripts\activate.bat

set a=8
set max_a=%1

:loop
if %a% gtr %max_a% goto end

echo No. of trajectories = %a%
python GPU_ODE_CUBIE\bench_cubie.py %a%

REM Increment the value
set /a a=%a%*4
goto loop

:end
REM Deactivate virtual environment
call deactivate

endlocal
