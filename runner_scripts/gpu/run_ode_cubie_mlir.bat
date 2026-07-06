@echo off
setlocal enabledelayedexpansion

REM Activate virtual environment
call GPU_ODE_CUBIE_MLIR\venv\Scripts\activate.bat

set a=8
set max_a=%1

:loop
if %a% gtr %max_a% goto end

echo No. of trajectories = %a%
python GPU_ODE_CUBIE_MLIR\bench_cubie_mlir.py %a%
if errorlevel 1 exit /b 1

REM Increment the value
set /a a=%a%*4
goto loop

:end
REM Deactivate virtual environment
call deactivate

endlocal
