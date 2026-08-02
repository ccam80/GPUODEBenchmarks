@echo off
setlocal enabledelayedexpansion
call "%~dp0..\parse_args.bat" %*
if errorlevel 1 exit /b 1

call GPU_ODE_CUBIE_MLIR\venv\Scripts\activate.bat

REM Cubie picks its backend from this at import time.
set CUBIE_CUDA_BACKEND=mlir

if /i "%ANALYSIS%"=="work-precision" (
    python GPU_ODE_CUBIE_MLIR\bench_cubie_mlir.py 32768 wp
    if errorlevel 1 exit /b 1
    call deactivate
    endlocal
    exit /b 0
)

set a=8
:loop
if %a% gtr %NMAX% goto end

echo No. of trajectories = %a%
python GPU_ODE_CUBIE_MLIR\bench_cubie_mlir.py %a%
if errorlevel 1 exit /b 1

set /a a=%a%*4
goto loop

:end
call deactivate
endlocal
