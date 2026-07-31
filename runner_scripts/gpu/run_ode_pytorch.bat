@echo off
setlocal enabledelayedexpansion
call "%~dp0..\parse_args.bat" %*
if errorlevel 1 exit /b 1

call GPU_ODE_PyTorch\venv\Scripts\activate.bat

REM Fixed-step only: torchdiffeq adaptive solvers are incompatible with torch.vmap.
if /i "%ANALYSIS%"=="work-precision" (
    python GPU_ODE_PyTorch\bench_torchdiffeq.py 32768 wp
    if errorlevel 1 exit /b 1
    call deactivate
    endlocal
    exit /b 0
)

set a=8
:loop
if %a% gtr %NMAX% goto end

echo No. of trajectories = %a%
python GPU_ODE_PyTorch\bench_torchdiffeq.py %a%
if errorlevel 1 exit /b 1

set /a a=%a%*4
goto loop

:end
call deactivate
endlocal
