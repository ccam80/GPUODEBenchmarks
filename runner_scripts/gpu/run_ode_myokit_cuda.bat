@echo off
setlocal enabledelayedexpansion
call "%~dp0..\parse_args.bat" %*
if errorlevel 1 exit /b 1

call GPU_ODE_MYOKIT_CUDA\venv\Scripts\activate.bat

REM Myokit CUDA exposes float32 forward Euler only, so work-precision is fixed-step.
if /i "%ANALYSIS%"=="work-precision" (
    python GPU_ODE_MYOKIT_CUDA\bench_myokit_cuda.py 32768 wp
    if errorlevel 1 exit /b 1
    call deactivate
    endlocal
    exit /b 0
)

set a=8
:loop
if %a% gtr %NMAX% goto end

echo No. of trajectories = %a%
python GPU_ODE_MYOKIT_CUDA\bench_myokit_cuda.py %a%
if errorlevel 1 exit /b 1

set /a a=%a%*4
goto loop

:end
call deactivate
endlocal
