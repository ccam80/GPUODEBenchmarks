@echo off
setlocal enabledelayedexpansion

call GPU_ODE_MYOKIT_CUDA\venv\Scripts\activate.bat

REM Myokit CUDA exposes float32 forward Euler only. Its work-precision mode
REM therefore writes only the fixed-step sweep.
if /i "%~1"=="wp" (
    python GPU_ODE_MYOKIT_CUDA\bench_myokit_cuda.py 32768 wp
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
python GPU_ODE_MYOKIT_CUDA\bench_myokit_cuda.py %a%
if errorlevel 1 exit /b 1

set /a a=%a%*4
goto loop

:end
call deactivate
endlocal
